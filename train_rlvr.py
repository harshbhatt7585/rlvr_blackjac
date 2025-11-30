"""
RLVR Training Script for Blackjack with Gemma

This script implements Reinforcement Learning with Verifiable Rewards (RLVR)
to train a Gemma model to play Blackjack optimally.

RLVR approach:
1. Collect rollouts: Have the model play Blackjack episodes
2. Weight by rewards: Prioritize successful trajectories
3. Fine-tune: Update model on reward-weighted examples
4. Iterate: Repeat to improve performance
"""

import argparse
import logging
import os
import json
import time
import copy
from pathlib import Path

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch.nn as nn
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)



from env import BlackjackEnv


class WeightedTrainer(Trainer):
    """Trainer that applies per-example weights coming from the dataset."""

    def compute_loss(self, model, inputs, return_outputs=False):
        weights = inputs.pop('weight', None)

        outputs = model(**inputs)
        loss = outputs.loss

        if weights is not None and inputs.get('labels') is not None:
            logits = outputs.logits
            labels = inputs['labels']

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            token_loss = token_loss.view(shift_labels.size(0), -1)

            seq_loss = token_loss.mean(dim=1)

            weights = weights.to(seq_loss.device)
            normalizer = weights.sum().clamp_min(1e-8)
            loss = torch.sum(seq_loss * (weights / normalizer))

        return (loss, outputs) if return_outputs else loss


def configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )


@dataclass
class RLVRConfig:
    """Configuration for RLVR training."""

    # Model settings
    model_name: str = "google/gemma-3-1b-it"  # or "google/gemma-7b-it"

    # Training settings
    num_iterations: int = 10
    episodes_per_iteration: int = 100
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs_per_iteration: int = 1

    # RLVR settings
    temperature: float = 0.7
    reward_threshold: float = 0.0  # Only train on episodes with reward >= this
    advantage_weighting: bool = True  # Weight examples by advantage

    # LoRA settings
    use_lora: bool = True  # Use LoRA for efficient fine-tuning
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling factor)
    lora_dropout: float = 0.05  # LoRA dropout
    lora_target_modules: Optional[List[str]] = None  # Auto-detect if None

    # Environment settings
    natural_reward: float = 1.5
    seed: int = 42

    # Logging
    output_dir: str = "./checkpoints"
    log_file: str = "./training_log.json"
    eval_frequency: int = 1  # Evaluate every N iterations
    eval_episodes: int = 100

    # Weights & Biases settings
    use_wandb: bool = True  # Enable W&B logging
    wandb_project: str = "blackjack-rlvr"  # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity (username/team)
    wandb_run_name: Optional[str] = None  # Run name (auto-generated if None)
    wandb_tags: Optional[List[str]] = None  # Tags for the run

    # Verbosity
    verbose: bool = False


@dataclass
class Episode:
    """Store a single episode trajectory."""
    states: List[str]  # State descriptions
    actions: List[int]  # Actions taken
    prompts: List[str]  # Full prompts sent to model
    responses: List[str]  # Model responses
    rewards: List[float]  # Intermediate rewards (0 until final)
    total_reward: float  # Final episode reward
    reasonings: List[str]  # Reasoning for each action
    logprobs: List[float]  # Log probabilities of each action
    message_histories: List[List[Dict[str, str]]]  # Conversation context per turn


class RLVRTrainer:
    """RLVR trainer for Blackjack."""

    def __init__(self, config: RLVRConfig):
        self.config = config
        if not logging.getLogger().handlers:
            configure_logging(config.verbose)
        elif config.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        self.logger = logging.getLogger("RLVRTrainer")

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.env = BlackjackEnv(
            natural_reward=config.natural_reward,
            seed=config.seed
        )
        self.logger.info("Loading model: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True  # Required for some models like DeepSeek
        )
        if torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True  # Required for some models like DeepSeek
        ).to(device_map)

        # Configure padding token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.logger.debug("Using unk_token as pad_token: %s", self.tokenizer.pad_token)
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.debug("Using eos_token as pad_token: %s", self.tokenizer.pad_token)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Apply LoRA if enabled
        if config.use_lora:
            self.logger.info("Applying LoRA configuration")

            # Auto-detect target modules if not specified
            if config.lora_target_modules is None:
                config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                self.logger.debug("Using default attention projections (q/k/v/o_proj)")

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                lora_dropout=config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )

            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()


        self.history = []

        os.makedirs(config.output_dir, exist_ok=True)

        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                tags=config.wandb_tags,
                config=asdict(config)
            )
            self.logger.info("W&B logging enabled: %s", wandb.run.url)


    def collect_episode(self, temperature: float = 0.7) -> Episode:
        obs = self.env.reset()

        states: List[str] = []
        actions: List[int] = []
        prompts: List[str] = []
        responses: List[str] = []
        rewards: List[float] = []
        reasonings: List[str] = []
        all_logprobs: List[torch.Tensor] = []
        history_snapshots: List[List[Dict[str, str]]] = []

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a clever Blackjack player."}
        ]

        done = False

        while not done:
            prompt = self.env.get_prompt_for_llm()
            conversation.append({"role": "user", "content": prompt})

            history_snapshot = copy.deepcopy(conversation)
            history_snapshots.append(history_snapshot)

            try:
                formatted = self.tokenizer.apply_chat_template(
                    history_snapshot,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                self.logger.debug("Chat template failed (%s), using simple format", e)
                parts = []
                for message in history_snapshot:
                    role = message['role'].capitalize()
                    parts.append(f"{role}: {message['content']}")
                parts.append("Assistant:")
                formatted = "\n\n".join(parts)

            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=756,
                    temperature=temperature,
                    do_sample=temperature > 0.0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            scores = outputs.scores
            generated_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
            transition_scores = []
            for i in range(len(scores)):
                logprobs = torch.log_softmax(scores[i], dim=-1)
                token_id = generated_ids[:, i].unsqueeze(-1)
                token_logprob = logprobs.gather(-1, token_id).squeeze(-1)
                transition_scores.append(token_logprob)

            logprobs = torch.stack(transition_scores)

            response = self.tokenizer.decode(
                outputs.sequences[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            conversation.append({"role": "assistant", "content": response})

            response_str = response
            try:
                cleaned = response.split('</think>')[-1].strip()
                if cleaned.startswith('```'):
                    cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
                    cleaned = cleaned.rsplit('```', 1)[0] if '```' in cleaned else cleaned

                parsed = json.loads(cleaned.strip())
                action = parsed.get('action')
                reasoning = parsed.get('reasoning')
            except Exception:
                self.logger.debug("Error parsing response: %s", response)
                action = None
                reasoning = None

            if action not in (0, 1):
                self.logger.warning("Invalid action '%s', defaulting to stand (0)", action)
                action = 0

            states.append(obs['description'])
            actions.append(action)
            reasonings.append(reasoning)
            prompts.append(prompt)
            responses.append(response_str)

            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            all_logprobs.append(logprobs)

        total_reward = sum(rewards)

        return Episode(
            states=states,
            actions=actions,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            total_reward=total_reward,
            reasonings=reasonings,
            logprobs=all_logprobs,
            message_histories=history_snapshots
        )

    def collect_rollouts(self, num_episodes: int) -> List[Episode]:

        episodes = []
        total_actions = 0

        self.logger.debug("Collecting %d episodes", num_episodes)
        for _ in tqdm(range(num_episodes)):
            episode = self.collect_episode(temperature=self.config.temperature)
            episodes.append(episode)


        return episodes

    def create_training_dataset(self, episodes: List[Episode]) -> Dataset:
        training_examples = []

        avg_reward = np.mean([ep.total_reward for ep in episodes]) if self.config.advantage_weighting else 0.0

        for episode in episodes:
            # Skip low-reward episodes
            if episode.total_reward < self.config.reward_threshold:
                continue

            # Simple weight calculation: higher reward = higher weight
            if self.config.advantage_weighting:
                weight = max(0.1, episode.total_reward - avg_reward + 1.0)
            else:
                weight = 1.0

            for i, action in enumerate(episode.actions):
                # Skip malformed responses
                if action is None or episode.reasonings[i] is None:
                    continue

                # Create target response
                target = json.dumps({"action": action, "reasoning": episode.reasonings[i]})

                # Build conversation messages
                messages = episode.message_histories[i] + [{"role": "assistant", "content": target}]

                # Format as text
                try:
                    full_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                except Exception:
                    full_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])

                training_examples.append({
                    'text': full_text,
                    'weight': float(weight),
                    'reward': float(episode.total_reward),
                    'action': action
                })

        # Return empty or filled dataset
        if not training_examples:
            return Dataset.from_dict({'text': [], 'weight': [], 'reward': [], 'action': []})

        return Dataset.from_list(training_examples)

    def _log_iteration(self, record: Dict):
        self.history.append(record)

        log_path = self.config.log_file
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, name: str) -> str:
        save_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(save_dir, exist_ok=True)

        if isinstance(self.model, PeftModel):
            self.model.save_pretrained(save_dir)
        else:
            self.model.save_pretrained(save_dir)

        self.tokenizer.save_pretrained(save_dir)
        return save_dir

    def train_on_dataset(self, dataset: Dataset):
        """
        Fine-tune the model on collected dataset.

        Args:
            dataset: Training dataset
        """
        if len(dataset) == 0:
            self.logger.warning("Empty dataset, skipping training")
            return

        self.logger.info("Training on %d examples", len(dataset))

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=512,
                padding=False  # Use dynamic padding via data collator
            )

        columns_to_keep = {"weight"}
        columns_to_remove = [col for col in dataset.column_names if col not in columns_to_keep]

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove
        )

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs_per_iteration,
            per_device_train_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=10,
            logging_steps=5,
            logging_first_step=True,
            save_strategy="no",  # Don't save intermediate checkpoints
            report_to="wandb" if self.config.use_wandb else "none",
            bf16=True,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        train_result = trainer.train()

        if self.config.use_wandb:
            train_metrics = {
                "train/loss": train_result.training_loss,
                "train/learning_rate": self.config.learning_rate,
                "train/num_examples": len(dataset),
            }
            wandb.log(train_metrics)

        return train_result


    def evaluate(self, num_episodes: int = 100) -> Dict:
        self.logger.info("Evaluating on %d episodes", num_episodes)

        episodes = []
        for _ in tqdm(range(num_episodes)):
            # Use lower temperature for evaluation (more deterministic)
            episode = self.collect_episode(temperature=0.3)
            episodes.append(episode)

        # Calculate metrics
        total_rewards = [ep.total_reward for ep in episodes]

        metrics = {
            'mean_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'min_reward': np.min(total_rewards),
            'max_reward': np.max(total_rewards),
            'win_rate': np.mean([r > 0 for r in total_rewards]),
            'lose_rate': np.mean([r < 0 for r in total_rewards]),
            'draw_rate': np.mean([r == 0 for r in total_rewards]),
        }

        return metrics


    def train(self):
        self.model.train()

        for iteration in range(1, self.config.num_iterations + 1):
            self.logger.info("Iteration %d/%d", iteration, self.config.num_iterations)
            episodes = self.collect_rollouts(self.config.episodes_per_iteration)

            if len(episodes) == 0:
                self.logger.warning("No episodes collected, skipping iteration")
                continue

            total_rewards = np.array([ep.total_reward for ep in episodes], dtype=np.float32)
            reward_stats = {
                'reward/mean': float(np.mean(total_rewards)),
                'reward/std': float(np.std(total_rewards)),
                'reward/max': float(np.max(total_rewards)),
                'reward/min': float(np.min(total_rewards)),
                'reward/win_rate': float(np.mean(total_rewards > 0.0)),
            }

            dataset = self.create_training_dataset(episodes)

            print("Dataset", dataset)
            num_examples = len(dataset)

            train_loss = None
            if num_examples > 0:
                train_result = self.train_on_dataset(dataset)
                if train_result is not None:
                    train_loss = float(train_result.training_loss)
            else:
                self.logger.warning("No valid training examples (possible reward threshold filter)")

            eval_metrics = None
            if self.config.eval_frequency > 0 and iteration % self.config.eval_frequency == 0:
                self.model.eval()
                eval_metrics = self.evaluate(self.config.eval_episodes)
                self.model.train()

            iteration_record = {
                'iteration': iteration,
                'episodes': len(episodes),
                'examples': num_examples,
                'train/loss': train_loss,
                **reward_stats
            }

            if eval_metrics is not None:
                iteration_record.update({f"eval/{k}": float(v) for k, v in eval_metrics.items()})

            self._log_iteration(iteration_record)

            checkpoint_path = self._save_checkpoint(f"iter_{iteration:03d}")
            self.logger.info("Saved checkpoint: %s", checkpoint_path)

            if self.config.use_wandb:
                wandb_metrics = {k: v for k, v in iteration_record.items() if v is not None}
                wandb.log(wandb_metrics, step=iteration)

        final_path = self._save_checkpoint("final_model")
        self.logger.info("Final model saved to: %s", final_path)

        if self.config.use_wandb:
            wandb.finish()


def _format_record(record: Dict) -> str:
    iteration = record.get('iteration', '?')
    parts = [f"iter {iteration}"]

    episodes = record.get('episodes')
    if episodes is not None:
        parts.append(f"episodes={episodes}")

    examples = record.get('examples')
    if examples is not None:
        parts.append(f"examples={examples}")

    train_loss = record.get('train/loss')
    if train_loss is not None:
        parts.append(f"train_loss={train_loss:.4f}")

    reward_mean = record.get('reward/mean')
    if reward_mean is not None:
        parts.append(f"reward_mean={reward_mean:.3f}")

    win_rate = record.get('reward/win_rate')
    if win_rate is not None:
        parts.append(f"win_rate={win_rate:.2f}")

    eval_mean = record.get('eval/mean_reward')
    if eval_mean is not None:
        parts.append(f"eval_mean={eval_mean:.3f}")

    return " | ".join(parts)


def watch_training_log(log_file: str, lines: int = 20, follow: bool = False, interval: float = 2.0):
    path = Path(log_file)
    if not path.exists():
        print(f"No log file found at {path}")
        return

    def iter_records():
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    records = list(iter_records())
    if lines > 0:
        records = records[-lines:]

    for record in records:
        print(_format_record(record))

    if not follow:
        return

    with path.open('r', encoding='utf-8') as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(interval)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            print(_format_record(record))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLVR Blackjack Trainer")
    subparsers = parser.add_subparsers(dest='command')

    default_cfg = RLVRConfig()

    train_parser = subparsers.add_parser('train', help='Run RLVR training (default)')
    train_parser.add_argument('--model-name', default=default_cfg.model_name, help='Model checkpoint to fine-tune')
    train_parser.add_argument('--iterations', type=int, default=default_cfg.num_iterations, help='Training iterations')
    train_parser.add_argument('--episodes', type=int, default=default_cfg.episodes_per_iteration, help='Episodes per iteration')
    train_parser.add_argument('--output-dir', default=default_cfg.output_dir, help='Directory for checkpoints')
    train_parser.add_argument('--log-file', default=default_cfg.log_file, help='Path to JSONL training log')
    train_parser.add_argument('--temperature', type=float, default=default_cfg.temperature, help='Sampling temperature for rollouts')
    train_parser.add_argument('--eval-frequency', type=int, default=default_cfg.eval_frequency, help='Evaluate every N iterations (0 to disable)')
    train_parser.add_argument('--eval-episodes', type=int, default=default_cfg.eval_episodes, help='Episodes per evaluation run')
    train_parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    train_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    watch_parser = subparsers.add_parser('watch', help='Watch the training log')
    watch_parser.add_argument('--log-file', default=default_cfg.log_file, help='Path to JSONL training log')
    watch_parser.add_argument('--lines', type=int, default=20, help='Number of recent lines to display')
    watch_parser.add_argument('--follow', action='store_true', help='Follow the log (like tail -f)')
    watch_parser.add_argument('--interval', type=float, default=2.0, help='Refresh interval when following the log')

    parser.set_defaults(command='train')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'watch':
        configure_logging(False)
        watch_training_log(
            log_file=args.log_file,
            lines=args.lines,
            follow=args.follow,
            interval=args.interval
        )
        return

    configure_logging(args.verbose)

    config = RLVRConfig(
        model_name=args.model_name,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        output_dir=args.output_dir,
        log_file=args.log_file,
        temperature=args.temperature,
        eval_frequency=args.eval_frequency,
        eval_episodes=args.eval_episodes,
        use_wandb=not args.no_wandb,
        verbose=args.verbose
    )

    trainer = RLVRTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
