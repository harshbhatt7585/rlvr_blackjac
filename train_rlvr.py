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

import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import re
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)



from env import BlackjackEnv


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


class RLVRTrainer:
    """RLVR trainer for Blackjack."""

    def __init__(self, config: RLVRConfig):
        self.config = config

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.env = BlackjackEnv(
            natural_reward=config.natural_reward,
            seed=config.seed
        )
        print(f"Loading model: {config.model_name}")
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
                print(f"  Using unk_token as pad_token: {self.tokenizer.pad_token}")
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"  Using eos_token as pad_token: {self.tokenizer.pad_token}")
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Apply LoRA if enabled
        if config.use_lora:
            print("Applying LoRA configuration...")

            # Auto-detect target modules if not specified
            if config.lora_target_modules is None:
                config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
                print("  Using default attention projections (q/k/v/o_proj)")

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
            print(f"✓ W&B logging enabled: {wandb.run.url}")


    def collect_episode(self, temperature: float = 0.7) -> Episode:
        obs = self.env.reset()

        states = []
        actions = []
        prompts = []
        responses = []
        rewards = []
        reasonings = []
        done = False
        all_logprobs = []

        while not done:
            prompt = self.env.get_prompt_for_llm()

            messages = [
                {"role": "system", "content": "You are a clever Blackjack player."},
                {"role": "user", "content": prompt}
            ]

            # Try to use chat template, fallback to simple format if not available
            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback for models without chat template
                print(f"Warning: Chat template failed ({e}), using simple format")
                formatted = f"System: You are a clever Blackjack player.\n\nUser: {prompt}\n\nAssistant:"

            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        temperature=temperature,
                        do_sample=temperature > 0.0,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        output_scores=True,  # Enable logits output
                        return_dict_in_generate=True  # Structured output
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
                print(f"Logprobs: {logprobs}")
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            print(f"Response: {response}")
            

            response_str = response  # Keep original string
            try:
                # Strip markdown code blocks if present
                cleaned = response.split('</think>')[-1].strip()
                if cleaned.startswith('```'):
                    # Remove ```json or ``` at start and ``` at end
                    cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
                    cleaned = cleaned.rsplit('```', 1)[0] if '```' in cleaned else cleaned

                parsed = json.loads(cleaned.strip())
                action = parsed['action']
                reasoning = parsed['reasoning']
            except Exception as e:
                print(f"Error parsing response: {e}")
                action = None
                reasoning = None

            

            print(f"Action: {action}")
            print(f"Reasoning: {reasoning}")

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
            logprobs=all_logprobs
        )

    def collect_rollouts(self, num_episodes: int) -> List[Episode]:

        episodes = []
        total_actions = 0

        print(f"Collecting {num_episodes} episodes...")
        for _ in tqdm(range(num_episodes)):
            episode = self.collect_episode(temperature=self.config.temperature)
            episodes.append(episode)


        return episodes

    def create_training_dataset(self, episodes: List[Episode]) -> Dataset:
        training_examples = []

        for episode in episodes:
            # Skip low-reward episodes if threshold is set
            if episode.total_reward < self.config.reward_threshold:
                continue

            # Calculate advantage (how much better than average)
            if self.config.advantage_weighting:
                avg_reward = np.mean([ep.total_reward for ep in episodes])
                advantage = episode.total_reward - avg_reward
                weight = max(0.1, advantage + 1.0)  # Ensure positive weight
            else:
                weight = 1.0

            for i in range(len(episode.actions)):
                # Create target response in JSON format with reasoning
                action = episode.actions[i]
                state = episode.states[i]
                reasoning = episode.reasonings[i]

                # Create JSON formatted target
                target = json.dumps({
                    "action": action,
                    "reasoning": reasoning
                })

                # Create full training text (prompt + response)
                full_text = episode.prompts[i] + target

                training_examples.append({
                    'text': full_text,
                    'weight': weight,
                    'reward': episode.total_reward,
                    'action': action
                })

        return Dataset.from_list(training_examples)

    def train_on_dataset(self, dataset: Dataset):
        """
        Fine-tune the model on collected dataset.

        Args:
            dataset: Training dataset
        """
        if len(dataset) == 0:
            print("Warning: Empty dataset, skipping training")
            return

        print(f"Training on {len(dataset)} examples...")

        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=512,
                padding=False  # Use dynamic padding via data collator
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

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
        trainer = Trainer(
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
        """
        Evaluate current model performance.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating on {num_episodes} episodes...")

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
        """Train using RLVR (supervised fine-tuning on good trajectories)."""
        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.config.num_iterations}")
            print('='*60)

            # Collect rollouts
            episodes = self.collect_rollouts(self.config.episodes_per_iteration)


            avg_reward = np.mean([ep.total_reward for ep in episodes])
            

            current_episode = self.collect_episode(temperature=0.3)
            reward = current_episode.total_reward
            advantage = reward - avg_reward

            print(f"Advantage: {advantage}")
            

            # Print rollout statistics
            rewards = [ep.total_reward for ep in episodes]

            # Calculate action distribution
            all_actions = []
            for ep in episodes:
                all_actions.extend(ep.actions)
            # Filter out None values (from failed parses)
            valid_actions = [a for a in all_actions if a is not None]
            hit_rate = sum(valid_actions) / len(valid_actions) if valid_actions else 0
            stand_rate = 1 - hit_rate

            rollout_stats = {
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'win_rate': np.mean([r > 0 for r in rewards]),
                'lose_rate': np.mean([r < 0 for r in rewards]),
                'draw_rate': np.mean([r == 0 for r in rewards]),
                'hit_rate': hit_rate,
                'stand_rate': stand_rate,
            }

            print(f"\nRollout Statistics:")
            print(f"  Mean Reward: {rollout_stats['mean_reward']:.3f}")
            print(f"  Std Reward: {rollout_stats['std_reward']:.3f}")
            print(f"  Win Rate: {rollout_stats['win_rate']:.1%}")
            print(f"  Action Distribution: Hit {hit_rate:.1%}, Stand {stand_rate:.1%}")

            # Warn about pathological action distributions
            if hit_rate > 0.9 or stand_rate > 0.9:
                print(f"  ⚠️  Warning: Highly biased action distribution!")
                print(f"      Model may not be learning proper strategy")

            # Log rollout statistics to wandb
            if self.config.use_wandb:
                wandb.log({
                    "iteration": iteration,
                    "rollout/mean_reward": rollout_stats['mean_reward'],
                    "rollout/std_reward": rollout_stats['std_reward'],
                    "rollout/min_reward": rollout_stats['min_reward'],
                    "rollout/max_reward": rollout_stats['max_reward'],
                    "rollout/win_rate": rollout_stats['win_rate'],
                    "rollout/lose_rate": rollout_stats['lose_rate'],
                    "rollout/draw_rate": rollout_stats['draw_rate'],
                    "rollout/hit_rate": rollout_stats['hit_rate'],
                    "rollout/stand_rate": rollout_stats['stand_rate'],
                })

            # Create training dataset
            dataset = self.create_training_dataset(episodes)
            print(f"\nCreated dataset with {len(dataset)} examples")

            # Log dataset statistics
            if self.config.use_wandb:
                wandb.log({
                    "iteration": iteration,
                    "dataset/num_examples": len(dataset),
                    "dataset/num_episodes": len(episodes),
                })

            self.train_on_dataset(dataset)

            if iteration % self.config.eval_frequency == 0:
                print(f"\nEvaluation after iteration {iteration}:")
                metrics = self.evaluate(self.config.eval_episodes)
                self._print_metrics(metrics)

                self.history.append({
                    'iteration': iteration,
                    'metrics': metrics
                })

                if self.config.use_wandb:
                    wandb.log({
                        "iteration": iteration,
                        "eval/mean_reward": metrics['mean_reward'],
                        "eval/std_reward": metrics['std_reward'],
                        "eval/win_rate": metrics['win_rate'],
                        "eval/lose_rate": metrics['lose_rate'],
                        "eval/draw_rate": metrics['draw_rate'],
                        "eval/min_reward": metrics['min_reward'],
                        "eval/max_reward": metrics['max_reward'],
                    })

                # Save checkpoint
                checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint_iter_{iteration}"
                )
                self.save_checkpoint(checkpoint_path)

        # Final evaluation
        print(f"\n{'='*60}")
        print("Final Evaluation:")
        print('='*60)
        final_metrics = self.evaluate(self.config.eval_episodes)
        self._print_metrics(final_metrics)

        self.history.append({
            'iteration': self.config.num_iterations,
            'metrics': final_metrics,
            'final': True
        })

        if self.config.use_wandb:
            wandb.log({
                "iteration": self.config.num_iterations,
                "final/mean_reward": final_metrics['mean_reward'],
                "final/std_reward": final_metrics['std_reward'],
                "final/win_rate": final_metrics['win_rate'],
                "final/lose_rate": final_metrics['lose_rate'],
                "final/draw_rate": final_metrics['draw_rate'],
                "final/min_reward": final_metrics['min_reward'],
                "final/max_reward": final_metrics['max_reward'],
            })

        self.save_checkpoint(os.path.join(self.config.output_dir, "final_model"))
        self.save_history()

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Final model saved to: {self.config.output_dir}/final_model")
        print(f"Training history saved to: {self.config.log_file}")
        print('='*60)

        if self.config.use_wandb:
            wandb.finish()
            print("W&B run finished")

    def _print_metrics(self, metrics: Dict):
        """Print evaluation metrics."""
        print(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Lose Rate: {metrics['lose_rate']:.1%}")
        print(f"  Draw Rate: {metrics['draw_rate']:.1%}")
        print(f"  Min/Max Reward: {metrics['min_reward']:.1f} / {metrics['max_reward']:.1f}")

    def save_checkpoint(self, path: str):
        print(f"Saving checkpoint to {path}...")

        if self.config.use_lora:
            self.model.save_pretrained(path)
            print(f"  LoRA adapters saved (trainable params only)")
        else:
            self.model.save_pretrained(path)
            print(f"  Full model saved")

        self.tokenizer.save_pretrained(path)

    def save_history(self):
        """Save training history to JSON."""
        with open(self.config.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    """Main training function."""

    # Create configuration
    config = RLVRConfig(
        model_name="google/gemma-2b-it",  # Use smaller model for faster training
        num_iterations=10,
        episodes_per_iteration=50,  # Start with fewer episodes
        batch_size=4,
        learning_rate=2e-5,
        num_epochs_per_iteration=1,
        temperature=0.3,  # Lower temperature for less random exploration
        reward_threshold=0.0,  # Only train on winning episodes
        advantage_weighting=True,
        eval_episodes=50,
        output_dir="./checkpoints",
        log_file="./training_log.json"
    )

    trainer = RLVRTrainer(config)

    trainer.train()


if __name__ == "__main__":
    main()
