"""
RLVR Training Script for Blackjack

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
from threading import Thread

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer
)
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)



from env import BlackjackEnv

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

    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"  

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
    ppo_clip_ratio: float = 0.2  # PPO clipping parameter

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
    stream_rollouts: bool = False  # Stream model tokens during rollouts

    # Replay buffer settings
    replay_buffer_capacity: int = 1000


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
    message_histories: List[List[Dict[str, str]]]  # Conversation context per turn
    logits: List  # List of log probability tensors for each step
    response_token_ids: List[List[int]]  # Token IDs for each response


import random
@dataclass
class ReplayBuffer:
    episodes: List[Episode]


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, episodes: List[Episode]):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.extend(episodes)
    
    def sample(self, batch_size: int) -> List[Episode]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        self.buffer = []



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

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

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

        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)


    def _generate_response(
        self,
        inputs,
        temperature: float,
        stream: bool,
        stream_prefix: Optional[str] = None
    ) -> str:
        generation_kwargs = {
            **{k: v for k, v in inputs.items()},
            "max_new_tokens": 756,
            "temperature": temperature,
            "do_sample": temperature > 0.0,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "output_scores": True,
            "return_dict_in_generate": True,
        }

        with torch.no_grad():
            if stream:
                if stream_prefix is not None:
                    print(stream_prefix, end="", flush=True)
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                generation_kwargs["streamer"] = streamer


                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                chunks = []
                for text in streamer:
                    print(text, end="", flush=True)
                    chunks.append(text)

                thread.join()
                print()

                return "".join(chunks), None, None

            outputs = self.model.generate(**generation_kwargs)

        # Extract generated tokens
        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:].tolist()

        # Extract log probabilities for the actual generated tokens
        logprobs_for_tokens = []
        for j, token_id in enumerate(generated_tokens):
            if j < len(outputs.scores):
                logit = outputs.scores[j][0]  # Get logits for first (and only) batch element
                logprob_dist = F.log_softmax(logit, dim=-1)
                token_logprob = logprob_dist[token_id]
                logprobs_for_tokens.append(token_logprob)

        if logprobs_for_tokens:
            logprobs = torch.stack(logprobs_for_tokens)  # Shape: [seq_len]
        else:
            logprobs = None

        decoded_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return decoded_text, logprobs, generated_tokens

    def _generate_responses_batched(
        self,
        formatted_prompts: List[str],
        temperature: float
    ) -> List[tuple]:
        """Generate responses for multiple prompts sequentially (not truly batched due to padding issues)."""

        results_texts = []
        results_logprobs = []
        results_token_ids = []

        # Generate each prompt individually to avoid left-padding alignment issues
        for prompt in formatted_prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            generation_kwargs = {
                **{k: v for k, v in inputs.items()},
                "max_new_tokens": 756,
                "temperature": temperature,
                "do_sample": temperature > 0.0,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "output_scores": True,
                "return_dict_in_generate": True,
            }

            with torch.no_grad():
                outputs = self.model.generate(**generation_kwargs)

            # Extract generated tokens
            generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:].tolist()

            # Decode the response
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Extract log probabilities for the actual generated tokens
            token_ids_list = []
            logprobs_list = []

            for j, token_id in enumerate(generated_tokens):
                if j < len(outputs.scores):
                    token_ids_list.append(token_id)

                    logit = outputs.scores[j][0]  # Single batch element
                    logprob_dist = F.log_softmax(logit, dim=-1)
                    token_logprob = logprob_dist[token_id]

                    logprobs_list.append(token_logprob)

            results_texts.append(response)
            results_logprobs.append(torch.stack(logprobs_list) if logprobs_list else torch.tensor([]))
            results_token_ids.append(torch.tensor(token_ids_list, device=self.model.device))

        # Stack into batch tensors
        batch_size = len(formatted_prompts)
        max_len = max(len(tids) for tids in results_token_ids) if results_token_ids else 0

        batch_token_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.model.device)
        batch_logprobs = torch.zeros((batch_size, max_len), dtype=torch.float, device=self.model.device)

        for i in range(batch_size):
            seq_len = len(results_token_ids[i])
            batch_token_ids[i, :seq_len] = results_token_ids[i]
            batch_logprobs[i, :seq_len] = results_logprobs[i]

        return results_texts, batch_logprobs, batch_token_ids

    def collect_episode(
        self,
        temperature: float = 0.7,
        verbose: bool = False,
        stream: Optional[bool] = None,
        context: Optional[str] = None,
        episode_index: Optional[int] = None,
        episode_total: Optional[int] = None
    ) -> Episode:
        obs = self.env.reset()

        states: List[str] = []
        actions: List[int] = []
        prompts: List[str] = []
        responses: List[str] = []
        rewards: List[float] = []
        reasonings: List[str] = []
        history_snapshots: List[List[Dict[str, str]]] = []
        all_logprobs: List = []
        all_token_ids: List[List[int]] = []

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a clever Blackjack player."}
        ]

        done = False
        step = 0

        stream_enabled = stream if stream is not None else (self.config.stream_rollouts or verbose)

        if verbose or stream_enabled:
            header_parts = []
            if context:
                header_parts.append(context)
            if episode_index is not None:
                total_text = f"{episode_index + 1}" if episode_total is None else f"{episode_index + 1}/{episode_total}"
                header_parts.append(f"Episode {total_text}")
            else:
                header_parts.append("Episode")
            header = " - ".join(header_parts)
            print(f"\n{'='*60}\n{header}\n{'='*60}")

        while not done:
            step += 1
            prompt = self.env.get_prompt_for_llm()
            conversation.append({"role": "user", "content": prompt})

            if verbose or stream_enabled:
                print(f"\n[Step {step}]")
                print(f"State: {prompt}")

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

            response, logprobs, token_ids = self._generate_response(
                inputs,
                temperature,
                stream=stream_enabled,
                stream_prefix="Model response: " if stream_enabled else None
            )


            if verbose and not stream_enabled:
                print(f"Model response: {response}")

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

                self.logger.info("Action: %s, Reasoning: %s", action, reasoning)
            except Exception:
                self.logger.debug("Error parsing response: %s", response, exc_info=True)
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
            all_logprobs.append(logprobs)
            all_token_ids.append(token_ids if token_ids is not None else [])

            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)

        total_reward = sum(rewards)

        if verbose or stream_enabled:
            print(f"\nEpisode complete | Total reward: {total_reward:.2f}")
            print("-" * 60)

        return Episode(
            states=states,
            actions=actions,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            total_reward=total_reward,
            reasonings=reasonings,
            message_histories=history_snapshots,
            logits=all_logprobs,
            response_token_ids=all_token_ids
        )

    def collect_episodes_batched(
        self,
        num_episodes: int,
        temperature: float = 0.7,
        batch_size: Optional[int] = None
    ) -> List[Episode]:
        """Collect multiple episodes in parallel using batched generation."""

        if batch_size is None:
            batch_size = min(num_episodes, 8)  # Default batch size

        episodes = []

        # Process episodes in batches
        for batch_start in range(0, num_episodes, batch_size):
            batch_end = min(batch_start + batch_size, num_episodes)
            current_batch_size = batch_end - batch_start

            # Initialize environments and state tracking for this batch
            envs = [BlackjackEnv(natural_reward=self.config.natural_reward) for _ in range(current_batch_size)]
            observations = [env.reset() for env in envs]

            # Track state for each episode in the batch
            batch_states = {i: {
                'env': envs[i],
                'obs': observations[i],
                'done': False,
                'states': [],
                'actions': [],
                'prompts': [],
                'responses': [],
                'rewards': [],
                'reasonings': [],
                'history_snapshots': [],
                'all_logprobs': [],
                'all_token_ids': [],
                'conversation': [{"role": "system", "content": "You are a clever Blackjack player."}]
            } for i in range(current_batch_size)}

            max_steps = 50  # Safety limit
            step = 0

            while any(not batch_states[i]['done'] for i in range(current_batch_size)) and step < max_steps:
                step += 1

                # Collect prompts from all active (non-done) episodes
                active_indices = [i for i in range(current_batch_size) if not batch_states[i]['done']]

                if not active_indices:
                    break

                formatted_prompts = []
                for i in active_indices:
                    state = batch_states[i]
                    prompt = state['env'].get_prompt_for_llm()
                    state['conversation'].append({"role": "user", "content": prompt})

                    # Store snapshot before generation
                    history_snapshot = copy.deepcopy(state['conversation'])
                    state['history_snapshots'].append(history_snapshot)

                    # Format the prompt
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

                    formatted_prompts.append(formatted)
                    state['prompts'].append(prompt)

                # Generate responses in batch
                responses_data = self._generate_responses_batched(formatted_prompts, temperature)

                # responses_data returns: (texts, token_logprobs, generated_ids)
                # texts: list of strings
                # token_logprobs: [B, seq_len] - logprobs for sampled tokens
                # generated_ids: [B, seq_len] - token IDs

                # Process each response
                for idx, i in enumerate(active_indices):
                    state = batch_states[i]

                    # Extract data for this specific batch element
                    response = responses_data[0][idx] if isinstance(responses_data[0], list) else responses_data[0]
                    logprobs = responses_data[1][idx] if len(responses_data[1].shape) > 1 else responses_data[1]  # [seq_len]
                    token_ids = responses_data[2][idx] if len(responses_data[2].shape) > 1 else responses_data[2]  # [seq_len]

                    state['conversation'].append({"role": "assistant", "content": response})
                    state['responses'].append(response)
                    state['all_logprobs'].append(logprobs)
                    state['all_token_ids'].append(token_ids)

                    # Parse the response
                    try:
                        cleaned = response.split('</think>')[-1].strip()
                        if cleaned.startswith('```'):
                            cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned[3:]
                            cleaned = cleaned.rsplit('```', 1)[0] if '```' in cleaned else cleaned

                        parsed = json.loads(cleaned.strip())
                        action = parsed.get('action')
                        reasoning = parsed.get('reasoning')
                    except Exception:
                        self.logger.debug("Error parsing response: %s", response, exc_info=True)
                        action = None
                        reasoning = None

                    if action not in (0, 1):
                        self.logger.warning("Invalid action '%s', defaulting to stand (0)", action)
                        action = 0

                    state['states'].append(state['obs']['description'])
                    state['actions'].append(action)
                    state['reasonings'].append(reasoning)

                    # Step the environment
                    obs, reward, done, info = state['env'].step(action)
                    state['obs'] = obs
                    state['rewards'].append(reward)
                    state['done'] = done

            # Create Episode objects from completed batch
            for i in range(current_batch_size):
                state = batch_states[i]
                total_reward = sum(state['rewards'])

                episode = Episode(
                    states=state['states'],
                    actions=state['actions'],
                    prompts=state['prompts'],
                    responses=state['responses'],
                    rewards=state['rewards'],
                    total_reward=total_reward,
                    reasonings=state['reasonings'],
                    message_histories=state['history_snapshots'],
                    logits=state['all_logprobs'],  # Keep as list of tensors
                    response_token_ids=state['all_token_ids']
                )
                episodes.append(episode)

        return episodes

    def collect_rollouts(self, num_episodes: int, iteration: int) -> List[Episode]:
        """Collect episodes sequentially to ensure correct trajectories."""

        episodes = []
        self.logger.info("Collecting %d episodes sequentially", num_episodes)

        iterator = tqdm(
            range(num_episodes),
            desc=f"Iteration {iteration}/{self.config.num_iterations}",
            disable=self.config.verbose or self.config.stream_rollouts,
        )

        for idx in iterator:
            episode = self.collect_episode(
                temperature=self.config.temperature,
                verbose=self.config.verbose,
                stream=self.config.stream_rollouts,
                context=f"Iteration {iteration}/{self.config.num_iterations}",
                episode_index=idx,
                episode_total=num_episodes
            )
            episodes.append(episode)

        return episodes


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


    def evaluate(self, num_episodes: int = 100, iteration: Optional[int] = None) -> Dict:
        iteration_text = f"Iteration {iteration}/{self.config.num_iterations}" if iteration is not None else "Evaluation"
        print(f"\n{'.'*60}\n{iteration_text} | Evaluation\n{'.'*60}")
        print(f"Evaluating {num_episodes} episodes...")

        self.logger.info("Evaluating on %d episodes", num_episodes)

        episodes = []
        iterator = tqdm(
            range(num_episodes),
            disable=self.config.verbose or self.config.stream_rollouts
        )
        for idx in iterator:
            # Use lower temperature for evaluation (more deterministic)
            episode = self.collect_episode(
                temperature=0.3,
                verbose=self.config.verbose,
                stream=self.config.stream_rollouts,
                context=f"{iteration_text} | Eval",
                episode_index=idx,
                episode_total=num_episodes
            )
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

        print(
            "Evaluation summary → "
            f"mean: {metrics['mean_reward']:.3f}, "
            f"std: {metrics['std_reward']:.3f}, "
            f"min/max: {metrics['min_reward']:.3f}/{metrics['max_reward']:.3f}, "
            f"win-rate: {metrics['win_rate']:.1%}, "
            f"draw-rate: {metrics['draw_rate']:.1%}"
        )

        return metrics

    def ppo_train(self):
        import random

        for iteration in range(1, self.config.num_iterations + 1):
            self.logger.info("Starting iteration %d/%d", iteration, self.config.num_iterations)

            episodes = self.collect_rollouts(self.config.episodes_per_iteration, iteration=iteration)
            self.replay_buffer.add(episodes)

            # Calculate baseline (mean reward)
            mean_reward = sum([ep.total_reward for ep in episodes]) / len(episodes)
            std_reward = np.std([ep.total_reward for ep in episodes])
            min_reward = min([ep.total_reward for ep in episodes])
            max_reward = max([ep.total_reward for ep in episodes])
            win_rate = np.mean([ep.total_reward > 0 for ep in episodes])
            lose_rate = np.mean([ep.total_reward < 0 for ep in episodes])
            draw_rate = np.mean([ep.total_reward == 0 for ep in episodes])
            avg_episode_length = np.mean([len(ep.actions) for ep in episodes])

            self.logger.info("Mean reward for iteration %d: %.3f", iteration, mean_reward)

            if self.config.use_wandb:
                train_logs = {
                    'train/reward_mean': mean_reward,
                    'train/reward_std': std_reward,
                    'train/reward_min': min_reward,
                    'train/reward_max': max_reward,
                    'train/win_rate': win_rate,
                    'train/lose_rate': lose_rate,
                    'train/draw_rate': draw_rate,
                    'train/episode_length': avg_episode_length,
                    'train/num_episodes': len(episodes),
                }

                # Log histogram of episode rewards
                episode_rewards = [ep.total_reward for ep in episodes]
                train_logs['train/reward_histogram'] = wandb.Histogram(episode_rewards)

                wandb.log(train_logs, step=iteration)

            # Train on the batch
            total_loss = 0.0
            step_losses = []
            gradient_norms = []
            num_training_steps = 0

            actual_batch_size = min(self.config.batch_size, len(self.replay_buffer))
            if actual_batch_size == 0:
                self.logger.warning("Replay buffer is empty, skipping training")
                continue

            batch_episodes = self.replay_buffer.sample(actual_batch_size)

            # For PPO, we need to recompute logprobs for the same trajectories with current policy
            # Collect old logprobs (detached) and advantages
            batch_size = len(batch_episodes)
            batch_losses = []

            for ep_idx, episode in enumerate(batch_episodes):
                advantage = episode.total_reward - mean_reward

                # Recompute logprobs for this trajectory with current policy (with gradients)
                for step_idx, history in enumerate(episode.message_histories):
                    response_token_ids = episode.response_token_ids[step_idx]

                    formatted = self.tokenizer.apply_chat_template(
                        history,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # Tokenize prompt
                    inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
                    prompt_length = inputs['input_ids'].shape[1]

                    # Create full input (prompt + response tokens)
                    response_token_ids = torch.tensor([response_token_ids], device=self.model.device)

                    full_input_ids = torch.cat([inputs['input_ids'], response_token_ids], dim=1)

                    # Forward pass with gradients on the full sequence
                    outputs = self.model(input_ids=full_input_ids)

                    seq_len = response_token_ids.shape[0]

                    # Get logits for the response part (shift by 1 for next token prediction)
                    response_logits = outputs.logits[0, prompt_length-1:prompt_length-1+seq_len, :]

                    # Get log probabilities for the actual generated tokens
                    log_probs = F.log_softmax(response_logits, dim=-1)

                    # Extract log probabilities for the actual tokens that were generated
                    selected_log_probs = log_probs[range(seq_len), response_token_ids]

                    # Get old log probabilities for this step (detached)
                    old_step_logprobs = episode.logits[step_idx]

                    if old_step_logprobs is None:
                        # Skip if we don't have old logprobs (e.g., from streaming)
                        continue

                    old_step_logprobs = old_step_logprobs.detach().to(self.model.device)

                    # The old logprobs might be:
                    # 1. [seq_len] - just the logprobs for sampled tokens (from your custom generation)
                    # 2. [seq_len, vocab_size] - full distribution

                    if len(old_step_logprobs.shape) == 1:
                        # Case 1: Already extracted logprobs for sampled tokens
                        old_selected_log_probs = old_step_logprobs
                    else:
                        # Case 2: Full distribution, need to extract for specific tokens
                        old_selected_log_probs = old_step_logprobs[range(seq_len), response_token_ids]

                    # Average over tokens
                    new_logprob_mean = selected_log_probs.mean()
                    old_logprob_mean = old_selected_log_probs.mean()

                    # Compute ratio
                    ratio = torch.exp(new_logprob_mean - old_logprob_mean)

                    # Check for invalid values
                    if torch.isnan(ratio) or torch.isinf(ratio):
                        self.logger.warning(f"Invalid ratio detected (nan/inf), skipping step. new_logprob: {new_logprob_mean.item():.4f}, old_logprob: {old_logprob_mean.item():.4f}")
                        continue

                    # Clip ratio
                    clip_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio)

                    # Convert advantage to tensor
                    advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.model.device)

                    # Compute PPO loss (we want to maximize, so minimize negative)
                    ppo_loss = -torch.min(ratio * advantage_tensor, clip_ratio * advantage_tensor)

                    batch_losses.append(ppo_loss)

            if batch_losses:
                total_loss = torch.stack(batch_losses).mean()
                num_training_steps = len(batch_losses)

                self.optimizer.zero_grad()
                total_loss.backward()

                # Compute gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                gradient_norms.append(grad_norm.item())

                self.optimizer.step()
                step_losses.append(total_loss.item())
            else:
                total_loss = torch.tensor(0.0)

            avg_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
            avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
            self.logger.info("Iteration %d complete. Avg loss: %.4f, Avg grad norm: %.4f",
                           iteration, avg_loss, avg_grad_norm)

            # Log aggregated training metrics
            if self.config.use_wandb and num_training_steps > 0:
                log_dict = {
                    'train/loss': avg_loss,
                    'train/loss_std': np.std(step_losses) if step_losses else 0.0,
                    'train/gradient_norm_mean': avg_grad_norm,
                    'train/gradient_norm_std': np.std(gradient_norms) if gradient_norms else 0.0,
                    'train/num_training_steps': num_training_steps,
                }
                # Only add histogram if we have valid step losses
                if step_losses and all(np.isfinite(loss) for loss in step_losses):
                    log_dict['train/loss_histogram'] = wandb.Histogram(step_losses)
                wandb.log(log_dict, step=iteration)

            # Evaluate if needed
            if self.config.eval_frequency > 0 and iteration % self.config.eval_frequency == 0 and self.config.eval_episodes > 0:
                eval_metrics = self.evaluate(self.config.eval_episodes, iteration=iteration)

                # Log metrics
                record = {
                    'iteration': iteration,
                    'episodes': len(episodes),
                    'train/loss': avg_loss,
                    'reward/mean': mean_reward,
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/win_rate': eval_metrics['win_rate']
                }
                self._log_iteration(record)

                if self.config.use_wandb:
                    # Log all evaluation metrics
                    eval_logs = {
                        'eval/mean_reward': eval_metrics['mean_reward'],
                        'eval/std_reward': eval_metrics['std_reward'],
                        'eval/min_reward': eval_metrics['min_reward'],
                        'eval/max_reward': eval_metrics['max_reward'],
                        'eval/win_rate': eval_metrics['win_rate'],
                        'eval/lose_rate': eval_metrics['lose_rate'],
                        'eval/draw_rate': eval_metrics['draw_rate'],
                    }
                    wandb.log(eval_logs, step=iteration)

            # Save checkpoint
            if iteration % 5 == 0 or iteration == self.config.num_iterations:
                checkpoint_name = f"iteration_{iteration}"
                save_path = self._save_checkpoint(checkpoint_name)
                self.logger.info("Saved checkpoint to %s", save_path)



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
    train_parser.add_argument('--no-stream', action='store_true', help='Disable token streaming during rollouts')
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
        verbose=args.verbose,
        stream_rollouts=not args.no_stream
    )

    trainer = RLVRTrainer(config)
    trainer.ppo_train()


if __name__ == "__main__":
    main()
