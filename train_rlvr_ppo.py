import logging
import os
import json
import copy
import random
from threading import Thread

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

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

# Import render server if rendering is enabled

from render_server import update_game_state, enable_rendering, run_server, setup_logging_handler, setup_print_capture
from threading import Thread
RENDER_AVAILABLE = True

def configure_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

@dataclass
class RLVRConfig:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    num_iterations: int = 10
    episodes_per_iteration: int = 100
    batch_size: int = 8
    learning_rate: float = 2e-5
    temperature: float = 0.7
    ppo_clip_ratio: float = 0.2
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    natural_reward: float = 1.5
    seed: int = 42
    output_dir: str = "./checkpoints"
    log_file: str = "./training_log.json"
    eval_frequency: int = 1
    eval_episodes: int = 100
    verbose: bool = False
    stream_rollouts: bool = False
    replay_buffer_capacity: int = 1000
    discount_factor: float = 1.0
    value_loss_coef: float = 0.5
    use_wandb: bool = True
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    gae_lambda: float = 0.95
    enable_render: bool = True
    render_port: int = 5000


@dataclass
class Episode:
    states: List[str]
    actions: List[int]
    prompts: List[str]
    responses: List[str]
    rewards: List[float]
    total_reward: float
    reasonings: List[str]
    message_histories: List[List[Dict[str, str]]]
    logits: List
    response_token_ids: List[List[int]]
    value_estimates: List[float]


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
        
        # Set up rendering if enabled
        self.render_enabled = config.enable_render and RENDER_AVAILABLE
        if self.render_enabled:
            self.logger.info("Rendering enabled - starting render server on port %d", config.render_port)
            enable_rendering()
            # Start render server in background thread
            self.render_thread = Thread(
                target=run_server,
                args=('127.0.0.1', config.render_port, False),
                daemon=True
            )
            self.render_thread.start()
            # Give server a moment to start
            import time
            time.sleep(1)
            # Set render callback
            self.env.set_render_callback(update_game_state)
            # Add WebSocket log handler to send logs to frontend
            ws_handler = setup_logging_handler()
            logging.getLogger().addHandler(ws_handler)
            # Capture print statements as well
            setup_print_capture()
            self.logger.info("WebSocket logging enabled - logs will appear in frontend")
        elif config.enable_render and not RENDER_AVAILABLE:
            self.logger.warning("Rendering requested but render_server not available. Install flask-socketio to enable rendering.")
        
        self.logger.info("Loading model: %s", config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        if torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True
        ).to(device_map)

        if self.tokenizer.pad_token is None:
            if self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                self.logger.debug("Using unk_token as pad_token: %s", self.tokenizer.pad_token)
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.debug("Using eos_token as pad_token: %s", self.tokenizer.pad_token)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if config.use_lora:
            self.logger.info("Applying LoRA configuration")

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

        self.device = next(self.model.parameters()).device

        self.value_head = nn.Linear(self.model.config.hidden_size, 1).to(self.device)

        self.trainable_parameters = list(self.model.parameters()) + list(self.value_head.parameters())
        self.optimizer = torch.optim.AdamW(self.trainable_parameters, lr=config.learning_rate)
        self.history = []
        os.makedirs(config.output_dir, exist_ok=True)
        self.replay_buffer = ReplayBuffer(capacity=config.replay_buffer_capacity)
        self.wandb_run = None

        if self.config.use_wandb:
            try:
                import wandb
            except ImportError as exc:  # pragma: no cover - configuration guard
                raise RuntimeError(
                    "Weights & Biases logging is enabled but the 'wandb' package is not installed."
                ) from exc

            init_kwargs = {
                "project": self.config.wandb_project or "rlvr_blackjack",
                "config": asdict(self.config),
            }
            if self.config.wandb_run_name:
                init_kwargs["name"] = self.config.wandb_run_name

            self.wandb_run = wandb.init(**init_kwargs)

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

        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:].tolist()

        logprobs_for_tokens = []
        for j, token_id in enumerate(generated_tokens):
            if j < len(outputs.scores):
                logit = outputs.scores[j][0]
                scaled_logits = logit / temperature if temperature > 0.0 else logit
                logprob_dist = F.log_softmax(scaled_logits, dim=-1)
                token_logprob = logprob_dist[token_id]
                logprobs_for_tokens.append(token_logprob)

        if logprobs_for_tokens:
            logprobs = torch.stack(logprobs_for_tokens)
        else:
            logprobs = None

        decoded_text = self.tokenizer.decode(
            outputs.sequences[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return decoded_text, logprobs, generated_tokens

    def _estimate_state_value(self, inputs: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
            last_hidden = hidden_states[:, -1, :]
            value = self.value_head(last_hidden.to(torch.float32)).squeeze(-1)
        return value.item()

    def _compute_returns(self, rewards: List[float]) -> List[float]:
        returns: List[float] = []
        running_return = 0.0
        gamma = self.config.discount_factor
        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.insert(0, running_return)
        return returns

    def _wandb_log(self, data: Dict[str, float]) -> None:
        if self.wandb_run is not None:
            self.wandb_run.log(data)

    def collect_episode(
        self,
        temperature: float = 0.7,
        verbose: bool = False,
        stream: Optional[bool] = None,
        context: Optional[str] = None,
        episode_index: Optional[int] = None,
        episode_total: Optional[int] = None,
        log_rewards: bool = False
    ) -> Episode:
        obs = self.env.reset()
        
        # Render initial state
        if self.render_enabled:
            self.env.render(obs, action=None, reward=None, info=None, reasoning=None)

        was_training_model = self.model.training
        was_training_value_head = self.value_head.training
        self.model.eval()
        self.value_head.eval()

        states: List[str] = []
        actions: List[int] = []
        prompts: List[str] = []
        responses: List[str] = []
        rewards: List[float] = []
        reasonings: List[str] = []
        history_snapshots: List[List[Dict[str, str]]] = []
        all_logprobs: List = []
        all_token_ids: List[List[int]] = []
        value_estimates: List[float] = []

        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a clever Blackjack player."}
        ]

        done = False
        step = 0

        stream_enabled = stream if stream is not None else (self.config.stream_rollouts or verbose)
        
        # Render initial state
        if self.render_enabled:
            self.env.render(obs, action=None, reward=None, info=None, reasoning=None)

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

            formatted = self.tokenizer.apply_chat_template(
                history_snapshot,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

            state_value = self._estimate_state_value(inputs)
            value_estimates.append(state_value)

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
            
            # Render game state after step
            if self.render_enabled:
                self.env.render(obs, action=action, reward=reward, info=info, reasoning=reasoning)

            if log_rewards and self.wandb_run is not None:
                self._wandb_log({"reward": reward})

        total_reward = sum(rewards)

        if verbose or stream_enabled:
            print(f"\nEpisode complete | Total reward: {total_reward:.2f}")
            print("-" * 60)

        if was_training_model:
            self.model.train()
        if was_training_value_head:
            self.value_head.train()

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
            response_token_ids=all_token_ids,
            value_estimates=value_estimates
        )

    def collect_rollouts(self, num_episodes: int, iteration: int) -> List[Episode]:
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
                episode_total=num_episodes,
                log_rewards=self.wandb_run is not None
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

        torch.save(self.value_head.state_dict(), os.path.join(save_dir, "value_head.pt"))
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
            episode = self.collect_episode(
                temperature=0.3,
                verbose=self.config.verbose,
                stream=self.config.stream_rollouts,
                context=f"{iteration_text} | Eval",
                episode_index=idx,
                episode_total=num_episodes
            )
            episodes.append(episode)

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

    def compute_gae(self, rewards: List[float], value_estimates: List[float], next_value: float) -> List[float]:
        advantages = []
        gae = 0.0
        gamma = self.config.discount_factor
        lambda_ = self.config.gae_lambda

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_value
            else:
                next_value = value_estimates[t + 1]
            
            delta = rewards[t] + gamma * next_value - value_estimates[t]
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)
        
        returns = [adv + value for adv, value in zip(advantages, value_estimates)]
        
        return advantages, returns

    def ppo_train(self):
        for iteration in range(1, self.config.num_iterations + 1):
            self.logger.info("Starting iteration %d/%d", iteration, self.config.num_iterations)

            episodes = self.collect_rollouts(self.config.episodes_per_iteration, iteration=iteration)
            self.replay_buffer.add(episodes)

            mean_reward = sum([ep.total_reward for ep in episodes]) / len(episodes)
            std_reward = np.std([ep.total_reward for ep in episodes])
            win_rate = np.mean([ep.total_reward > 0 for ep in episodes])
            self._wandb_log({
                "train/win_rate": win_rate,
            })
            lose_rate = np.mean([ep.total_reward < 0 for ep in episodes])
            draw_rate = np.mean([ep.total_reward == 0 for ep in episodes])
            avg_episode_length = np.mean([len(ep.actions) for ep in episodes])

            self.logger.info("Mean reward for iteration %d: %.3f", iteration, mean_reward)

            total_loss = 0.0
            step_losses = []
            num_training_steps = 0

            actual_batch_size = min(self.config.batch_size, len(self.replay_buffer))
            if actual_batch_size == 0:
                self.logger.warning("Replay buffer is empty, skipping training")
                continue

            batch_episodes = self.replay_buffer.sample(actual_batch_size)

            batch_policy_losses = []
            batch_value_losses = []

            self.model.train()
            self.value_head.train()

            for ep_idx, episode in enumerate(batch_episodes):
                advantages, returns = self.compute_gae(episode.rewards, episode.value_estimates, 0.0)

                for step_idx, history in enumerate(episode.message_histories):
                    response_token_ids = episode.response_token_ids[step_idx]
                    if not response_token_ids:
                        continue

                    formatted = self.tokenizer.apply_chat_template(
                        history,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = self.tokenizer(formatted, return_attention_mask=True, return_tensors="pt").to(self.model.device)
                    prompt_length = inputs['input_ids'].shape[1]

                    response_token_tensor = torch.tensor(
                        response_token_ids,
                        dtype=torch.long,
                        device=self.model.device
                    ).unsqueeze(0)

                    full_input_ids = torch.cat([inputs['input_ids'], response_token_tensor], dim=1)

                    outputs = self.model(
                        input_ids=full_input_ids,
                        output_hidden_states=True,
                        return_dict=True,
                        attention_mask=inputs['attention_mask'],
                    )

                    seq_len = response_token_tensor.shape[1]

                    response_logits = outputs.logits[0, prompt_length - 1:prompt_length - 1 + seq_len, :]

                    scaled_logits = response_logits / self.config.temperature if self.config.temperature > 0.0 else response_logits
                    log_probs = F.log_softmax(scaled_logits, dim=-1).to(torch.float32)

                    response_tokens_flat = response_token_tensor.squeeze(0)
                    selected_log_probs = log_probs[range(seq_len), response_tokens_flat]

                    old_step_logprobs = episode.logits[step_idx]
                    if old_step_logprobs is None or len(old_step_logprobs) != seq_len:
                        continue

                    old_selected_log_probs = old_step_logprobs.detach().to(self.model.device, dtype=torch.float32)

                    # Per-token PPO computation (fixed)
                    logprob_diff = selected_log_probs - old_selected_log_probs
                    # print("logprob_diff: ", logprob_diff)
                    ratios = torch.exp(logprob_diff)

                    old_value = episode.value_estimates[step_idx]
                    advantage = advantages[step_idx]
                    advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.model.device)

                    surrog1 = ratios * advantage_tensor
                    surrog2 = torch.clamp(ratios, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio) * advantage_tensor
                    policy_losses = -torch.min(surrog1, surrog2)
                    policy_loss = policy_losses.mean()  # Mean over tokens
                    batch_policy_losses.append(policy_loss)

                    # Value loss (keep this)
                    hidden_states = outputs.hidden_states[-1]
                    prompt_hidden = hidden_states[0, prompt_length - 1, :].to(torch.float32)
                    value_pred = self.value_head(prompt_hidden)
                    return_tensor = torch.tensor(returns[step_idx], dtype=torch.float32, device=self.model.device)
                    value_loss = F.mse_loss(value_pred, return_tensor)
                    batch_value_losses.append(value_loss)

            if not batch_policy_losses:
                self.logger.warning("No valid policy updates in this batch, skipping optimizer step")
                continue

            policy_loss = torch.stack(batch_policy_losses).mean()

            value_loss = torch.stack(batch_value_losses).mean() if batch_value_losses else torch.tensor(0.0, device=self.model.device)
            total_loss = policy_loss + self.config.value_loss_coef * value_loss
            num_training_steps = len(batch_policy_losses)

            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.trainable_parameters, max_norm=1.0)

            self.optimizer.step()


            avg_total_loss = total_loss.detach().item()
            avg_policy_loss = policy_loss.detach().item()
            avg_value_loss = value_loss.detach().item() if batch_value_losses else 0.0
            step_losses.append(avg_total_loss)

            self.logger.info(
                "Iteration %d complete. Total loss: %.4f | Policy: %.4f | Value: %.4f",
                iteration,
                avg_total_loss,
                avg_policy_loss,
                avg_value_loss
            )

            self._wandb_log(
                {
                    "train/average_reward": mean_reward,
                    "train/policy_loss": avg_policy_loss,
                    "train/value_loss": avg_value_loss,
                }
            )

            if self.config.eval_frequency > 0 and iteration % self.config.eval_frequency == 0 and self.config.eval_episodes > 0:
                eval_metrics = self.evaluate(self.config.eval_episodes, iteration=iteration)

                record = {
                    'iteration': iteration,
                    'episodes': len(episodes),
                    'train/loss': avg_total_loss,
                    'train/policy_loss': avg_policy_loss,
                    'train/value_loss': avg_value_loss,
                    'reward/mean': mean_reward,
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/win_rate': eval_metrics['win_rate']
                }
                self._log_iteration(record)

            if iteration % 5 == 0 or iteration == self.config.num_iterations:
                checkpoint_name = f"iteration_{iteration}"
                save_path = self._save_checkpoint(checkpoint_name)
                self.logger.info("Saved checkpoint to %s", save_path)

        if self.wandb_run is not None:
            self.wandb_run.finish()
            self.wandb_run = None


def main():
    configure_logging(verbose=False)

    config = RLVRConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        num_iterations=10,
        episodes_per_iteration=30,
        batch_size=8,
        learning_rate=2e-5,
        temperature=0.7,
        ppo_clip_ratio=0.2,
        use_lora=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        natural_reward=1.5,
        seed=42,
        output_dir="./checkpoints",
        log_file="./training_log.json",
        eval_frequency=1,
        eval_episodes=30,
        verbose=False,
        stream_rollouts=False,
        replay_buffer_capacity=1000,
        enable_render=True,
    )

    trainer = RLVRTrainer(config)
    trainer.ppo_train()


if __name__ == "__main__":
    main()
