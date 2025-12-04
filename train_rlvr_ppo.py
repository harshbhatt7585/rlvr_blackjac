import logging
import os
import json
import copy
import random
from threading import Thread

import torch
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
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

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.history = []
        os.makedirs(config.output_dir, exist_ok=True)
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

        generated_tokens = outputs.sequences[0][inputs['input_ids'].shape[1]:].tolist()

        logprobs_for_tokens = []
        for j, token_id in enumerate(generated_tokens):
            if j < len(outputs.scores):
                logit = outputs.scores[j][0]
                logprob_dist = F.log_softmax(logit, dim=-1)
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

    def ppo_train(self):
        for iteration in range(1, self.config.num_iterations + 1):
            self.logger.info("Starting iteration %d/%d", iteration, self.config.num_iterations)

            episodes = self.collect_rollouts(self.config.episodes_per_iteration, iteration=iteration)
            self.replay_buffer.add(episodes)

            mean_reward = sum([ep.total_reward for ep in episodes]) / len(episodes)
            std_reward = np.std([ep.total_reward for ep in episodes])
            win_rate = np.mean([ep.total_reward > 0 for ep in episodes])
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

            batch_losses = []

            for ep_idx, episode in enumerate(batch_episodes):
                advantage = episode.total_reward - mean_reward

                for step_idx, history in enumerate(episode.message_histories):
                    response_token_ids = episode.response_token_ids[step_idx]

                    formatted = self.tokenizer.apply_chat_template(
                        history,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)
                    prompt_length = inputs['input_ids'].shape[1]

                    response_token_ids = torch.tensor([response_token_ids], device=self.model.device)

                    full_input_ids = torch.cat([inputs['input_ids'], response_token_ids], dim=1)

                    outputs = self.model(input_ids=full_input_ids)

                    seq_len = response_token_ids.shape[0]

                    response_logits = outputs.logits[0, prompt_length-1:prompt_length-1+seq_len, :]

                    log_probs = F.log_softmax(response_logits, dim=-1)

                    selected_log_probs = log_probs[range(seq_len), response_token_ids]

                    old_step_logprobs = episode.logits[step_idx]

                    old_step_logprobs = old_step_logprobs.detach().to(self.model.device)

                    old_selected_log_probs = old_step_logprobs

                    new_logprob_mean = selected_log_probs.mean()
                    old_logprob_mean = old_selected_log_probs.mean()

                    ratio = torch.exp(new_logprob_mean - old_logprob_mean)

                    if torch.isnan(ratio) or torch.isinf(ratio):
                        self.logger.warning(f"Invalid ratio detected (nan/inf), skipping step. new_logprob: {new_logprob_mean.item():.4f}, old_logprob: {old_logprob_mean.item():.4f}")
                        continue

                    clip_ratio = torch.clamp(ratio, 1 - self.config.ppo_clip_ratio, 1 + self.config.ppo_clip_ratio)

                    advantage_tensor = torch.tensor(advantage, dtype=torch.float32, device=self.model.device)

                    ppo_loss = -torch.min(ratio * advantage_tensor, clip_ratio * advantage_tensor)

                    batch_losses.append(ppo_loss)

            total_loss = torch.stack(batch_losses).mean()
            num_training_steps = len(batch_losses)

            self.optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            step_losses.append(total_loss.item())

            avg_loss = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
            self.logger.info("Iteration %d complete. Avg loss: %.4f", iteration, avg_loss)

            if self.config.eval_frequency > 0 and iteration % self.config.eval_frequency == 0 and self.config.eval_episodes > 0:
                eval_metrics = self.evaluate(self.config.eval_episodes, iteration=iteration)

                record = {
                    'iteration': iteration,
                    'episodes': len(episodes),
                    'train/loss': avg_loss,
                    'reward/mean': mean_reward,
                    'eval/mean_reward': eval_metrics['mean_reward'],
                    'eval/win_rate': eval_metrics['win_rate']
                }
                self._log_iteration(record)

            if iteration % 5 == 0 or iteration == self.config.num_iterations:
                checkpoint_name = f"iteration_{iteration}"
                save_path = self._save_checkpoint(checkpoint_name)
                self.logger.info("Saved checkpoint to %s", save_path)


def main():
    configure_logging(verbose=False)

    config = RLVRConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        num_iterations=10,
        episodes_per_iteration=100,
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
        eval_episodes=100,
        verbose=False,
        stream_rollouts=False,
        replay_buffer_capacity=1000
    )

    trainer = RLVRTrainer(config)
    trainer.ppo_train()


if __name__ == "__main__":
    main()
