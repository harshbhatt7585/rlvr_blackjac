"""
Demo script to play Blackjack with a trained Gemma model.

This script allows you to:
1. Watch the trained model play Blackjack
2. Compare against a baseline strategy
3. Analyze model decisions
"""

import os
import torch
import re
from threading import Thread
from typing import Optional, Callable
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel, PeftConfig

from env import BlackjackEnv


class BlackjackPlayer:
    """Player using a trained Gemma model."""

    def __init__(self, model_path: str = "google/gemma-2b-it", base_model: Optional[str] = None):
        """
        Initialize the player with a model.

        Args:
            model_path: Path to model checkpoint or Hugging Face model name
            base_model: Base model name if loading LoRA adapters (auto-detected if None)
        """
        print(f"Loading model from: {model_path}")

        # Check if this is a LoRA adapter checkpoint
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)

        if is_lora:
            print("Detected LoRA adapter checkpoint")

            # Load LoRA config to get base model
            peft_config = PeftConfig.from_pretrained(model_path)

            if base_model is None:
                base_model = peft_config.base_model_name_or_path
                print(f"Base model from config: {base_model}")

            # Load base model
            print(f"Loading base model: {base_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)
            base_model_obj = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )

            # Load LoRA adapters
            print("Loading LoRA adapters...")
            self.model = PeftModel.from_pretrained(base_model_obj, model_path)
            self.model.eval()
            print("✓ LoRA model loaded successfully")

        else:
            # Load full model
            print("Loading full model...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            print("✓ Full model loaded successfully")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def extract_action(self, response: str) -> Optional[int]:
        """Extract action from model response."""
        # Look for "0" or "1" in the response
        match = re.search(r'\b([01])\b', response)
        if match:
            return int(match.group(1))

        # If no explicit number, try to infer from keywords
        response_lower = response.lower()
        if 'stand' in response_lower:
            return 0
        elif 'hit' in response_lower:
            return 1

        return None

    def choose_action(
        self,
        prompt: str,
        temperature: float = 0.3,
        stream: bool = False,
        stream_handler: Optional[Callable[[str], None]] = None
    ) -> tuple[int, str]:
        """
        Choose an action given a game state prompt.

        Args:
            prompt: State description and action request
            temperature: Sampling temperature
            stream: Whether to stream tokens as they are generated
            stream_handler: Optional callable that receives streamed text chunks

        Returns:
            Tuple of (action, raw_response)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            if stream:
                streamer = TextIteratorStreamer(
                    self.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )

                generation_kwargs = {
                    **{k: v for k, v in inputs.items()},
                    "max_new_tokens": 100,
                    "temperature": temperature,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "streamer": streamer
                }

                # Launch generation in a separate thread so we can consume the stream
                thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
                thread.start()

                response_chunks = []
                for new_text in streamer:
                    if stream_handler is not None:
                        stream_handler(new_text)
                    else:
                        print(new_text, end="", flush=True)
                    response_chunks.append(new_text)

                thread.join()

                if stream_handler is None:
                    print()

                response = "".join(response_chunks)

            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )

        # Extract action
        action = self.extract_action(response)

        # Default to stand if invalid
        if action is None:
            action = 0
            response += " [DEFAULTED TO STAND]"

        return action, response


def baseline_strategy(player_sum: int, dealer_card: int, usable_ace: bool) -> int:
    """
    Basic Blackjack strategy.

    Args:
        player_sum: Player's hand total
        dealer_card: Dealer's visible card
        usable_ace: Whether player has usable ace

    Returns:
        Action (0=Stand, 1=Hit)
    """
    # Simple strategy: hit if under 17, stand otherwise
    # More sophisticated strategies consider dealer's card
    if player_sum < 12:
        return 1  # Always hit
    elif player_sum >= 17:
        return 0  # Always stand
    elif dealer_card >= 7 and player_sum < 17:
        return 1  # Hit if dealer shows strong card
    elif dealer_card <= 6 and player_sum >= 12:
        return 0  # Stand if dealer shows weak card
    else:
        return 1  # Default: hit


def play_episode(player: BlackjackPlayer, env: BlackjackEnv,
                 verbose: bool = True) -> tuple[float, list]:
    """
    Play a single episode.

    Args:
        player: BlackjackPlayer instance
        env: BlackjackEnv instance
        verbose: Whether to print game progress

    Returns:
        Tuple of (total_reward, decision_log)
    """
    obs = env.reset()
    decision_log = []

    if verbose:
        print("\n" + "="*60)
        env.render()

    done = False
    total_reward = 0

    while not done:
        # Get prompt
        prompt = env.get_prompt_for_llm()

        if verbose:
            print(f"\n{obs['description']}")

        stream_output = verbose

        if stream_output:
            print("\nModel response: ", end="", flush=True)

        # Get model's action
        action, response = player.choose_action(prompt, stream=stream_output)

        action_name = "STAND" if action == 0 else "HIT"

        if verbose:
            if not stream_output:
                print(f"\nModel response: {response}")
            print(f"Action: {action_name}")

        # Compare with baseline
        baseline_action = baseline_strategy(
            obs['player_sum'],
            obs['dealer_card'],
            obs['usable_ace']
        )
        baseline_name = "STAND" if baseline_action == 0 else "HIT"

        decision_log.append({
            'state': obs['description'],
            'model_action': action,
            'baseline_action': baseline_action,
            'agreement': action == baseline_action
        })

        if verbose and action != baseline_action:
            print(f"⚠️  Baseline would choose: {baseline_name}")

        # Take action
        obs, reward, done, info = env.step(action)
        total_reward += reward

        if verbose:
            env.render()

        if done:
            if verbose:
                print(f"\n{info.get('explanation', '')}")
                if 'dealer_hand' in info:
                    print(f"Dealer's final hand: {info['dealer_hand']}")
                print(f"\nReward: {reward}")
                print(f"Total Reward: {total_reward}")
                print("="*60)

    return total_reward, decision_log


def evaluate_player(player: BlackjackPlayer, num_episodes: int = 100) -> dict:
    """
    Evaluate player performance over multiple episodes.

    Args:
        player: BlackjackPlayer instance
        num_episodes: Number of episodes to play

    Returns:
        Dictionary of performance metrics
    """
    env = BlackjackEnv(seed=None)  # Random seed for evaluation

    rewards = []
    agreements = []

    print(f"\nEvaluating over {num_episodes} episodes...")

    for i in range(num_episodes):
        if (i + 1) % 20 == 0:
            print(f"Episode {i + 1}/{num_episodes}")

        reward, decision_log = play_episode(player, env, verbose=False)
        rewards.append(reward)

        # Track agreement with baseline
        episode_agreements = [d['agreement'] for d in decision_log]
        agreements.extend(episode_agreements)

    metrics = {
        'mean_reward': sum(rewards) / len(rewards),
        'win_rate': sum(r > 0 for r in rewards) / len(rewards),
        'lose_rate': sum(r < 0 for r in rewards) / len(rewards),
        'draw_rate': sum(r == 0 for r in rewards) / len(rewards),
        'baseline_agreement': sum(agreements) / len(agreements) if agreements else 0
    }

    return metrics


def main():
    """Main demo function."""
    import argparse

    parser = argparse.ArgumentParser(description="Play Blackjack with trained Gemma")
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b-it",
        help="Path to model checkpoint or HF model name"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run full evaluation instead of watching games"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of episodes for evaluation"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name for LoRA adapters (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Load player
    player = BlackjackPlayer(model_path=args.model, base_model=args.base_model)

    if args.evaluate:
        # Full evaluation
        metrics = evaluate_player(player, num_episodes=args.eval_episodes)

        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Reward: {metrics['mean_reward']:.3f}")
        print(f"Win Rate: {metrics['win_rate']:.1%}")
        print(f"Lose Rate: {metrics['lose_rate']:.1%}")
        print(f"Draw Rate: {metrics['draw_rate']:.1%}")
        print(f"Baseline Agreement: {metrics['baseline_agreement']:.1%}")
        print("="*60)

    else:
        # Watch games
        env = BlackjackEnv(seed=42)

        print("\n" + "="*60)
        print(f"WATCHING {args.episodes} GAMES")
        print("="*60)

        total_rewards = []

        for episode in range(args.episodes):
            print(f"\n### EPISODE {episode + 1}/{args.episodes} ###")
            reward, _ = play_episode(player, env, verbose=True)
            total_rewards.append(reward)

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.3f}")
        print(f"Wins: {sum(r > 0 for r in total_rewards)}/{args.episodes}")
        print(f"Losses: {sum(r < 0 for r in total_rewards)}/{args.episodes}")
        print(f"Draws: {sum(r == 0 for r in total_rewards)}/{args.episodes}")
        print("="*60)


if __name__ == "__main__":
    main()
