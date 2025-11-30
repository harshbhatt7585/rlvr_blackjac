import random
from typing import Tuple, Dict, List, Optional
import numpy as np



class BlackjackEnv:
    """
    Blackjack environment for training LLMs using RLVR.

    The environment follows standard Blackjack rules:
    - Goal: Get as close to 21 as possible without going over
    - Face cards (J, Q, K) are worth 10
    - Aces can be worth 1 or 11
    - Dealer hits until reaching 17 or higher

    Action space:
        0: Stand (end turn)
        1: Hit (draw another card)

    State representation:
        - player_sum: Current sum of player's hand
        - dealer_card: Dealer's visible card value
        - usable_ace: Whether player has a usable ace (ace counted as 11)
        - player_cards: List of player's cards
        - dealer_cards: List of dealer's cards (only first card visible initially)

    Rewards:
        +1.0: Win (player closer to 21 than dealer, or dealer busts)
        0.0: Draw (same total as dealer)
        -1.0: Lose (player busts or dealer closer to 21)
        +1.5: Natural blackjack (21 with first two cards, dealer doesn't have it)
    """

    def __init__(self, natural_reward: float = 1.5, seed: Optional[int] = None):
        self.natural_reward = natural_reward
        self.action_space_n = 2  # 0: Stand, 1: Hit

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Game state
        self.player_cards: List[int] = []
        self.dealer_cards: List[int] = []
        self.deck: List[int] = []
        self.done: bool = False
        self.usable_ace: bool = False
        self.total_steps: int = 0
        self.dealer_sec_card_up = False

        self.history: List[Dict] = []

    def _create_deck(self) -> List[int]:
        """Create a standard 52-card deck (values only)."""
        # 1 = Ace, 2-10 = number cards, 11-13 = J, Q, K (all worth 10)
        deck = []
        for _ in range(4):  # 4 suits
            deck.extend([1] + list(range(2, 11)) + [10, 10, 10])  # A, 2-10, J, Q, K
        random.shuffle(deck)
        return deck

    def _draw_card(self) -> int:
        if not self.deck:
            self.deck = self._create_deck()
        return self.deck.pop()

    def _calculate_hand(self, cards: List[int]) -> Tuple[int, bool]:

        total = sum(cards)
        usable_ace = False

        # Check if there's an ace that can be counted as 11
        if 1 in cards:
            if total + 10 <= 21:
                total += 10
                usable_ace = True

        return total, usable_ace

    def _is_bust(self, hand_sum: int) -> bool:
        return hand_sum > 21

    def reset(self) -> Dict:
        self.deck = self._create_deck()
        self.done = False

        self.player_cards = [self._draw_card(), self._draw_card()]
        self.dealer_cards = [self._draw_card(), self._draw_card()]

        self.player_sum, self.usable_ace = self._calculate_hand(self.player_cards)

        return self._get_observation()

    def _get_observation(self) -> Dict:
        dealer_visible = self.dealer_cards[0]

        player_cards_str = ", ".join(self._card_to_string(c) for c in self.player_cards)
        dealer_card_str = self._card_to_string(dealer_visible)

        description = (
            f"Your hand: [{player_cards_str}] (Total: {self.player_sum}"
            f"{' with usable ace' if self.usable_ace else ''})\n"
            f"Dealer's visible card: {dealer_card_str}"
        )

        return {
            "player_sum": self.player_sum,
            "dealer_card": dealer_visible,
            "usable_ace": self.usable_ace,
            "player_cards": self.player_cards.copy(),
            "dealer_visible_card": dealer_visible,
            "done": self.done,
            "description": description
        }

    def _card_to_string(self, card: int) -> str:
        """Convert card value to string representation."""
        if card == 1:
            return "A"
        elif card == 10:
            return "10/J/Q/K"
        else:
            return str(card)

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        reward = 0.0
        info = {}

        if action == 1:  
            new_card = self._draw_card()
            self.player_cards.append(new_card)
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_cards)

            if self._is_bust(self.player_sum):
                self.done = True
                reward = -1.0
                info["result"] = "bust"
            
            if self.player_sum == 21:
                self.done = True
                reward = +1.0

        else: 
            self.dealer_sum, _ = self._calculate_hand(self.dealer_cards)

            while self.dealer_sum < 17:
                self.dealer_cards.append(self._draw_card())
                self.dealer_sum, _ = self._calculate_hand(self.dealer_cards)

            self.done = True

            dealer_bust = self._is_bust(self.dealer_sum)

            if dealer_bust:
                reward = 1.0
            elif self.player_sum > self.dealer_sum:
                reward = 1.0
            else:
                reward = -1.0

        new_state = self._get_observation()
        return new_state, reward, self.done, info

    def render(self, mode: str = "human") -> Optional[str]:
        player_cards_str = ", ".join(self._card_to_string(c) for c in self.player_cards)
        dealer_cards_str = ", ".join(self._card_to_string(c) for c in self.dealer_cards)

        output = "=" * 50 + "\n"
        output += f"Player's hand: [{player_cards_str}] (Total: {self.player_sum})\n"

        if self.done:
            output += f"Dealer's hand: [{dealer_cards_str}] (Total: {self.dealer_sum})\n"
        else:
            dealer_visible = self._card_to_string(self.dealer_cards[0])
            output += f"Dealer's hand: [{dealer_visible}, ?]\n"

        output += "=" * 50

        if mode == "ansi":
            return output
        else:
            print(output)
            return None

    def get_prompt_for_llm(self) -> str:
        obs = self._get_observation()
        prompt = f"""You are playing Blackjack. Here is the current situation:

{obs['description']}
Available actions:
0: Stand (end your turn and let the dealer play)
1: Hit (draw another card)

What action do you choose? Respond in JSON format with your action and reasoning.

response format:
```json
{{
    "action": <0 or 1>,
    "reasoning": <string describing your reasoning under 50 words>
}}
```
"""
        return prompt


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = BlackjackEnv(seed=42)

    # Play a few episodes
    for episode in range(3):
        print(f"\n{'='*60}")
        print(f"EPISODE {episode + 1}")
        print('='*60)

        obs = env.reset()
        env.render()
        print(f"\n{obs['description']}\n")

        done = False
        total_reward = 0

        while not done:
            # Simple policy: hit if under 17, stand otherwise
            action = 1 if obs['player_sum'] < 17 else 0
            action_name = "HIT" if action == 1 else "STAND"
            print(f"Action: {action_name}")

            obs, reward, done, info = env.step(action)
            total_reward += reward

            env.render()

            if done:
                print(f"\n{info.get('explanation', '')}")
                if 'dealer_hand' in info:
                    print(f"Dealer's final hand: {info['dealer_hand']}")
                print(f"Reward: {reward}")
                print(f"Total Reward: {total_reward}")

        print()

    # Example of LLM prompt generation
    print(f"\n{'='*60}")
    print("EXAMPLE LLM PROMPT")
    print('='*60)
    env.reset()
    print(env.get_prompt_for_llm())
