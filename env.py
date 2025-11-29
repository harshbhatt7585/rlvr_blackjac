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
        """
        Initialize the Blackjack environment.

        Args:
            natural_reward: Reward multiplier for natural blackjack (default 1.5)
            seed: Random seed for reproducibility
        """
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

        # Episode tracking
        self.done = False
        self.player_sum = 0
        self.dealer_sum = 0
        self.usable_ace = False

    def _create_deck(self) -> List[int]:
        """Create a standard 52-card deck (values only)."""
        # 1 = Ace, 2-10 = number cards, 11-13 = J, Q, K (all worth 10)
        deck = []
        for _ in range(4):  # 4 suits
            deck.extend([1] + list(range(2, 11)) + [10, 10, 10])  # A, 2-10, J, Q, K
        random.shuffle(deck)
        return deck

    def _draw_card(self) -> int:
        """Draw a card from the deck."""
        if not self.deck:
            self.deck = self._create_deck()
        return self.deck.pop()

    def _calculate_hand(self, cards: List[int]) -> Tuple[int, bool]:
        """
        Calculate the value of a hand.

        Args:
            cards: List of card values

        Returns:
            Tuple of (hand_sum, usable_ace)
            - hand_sum: Total value of the hand
            - usable_ace: Whether there's an ace counted as 11
        """
        total = sum(cards)
        usable_ace = False

        # Check if there's an ace that can be counted as 11
        if 1 in cards:
            if total + 10 <= 21:
                total += 10
                usable_ace = True

        return total, usable_ace

    def _is_bust(self, hand_sum: int) -> bool:
        """Check if a hand has busted (over 21)."""
        return hand_sum > 21

    def _is_natural(self, cards: List[int]) -> bool:
        """Check if hand is a natural blackjack (21 with first two cards)."""
        return len(cards) == 2 and sum(cards) + (10 if 1 in cards else 0) == 21

    def reset(self) -> Dict:
        """
        Reset the environment for a new episode.

        Returns:
            Initial state observation
        """
        self.deck = self._create_deck()
        self.done = False

        # Deal initial cards
        self.player_cards = [self._draw_card(), self._draw_card()]
        self.dealer_cards = [self._draw_card(), self._draw_card()]

        # Calculate initial sums
        self.player_sum, self.usable_ace = self._calculate_hand(self.player_cards)

        return self._get_observation()

    def _get_observation(self) -> Dict:
        """
        Get the current state observation.

        Returns:
            Dictionary containing:
            - player_sum: Current sum of player's hand
            - dealer_card: Dealer's visible card
            - usable_ace: Whether player has usable ace
            - player_cards: Player's cards
            - dealer_visible_card: Dealer's first card
            - done: Whether episode is finished
            - description: Text description of the state (for LLM)
        """
        dealer_visible = self.dealer_cards[0]

        # Create text description for LLM
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
        """
        Execute one step in the environment.

        Args:
            action: 0 for Stand, 1 for Hit

        Returns:
            Tuple of (observation, reward, done, info)
            - observation: Current state
            - reward: Reward received
            - done: Whether episode is finished
            - info: Additional information
        """
        if self.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")

        info = {
            "player_natural": self._is_natural(self.player_cards),
            "dealer_natural": False,
            "result": None
        }

        reward = 0.0

        if action == 1:  # Hit
            # Player draws a card
            new_card = self._draw_card()
            self.player_cards.append(new_card)
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_cards)

            # Check if player busts
            if self._is_bust(self.player_sum):
                self.done = True
                reward = -1.0
                info["result"] = "bust"
                info["explanation"] = f"Player busts with {self.player_sum}"

        else: 
            self.dealer_sum, _ = self._calculate_hand(self.dealer_cards)
            info["dealer_natural"] = self._is_natural(self.dealer_cards)

            while self.dealer_sum < 17:
                self.dealer_cards.append(self._draw_card())
                self.dealer_sum, _ = self._calculate_hand(self.dealer_cards)

            self.done = True

            dealer_bust = self._is_bust(self.dealer_sum)
            player_natural = info["player_natural"]
            dealer_natural = info["dealer_natural"]

            if dealer_bust:
                reward = 1.0
                info["result"] = "dealer_bust"
                info["explanation"] = f"Dealer busts with {self.dealer_sum}"
            elif player_natural and not dealer_natural:
                reward = self.natural_reward
                info["result"] = "natural_win"
                info["explanation"] = "Natural blackjack!"
            elif self.player_sum > self.dealer_sum:
                reward = 1.0
                info["result"] = "win"
                info["explanation"] = f"Player wins ({self.player_sum} vs {self.dealer_sum})"
            elif self.player_sum == self.dealer_sum:
                reward = 0.0
                info["result"] = "draw"
                info["explanation"] = f"Draw ({self.player_sum})"
            else:
                reward = -1.0
                info["result"] = "lose"
                info["explanation"] = f"Dealer wins ({self.dealer_sum} vs {self.player_sum})"

            dealer_cards_str = ", ".join(self._card_to_string(c) for c in self.dealer_cards)
            info["dealer_hand"] = f"[{dealer_cards_str}] (Total: {self.dealer_sum})"

        observation = self._get_observation()
        return observation, reward, self.done, info

    def render(self, mode: str = "human") -> Optional[str]:
        """
        Render the current state.

        Args:
            mode: Render mode ('human' for print, 'ansi' for string)

        Returns:
            String representation if mode is 'ansi', None otherwise
        """
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

    def get_action_mask(self) -> List[bool]:
        """
        Get valid actions in current state.

        Returns:
            List of booleans indicating valid actions
        """
        if self.done:
            return [False, False]
        return [True, True]  # Both stand and hit are always valid when not done

    def get_prompt_for_llm(self) -> str:
        """
        Generate a prompt suitable for LLM decision making.

        Returns:
            String prompt describing the current state and asking for action
        """
        obs = self._get_observation()
        prompt = f"""You are playing Blackjack. Here is the current situation:

{obs['description']}

Available actions:
0: Stand (end your turn and let the dealer play)
1: Hit (draw another card)

What action do you choose? Respond in JSON format with your action and reasoning.

Example response:
```json
{{
    "action": 0,
    "reasoning": "Your reasoning here"
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
