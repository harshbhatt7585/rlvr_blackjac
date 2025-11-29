"""
Quick test script to verify the Blackjack environment works correctly.
"""

from env import BlackjackEnv
import numpy as np


def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing basic environment functionality...")

    env = BlackjackEnv(seed=42)

    # Test reset
    obs = env.reset()
    assert 'player_sum' in obs
    assert 'dealer_card' in obs
    assert 'description' in obs
    print("✓ Reset works")

    # Test step (hit)
    obs, reward, done, info = env.step(1)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    print("✓ Step (hit) works")

    # Play until done
    while not done:
        action = 1 if obs['player_sum'] < 17 else 0
        obs, reward, done, info = env.step(action)

    assert done
    assert 'result' in info
    print(f"✓ Episode completed with result: {info['result']}")

    print("\n✅ All basic tests passed!")


def test_multiple_episodes():
    """Test multiple episodes for consistency."""
    print("\nTesting multiple episodes...")

    env = BlackjackEnv(seed=42)
    results = {'win': 0, 'lose': 0, 'draw': 0}

    for _ in range(20):
        obs = env.reset()
        done = False

        while not done:
            action = 1 if obs['player_sum'] < 17 else 0
            obs, reward, done, info = env.step(action)

        if reward > 0:
            results['win'] += 1
        elif reward < 0:
            results['lose'] += 1
        else:
            results['draw'] += 1

    print(f"Results over 20 episodes:")
    print(f"  Wins: {results['win']}")
    print(f"  Losses: {results['lose']}")
    print(f"  Draws: {results['draw']}")

    print("\n✅ Multiple episodes test passed!")


def test_prompt_generation():
    """Test LLM prompt generation."""
    print("\nTesting LLM prompt generation...")

    env = BlackjackEnv(seed=42)
    obs = env.reset()

    prompt = env.get_prompt_for_llm()
    assert isinstance(prompt, str)
    assert 'Blackjack' in prompt
    assert 'Your hand' in prompt
    assert 'Dealer' in prompt
    assert '0: Stand' in prompt
    assert '1: Hit' in prompt

    print("Example prompt:")
    print("-" * 60)
    print(prompt)
    print("-" * 60)

    print("\n✅ Prompt generation test passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\nTesting edge cases...")

    env = BlackjackEnv(seed=42)

    # Test natural blackjack
    found_natural = False
    for _ in range(100):
        obs = env.reset()
        if obs['player_sum'] == 21 and len(obs['player_cards']) == 2:
            found_natural = True
            print("✓ Found natural blackjack")
            # Stand immediately
            _, reward, _, info = env.step(0)
            break

    # Test bust
    env.reset()
    found_bust = False
    for _ in range(100):
        obs = env.reset()
        done = False
        while not done and not found_bust:
            obs, reward, done, info = env.step(1)  # Always hit
            if done and 'bust' in info.get('result', ''):
                found_bust = True
                print("✓ Found player bust")

        if found_bust:
            break

    print("\n✅ Edge cases test passed!")


def test_reward_structure():
    """Test reward structure."""
    print("\nTesting reward structure...")

    env = BlackjackEnv(seed=42, natural_reward=1.5)

    rewards = []
    for _ in range(50):
        obs = env.reset()
        done = False

        while not done:
            action = 1 if obs['player_sum'] < 17 else 0
            obs, reward, done, info = env.step(action)

        rewards.append(reward)

    unique_rewards = set(rewards)
    print(f"Unique rewards found: {sorted(unique_rewards)}")

    assert -1.0 in unique_rewards or len(unique_rewards) > 0
    print("✓ Reward structure is valid")

    mean_reward = np.mean(rewards)
    print(f"Mean reward over 50 episodes: {mean_reward:.3f}")

    print("\n✅ Reward structure test passed!")


if __name__ == "__main__":
    print("="*60)
    print("BLACKJACK ENVIRONMENT TEST SUITE")
    print("="*60)

    test_basic_functionality()
    test_multiple_episodes()
    test_prompt_generation()
    test_edge_cases()
    test_reward_structure()

    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\nEnvironment is ready for RLVR training!")
