"""
Example script demonstrating different LoRA configurations for Blackjack training.

This shows three preset configurations:
1. Lightweight: Fast training, minimal memory
2. Standard: Balanced performance and efficiency
3. High-capacity: Maximum performance, more resources
"""

from train_rlvr import RLVRConfig, RLVRTrainer


def train_with_lora():
    """Standard configuration - recommended for most users."""
    print("\n" + "="*60)
    print("STANDARD CONFIGURATION")
    print("Best for: Consumer GPUs (16GB), balanced training")
    print("="*60)

    config = RLVRConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",

        # LoRA settings - balanced
        use_lora=True,
        lora_r=16,                  # Standard rank
        lora_alpha=32,
        lora_dropout=0.05,

        # Training settings - standard
        num_iterations=10,
        episodes_per_iteration=50,
        batch_size=8,
        learning_rate=2e-5,
        temperature=0.3,            # Lower temperature for less random exploration
        reward_threshold=0.0,       # Only train on winning episodes

        # Evaluation
        eval_frequency=1,
        eval_episodes=30,

        output_dir="./checkpoints_deepseek",
        log_file="./training_log_deepseek.json"
    )

    trainer = RLVRTrainer(config)
    trainer.train()



if __name__ == "__main__":

    print("\n" + "="*60)
    print("RLVR BLACKJACK TRAINING WITH LORA")
    print("="*60)

    train_with_lora()

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nTo evaluate your model:")
    print(f"  python play_blackjack.py --model ./checkpoints_deepseek/final_model --evaluate")
    print()
