"""PPO fine-tuning script configured for Apple Silicon GPUs."""

from types import SimpleNamespace

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import PPOConfig, PPOTrainer


# LoRA configuration keeps the policy lightweight
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Model + tokenizer ids
model_id = "meta-llama/Llama-3.2-1B-Instruct"

# Select dtype/device based on MPS availability
use_mps = torch.backends.mps.is_available()
dtype = torch.bfloat16
device_map = {"": "mps"} if use_mps else "auto"

# Load tokenizer and ensure padding token exists
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Policy model (with LoRA adapters)
policy_base = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device_map,
)
policy_base.config.use_cache = False
policy_model = get_peft_model(policy_base, lora_config)
policy_model.print_trainable_parameters()


# Value network (kept light but needed for PPO advantage estimates)
value_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=dtype,
    device_map=device_map,
)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=1,
    torch_dtype=dtype,
    device_map=device_map,
)
value_model.config.use_cache = False

if use_mps:
    policy_model = policy_model.to("mps")
    value_model = value_model.to("mps")

# Load a simple prompt-only dataset (toy example)
dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")


def tokenize(example):
    tokenized = tokenizer(text=example["prompt"], truncation=True, padding="max_length", max_length=256)
    if tokenized["input_ids"][-1] != tokenizer.eos_token_id:
        tokenized["input_ids"].append(tokenizer.eos_token_id)
        tokenized["attention_mask"].append(1)
    return tokenized


tokenized_dataset = dataset.map(tokenize, remove_columns="prompt")

# PPO configuration tuned for small-batch hardware
training_args = PPOConfig(
    output_dir="ppo-model",
    per_device_train_batch_size=1,
    batch_size=1,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
    num_ppo_epochs=1,
    learning_rate=5e-6,
    report_to="none",
)

# Create trainer
trainer = PPOTrainer(
    args=training_args,
    processing_class=tokenizer,
    model=policy_model,
    ref_model=None,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

if use_mps:
    torch.mps.empty_cache()

trainer.train()
