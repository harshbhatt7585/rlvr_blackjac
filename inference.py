"""Run inference with a base model plus optional LoRA adapters."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from env import done
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def select_device(prefer_mps: bool = True) -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    base_model: str,
    adapter_path: Optional[str],
    device: torch.device,
    merge_adapters: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    """Load base model and optionally attach LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype is None:
        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    model.eval()

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapters:
            model = model.merge_and_unload()
        else:
            model.to(device)
        model.eval()

    return model, tokenizer


def read_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt:
        return prompt
    if prompt_file:
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return path.read_text(encoding="utf-8")
    raise ValueError("No prompt provided. Use --prompt or --prompt-file, or --interactive mode.")


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    deterministic: bool,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    do_sample = not deterministic and temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def interactive_loop(args, model, tokenizer, device):
    print("Entering interactive mode. Press Ctrl+C or send an empty prompt to exit.")
    try:
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                break
            completion = generate(
                model,
                tokenizer,
                prompt,
                device,
                args.max_new_tokens,
                args.temperature,
                args.top_p,
                args.repetition_penalty,
                deterministic=args.deterministic,
            )
            print(f"Model: {completion}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting interactive mode.")



def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with PPO-trained LoRA adapters.")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct", help="Base model name or path")
    parser.add_argument("--adapter", default=None, help="Directory with LoRA adapters (e.g. PPO checkpoint)")
    parser.add_argument("--prompt", default=None, help="Prompt text to feed the model")
    parser.add_argument("--prompt-file", default=None, help="Path to file containing prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty factor")
    parser.add_argument("--deterministic", action="store_true", help="Disable sampling for greedy generation")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapters into the base model for inference")
    parser.add_argument("--interactive", action="store_true", help="Run in an interactive REPL loop")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-mps", action="store_true", help="Disable MPS even if available")
    return parser.parse_args(argv)



def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = select_device(prefer_mps=not args.no_mps)
    print(f"Using device: {device}")

    model, tokenizer = load_model(
        base_model=args.base_model,
        adapter_path=args.adapter,
        device=device,
        merge_adapters=args.merge,
    )

    if args.interactive:
        interactive_loop(args, model, tokenizer, device)
        return

    prompt = read_prompt(args.prompt, args.prompt_file)
    completion = generate(
        model,
        tokenizer,
        prompt,
        device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.repetition_penalty,
        deterministic=args.deterministic,
    )

    print("=== Prompt ===")
    print(prompt)
    print("\n=== Completion ===")
    print(completion)


if __name__ == "__main__":
    main(sys.argv[1:])
