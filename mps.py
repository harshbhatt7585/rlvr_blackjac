import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
prompt = "What is the meaning of life?"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load model in BF16 on MPS
device = "mps"
torch_dtype = torch.bfloat16

print("Loading model on MPS (bf16)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch_dtype,
    device_map={"": device},
)

model.eval()

# Prepare input
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Create streamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Launch generation in background thread
def generate_async():
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            streamer=streamer,
        )

thread = Thread(target=generate_async)
thread.start()

# Stream tokens as they come
print("\n=== Streaming Output (MPS, BF16) ===\n")
for token in streamer:
    print(token, end="", flush=True)

thread.join()
print("\n\n=== Done ===")
