import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZES = range(55, 70, 1)  # Batch sizes from 1 to 64
OUTPUT_SIZE = 1
INPUT_SIZE = 128
WARMUP_ITERATIONS = 5
MODEL_ID = "Qwen/Qwen2.5-7B"
OUT_CSV = "latency_results55_70.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda")

@torch.no_grad()
def time_pytorch_function(func, inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(WARMUP_ITERATIONS):
        func(**inputs)
    start.record()
    func(**inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

@torch.no_grad()
def run_generate(model, input_ids, attention_mask):
    return model.generate(input_ids=input_ids, attention_mask=attention_mask,
                          max_new_tokens=OUTPUT_SIZE, min_new_tokens=OUTPUT_SIZE)

@torch.no_grad()
def run_forward(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)

results = []
for bs in tqdm(BATCH_SIZES, desc="Processing batch sizes"):
    input_ids = torch.randint(0, tokenizer.vocab_size, (bs, INPUT_SIZE), device="cuda")
    attn_mask = torch.ones_like(input_ids)

    try:
        gen_latency = time_pytorch_function(run_generate, {
            "model": model,
            "input_ids": input_ids[:, -1:],
            "attention_mask": torch.ones((bs, INPUT_SIZE + 1), device="cuda")
        })

        fwd_latency = time_pytorch_function(run_forward, {
            "model": model,
            "input_ids": input_ids,
            "attention_mask": attn_mask
        })

        results.append({
            "batch_size": bs,
            "generate_latency_ms": gen_latency,
            "forward_latency_ms": fwd_latency
        })
    except Exception as e:
        print(f"Failed for batch size {bs}: {e}")

df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)

# plt.plot(df["batch_size"], INPUT_SIZE*df["batch_size"]/(1000*df["generate_latency_ms"]), label="generate")
# plt.plot(df["batch_size"], INPUT_SIZE*df["batch_size"]/(1000*df["forward_latency_ms"]), label="forward")
# plt.xlabel("Batch Size")
# plt.ylabel("throughput (tokens/sec)")
# plt.title("Latency vs Batch Size")
# plt.legend()
# plt.grid(True)
# plt.savefig("latency_plot.png")
# plt.show()

df = pd.read_csv(OUT_CSV)
plt.plot(df["batch_size"], df["generate_latency_ms"], label="generate")
plt.plot(df["batch_size"], df["forward_latency_ms"], label="forward")
plt.xlabel("Batch Size")
plt.ylabel("Latency (ms)")
plt.title("Latency vs Batch Size")
plt.legend()
plt.grid(True)
plt.savefig(OUT_CSV.replace(".csv", ".png"))
plt.show()