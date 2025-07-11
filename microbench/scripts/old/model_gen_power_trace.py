import pynvml
import threading
import time
import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
batch_size = 1
input_length = 193
generate_length = 1
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.float32
)
model.eval()
model.compile()

# Random input generation
input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, input_length),
    dtype=torch.long,
    device="cuda"
)

samples = []
logging_enabled = True

def log_power(sampling_interval_us=100):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start = time.perf_counter_ns()

    while logging_enabled:
        ts = time.perf_counter_ns() - start
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # mW → W
        samples.append((ts, power))
        time.sleep(sampling_interval_us / 1_000_000)  # µs → s

def run_generate(inputs=input_ids, generate_length=generate_length):
    nvtx.range_push(f"generate_bs{batch_size}_input_length{input_length}_newtokens{generate_length}")
    _ = model.generate(
        inputs=input_ids,
        max_new_tokens=generate_length,
    )
    nvtx.range_pop()



# 🔁 Appel au modèle
for _ in range(2):  # Warmup
    run_generate(generate_length=3)

# 🚀 Lance le logger dans un thread
nvtx.mark("logging_started")
t = threading.Thread(target=log_power, kwargs={'sampling_interval_us': 100})  # 100 µs interval
t.start()

torch.cuda.synchronize()
start_gen = torch.cuda.Event(enable_timing=True)
end_gen = torch.cuda.Event(enable_timing=True)
start_gen.record()
for _ in range(200):
    run_generate()
end_gen.record()
torch.cuda.synchronize()

# ⏹️ Stop logging
logging_enabled = False
t.join()

import pandas as pd

df = pd.DataFrame(samples, columns=["time_ns", "power_w"])
df["time_ms"] = df["time_ns"] / 1_000_000

date = time.strftime("%Y-%m-%d-%H-%M-%S")
df.to_csv(f"/fsx/jdelavande/benchlab/microbench/tmp/llama3_power_trace_{date}.csv", index=False)
print(f"\n💾 Power log saved to /fsx/jdelavande/benchlab/microbench/tmp/llama3_power_trace_{date}.csv — {len(df)} samples")