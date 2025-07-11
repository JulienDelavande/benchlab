import pynvml
import threading
import time
import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.float32
)
model.eval()
model.compile()

output_dir = "/fsx/jdelavande/benchlab/microbench/power_data2"
os.makedirs(output_dir, exist_ok=True)

# Power logging
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
        time.sleep(sampling_interval_us / 1_000_000)

def run_prefill(input_ids):
    nvtx.range_push("prefill")
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    nvtx.range_pop()

def run_decode(input_ids, max_new_tokens):
    nvtx.range_push("decode")
    _ = model.generate(inputs=input_ids, max_new_tokens=max_new_tokens)
    nvtx.range_pop()

def log_and_run(run_fn, label):
    global samples, logging_enabled
    samples = []
    logging_enabled = True

    t = threading.Thread(target=log_power, kwargs={'sampling_interval_us': 100})
    t.start()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    run_fn()

    end.record()
    torch.cuda.synchronize()
    logging_enabled = False
    t.join()

    df = pd.DataFrame(samples, columns=["time_ns", "power_w"])
    df["time_ms"] = df["time_ns"] / 1_000_000
    df.to_csv(os.path.join(output_dir, f"{label}.csv"), index=False)
    print(f"💾 Saved {label}.csv with {len(df)} samples")

# -------------------
# Phase 1: Prefill sweep over input lengths s
# -------------------
s_list = [i for i in range(1, 1000, 1)] 

for s in s_list:
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(1, s),
        dtype=torch.long,
        device="cuda"
    )
    # Warmup for each input length
    run_prefill(input_ids)
    run_fn = lambda: [run_prefill(input_ids) for _ in range(200)]
    log_and_run(run_fn, f"prefill_s{s}")

# -------------------
# Phase 2: Decode sweep over batch sizes b
# -------------------
b_list = [i for i in range(1, 200, 1)]  # Batch sizes from 1 to 99
generate_length = 3000
prompt_len = 1

# Warmup
input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(1, prompt_len),
    dtype=torch.long,
    device="cuda"
)
run_decode(input_ids, max_new_tokens=10)

for b in b_list:
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(b, prompt_len),
        dtype=torch.long,
        device="cuda"
    )
    run_fn = lambda: run_decode(input_ids, max_new_tokens=generate_length)
    log_and_run(run_fn, f"decode_b{b}_g{generate_length}")
