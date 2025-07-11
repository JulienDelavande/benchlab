import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker
import pandas as pd

model_id = "meta-llama/Llama-3.1-8B-Instruct"
batch_size = 1
input_length = 128
generate_length = 10
runs = 500
folder = '/fsx/jdelavande/benchlab/microbench/tmp'
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

tracker = EmissionsTracker(
        log_level="warning",
        tracking_mode="machine",
        gpu_ids=[0],
        allow_multiple_runs=True,
        measure_power_secs=1,
    )


# # -------------------------
# # Prefill phase with forward()
# # -------------------------
def run_prefill():
    nvtx.range_push(f"prefill_bs{batch_size}_input_length{input_length}")
    out = model.generate(
        inputs=input_ids,
        max_new_tokens=1,
    )
    nvtx.range_pop()
    return out


print(f"Running prefill with batch size {batch_size} and input length {input_length}")
for _ in range(2):
    _ = run_prefill()

torch.cuda.synchronize()
start_prefill = torch.cuda.Event(enable_timing=True)
end_prefill = torch.cuda.Event(enable_timing=True)
tracker.start_task("prefill")
start_prefill.record()
for _ in range(runs):
    _ = run_prefill()
end_prefill.record()
emissions = tracker.stop_task()
torch.cuda.synchronize()
elapsed_time_ms_prefill = start_prefill.elapsed_time(end_prefill)
duration_prefill = elapsed_time_ms_prefill / runs
energy_prefill_cpu = emissions.cpu_energy / runs
energy_prefill_gpu = emissions.gpu_energy / runs
energy_prefill_ram = emissions.ram_energy / runs

results_prefill = {
    "duration_prefill": duration_prefill,
    "energy_prefill_cpu": energy_prefill_cpu,
    "energy_prefill_gpu": energy_prefill_gpu,
    "energy_prefill_ram": energy_prefill_ram,
}


# -------------------------
# Decode phase with generate()
# -------------------------
generate_kwargs = {
    "inputs": input_ids,
    "max_new_tokens": generate_length,
}

def run_generate():
    nvtx.range_push(f"generate_bs{batch_size}_input_length{input_length}")
    _ = model.generate(**generate_kwargs)
    nvtx.range_pop()

print(f"\nRunning generate with batch size {batch_size} and input length {input_length}")
for _ in range(2):
    run_generate()

torch.cuda.synchronize()
start_gen = torch.cuda.Event(enable_timing=True)
end_gen = torch.cuda.Event(enable_timing=True)
tracker.start_task("generate")
start_gen.record()
for _ in range(runs):
    run_generate()
end_gen.record()
torch.cuda.synchronize()
emissions = tracker.stop_task()

duration_generate = (start_gen.elapsed_time(end_gen)) / runs
energy_generate_cpu = emissions.cpu_energy / runs
energy_generate_gpu = emissions.gpu_energy / runs
energy_generate_ram = emissions.ram_energy / runs

results_generate = {
    "duration_generate": duration_generate,
    "energy_generate_cpu": energy_generate_cpu,
    "energy_generate_gpu": energy_generate_gpu,
    "energy_generate_ram": energy_generate_ram,
}

results_decode = {
    "duration_decode": duration_generate - duration_prefill,
    "energy_decode_cpu": energy_generate_cpu - energy_prefill_cpu,
    "energy_decode_gpu": energy_generate_gpu - energy_prefill_gpu,
    "energy_decode_ram": energy_generate_ram - energy_prefill_ram,
}

results = {
    "model": model_id,
    "batch_size": batch_size,
    "input_length": input_length,
    "generate_length": generate_length,
    "runs": runs,
    **results_prefill,
    **results_generate,
    **results_decode
}

now = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
output_file = f"{folder}/energyllama8Bcompile_decodeprefillgenerate_results_bs{batch_size}_input_length{input_length}_runs{runs}_{now}.csv"
df = pd.DataFrame([results])
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")