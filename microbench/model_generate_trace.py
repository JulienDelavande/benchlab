import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct"
batch_size = 1
input_length = 1_000
generate_length = 4
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.float32
)
model.eval()

# Random input generation
input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, input_length),
    dtype=torch.long,
    device="cuda"
)

# # -------------------------
# # Prefill phase with forward()
# # -------------------------
# def run_prefill():
#     nvtx.range_push(f"prefill_bs{batch_size}_input_length{input_length}")
#     out = model.generate(
#         inputs=input_ids,
#         max_new_tokens=1,
#     )
#     nvtx.range_pop()
#     return out


# print(f"Running prefill with batch size {batch_size} and input length {input_length}")
# for _ in range(2):
#     _ = run_prefill()

# torch.cuda.synchronize()
# start_prefill = torch.cuda.Event(enable_timing=True)
# end_prefill = torch.cuda.Event(enable_timing=True)
# start_prefill.record()
# _ = run_prefill()
# torch.cuda.synchronize()
# end_prefill.record()
# elapsed_time_ms_prefill = start_prefill.elapsed_time(end_prefill)
# print(f"\n⏱️ Time taken for prefill (forward): {elapsed_time_ms_prefill:.3f} ms")


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
start_gen.record()
run_generate()
torch.cuda.synchronize()
end_gen.record()
elapsed_time_ms_gen = start_gen.elapsed_time(end_gen)
print(f"\n🚀 Time taken with generate(): {elapsed_time_ms_gen:.3f} ms")
