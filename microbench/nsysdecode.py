import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-7B"
batch_size = 1
input_length = 1

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


attention_mask = torch.ones((batch_size, input_length + 1), device="cuda", dtype=torch.long)
# Prefill phase setup
def run_prefill():
    nvtx.range_push(f"prefill_bs{batch_size}_input_length{input_length}")
    out = model(
        input_ids=input_ids,
        past_key_values=None,
        attention_mask=attention_mask,
        use_cache=True,
    )
    nvtx.range_pop()
    return out


### Prefill phase
# warmup_prefill = 5
print(f"Running prefill with batch size {batch_size} and input length {input_length}")
for _ in range(5):
    _ = run_prefill()

# Profile_prefill
torch.cuda.synchronize()
start_prefill = torch.cuda.Event(enable_timing=True)
end_prefill = torch.cuda.Event(enable_timing=True)

start_prefill.record()
_ = run_prefill()
torch.cuda.synchronize()
end_prefill.record()

elapsed_time_ms_prefill = start_prefill.elapsed_time(end_prefill)
print(f"\n⏱️ Time taken for prefill: {elapsed_time_ms_prefill:.3f} ms")



# ### Decode phase
# # Prefill phase: generate past_key_values
# with torch.no_grad():
#     out = model(input_ids=input_ids, use_cache=True)
#     past_key_values = out.past_key_values

# # Token to decode
# decode_token = torch.randint(
#     low=0,
#     high=tokenizer.vocab_size,
#     size=(batch_size, 1),
#     dtype=torch.long,
#     device="cuda"
# )

# # Attention mask (1 for all tokens, assuming no padding)
# 

# # Decode phase setup 
# def run_single_decode():
#     nvtx.range_push(f"decode_bs{batch_size}_input_length{input_length}")
#     out = model(
#         input_ids=decode_token,
#         past_key_values=past_key_values,
#         attention_mask=attention_mask,
#         use_cache=True,
#     )
#     nvtx.range_pop()
#     return out


# # Warmup decode
# print(f"Running decode with batch size {batch_size} and input length {input_length}")
# for _ in range(5):
#     _ = run_single_decode()

# # Profile decode
# torch.cuda.synchronize()
# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

# start.record()
# _ = run_single_decode()
# torch.cuda.synchronize()
# end.record()

# elapsed_time_ms = start.elapsed_time(end)
# print(f"\n⏱️ Time taken for decode: {elapsed_time_ms:.3f} ms")
