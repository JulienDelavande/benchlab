import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B"
batch_size = 63

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    # torch_dtype=torch.float16
)
model.eval()

# 1. Séquence aléatoire de 128 tokens (prompt simulé)
input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, 128),
    dtype=torch.long,
    device="cuda"
)

# 2. Premier passage : init du KV cache
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values

# 3. Token à décoder (1 seul)
decode_token = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, 1),
    dtype=torch.long,
    device="cuda"
)

# 4. Mask mis à jour pour 129 tokens
attention_mask = torch.ones((batch_size, 129), device="cuda", dtype=torch.long)

# 5. Fonction de decode unique token
def run_single_decode():
    return model(
        input_ids=decode_token,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
    )

# 6. Warmup
for _ in range(5):
    _ = run_single_decode()

# 7. Mesure du temps CUDA
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
_ = run_single_decode()
end.record()
torch.cuda.synchronize()

elapsed_time_ms = start.elapsed_time(end)
print(f"\n⏱️ Time taken for decode: {elapsed_time_ms:.3f} ms")

# 8. Profiling PyTorch
print("\n=============")
print("🔍 Profiling decode")
print("=============\n")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    _ = run_single_decode()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
