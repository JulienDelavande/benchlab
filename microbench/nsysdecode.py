import torch
import torch.cuda.nvtx as nvtx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen2.5-7B"
batch_size = 64

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=torch.float16  # conseillé avec Qwen 7B
)
model.eval()

# 1. Séquence aléatoire de 128 tokens
input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, 128),
    dtype=torch.long,
    device="cuda"
)

# 2. Initialisation du KV cache
with torch.no_grad():
    out = model(input_ids=input_ids, use_cache=True)
    past_key_values = out.past_key_values

# 3. Token à décoder (1 token par élément de batch)
decode_token = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(batch_size, 1),
    dtype=torch.long,
    device="cuda"
)

# 4. Attention mask mis à jour
attention_mask = torch.ones((batch_size, 129), device="cuda", dtype=torch.long)

# 5. Fonction de decode unique token, avec balise NVTX pour nsys
def run_single_decode():
    nvtx.range_push(f"decode_bs{batch_size}")
    out = model(
        input_ids=decode_token,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
    )
    nvtx.range_pop()
    return out

# 6. Warmup
for _ in range(5):
    _ = run_single_decode()

# 7. Passage profilé
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
_ = run_single_decode()
end.record()
torch.cuda.synchronize()

elapsed_time_ms = start.elapsed_time(end)
print(f"\n⏱️ Time taken for decode: {elapsed_time_ms:.3f} ms")
