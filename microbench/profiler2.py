import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def time_pytorch_function(func, inputs):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(**inputs)

    start.record()
    func(**inputs)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

b = torch.randn(10000, 10000).cuda()
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', device_map="cuda", torch_dtype=torch.float16)
prompt = "Bonjour, comment vas-tu aujourd'hui ? Je voulais te parler de..."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
attention_mask = torch.ones_like(input_ids)

def run_decode(model, input_ids, attention_mask):
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1, min_new_tokens=1)

out = time_pytorch_function(run_decode, {'model': model, 'input_ids': input_ids, 'attention_mask': attention_mask})

print("=============")
print("Profiling run_decode")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_device='cuda') as prof:
    run_decode(model, input_ids, attention_mask)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
print(f'Time taken: {out} ms')
