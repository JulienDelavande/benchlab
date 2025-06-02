import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import nvtx
import time

@nvtx.annotate("prefill", color="blue")
def run_prefill(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask)

@nvtx.annotate("decode", color="green")
def run_decode(model, input_ids, attention_mask):
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B', device_map="cuda", torch_dtype=torch.float16)

prompt = "Bonjour, comment vas-tu aujourd'hui ? Je voulais te parler de..."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
attention_mask = torch.ones_like(input_ids)

# Prefill
out = run_prefill(model, input_ids, attention_mask)
torch.cuda.synchronize()

# Decode
out = run_decode(model, input_ids, attention_mask)
torch.cuda.synchronize()
