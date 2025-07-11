"""
export output_dir=/fsx/jdelavande/benchlab/microbench/nsys_results/flash_attention2
nsys profile -t cuda,nvtx,cublas --sample=none \
    -o ${output_dir} \
    python flash_attention.py

nsys stats ${output_dir}.nsys-rep >> ${output_dir}.txt
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.cuda.nvtx as nvtx

model_name = "meta-llama/Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"  # <- Active FlashAttention2
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prompt court pour test
inputs = tokenizer("Paris is the capital of", return_tensors="pt").to(model.device)

# Profiling avec NVTX (à visualiser avec Nsight Systems)
nvtx.range_push(f"generate_{model_name}")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=5)
nvtx.range_pop()

print(tokenizer.decode(output[0]))
