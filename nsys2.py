import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.cuda.nvtx as nvtx

BATCH_SIZE = 65
OUTPUT_SIZE = 1
INPUT_SIZE = 128
MODEL_ID = "Qwen/Qwen2.5-7B"
WARMUP_ITERATIONS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda",
#    torch_dtype=torch.float16
)
pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def time_pytorch_function(func, inputs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(WARMUP_ITERATIONS):
        func(**inputs)

    nvtx.range_push(f"generate_bs{BATCH_SIZE}")
    start.record()
    func(**inputs)
    end.record()
    torch.cuda.synchronize()
    nvtx.range_pop()
    return start.elapsed_time(end)

# 2. Initialisation du KV cache
# with torch.no_grad():
#     out = model(input_ids=input_ids, use_cache=True)
#     past_key_values = out.past_key_values

@torch.no_grad()
def run_generate(model, input_ids, attention_mask=None):
    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=OUTPUT_SIZE,
        min_new_tokens=OUTPUT_SIZE,
        pad_token_id=pad_id,
    )


@torch.no_grad()
def run_manual_decode(model, input_ids, attention_mask=None, past_key_values=None):
    generated = []

    for _ in range(OUTPUT_SIZE):
        outputs = model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, -1:, :]
        input_ids = torch.argmax(logits, dim=-1)
        past_key_values = outputs.past_key_values
        # attention_mask = torch.cat(
        #     [attention_mask, torch.ones((BATCH_SIZE, 1), device="cuda", dtype=torch.long)],
        #     dim=1
        # )
        generated.append(input_ids)

    return torch.cat(generated, dim=1)

batch_sizes = [62, 63, 64, 65, 66, 67, 68, 69]
results = []
for bs in batch_sizes:
    BATCH_SIZE = bs

    # Filling the kvcache
    input_ids = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(bs, INPUT_SIZE),
        dtype=torch.long,
        device="cuda"
    )

    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past_key_values = out.past_key_values


    # Next token to decode 
    next_token = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(bs, 1),
        dtype=torch.long,
        device="cuda"
    )
    attention_mask = None
    
    out_call = time_pytorch_function(
        run_manual_decode,
        {
            'model': model,
            'input_ids': next_token,
            'attention_mask': torch.ones((bs, 1), device="cuda", dtype=torch.long),
            'past_key_values': past_key_values,
        }
    )

    print(f"Time taken for manual decode with batch size {bs}: {out_call} ms")
    out_gen = time_pytorch_function(
        run_generate,
        {
            'model': model,
            'input_ids': next_token,
#            'past_key_values': past_key_values,
            'attention_mask': attention_mask,
        }
    )
    print(f"Time taken for generate with batch size {bs}: {out_gen} ms")
print("Profiling completed.")
