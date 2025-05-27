import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.cuda.nvtx as nvtx

BATCH_SIZE = 65
INPUT_SIZE = 1
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

@torch.no_grad()
def run_one_decode(model, input_ids):
    outputs = model(
        input_ids=input_ids,
        use_cache=False,
    )
    return outputs

if __name__ == "__main__":
    input_ids = torch.randint(0, tokenizer.vocab_size, (BATCH_SIZE, INPUT_SIZE), device="cuda")

    # Profiling
    elapsed_time_ms = time_pytorch_function(run_one_decode, {
        "model": model,
        "input_ids": input_ids,
    })

    print(f"Elapsed time for decoding {BATCH_SIZE} tokens: {elapsed_time_ms:.2f} ms")