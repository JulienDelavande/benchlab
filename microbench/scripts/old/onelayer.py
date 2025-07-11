import torch
import time
import torch.cuda.nvtx as nvtx
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

hidden_dim = 4096
ffn_dim = 11008
seq_len = 1

BATCH_SIZE = 64  # This is the batch size to test


def benchmark(batch_size, runs=10):

    torch.cuda.synchronize()
    nvtx.range_push(f"Create Linear Batch Size {batch_size}")
    x = torch.randn(batch_size * seq_len, hidden_dim, device=device)
    linear = torch.nn.Linear(hidden_dim, ffn_dim, bias=False).to(device)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # Warmup
    torch.cuda.synchronize()
    nvtx.range_push(f"Warmup Linear Batch Size {batch_size}")
    for _ in range(5):
        _ = linear(x)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # Actual benchmark
    torch.cuda.synchronize()
    nvtx.range_push(f"Linear Batch Size {batch_size}")
    for _ in range(runs):
        _ = linear(x)
    torch.cuda.synchronize()
    nvtx.range_pop()


    # Measure time
        # Timing
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # torch.cuda.synchronize()

        # nvtx.range_push(f"Linear Batch Size {batch_size}")
        # start.record()  
        # for _ in range(runs):
        #     _ = linear(x)
        # end.record()
        # torch.cuda.synchronize()
        # nvtx.range_pop()

        # avg_time_ms = start.elapsed_time(end) / runs
        # print(f"Batch size: {batch_size:>3} | Avg latency: {avg_time_ms:.2f} ms")
        # return avg_time_ms

if __name__ == "__main__":
    if len(sys.argv) > 1:
        BATCH_SIZE = int(sys.argv[1])
    else:
        print("No batch size provided, using default BATCH_SIZE = 64")
    torch.cuda.synchronize()
    nvtx.range_push(f"Benchmark Linear Layer Batch Size {BATCH_SIZE}")
    benchmark(BATCH_SIZE)
    torch.cuda.synchronize()
    nvtx.range_pop()

    torch.cuda.synchronize()
    nvtx.range_push(f"Benchmark Linear Layer Batch Size {BATCH_SIZE + 1}")
    benchmark(BATCH_SIZE + 1)
    torch.cuda.synchronize()
    nvtx.range_pop()

