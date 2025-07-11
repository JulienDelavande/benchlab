import torch
import time
import csv
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

hidden_dim = 4096
ffn_dim = 11008
seq_len = 1  # Default sequence length for the benchmark
batch_size = 1  # Default batch size for the benchmark
runs = 10
warmup_runs = 5


batch_sizes = [batch_size]
seq_lens = [seq_len]  # List of sequence lengths, can be extended if needed)
hidden_dims = [hidden_dim]  # List of hidden dimensions, can be extended if needed
ffn_dims = list(range(1, 65536))  # List of feed-forward dimensions, can be extended if needed

METRIC = "ffn_dims"  # Metric to benchmark, can be changed to "batch_size" or "hidden_dim"
metric = ffn_dims
FOLDER_OUTPUT = "tmp"
NAME_OUTPUT = f"{FOLDER_OUTPUT}/linear_latency_results_{METRIC}"

def benchmark(batch_size, seq_len=1, ffn_dim=ffn_dim, hidden_dim=hidden_dim, runs=runs):
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    linear = torch.nn.Linear(hidden_dim, ffn_dim, bias=False).to(device)

    # Warmup
    for _ in range(warmup_runs):
        _ = linear(x)
    torch.cuda.synchronize()

    # Mesure de la latence
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(runs):
        _ = linear(x)
    end.record()

    torch.cuda.synchronize()
    avg_time_ms = start.elapsed_time(end) / runs
    return avg_time_ms

if __name__ == "__main__":
    
    results = []

    for hidden_dim in hidden_dims:
        for ffn_dim in ffn_dims:
            for seq_len in seq_lens:
                for bs in batch_sizes:
                    avg_time = benchmark(bs, seq_len=seq_len, ffn_dim=ffn_dim, hidden_dim=hidden_dim, runs=runs)
                    print(f"{METRIC}: {ffn_dim:>3} | Avg latency: {avg_time:.2f} ms")
                    results.append((hidden_dim, ffn_dim, seq_len, bs, avg_time))

    # Sauvegarde dans un CSV
    with open(f"{NAME_OUTPUT}.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["hidden_dim", "ffn_dim", "seq_len", "batch_size", "avg_latency_ms"])
        writer.writerows(results)

    # Tracé du graphe
    
    hidden_dims, ffn_dims, seq_lens, batch_sizes, avg_latencies = zip(*results)
    plt.figure(figsize=(14, 6))
    plt.plot(metric, avg_latencies, linestyle='-', color='b')
    plt.xlabel(METRIC)
    plt.ylabel("Average Latency (ms)")
    plt.title(f"Latency vs {METRIC} (Linear Layer)")
    plt.grid(True)
    plt.savefig(f"{NAME_OUTPUT}.png")
    plt.show()
