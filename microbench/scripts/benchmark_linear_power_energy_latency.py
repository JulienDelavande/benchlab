import argparse
import time
import torch
import torch.cuda.nvtx as nvtx
import pandas as pd
import threading
import pynvml
from codecarbon import EmissionsTracker

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[4096])
    parser.add_argument("--ffn_dims", type=int, nargs="+", default=[11008])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[1])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--log_power", action="store_true")
    parser.add_argument("--log_energy", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./")
    return parser.parse_args()

def log_power(samples, stop_flag, interval_us=100):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start = time.perf_counter_ns()
    while not stop_flag.is_set():
        ts = time.perf_counter_ns() - start
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        samples.append((ts, power))
        time.sleep(interval_us / 1_000_000)

def benchmark_linear(batch_size, hidden_dim, ffn_dim, seq_len, runs, log_power=False, tracker=None):
    torch.cuda.synchronize()

    x = torch.randn(batch_size * seq_len, hidden_dim, device=device)
    linear = torch.nn.Linear(hidden_dim, ffn_dim, bias=False).to(device)

    # Warmup
    for _ in range(5):
        _ = linear(x)

    # Power logging
    samples, stop_flag, power_thread = [], threading.Event(), None
    if log_power:
        power_thread = threading.Thread(target=log_power, args=(samples, stop_flag))
        power_thread.start()

    # Energy tracking
    if tracker:
        tracker.start_task("linear")

    # Latency timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    for _ in range(runs):
        _ = linear(x)
    end_event.record()
    torch.cuda.synchronize()

    if tracker:
        emissions = tracker.stop_task()
    else:
        emissions = None

    if log_power:
        stop_flag.set()
        power_thread.join()

    latency_ms = start_event.elapsed_time(end_event) / runs

    result = {
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "ffn_dim": ffn_dim,
        "seq_len": seq_len,
        "latency_ms": latency_ms
    }

    if emissions:
        result.update({
            "energy_cpu_Wh": emissions.cpu_energy / 3600,
            "energy_gpu_Wh": emissions.gpu_energy / 3600,
            "energy_ram_Wh": emissions.ram_energy / 3600
        })

    if log_power:
        df_power = pd.DataFrame(samples, columns=["time_ns", "power_w"])
        df_power["time_ms"] = df_power["time_ns"] / 1_000_000
        filename = f"power_linear_b{batch_size}_h{hidden_dim}_d{ffn_dim}_s{seq_len}.csv"
        df_power.to_csv(f"{args.output_dir}/{filename}", index=False)
        print(f"✅ Power trace saved to {filename}")

    return result

def main():
    global args
    args = parse_args()

    tracker = EmissionsTracker(
        log_level="warning",
        tracking_mode="machine",
        gpu_ids=[0],
        allow_multiple_runs=True,
        measure_power_secs=1
    ) if args.log_energy else None

    all_results = []
    for b in args.batch_sizes:
        for h in args.hidden_dims:
            for d in args.ffn_dims:
                for s in args.seq_lens:
                    print(f"▶ Benchmarking: B={b}, H={h}, D={d}, S={s}")
                    result = benchmark_linear(
                        batch_size=b,
                        hidden_dim=h,
                        ffn_dim=d,
                        seq_len=s,
                        runs=args.runs,
                        log_power=args.log_power,
                        tracker=tracker
                    )
                    all_results.append(result)

    df = pd.DataFrame(all_results)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    out_csv = f"{args.output_dir}/linear_benchmark_{timestamp}.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ Results saved to {out_csv}")

if __name__ == "__main__":
    main()
