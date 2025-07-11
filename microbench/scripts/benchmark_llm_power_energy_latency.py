import argparse
import time
import threading
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx
import pynvml
from codecarbon import EmissionsTracker
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--input_lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--generate_lengths", type=int, nargs="+", default=[10])
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--warmup_runs", type=int, default=2)
    parser.add_argument("--log_power", action="store_true")
    parser.add_argument("--log_energy", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./")
    parser.add_argument("--compile", action="store_true")
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

def run_phase(name, func, args, tracker, config_suffix):
    for _ in range(args.warmup_runs): func()
    torch.cuda.synchronize()

    samples, stop_flag, power_thread = [], threading.Event(), None
    if args.log_power:
        power_thread = threading.Thread(target=log_power, args=(samples, stop_flag))
        power_thread.start()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    if tracker: tracker.start_task(name)

    start_event.record()
    for _ in range(args.runs): func()
    end_event.record()
    torch.cuda.synchronize()

    emissions = tracker.stop_task() if tracker else None

    if args.log_power:
        stop_flag.set()
        power_thread.join()
        df_power = pd.DataFrame(samples, columns=["time_ns", "power_w"])
        df_power["time_ms"] = df_power["time_ns"] / 1_000_000
        date = time.strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{args.output_dir}/power_trace_{name}_{config_suffix}_{date}.csv"
        df_power.to_csv(filename, index=False)
        print(f"✅ Power log saved to {filename}")

    duration = start_event.elapsed_time(end_event) / args.runs
    return duration, emissions

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="cuda", torch_dtype=torch.float32
    )
    model.eval()
    if args.compile:
        model = torch.compile(model)

    tracker = EmissionsTracker(
        log_level="warning",
        tracking_mode="machine",
        gpu_ids=[0],
        allow_multiple_runs=True,
        measure_power_secs=1
    ) if args.log_energy else None

    all_results = []

    for b in args.batch_sizes:
        for s in args.input_lengths:
            for g in args.generate_lengths:
                try:
                    input_ids = torch.randint(
                        low=0, high=tokenizer.vocab_size,
                        size=(b, s),
                        device="cuda", dtype=torch.long
                    )

                    config_suffix = f"b{b}_s{s}_g{g}"
                    result = {
                        "model": args.model_id,
                        "batch_size": b,
                        "input_length": s,
                        "generate_length": g,
                        "runs": args.runs,
                        "warmup_runs": args.warmup_runs,
                        "compile": args.compile
                    }

                    if args.prefill:
                        def prefill():
                            nvtx.range_push("prefill")
                            model.generate(inputs=input_ids, max_new_tokens=1)
                            nvtx.range_pop()

                        duration_prefill, emissions_prefill = run_phase("prefill", prefill, args, tracker, config_suffix)
                        result["duration_prefill"] = duration_prefill
                        if emissions_prefill:
                            result.update({
                                "energy_prefill_cpu": emissions_prefill.cpu_energy / args.runs,
                                "energy_prefill_gpu": emissions_prefill.gpu_energy / args.runs,
                                "energy_prefill_ram": emissions_prefill.ram_energy / args.runs,
                            })

                    if args.decode:
                        def decode():
                            nvtx.range_push("decode")
                            model.generate(inputs=input_ids, max_new_tokens=g)
                            nvtx.range_pop()

                        duration_generate, emissions_generate = run_phase("decode", decode, args, tracker, config_suffix)
                        result["duration_generate"] = duration_generate
                        if emissions_generate:
                            result.update({
                                "energy_generate_cpu": emissions_generate.cpu_energy / args.runs,
                                "energy_generate_gpu": emissions_generate.gpu_energy / args.runs,
                                "energy_generate_ram": emissions_generate.ram_energy / args.runs,
                            })

                        if args.prefill and emissions_generate and emissions_prefill:
                            result["duration_decode"] = duration_generate - duration_prefill
                            result["energy_decode_cpu"] = result["energy_generate_cpu"] - result["energy_prefill_cpu"]
                            result["energy_decode_gpu"] = result["energy_generate_gpu"] - result["energy_prefill_gpu"]
                            result["energy_decode_ram"] = result["energy_generate_ram"] - result["energy_prefill_ram"]

                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error processing batch size {b}, input length {s}, generate length {g}: {e}")
                    continue

    df = pd.DataFrame(all_results)
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_file = f"{args.output_dir}/benchmark_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Benchmark results saved to {output_file}")

if __name__ == "__main__":
    main()
