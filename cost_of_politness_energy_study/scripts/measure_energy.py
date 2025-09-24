import argparse
import time
from tqdm import tqdm
import os
from typing import Callable, Any
from pathlib import Path
import torch
from transformers import pipeline
from datasets import load_dataset
from codecarbon import EmissionsTracker
import pandas as pd

DEVICE = 'cuda'

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the energy measurement script.

    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.n_samples > 0:
        dataset = dataset.select(range(args.start_index, args.start_index + args.n_samples))
    else:
        dataset = dataset.select(range(args.start_index, len(dataset)))

    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        device=DEVICE,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.dtype
    )

    results = []
    out_csv_path = Path(args.out_csv)
    out_generated_path = Path(args.out_generated)
    os.makedirs(out_csv_path.parent, exist_ok=True)
    os.makedirs(out_generated_path.parent, exist_ok=True)

    pbar = tqdm(total=len(dataset), desc="Processing dataset")
    pbar.set_postfix({"model": args.model_name, "dataset": args.dataset_name})

    if len(dataset) == 0:
        raise ValueError("The dataset is empty. Please check the dataset name and split.")

    #### WARMUP #####
    for i in range(args.warmup):
        item = dataset[i % len(dataset)]
        prompt = item[args.column]
        inputs = pipe.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            pipe.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
            )
    pbar.update(args.warmup)

    #### MEASURE ENERGY CONSUMPTION #####
    pbar.set_description("Measuring energy consumption")

    tracker = EmissionsTracker(
        log_level="warning",
        tracking_mode="machine",
        gpu_ids=gpu_ids,
        allow_multiple_runs=True,
        measure_power_secs=1,
    )

    for item in dataset:
        inputs = pipe.tokenizer(item[args.column], return_tensors="pt").to(device)
        pbar.update(1)
        prompt = item[args.column]

        #### PREFILL #####
        duration_prefill, emissions_prefill, _ = measure_energy(
            tracker, 
            "prefill", 
            lambda: pipe.model.generate(**inputs, max_new_tokens=1, do_sample=False, return_dict_in_generate=True), 
            args.runs
        )
        energy_prefill_cpu = emissions_prefill.cpu_energy / args.runs
        energy_prefill_gpu = emissions_prefill.gpu_energy / args.runs
        energy_prefill_ram = emissions_prefill.ram_energy / args.runs
        
        #### GENERATE #####
        duration_generate, emissions_generate, generated_ = measure_energy(
            tracker,
            "generate",
            lambda: pipe.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, return_dict_in_generate=True),
            args.runs
        )
        energy_generate_cpu = emissions_generate.cpu_energy / args.runs
        energy_generate_gpu = emissions_generate.gpu_energy / args.runs
        energy_generate_ram = emissions_generate.ram_energy / args.runs
        generated = pipe.tokenizer.decode(generated_.sequences[0], skip_special_tokens=False)

        #### DECODE #####
        duration_decode = duration_generate - duration_prefill
        energy_decode_cpu = energy_generate_cpu - energy_prefill_cpu
        energy_decode_gpu = energy_generate_gpu - energy_prefill_gpu
        energy_decode_ram = energy_generate_ram - energy_prefill_ram

        prompt_tokens = len(pipe.tokenizer(prompt)["input_ids"])
        response_tokens = len(pipe.tokenizer(generated)["input_ids"]) - prompt_tokens

        result = {
            'model': args.model_name,
            'dataset': args.dataset_name,
            'split': args.split,
            'column': args.column,
            'dtype': args.dtype,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "duration_prefill": duration_prefill,
            "duration_generate": duration_generate,
            "duration_decode": duration_decode,
            "energy_prefill_cpu": energy_prefill_cpu,
            "energy_prefill_gpu": energy_prefill_gpu,
            "energy_prefill_ram": energy_prefill_ram,
            "energy_generate_cpu": energy_generate_cpu,
            "energy_generate_gpu": energy_generate_gpu,
            "energy_generate_ram": energy_generate_ram,
            "energy_decode_cpu": energy_decode_cpu,
            "energy_decode_gpu": energy_decode_gpu,
            "energy_decode_ram": energy_decode_ram,
        }
        results.append(result)

        df = pd.DataFrame(results)
        df.index = range(args.start_index, args.start_index + len(df))
        df.to_csv(args.out_csv, index=True)
        generated_df = pd.DataFrame({
            'prompt': [prompt],
            'generated': [generated]
        })
        generated_df.to_csv(args.out_generated, mode='a', header=not os.path.exists(args.out_generated), index=True)

    pbar.close()

def measure_energy(tracker: EmissionsTracker, task_name: str, fn: Callable[[], Any], runs: int = 10) -> tuple[float, EmissionsTracker, Any]:
    """ 
    Measure the energy consumption of a function.
    Args:
        tracker (EmissionsTracker): The emissions tracker.
        task_name (str): Name of the task for tracking.
        fn (Callable): The function to measure.
        runs (int): Number of runs to average the measurement.
    Returns:
        tuple: A tuple containing the average duration, emissions tracker, and the result of the function.
    """
    tracker.start_task(task_name)
    start = time.time()
    with torch.no_grad():
        for _ in range(runs):
            generated = fn()
    torch.cuda.synchronize()
    end = time.time()
    emissions = tracker.stop_task()
    torch.cuda.empty_cache()
    return (end - start) / runs, emissions, generated

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure energy consumption of text generation models.")
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Name of the model to use on Hugging Face Hub")
    parser.add_argument("--dataset_name", type=str, default="Anonyme162325/ultrachat_200k-Llama-3-8B-Instruct-with-thanks", help="Name of the dataset to use on Hugging Face Hub")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'test')")
    parser.add_argument("--column", type=str, default="conversation_with_thanks", help="Column in the dataset to use for prompts")
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs for averaging measurements")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs before starting measurements")
    parser.add_argument("--out_csv", type=str, default=f"../data/Llama-3.1-8B-Instruct-ultrachat_200k-Llama-3-8B-Instruct-with-thanks-energy_{now}.csv", help="Path to the output CSV file for energy measurements")
    parser.add_argument("--out_generated", type=str, default=f"../data/Llama-3.1-8B-Instruct-ultrachat_200k-Llama-3-8B-Instruct-with-thanks-generated_{now}.csv", help="Path to the output CSV file for generated samples")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU device IDs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing samples")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for the model (e.g., float16, bfloat16, float32)")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for dataset selection")
    
    args = parser.parse_args()
    main(args)
