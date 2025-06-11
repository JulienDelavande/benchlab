import argparse
from transformers import pipeline
from datasets import load_dataset
from codecarbon import EmissionsTracker
import pandas as pd
import torch
import time
from tqdm import tqdm
import os

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    dataset = load_dataset(args.dataset_name, split=args.split).select(range(args.start_index, args.n_samples)) if args.n_samples > 0 \
        else load_dataset(args.dataset_name, split=args.split).select(range(args.start_index, len(load_dataset(args.dataset_name, split=args.split))))

    pipe = pipeline(
        "text-generation",
        model=args.model_name,
        device=0,
        max_new_tokens=args.max_new_tokens,
    )
    if pipe.tokenizer.pad_token is None:
        pipe.tokenizer.pad_token = pipe.tokenizer.eos_token


    results = []

    pbar = tqdm(total=len(dataset), desc="Processing dataset")
    pbar.set_postfix({"model": args.model_name, "dataset": args.dataset_name})

    #### WARMUP #####
    for _ in range(args.warmup):
        batch = dataset.select(range(0, min(args.batch_size, len(dataset))))
        prompts = batch[args.column]  # Accès direct à la colonne (renvoie une liste)
        inputs = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(pipe.device)
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
    for i in range(0, len(dataset), args.batch_size):
        try:
            batch = dataset.select(range(i, min(i + args.batch_size, len(dataset))))
            prompts = batch[args.column]  # Accès direct à la colonne (renvoie une liste)

            tracker = EmissionsTracker(
                log_level="warning",
                tracking_mode="machine",
                gpu_ids=[0],
                allow_multiple_runs=True,
                measure_power_secs=1,
            )

            inputs = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(pipe.device)
            
            #### PREFILL #####
            torch.cuda.synchronize()
            tracker.start_task("prefill")
            start = time.time()
            with torch.no_grad():
                for _ in range(args.runs):
                    __ = pipe.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=False,
                        return_dict_in_generate=True,
                    )
            torch.cuda.synchronize()
            end = time.time()
            emissions = tracker.stop_task()

            duration_prefill = (end - start) / args.runs
            energy_prefill_cpu = emissions.cpu_energy / args.runs
            energy_prefill_gpu = emissions.gpu_energy / args.runs
            energy_prefill_ram = emissions.ram_energy / args.runs
            
            #### GENERATE #####
            torch.cuda.synchronize()
            tracker.start_task("generate")
            start = time.time()
            with torch.no_grad():
                for _ in range(args.runs):
                    generated_ = pipe.model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                    )
            generated_texts = pipe.tokenizer.batch_decode(generated_.sequences, skip_special_tokens=False)
    
            end = time.time()

            emissions = tracker.stop_task()

            duration_generate = (end - start) / args.runs
            energy_generate_cpu = emissions.cpu_energy / args.runs
            energy_generate_gpu = emissions.gpu_energy / args.runs
            energy_generate_ram = emissions.ram_energy / args.runs

            #### DECODE #####
            duration_decode = duration_generate - duration_prefill
            energy_decode_cpu = energy_generate_cpu - energy_prefill_cpu
            energy_decode_gpu = energy_generate_gpu - energy_prefill_gpu
            energy_decode_ram = energy_generate_ram - energy_prefill_ram

            prompt_tokens = [len(pipe.tokenizer(prompt)["input_ids"]) for prompt in prompts]
            response_tokens = [len(pipe.tokenizer(generated)["input_ids"]) - pt for generated, pt in zip(generated_texts, prompt_tokens)]

            result = {
                'model': args.model_name,
                'dataset': args.dataset_name,
                'split': args.split,
                'column': args.column,
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
                "batch_size": len(batch),
            }
            results.append(result)

            # Save result to CSV
            df = pd.DataFrame(results)
            # modify the index to start from --start_index
            df.index = range(args.start_index, args.start_index + len(df))
            df.to_csv(args.out_csv, index=True)
            # Save generated text to CSV
            # generated_df = pd.DataFrame({
            #     'prompt': [prompt],
            #     'generated': [generated]
            # })
            # generated_df.to_csv(args.out_generated, mode='a', header=not os.path.exists(args.out_generated), index=True)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--column", type=str, default="conversation_with_thanks")
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out_csv", type=str, default="/fsx/jdelavande/benchlab/thank_you/data/Llama-3.1-8B-Instruct-ultrachat_200k-Llama-3-8B-Instruct-with-thanks-energy.csv")
    parser.add_argument("--out_generated", type=str, default="/fsx/jdelavande/benchlab/thank_you/data/Llama-3.1-8B-Instruct-ultrachat_200k-Llama-3-8B-Instruct-with-thanks-generated.csv")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for dataset selection")
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated list of GPU device IDs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing samples")
    

    args = parser.parse_args()
    main(args)
