import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from datasets import load_dataset
from codecarbon import EmissionsTracker
import pandas as pd
import torch
import time
from tqdm import tqdm
import os

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # Chargement du dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.n_samples > 0:
        dataset = dataset.select(range(args.start_index, args.start_index + args.n_samples))
    else:
        dataset = dataset.select(range(args.start_index, len(dataset)))

    # Configuration de quantization
    quantization_config = None
    dtype = args.dtype

    if args.quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = None  # doit être None pour bitsandbytes
    elif args.quantization == "4bit":
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        dtype = None

    # Chargement modèle/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
    )

    results = []
    pbar = tqdm(total=len(dataset), desc="Processing dataset")
    pbar.set_postfix({"model": args.model_name, "quant": args.quantization})

    # Warmup
    for _ in range(args.warmup):
        item = dataset[0]
        prompt = item[args.column]
        inputs = tokenizer(prompt, return_tensors="pt").to(pipe.device)
        with torch.no_grad():
            pipe.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                return_dict_in_generate=True,
            )
    pbar.update(args.warmup)

    for item in dataset:
        prompt = item[args.column]
        inputs = tokenizer(prompt, return_tensors="pt").to(pipe.device)

        # Track energy during PREFILL
        tracker = EmissionsTracker(
            log_level="warning",
            tracking_mode="machine",
            gpu_ids=[0],
            allow_multiple_runs=True,
            measure_power_secs=1,
        )

        tracker.start_task("prefill")
        start = time.time()
        with torch.no_grad():
            for _ in range(args.runs):
                pipe.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
        torch.cuda.synchronize()
        end = time.time()
        emissions_prefill = tracker.stop_task()

        duration_prefill = (end - start) / args.runs
        energy_prefill_cpu = emissions_prefill.cpu_energy / args.runs
        energy_prefill_gpu = emissions_prefill.gpu_energy / args.runs
        energy_prefill_ram = emissions_prefill.ram_energy / args.runs

        # Track energy during GENERATE
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
        torch.cuda.synchronize()
        end = time.time()
        emissions_generate = tracker.stop_task()

        duration_generate = (end - start) / args.runs
        energy_generate_cpu = emissions_generate.cpu_energy / args.runs
        energy_generate_gpu = emissions_generate.gpu_energy / args.runs
        energy_generate_ram = emissions_generate.ram_energy / args.runs

        duration_decode = duration_generate - duration_prefill
        energy_decode_cpu = energy_generate_cpu - energy_prefill_cpu
        energy_decode_gpu = energy_generate_gpu - energy_prefill_gpu
        energy_decode_ram = energy_generate_ram - energy_prefill_ram

        generated_text = tokenizer.decode(generated_.sequences[0], skip_special_tokens=False)
        prompt_tokens = len(tokenizer(prompt)["input_ids"])
        response_tokens = len(tokenizer(generated_text)["input_ids"]) - prompt_tokens

        result = {
            'model': args.model_name,
            'quantization': args.quantization,
            'dataset': args.dataset_name,
            'split': args.split,
            'column': args.column,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'duration_prefill': duration_prefill,
            'duration_generate': duration_generate,
            'duration_decode': duration_decode,
            'energy_prefill_cpu': energy_prefill_cpu,
            'energy_prefill_gpu': energy_prefill_gpu,
            'energy_prefill_ram': energy_prefill_ram,
            'energy_generate_cpu': energy_generate_cpu,
            'energy_generate_gpu': energy_generate_gpu,
            'energy_generate_ram': energy_generate_ram,
            'energy_decode_cpu': energy_decode_cpu,
            'energy_decode_gpu': energy_decode_gpu,
            'energy_decode_ram': energy_decode_ram,
        }
        results.append(result)

        # Sauvegarde progressive
        df = pd.DataFrame(results)
        df.index = range(args.start_index, args.start_index + len(df))
        df.to_csv(args.out_csv, index=True)

        pd.DataFrame({
            'prompt': [prompt],
            'generated': [generated_text]
        }).to_csv(args.out_generated, mode='a', header=not os.path.exists(args.out_generated), index=True)

        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--column", type=str, default="conversation_with_thanks")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out_csv", type=str, default="/fsx/jdelavande/benchlab/thank_you/data/Llama-3.1-8B-Instruct-ultrachat_200k-Llama-3-8B-Instruct-with-thanks-energy-{}.csv".format(now))
    parser.add_argument("--out_generated", type=str, default="/fsx/jdelavande/benchlab/thank_you/data/Llama-3.1-8B-Instruct-ultrachat_200k-Llama-3-8B-Instruct-with-thanks-generated-{}.csv".format(now))
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--quantization", type=str, choices=["none", "8bit", "4bit"], default="8bit")
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated list of GPU device IDs to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing samples")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type for the model (e.g., float16, bfloat16, float32)")
    args = parser.parse_args()
    main(args)