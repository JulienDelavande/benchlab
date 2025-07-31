import random
import asyncio
import httpx
import pandas as pd
from datasets import load_dataset
import time
from codecarbon import EmissionsTracker
import argparse

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the energy measurement script.
    Args:
        args (argparse.Namespace): Parsed command line arguments.
    """
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]

    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.n_samples > 0:
        dataset = dataset.select(range(args.start_index, args.start_index + args.n_samples))
    prompts = dataset[args.column]

    tracker = EmissionsTracker(
        log_level="warning",
        tracking_mode="machine",
        gpu_ids=gpu_ids,
        allow_multiple_runs=True,
        measure_power_secs=1,
    )

    results, start_time, end_time, emissions = asyncio.run(run_all(
        prompts,
        tracker,
        args.warmup_runs,
        args.break_min,
        args.break_max,
        args.max_new_tokens,
        args.inference_url
    ))

    summary = pd.DataFrame([{
        "total_requests": len(results),
        "total_duration": end_time - start_time,
        "total_energy_cpu": emissions.cpu_energy,
        "total_energy_gpu": emissions.gpu_energy,
        "total_energy_ram": emissions.ram_energy,
    }])
    summary.to_csv(args.out_csv, index=False)


async def run_all(prompts: list[str], tracker: EmissionsTracker, warmup_runs: int, break_min: float, break_max: float, max_new_tokens: int, inference_url: str) -> tuple[list[dict], float, float, float]:
    """
    Run all requests with the given parameters.

    Args:
        prompts (list[str]): List of prompts to process.
        tracker (EmissionsTracker): The emissions tracker.
        warmup_runs (int): Number of warmup runs.
        break_min (float): Minimum break time between requests.
        break_max (float): Maximum break time between requests.
        max_new_tokens (int): Maximum number of new tokens to generate.
        inference_url (str): URL of the inference server.

    Returns:
        tuple: A tuple containing the results list, start time, end time, and emissions.
    """
    results = []
    tasks = []

    async with httpx.AsyncClient(timeout=30.0) as session:
        await warmup(session, prompts[0], tracker, warmup_runs, max_new_tokens, inference_url)

        print("Starting energy tracking after warmup...")
        tracker.start_task("dataset_requests")
        start_time = time.time()

        for idx, prompt in enumerate(prompts):
            delay = sum(random.uniform(break_min, break_max) for _ in range(idx))
            async def delayed_request(prompt=prompt, idx=idx, delay=delay):
                await asyncio.sleep(delay)
                try:
                    result = await send_request(session, prompt, idx, max_new_tokens, inference_url)
                except Exception:
                    result = {
                        "index": idx,
                        "prompt_length": len(prompt),
                        "status": "error",
                        "latency": -1
                    }
                results.append(result)

            tasks.append(asyncio.create_task(delayed_request()))

        await asyncio.gather(*tasks)

        end_time = time.time()
        emissions = tracker.stop_task()

    return results, start_time, end_time, emissions

async def send_request(session: httpx.AsyncClient, prompt: str, idx: int, max_new_tokens: int, inference_url: str) -> dict:
    """
    Send a request to the inference server and return the result.
    
    Args:        
        session (httpx.AsyncClient): The HTTP client session.
        prompt (str): The prompt to send.
        idx (int): The index of the prompt in the dataset.
        max_new_tokens (int): Maximum number of new tokens to generate.
        inference_url (str): URL of the inference server.
    Returns:
        dict: A dictionary containing the index, prompt length, status code, and latency.
    """

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False
        }
    }
    start = time.time()
    response = await session.post(inference_url, json=payload)
    end = time.time()
    return {
        "index": idx,
        "prompt_length": len(prompt),
        "status": response.status_code,
        "latency": end - start
    }

async def warmup(session: httpx.AsyncClient, prompt: str, tracker: EmissionsTracker, n: int, max_new_tokens: int, inference_url: str) -> None:
    """
    Perform warmup requests to the inference server.

    Args:
        session (httpx.AsyncClient): The HTTP client session.
        prompt (str): The prompt to use for warmup.
        tracker (EmissionsTracker): The emissions tracker.
        n (int): Number of warmup requests to perform.
        max_new_tokens (int): Maximum number of new tokens to generate.
        inference_url (str): URL of the inference server.
    """
    tracker.start_task("warmup")
    for _ in range(n):
        try:
            await send_request(session, prompt, 0, max_new_tokens, inference_url)
        except Exception as e:
            print(f"Warmup failed: {e}")
    tracker.stop_task()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_url", type=str, default="http://localhost:8080/generate", 
                        help="URL for the inference server")
    parser.add_argument("--dataset_name", type=str, default="jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks",
                        help="Name of the dataset to use on Hugging Face Hub")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (e.g., 'train', 'test')")
    parser.add_argument("--column", type=str, default="conversation_with_thanks",
                        help="Column in the dataset to use for prompts")
    parser.add_argument("--n_samples", type=int, default=-1,
                        help="Number of samples to process (-1 for all)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Index to start processing from")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--warmup_runs", type=int, default=5,
                        help="Number of warmup runs before starting measurements")
    parser.add_argument("--break_min", type=float, default=0.05,
                        help="Minimum break time between requests in seconds")
    parser.add_argument("--break_max", type=float, default=0.3,
                        help="Maximum break time between requests in seconds")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV file to save results")
    parser.add_argument("--gpu_ids", type=str, default="0",
                        help="Comma-separated list of GPU IDs to use for tracking")
    args = parser.parse_args()
    main(args)
