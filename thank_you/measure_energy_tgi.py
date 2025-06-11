import random
import asyncio
import httpx
import pandas as pd
from datasets import load_dataset
import time
from codecarbon import EmissionsTracker
import argparse


async def send_request(session, prompt, idx, max_new_tokens):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False
        }
    }
    start = time.time()
    response = await session.post("http://localhost:8080/generate", json=payload)
    end = time.time()
    return {
        "index": idx,
        "prompt_length": len(prompt),
        "status": response.status_code,
        "latency": end - start
    }


async def warmup(session, prompt, tracker, n, max_new_tokens):
    tracker.start_task("warmup")
    for _ in range(n):
        try:
            await send_request(session, prompt, 0, max_new_tokens)
        except Exception as e:
            print(f"Warmup failed: {e}")
    tracker.stop_task()


async def run_all(prompts, tracker, warmup_runs, break_min, break_max, max_new_tokens):
    results = []
    tasks = []

    async with httpx.AsyncClient(timeout=30.0) as session:
        await warmup(session, prompts[0], tracker, warmup_runs, max_new_tokens)

        print("Starting energy tracking after warmup...")
        tracker.start_task("dataset_requests")
        start_time = time.time()

        # Liste de tâches asyncio
        for idx, prompt in enumerate(prompts):
            delay = sum(random.uniform(break_min, break_max) for _ in range(idx))
            async def delayed_request(prompt=prompt, idx=idx, delay=delay):
                await asyncio.sleep(delay)
                try:
                    result = await send_request(session, prompt, idx, max_new_tokens)
                except Exception:
                    result = {
                        "index": idx,
                        "prompt_length": len(prompt),
                        "status": "error",
                        "latency": -1
                    }
                results.append(result)

            tasks.append(asyncio.create_task(delayed_request()))

        # Attendre la fin de TOUTES les requêtes
        await asyncio.gather(*tasks)

        end_time = time.time()
        emissions = tracker.stop_task()

    return results, start_time, end_time, emissions



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--column", type=str, default="conversation_with_thanks")
    parser.add_argument("--n_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup_runs", type=int, default=5)
    parser.add_argument("--break_min", type=float, default=0.05)
    parser.add_argument("--break_max", type=float, default=0.3)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--gpu_ids", type=str, default="0")
    args = parser.parse_args()

    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]

    # Charger le dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.n_samples > 0:
        dataset = dataset.select(range(args.start_index, args.start_index + args.n_samples))
    prompts = dataset[args.column]

    # Initialiser le tracker
    tracker = EmissionsTracker(
        log_level="warning",
        tracking_mode="machine",
        gpu_ids= gpu_ids,
        allow_multiple_runs=True,
        measure_power_secs=1,
    )

    # Lancer l'exécution
    results, start_time, end_time, emissions = asyncio.run(run_all(
        prompts,
        tracker,
        args.warmup_runs,
        args.break_min,
        args.break_max,
        args.max_new_tokens
    ))

    summary = pd.DataFrame([{
        "total_requests": len(results),
        "total_duration": end_time - start_time,
        "total_energy_cpu": emissions.cpu_energy,
        "total_energy_gpu": emissions.gpu_energy,
        "total_energy_ram": emissions.ram_energy,
    }])
    summary.to_csv(args.out_csv, index=False)



if __name__ == "__main__":
    main()
