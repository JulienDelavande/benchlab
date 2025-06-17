import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Charger l'adapter Motion
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(args.repo, args.ckpt), device=device))

    # Charger le pipeline AnimateDiff
    pipe = AnimateDiffPipeline.from_pretrained(
        args.base_model,
        motion_adapter=adapter,
        torch_dtype=dtype
    ).to(device)

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        beta_schedule="linear"
    )

    results = []

    for i in range(args.runs):
        print(f"Run {i+1}/{args.runs}")
        tracker = EmissionsTracker(gpu_ids=[0], log_level="warning", tracking_mode="machine", measure_power_secs=1)

        torch.cuda.synchronize()
        tracker.start_task("generate")
        start_generate = time.time()
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=torch.Generator().manual_seed(args.seed)
        )
        torch.cuda.synchronize()
        end_generate = time.time()
        emissions_generate = tracker.stop_task()
        tracker.stop()

        # Sauver le GIF si demandé
        if args.save_gif:
            export_to_gif(output.frames[0], f"{args.output_path}/animation_{i}.gif")

        results.append({
            "run": i,
            "duration_generate": end_generate - start_generate,
            "energy_generate_gpu": emissions_generate.gpu_energy,
            "energy_generate_cpu": emissions_generate.cpu_energy,
            "energy_generate_ram": emissions_generate.ram_energy
        })

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"Résultats sauvegardés dans {args.out_csv}")

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="ByteDance/AnimateDiff-Lightning")
    parser.add_argument("--ckpt", type=str, default="animatediff_lightning_4step_diffusers.safetensors")
    parser.add_argument("--base_model", type=str, default="emilianJR/epiCRealism")
    parser.add_argument("--prompt", type=str, default="A girl smiling")
    parser.add_argument("--negative_prompt", type=str, default="blurry, distorted, bad quality")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default=f"results_{now}.csv")
    parser.add_argument("--output_path", type=str, default="./data")
    parser.add_argument("--save_gif", action="store_true")

    args = parser.parse_args()
    main(args)
