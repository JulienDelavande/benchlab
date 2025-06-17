import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers.utils import export_to_video

# ✅ Remplace par le pipeline ContentV correct !
from contentv import ContentVPipeline  # Ex. ContentVPipeline.from_pretrained()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # Charger ContentV pipeline (attention : adapter le nom si nécessaire)
    pipe = ContentVPipeline.from_pretrained(
        args.model_name,
        torch_dtype=dtype
    ).to(device)

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
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(args.seed)
        )

        torch.cuda.synchronize()
        end_generate = time.time()
        emissions_generate = tracker.stop_task()
        tracker.stop()

        # ✅ Sauvegarde vidéo MP4
        if args.save_video:
            export_to_video(output.frames[0], f"{args.output_path}/contentv_{i}.mp4", fps=args.fps)

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
    parser.add_argument("--model_name", type=str, default="bytedance/ContentV")  # Adapter au nom réel sur HF Hub
    parser.add_argument("--prompt", type=str, default="A cinematic aerial shot of a futuristic city at sunset")
    parser.add_argument("--negative_prompt", type=str, default="blurry, bad quality, distorted")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default=f"contentv_results_{now}.csv")
    parser.add_argument("--output_path", type=str, default="./data")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--fps", type=int, default=24)

    args = parser.parse_args()
    main(args)
