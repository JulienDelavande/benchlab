import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.utils import export_to_video

def round_to_nearest_resolution_acceptable_by_vae(height, width, ratio):
    height = height - (height % ratio)
    width = width - (width % ratio)
    return height, width

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger les modèles
    pipe = LTXConditionPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(args.upsample_model_name, vae=pipe.vae, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe_upsample.to(device)
    pipe.vae.enable_tiling()

    expected_height, expected_width = args.height, args.width
    downscale_factor = args.downscale_factor
    num_frames = args.num_frames

    downscaled_height = int(expected_height * downscale_factor)
    downscaled_width = int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width, pipe.vae_spatial_compression_ratio)

    results = []

    for i in range(args.runs):
        print(f"Run {i + 1}/{args.runs}")
        tracker = EmissionsTracker(gpu_ids=[0], log_level="warning", tracking_mode="machine", measure_power_secs=1)

        # Étape 1 : génération basse résolution
        torch.cuda.synchronize()
        tracker.start_task("generate")
        start_generate = time.time()
        latents = pipe(
            conditions=None,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=args.generate_steps,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="latent",
        ).frames
        torch.cuda.synchronize()
        end_generate = time.time()
        emissions_generate = tracker.stop_task()

        # Étape 2 : upscaling latent
        torch.cuda.synchronize()
        tracker.start_task("upsample")
        start_upsample = time.time()
        upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames
        torch.cuda.synchronize()
        end_upsample = time.time()
        emissions_upsample = tracker.stop_task()

        # Étape 3 : denoise
        torch.cuda.synchronize()
        tracker.start_task("denoise")
        start_denoise = time.time()
        upscaled_height = downscaled_height * 2
        upscaled_width = downscaled_width * 2
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=args.denoise_strength,
            num_inference_steps=args.denoise_steps,
            latents=upscaled_latents,
            decode_timestep=args.decode_timestep,
            image_cond_noise_scale=args.image_cond_noise_scale,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="pil",
        ).frames[0]
        torch.cuda.synchronize()
        end_denoise = time.time()
        emissions_denoise = tracker.stop_task()

        tracker.stop()

        # Resize final
        video = [frame.resize((expected_width, expected_height)) for frame in video]

        if args.save_video:
            export_to_video(video, f"{args.output_path}/output_{i}.mp4", fps=args.fps)

        results.append({
            "run": i,
            "duration_generate": end_generate - start_generate,
            "duration_upsample": end_upsample - start_upsample,
            "duration_denoise": end_denoise - start_denoise,
            "energy_generate_gpu": emissions_generate.gpu_energy,
            "energy_upsample_gpu": emissions_upsample.gpu_energy,
            "energy_denoise_gpu": emissions_denoise.gpu_energy,
            "energy_generate_cpu": emissions_generate.cpu_energy,
            "energy_upsample_cpu": emissions_upsample.cpu_energy,
            "energy_denoise_cpu": emissions_denoise.cpu_energy,
            "energy_generate_ram": emissions_generate.ram_energy,
            "energy_upsample_ram": emissions_upsample.ram_energy,
            "energy_denoise_ram": emissions_denoise.ram_energy,
        })

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)
    print(f"Résultats sauvegardés dans {args.out_csv}")

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Lightricks/LTX-Video-0.9.7-dev")
    parser.add_argument("--upsample_model_name", type=str, default="Lightricks/ltxv-spatial-upscaler-0.9.7")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a river, high quality, detailed, cinematic")
    parser.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--downscale_factor", type=float, default=2/3)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--generate_steps", type=int, default=30)
    parser.add_argument("--denoise_steps", type=int, default=10)
    parser.add_argument("--denoise_strength", type=float, default=0.4)
    parser.add_argument("--decode_timestep", type=float, default=0.05)
    parser.add_argument("--image_cond_noise_scale", type=float, default=0.025)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default=f"results_{now}.csv")
    parser.add_argument("--output_path", type=str, default="./data")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--fps", type=int, default=24)

    args = parser.parse_args()
    main(args)
