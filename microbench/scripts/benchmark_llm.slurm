#!/bin/bash
#SBATCH --job-name=l8B_batch
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal

export output_dir="/fsx/jdelavande/benchlab/microbench/data/benchmark_results/llama8B_batch_$(date +"%Y-%m-%d-%H-%M-%S")"
mkdir -p ${output_dir}
#input_lengths=$(seq 1 100000)
#generate_lengths=$(seq 1 100000)
batch_sizes=$(seq 1 64)
python scripts/benchmark_llm_power_energy_latency.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --batch_sizes ${batch_sizes} \
  --input_lengths 1 10 100 1000 \
  --generate_lengths 1 10 100 1000 \
  --runs 20 \
  --warmup_runs 2 \
  --prefill \
  --decode \
  --log_power \
  --log_energy \
  --output_dir ${output_dir}