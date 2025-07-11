#!/bin/bash
#SBATCH --job-name=linearBenchmark
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal

export output_dir="/fsx/jdelavande/benchlab/microbench/data/benchmark_results/linear_slurm_$(date +"%Y-%m-%d-%H-%M-%S")"
seq_lens=$(seq 1 100000)
mkdir -p ${output_dir}
python scripts/benchmark_linear_power_energy_latency.py \
  --batch_sizes 1 \
  --hidden_dims 4096 \
  --ffn_dims 11008 \
  --seq_lens ${seq_lens} \
  --runs 20 \
  --log_power \
  --log_energy \
  --warmup_runs 5 \
  --prefix "exp1" \
  --output_dir ${output_dir}