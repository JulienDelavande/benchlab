#!/bin/bash
#SBATCH --job-name=ty_gma2B
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal

source ~/.bashrc

export model_name=google/gemma-2-2b-it
export dataset_name=jdelavande/ultrachat_200k-gemma-2-2b-it-with-thanks
export n_samples=-1
export start_index=59
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-${now}.log
python measure_energy.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --start_index ${start_index} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1