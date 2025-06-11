#!/bin/bash
#SBATCH --job-name=ty_b2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal

source ~/.bashrc

export model_name=meta-llama/Llama-3.1-8B-Instruct
export dataset_name=jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks
export n_samples=-1
export batch_size=2
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-batch${batch_size}-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-batch${batch_size}-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-batch${batch_size}-${now}.log
python measure_energy_batch.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --batch_size ${batch_size} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1