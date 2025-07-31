#!/bin/bash
#SBATCH --job-name=tyQmi7
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal

source ~/.bashrc

models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
)

datasets=(
  "jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks"
  "jdelavande/ultrachat_200k-Qwen2.5-7B-Instruct-with-thanks"
  "jdelavande/ultrachat_200k-Qwen2.5-7B-Instruct-with-thanks"
  "jdelavande/ultrachat_200k-Qwen2.5-7B-Instruct-with-thanks"
  "jdelavande/ultrachat_200k-Qwen2.5-7B-Instruct-with-thanks"
  "jdelavande/ultrachat_200k-Qwen2.5-7B-Instruct-with-thanks"
  "jdelavande/ultrachat_200k-Mistral-7B-Instruct-v0.3-with-thanks"
)


export index=6
export model_name="${models[$index]}"
export dataset_name="${datasets[$index]}"
export n_samples=-1
export batch_size=1
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")

for quant in 8bit 4bit; do
  export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-${quant}-${now}.csv
  export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-${quant}-${now}.csv
  export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-${quant}-${now}.log

  echo "Launching ${quant} quantized model..."
  python measure_energy_bsb.py \
    --model_name ${model_name} \
    --dataset_name ${dataset_name} \
    --n_samples ${n_samples} \
    --runs ${runs} \
    --warmup ${warmup} \
    --quantization ${quant} \
    --out_csv ${out_csv} \
    --out_generated ${out_generated}
done
