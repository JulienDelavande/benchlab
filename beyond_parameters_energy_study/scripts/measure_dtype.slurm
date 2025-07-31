#!/bin/bash
#SBATCH --job-name=6bf
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

dtypes=("float32" "float16" "bfloat16")
dtype="bfloat16"

export n_samples=-1
export batch_size=1
export runs=2
export warmup=1
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export index=6
export model_name="${models[$index]}"
export dataset_name="${datasets[$index]}"

# Fichier unique de sortie (résultats + générations)
export out_csv="/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-dtype3-${dtype}-energy-${now}.csv"
export out_generated="/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-dtype3-${dtype}-generated-${now}.csv"


echo "===== Starting ${model_name} | dtype=${dtype} =====" | tee -a ${log_file}
python measure_energy.py \
  --model_name "${model_name}" \
  --dataset_name "${dataset_name}" \
  --n_samples ${n_samples} \
  --batch_size ${batch_size} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv "${out_csv}" \
  --out_generated "${out_generated}" \
  --dtype ${dtype}
echo "===== Finished ${model_name} | dtype=${dtype} =====" | tee -a ${log_file}