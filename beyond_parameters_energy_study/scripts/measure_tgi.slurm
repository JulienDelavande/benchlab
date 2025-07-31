#!/bin/bash
#SBATCH --job-name=ty_tgi5
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal
#SBATCH --time=08:00:00


# --- ENV vars ---
export PORT=8080
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DATASET_NAME="jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks"
export SPLIT="train"
export COLUMN="conversation_with_thanks"
export N_SAMPLES=-1
export START_INDEX=0
export MAX_NEW_TOKENS=256
export WARMUP_RUNS=5
export BREAK_MIN=0.5
export BREAK_MAX=0.5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export gpu_ids='0'

safe_min=${BREAK_MIN/./_}
safe_max=${BREAK_MAX/./_}

export OUT_CSV="/fsx/jdelavande/benchlab/thank_you/data/tgi-${DATASET_NAME##*/}-${safe_min}-${safe_max}-${now}.csv"
export CUDA_VISIBLE_DEVICES=$gpu_ids
export num_shard=$(echo $gpu_ids | tr ',' '\n' | wc -l)

srun --container-image='ghcr.io#huggingface/text-generation-inference'   \
     --container-env=HF_TOKEN,PORT \
     --container-mounts="/scratch:/data,/fsx:/fsx" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     --qos normal \
     --gpus-per-node=$num_shard \
     /fsx/jdelavande/benchlab/thank_you/launch_tgi_and_run.sh