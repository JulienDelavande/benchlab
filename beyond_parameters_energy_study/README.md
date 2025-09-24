# Energy and Latency Measurement for Language Models

This repository provides tools and scripts to **measure energy consumption and latency** for various language models under different configurations. It is designed to evaluate the impact of **batch size**, **quantization**, **precision** (e.g., `float32`, `float16`, `bfloat16`), and **inference engines** on energy efficiency and latency.

This code supports a research paper on **energy-efficient AI inference**.

---

## Features

- Measure **energy consumption** and **latency** for different language models.
- Support for multiple configurations:
  - Varying **batch sizes**
  - **Quantization levels** (e.g., 8-bit, 4-bit, or none)
  - **Precision types**: `float32`, `float16`, `bfloat16`
  - With or without **inference engines tgi**
- Compatible with **Hugging Face models and datasets**
- Includes **scripts for batch experiments** or fine-grained configuration

---

## Installation

Installation uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python dependency management.

1. Install `uv` if needed:

```bash
pip install uv
```

2. Install the project dependencies:

```bash
uv sync
source .venv/bin/activate  # Activate the virtual environment
```
---

## Running Experiments

### Batch Energy Measurement

Run the batch experiment script:

```bash
python measure_energy_batch.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --dataset_name Anonyme162325/ultrachat_200k-Llama-3-8B-Instruct-with-thanks \
    --split train \
    --column conversation_with_thanks \
    --n_samples -1 \
    --max_new_tokens 256 \
    --runs 10 \
    --warmup 5 \
    --out_csv ../data/energy_results.csv \
    --gpu_ids 0 \  # Adjust based on available GPUs
    --batch_size 4 \  # Adjust batch size as needed
    --start_index 0
```

This script measures energy consumption and latency for a batch of prompts.

### Precision/quantization Measurement
Run the precision measurement script:

```bash
python measure_precision.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct
    --dataset_name Anonyme162325/ultrachat_200k-Llama-3-8B-Instruct-with-thanks \
    --split train \
    --column conversation_with_thanks \
    --n_samples -1 \
    --max_new_tokens 256 \
    --runs 10 \
    --warmup 5 \
    --out_csv ../data/precision_results.csv \
    --out_generated ../data/generated_samples.csv \
    --gpu_ids 0 \
    --batch_size 1 \
    --quantization 8bit \  # Options: none, 8bit, 4bit
    --dtype float32 \
    --gpu_ids 0 \
    --start_index 0
```

### Energy Measurement with Inference Engines

Use the inference engine (e.g., TGI) measurement script:
 1. Pull, Run and Enter TGI Docker image:
```bash
docker pull ghcr.io/huggingface/text-generation-inference:latest
docker run -d --gpus all --name tgi \
    -p 8000:80 \
    ghcr.io/huggingface/text-generation-inference:latest
docker exec -it tgi bash
```

2. Run the TGI measurement script:

```bash
export PORT=8080
export MODEL="meta-llama/Llama-3.1-8B-Instruct"
export DATASET_NAME="Anonyme162325/ultrachat_200k-Llama-3-8B-Instruct-with-thanks"
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
export OUT_CSV="../data/tgi-${DATASET_NAME##*/}-${safe_min}-${safe_max}-${now}.csv"
export CUDA_VISIBLE_DEVICES=$gpu_ids
export num_shard=$(echo $gpu_ids | tr ',' '\n' | wc -l)
source launch_tgi_and_run.sh
```

This will evaluate energy usage when using inference servers like Hugging Face's TGI.

### Measure Specific Data Types on SLURM

Use the provided SLURM scripts for cluster-based runs:

```bash
sbatch measure_batch.slurm # Batch script
sbatch measure_dtype.slurm # DTYPE and Precision scripts
sbatch measure_tgi.slurm # TGI script
```

---

## Custom Configurations

All scripts are configurable. You can:

* Change the model or dataset
* Adjust batch size, precision, or quantization settings
* Enable or disable inference engines

---

## Results

Results are saved as `.csv` files, including:

* **Energy consumption** (CPU, GPU, RAM)
* **Latency metrics** (prefill, decode, generate)
* **Token counts** (prompt and response)

These metrics allow detailed analysis of model behavior under different configurations.

## Reproducibility

The scripts are designed to be reproducible and easily adaptable to other models, datasets, or hardware configurations.

## References

See the associated article for full methodology, analysis, and results.

---

For any questions or contributions, feel free to open an issue or pull request.