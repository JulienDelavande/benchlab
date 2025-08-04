# Study of the Energy Cost of Politeness in LLMs

This repository accompanies the article on the energy impact of politeness (e.g., "thank you") during inference with large language models (LLMs). It contains code to generate datasets with and without polite messages, and to measure energy consumption during GPU inference.

## Project Structure

- `scripts/dataset_add_thanks.py`: Generates formatted datasets with or without "Thank you!" from a HuggingFace dataset.
- `scripts/measure_energy.py`: Measures energy consumption during LLM inference on a given dataset.

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

## Reproducing the results

### 1. Generate the dataset with or without "Thank you!"

```bash
cd scripts/
python dataset_add_thanks.py \
  --repo HuggingFaceH4 \
  --dataset_name ultrachat_200k \
  --split "train_sft[:10000]" \
  --model_name gemma-2-2b-it \
  --end_repo jdelavande \
  --template gemma
```

- Adjust the arguments as needed (see `--help` for the full list).

### 2. Measure energy consumption

```bash
cd scripts/
python measure_energy.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --dataset_name jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks \
  --split train \
  --column conversation_with_thanks \
  --n_samples 100 \
  --max_new_tokens 256 \
  --runs 10 \
  --warmup 5 \
  --out_csv ../data/energy_results.csv \
  --out_generated ../data/generated_samples.csv \
  --gpu_ids 0 \
  --batch_size 1 \
  --dtype float32 \
  --start_index 0
```

### Bonus: run on slurm cluster
- Use the provided `slurm` script to run the energy measurement on a slurm cluster:

```bash
cd scripts/
sbatch measure.slurm
```

- Adapt paths and parameters to your GPU setup and needs.

## Reproducibility

The scripts are designed to be reproducible and easily adaptable to other models, datasets, or hardware configurations.

## References

See the associated article for full methodology, analysis, and results.

---

For any questions or contributions, feel free to open an issue or pull request.
