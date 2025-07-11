# Microbench

## Utilization


```bash
# Activate the environment
srun --pty --gres=gpu:1 --cpus-per-task=8 --mem=32G bash
cd benchlab/microbench
source .venv/bin/activate
```

### Benchmark power, energy, latency for llm

```bash
# Without tracing
export output_dir="/fsx/jdelavande/benchlab/microbench/data/benchmark_results/llama8B_$(date +"%Y-%m-%d-%H-%M-%S")"
mkdir -p ${output_dir}
python scripts/benchmark_llm_power_energy_latency.py \
  --model_id meta-llama/Llama-3.1-8B-Instruct \
  --batch_sizes 1 \
  --input_lengths 256 \
  --generate_lengths 1 10 \
  --runs 20 \
  --warmup_runs 2 \
  --prefill \
  --decode \
  --log_power \
  --log_energy \
  --output_dir ${output_dir}

# With tracing
export output_dir="./data/nsys-results/llama8B_input1_$(date +"%Y-%m-%d-%H-%M-%S")"
nsys profile \
  --trace=cuda,nvtx,cublas \
  --sample=none \
  -o ${output_dir} \
  --force-overwrite true \
    python scripts/benchmark_llm_power_energy_latency.py
```


### Benchmark power, energy, latency for linear layer

```bash
# Without tracing
export output_dir="/fsx/jdelavande/benchlab/microbench/data/benchmark_results/linear_$(date +"%Y-%m-%d-%H-%M-%S")"
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
  --warmup_runs 2 \
  --prefix "exp0" \
  --output_dir ${output_dir}
 ```
