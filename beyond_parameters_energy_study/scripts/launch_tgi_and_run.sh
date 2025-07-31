#!/bin/bash

# Lancer TGI
/usr/local/bin/text-generation-launcher --model-id "${MODEL}" --num-shard "${num_shard}" &
TGI_PID=$!

# Attendre qu’il réponde à une requête /generate
echo 'Waiting for TGI...'
until curl -s -X POST http://localhost:${PORT}/generate \
       -H "Content-Type: application/json" \
       -d '{"inputs":"hello","parameters":{"max_new_tokens":1}}' | grep -q "generated_text"
do
    sleep 2
done
echo "TGI ready!"

python -m ensurepip --upgrade
python -m pip install --no-cache-dir httpx transformers datasets codecarbon pandas torch tqdm

# Lancer le script Python
python /fsx/jdelavande/benchlab/thank_you/measure_energy_tgi.py \
  --dataset_name "$DATASET_NAME" \
  --split "$SPLIT" \
  --column "$COLUMN" \
  --n_samples "$N_SAMPLES" \
  --start_index "$START_INDEX" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --warmup_runs "$WARMUP_RUNS" \
  --break_min "$BREAK_MIN" \
  --break_max "$BREAK_MAX" \
  --out_csv "$OUT_CSV" \
  --gpu_ids "$gpu_ids"

kill $TGI_PID
