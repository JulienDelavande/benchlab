# benchlab

srun --pty --gres=gpu:1 --cpus-per-task=6 --mem=32G bash
conda activate benchlab

ncu --nvtx --set full --target-processes all \
    --nvtx-include "prefill,decode" \
    -o profile_generate \
    python profiler.py
