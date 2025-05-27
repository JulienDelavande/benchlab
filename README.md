# benchlab

srun --pty --gres=gpu:1 --cpus-per-task=6 --mem=32G bash
conda activate benchlab

ncu --nvtx --set full --target-processes all \
    --nvtx-include "prefill,decode" \
    -o profile_generate \
    python profiler.py
ncu --set full python -c "import torch; a = torch.randn(10000, 10000, device='cuda'); a = a * a"


nsys profile -t cuda,nvtx,cublas --sample=none \
    -o decode_bs63 \
    python nsysdecode.py

nsys profile -t cuda,nvtx,cublas --sample=none \
    -o ./tmp/generate_bs64 \
    python nsysdecode.py

/opt/nvidia/nsight-compute/2024.1.1/ncu --set full python -c "import torch; a = torch.randn(10000, 10000, device='cuda'); a = a * a"

nsys profile -t cuda,nvtx,cudnn,cublas --force-overwrite=true -o decode_profile \
  python nsys3.py

export output_dir=nsys_results/linear_1runs_bs64
nsys profile -t cuda,nvtx,cudnn,cublas --force-overwrite=true -o ${output_dir} \
  python onelayer.py
nsys stats ${output_dir}.nsys-rep >> ${output_dir}.txt
nsys export --type=json --output=${output_dir}_trace.json ${output_dir}.nsys-rep

export batch_size=64
export output_dir=nsys_results/linear_1runs_bs${batch_size}
nsys profile -t cuda,nvtx,cublas,cudnn,osrt --cpuctxsw=true \
  --capture-range=nvtx --force-overwrite=true -o ${output_dir} \
  python onelayer.py ${batch_size}


