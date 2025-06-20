# benchlab

srun --pty --gres=gpu:1 --cpus-per-task=8 --mem=32G bash
squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
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
export runs=5
export dir=nsys_results
mkdir -p ${dir}
export output_dir=${dir}/linearA10g_${runs}runs_bs${batch_size}
nsys profile -t cuda,nvtx,cublas,cudnn,osrt --cpuctxsw=process-tree \
--force-overwrite=true -o ${output_dir} \
  python onelayer.py ${batch_size}


sudo nvidia-smi -pm 1
sudo nvidia-smi --gom=0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
sudo usermod -aG video $USER
sudo env "PATH=$PATH" ncu -set full python -c "import torch; a = torch.randn(10000, 10000, device='cuda'); a = a * a"

scp -i ~/.ssh/jdelavande-mac.pem ubuntu@10.90.52.154:/home/ubuntu/benchlab/nsys_results/linearA10g_5runs_bs64.nsys-rep ~/code/traces/

sudo env "PATH=$PATH" ncu --kernel-name ampere_sgemm_128x32_tn --launch-skip 7 --launch-count 1 "python" onelayer.py 64


## start node and then jupyter notebook server (then copy link to the vscode notebook kernel)
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0


optimum-benchmark --config-dir /fsx/jdelavande/benchlab/thank_you/scenarios \
            --config-name "text_generation_llama_3_1_8B" \
            backend.model="meta-llama/Llama-3.1-8B-Instruct" \
            backend.processor="meta-llama/Llama-3.1-8B-Instruct" \
            hydra.run.dir="/fsx/jdelavande/benchlab/thank_you/runs/llama3_8B/$(date +"%Y-%m-%d-%H-%M-%S")"


export model_name=meta-llama/Llama-3.1-8B-Instruct
export dataset_name=jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks
export n_samples=-1
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-${now}.log
python measure_energy.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1
  

export model_name=mistralai/Mistral-7B-Instruct-v0.3
export dataset_name=jdelavande/ultrachat_200k-Mistral-7B-Instruct-v0.3-with-thanks
export n_samples=-1
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-${now}.log
python measure_energy.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1

export model_name=Qwen/Qwen2.5-7B-Instruct
export dataset_name=jdelavande/ultrachat_200k-Qwen2.5-7B-Instruct-with-thanks
export n_samples=-1
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-${now}.log
python measure_energy.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1

export model_name=google/gemma-2-2b-it
export dataset_name=jdelavande/ultrachat_200k-gemma-2-2b-it-with-thanks
export n_samples=-1
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-${now}.log
python measure_energy.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1


## batch

export model_name=meta-llama/Llama-3.1-8B-Instruct
export dataset_name=jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks
export n_samples=-1
export batch_size=16
export runs=10
export warmup=5
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export out_csv=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-energy-batch-${now}.csv
export out_generated=/fsx/jdelavande/benchlab/thank_you/data/${model_name##*/}-${dataset_name##*/}-generated-batch-${now}.csv
export log_dir=/fsx/jdelavande/benchlab/thank_you/logs/${model_name##*/}-${dataset_name##*/}-energy-batch-${now}.log
python measure_energy_batch.py \
  --model_name ${model_name} \
  --dataset_name ${dataset_name} \
  --n_samples ${n_samples} \
  --batch_size ${batch_size} \
  --runs ${runs} \
  --warmup ${warmup} \
  --out_csv ${out_csv} \
  --out_generated ${out_generated} >> ${log_dir} 2>&1

## tgi

export PORT=8080
export MODEL=meta-llama/Llama-3.1-8B-Instruct
srun --container-image='ghcr.io#huggingface/text-generation-inference'   \
     --container-env=HF_TOKEN,PORT \
     --container-mounts="/scratch:/data" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     --qos normal \
     --gpus-per-node=1 \
	     /usr/local/bin/text-generation-launcher --model-id ${MODEL}



curl -X POST http://localhost:8080/generate \
     -H "Content-Type: application/json" \
     -d @request.json

export DATASET_NAME="jdelavande/ultrachat_200k-Llama-3-8B-Instruct-with-thanks"
export SPLIT="train"
export COLUMN="conversation_with_thanks"
export N_SAMPLES=-1
export START_INDEX=0
export MAX_NEW_TOKENS=256
export WARMUP_RUNS=5
export BREAK_MIN=0.005
export BREAK_MAX=0.05
export now=$(date +"%Y-%m-%d-%H-%M-%S")
export OUT_CSV="/fsx/jdelavande/benchlab/thank_you/data/tgi-${DATASET_NAME##*/}-${BREAK_MIN${VAR//./_}}-${BREAK_MAX${VAR//./_}}-${now}.csv"
export log_dir="/fsx/jdelavande/benchlab/thank_you/logs/tgi-${DATASET_NAME##*/}-energy-${now}.log"

python measure_energy_tgi.py \
  --dataset_name $DATASET_NAME \
  --split $SPLIT \
  --column $COLUMN \
  --n_samples $N_SAMPLES \
  --start_index $START_INDEX \
  --max_new_tokens $MAX_NEW_TOKENS \
  --warmup_runs $WARMUP_RUNS \
  --break_min $BREAK_MIN \
  --break_max $BREAK_MAX \
  --out_csv $OUT_CSV >> $log_dir 2>&1


