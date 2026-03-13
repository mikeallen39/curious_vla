#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lf

clear
sleep 5

: "${model_name_or_path=YOUR_MODEL_PATH}"
: "${template:=qwen2_vl}"

# Usage: ./lf_serve_cot.sh <num_instances>
# e.g.:  ./lf_serve_cot.sh 4   # start 4 instances
#        ./lf_serve_cot.sh 8   # start 8 instances

num_instances=$1
start_port=8192

if [ -z "$num_instances" ]; then
  echo "please provide num_instances: $0 4"
  exit 1
fi

cards_per_instance=$((8 / num_instances))

for i in $(seq 0 $((num_instances-1))); do
  start_card=$((i * cards_per_instance))
  end_card=$((start_card + cards_per_instance - 1))
  port=$((start_port + i))

  devices=$(seq -s, $start_card $end_card)

  echo "Start Instance $i: Use Devices: $devices, Port: $port"

  CUDA_VISIBLE_DEVICES=$devices API_VERBOSE=0 \
  API_PORT=$port \
  llamafactory-cli api \
    --model_name_or_path $model_name_or_path \
    --template $template \
    --infer_backend vllm \
    --image_max_pixels 262144 \
    --vllm_maxlen 65536 \
    --trust_remote_code true &
done

# --vllm_maxlen 16384
# Controls vllm kvcache max tokens (input+output). At current resolution,
# 1920x1080 after smart resize is about 1248 tokens.
# Keep prompt length in check, otherwise you may need tensor parallelism to avoid OOM.
# Can be ignored if you have 80GB VRAM.
