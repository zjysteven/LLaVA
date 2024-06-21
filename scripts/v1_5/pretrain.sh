#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <PER_DEVICE_BATCH_SIZE> <NUM_GPUS> <PORT> <PROJ> <NUM_QUERIES>"
    exit 1
fi

GLOBAL_BATCH_SIZE=256
PER_DEVICE_BATCH_SIZE="${1}"
NUM_GPUS="${2}"
GRAD_ACCUM=$((GLOBAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NUM_GPUS)))
PORT="${3}"
PROJ="${4}"
NUM_QUERIES="${5}"

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"
echo "Per device batch size: ${PER_DEVICE_BATCH_SIZE}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Gradient accumulation steps: ${GRAD_ACCUM}"
echo "PORT: ${PORT}"
echo "PROJ: ${PROJ}"
echo "NUM_QUERIES: ${NUM_QUERIES}"

if [ ${PROJ} = "perceiver" ]; then
    PROJ_NAME="perceiver${NUM_QUERIES}"
else
    PROJ_NAME="${PROJ}"
fi
echo "PROJ_NAME: ${PROJ_NAME}"

# generate a series of numbers from 0 to NUM_GPUS-1
# INCLUDE=""
# for ((i=0; i<$NUM_GPUS; i++)); do
#     INCLUDE+="$i"
#     if [ $i -lt $((NUM_GPUS-1)) ]; then
#         INCLUDE+=","
#     fi
# done

#deepspeed --include localhost:${INCLUDE} llava/train/train_mem.py \
#deepspeed --master_port ${PORT} llava/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node ${NUM_GPUS} --rdzv_backend c10d --rdzv_endpoint localhost:0 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data/llava_data/llava/llava_pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /data/llava_data/llava/llava_pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type ${PROJ} \
    --resampler_n_latents ${NUM_QUERIES} \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-${PROJ_NAME}-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name "llava-v1.5-7b-${PROJ_NAME}-pretrain"
