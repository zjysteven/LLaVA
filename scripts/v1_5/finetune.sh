#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <PER_DEVICE_BATCH_SIZE> <NUM_GPUS> <PORT> <PROJ> <NUM_QUERIES>"
    exit 1
fi

GLOBAL_BATCH_SIZE=128
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

# NOTE:
# - using torchrun because deepspeed is not compatible with `CUDA_VISIBLE_DEVICES`
# which will be called when using slurm

# - using local vicuna checkpoints, whose generation_config.do_sample is manually set to True
# so that when saving llava checkpoint there won't be an error due to mismatch between generation_config
# and transformers default config


#deepspeed llava/train/train_mem.py \
torchrun --nnodes=1 --nproc_per_node ${NUM_GPUS} --rdzv_backend c10d --rdzv_endpoint localhost:0 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/llava_data/text_files/llava_v1_5_mix665k.json \
    --image_folder /data/llava_data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-${PROJ_NAME}-pretrain/mm_projector.bin \
    --mm_projector_type ${PROJ} \
    --resampler_n_latents ${NUM_QUERIES} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-${PROJ_NAME}-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --run_name "llava-v1.5-7b-${PROJ_NAME}-finetune"
