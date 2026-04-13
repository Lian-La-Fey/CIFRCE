#!/bin/bash
set -e

# ==================================================
# Environment Settings
# ==================================================

export OMP_NUM_THREADS=1
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export DEEPSPEED_LOG_LEVEL=ERROR

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda


# ==================================================
# Kaggle GPU Detection
# ==================================================

NUM_GPUS=$(nvidia-smi -L | wc -l)

if [[ ${NUM_GPUS} -eq 1 ]]; then
    DEBUG=1
    NPROC_PER_NODE=1
else
    DEBUG=0
    NPROC_PER_NODE=${NUM_GPUS}
fi

echo "Detected ${NUM_GPUS} GPU(s)"
echo "DEBUG=${DEBUG}, NPROC_PER_NODE=${NPROC_PER_NODE}"

# ==================================================
# Distributed Configuration (Kaggle-safe)
# ==================================================

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NNODES=1
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

# ==================================================
# Model & Data Configuration
# ==================================================

llm=microsoft/MediPhi-Instruct
train_dataset_use=ct_rate_train,mimic_iv_train,radgraphXL_mimic_train,radgraphXL_stanford_train,radiopedia_train,rexgradient_train
val_dataset_use=ct_rate_val,mimic_iv_val,radgraphXL_mimic_val,radgraphXL_stanford_val,radiopedia_val,rexgradient_val
run_name="lgelcm_mediphi_it"
output_dir=/kaggle/working/lgelcm/checkpoints/${run_name}

# ==================================================
# Training Hyperparameters
# ==================================================

lr=2e-4
batch_size=1
grad_accum_steps=4

# ==================================================
# DeepSpeed (disabled by default)
# ==================================================

USE_DEEPSPEED=1
DEEPSPEED_CONFIG=./scripts/zero2.json
# DEEPSPEED_CONFIG=./scripts/zero3_offload.json

DS_ARGS=""
if [[ ${USE_DEEPSPEED} -eq 1 ]]; then
    DS_ARGS="--deepspeed ${DEEPSPEED_CONFIG}"
fi

# ==================================================
# Training Entry
# ==================================================

ENTRY_FILE=trainer.train

TRAIN_ARGS="
${DS_ARGS}
--model_name_or_path ${llm}
--train_dataset_use ${train_dataset_use}
--val_dataset_use ${val_dataset_use}
--output_dir ${output_dir}
--num_train_epochs 5
--fp16 True
--per_device_train_batch_size ${batch_size}
--gradient_accumulation_steps ${grad_accum_steps}
--lora_enable True
--lora_r 16
--lora_alpha 32
--eval_strategy no
--save_strategy steps
--save_steps 100
--save_total_limit 5
--learning_rate ${lr}
--weight_decay 0.01
--warmup_ratio 0.03
--max_grad_norm 1.0
--lr_scheduler_type cosine
--logging_steps 5
--model_max_length 8192
--gradient_checkpointing True
--dataloader_num_workers 4
--report_to none
--run_name ${run_name}
"

# ==================================================
# Launch
# ==================================================

if [[ ${DEBUG} -eq 1 ]]; then
    echo "Running in single GPU mode"
    python3 -m ${ENTRY_FILE} ${TRAIN_ARGS}
else
    echo "Running DDP on ${NPROC_PER_NODE} GPUs"
    torchrun \
        --nnodes=${NNODES} \
        --node_rank=${RANK} \
        --nproc_per_node=${NPROC_PER_NODE} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        -m ${ENTRY_FILE} ${TRAIN_ARGS}

fi