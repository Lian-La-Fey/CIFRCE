#!/bin/bash

#SBATCH --job-name=lgelcm_train
#SBATCH --partition=debug
#SBATCH --constraint=akya-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=4              # Total number of processes (GPUs) across all nodes
#SBATCH --ntasks-per-node=4     # Number of tasks (and GPUs) per node
#SBATCH --gres=gpu:4            # GPUs per node
#SBATCH --cpus-per-task=10
#SBATCH --time=0-04:00:00
#SBATCH --chdir=/arf/scratch/ibgulmez
#SBATCH --output=/arf/scratch/ibgulmez/lgelcm_logs/lgelcm_train.%j.out
#SBATCH --error=/arf/scratch/ibgulmez/lgelcm_logs/lgelcm_train.%j.err

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 2500))
export NNODES=$SLURM_NNODES
export GPUS_PER_NODE=$SLURM_NTASKS_PER_NODE
export WORLD_SIZE=$(($NNODES * $GPUS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "RANK=$RANK"
echo "LOCAL_RANK=$LOCAL_RANK"

# NCCL debugging
# export NCCL_DEBUG=INFO
# export LOGLEVEL=INFO
# export DS_LOG_LEVEL=debug
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

if [ -d /arf/scratch/$USER/projects/LGELCM ]; then
    rm -rf /arf/scratch/$USER/projects/LGELCM
fi

cp -r /arf/home/$USER/projects/LGELCM /arf/scratch/$USER/projects/

module purge
module load lib/cuda/11.8

CUDA_PATH=$(dirname $(dirname $(which nvcc)))
echo "CUDA path on host: $CUDA_PATH"

srun apptainer exec \
    --nv \
    --bind /arf/scratch/$USER:/scratch \
    --bind $CUDA_PATH:/usr/local/cuda \
    $HOME/container-gpu/miniconda3-gpu \
    bash -c "
    export CUDA_HOME=/usr/local/cuda &&
    export PATH=\$CUDA_HOME/bin:\$PATH &&
    export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH &&
    
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    
    export TORCH_EXTENSIONS_DIR=\"/scratch/torch_extensions_${SLURM_NODEID}\"
    mkdir -p \$TORCH_EXTENSIONS_DIR
    
    export DS_BUILD_OPS=1
    export DS_BUILD_FUSED_ADAM=1
    
    export MASTER_ADDR=\$MASTER_ADDR &&
    export MASTER_PORT=\$MASTER_PORT &&
    export WORLD_SIZE=\$WORLD_SIZE &&
    export RANK=\${SLURM_PROCID} &&
    export LOCAL_RANK=\${SLURM_LOCALID} &&
    
    source /opt/conda/bin/activate qwen3 &&
    cd /scratch/projects/LGELCM &&
    
    # ============================================================
    # QWEN3-VL TRAINING PARAMETERS
    # ============================================================
    
    # Model & Dataset
    llm=\"/scratch/LGELCM_MODELS/MediPhi-Instruct\"
    train_dataset_use=\"ct_rate_train,mimic_iv_train,radgraphXL_mimic_train,radgraphXL_stanford_train,radiopedia_train,rexgradient_train\"
    val_dataset_use=\"ct_rate_val,mimic_iv_val,radgraphXL_mimic_val,radgraphXL_stanford_val,radiopedia_val,rexgradient_val\"
    
    # Hyperparameters
    lr=2e-4
    batch_size=1
    grad_accum_steps=4
    
    run_name=\"lgelcm_lr\${lr}_bs\${batch_size}_ga\${grad_accum_steps}_\$(date +%Y_%m_%d_%H_%M)\"
    output_dir=\"/scratch/LGELCM_MODELS/\${run_name}/checkpoints\"
    
    DEEPSPEED_CONFIG=\"./scripts/zero2.json\"
    
    echo \"run_name=\${run_name}\"
    
    python -u -m trainer.train \
        --deepspeed \$DEEPSPEED_CONFIG \
        --model_name_or_path \${llm} \
        --train_dataset_use \${train_dataset_use} \
        --val_dataset_use \${val_dataset_use} \
        \
        --output_dir \${output_dir} \
        --fp16 True \
        --per_device_train_batch_size \${batch_size} \
        --per_device_eval_batch_size \${batch_size} \
        --gradient_accumulation_steps \${grad_accum_steps} \
        \
        --model_max_length 8192 \
        \
        --learning_rate \${lr} \
        --weight_decay 0.01 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --max_grad_norm 1.0 \
        \
        --num_train_epochs 10 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --save_total_limit 10 \
        --logging_steps 10  \
        \
        \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        \
        --lora_enable True \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.0 \
        \
        --report_to none \
        --run_name \${run_name}
    "