#!/bin/bash

#SBATCH --account=cs433
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=shot7m2-behave-mae
#SBATCH --output=logs/shot7m2/experiment2/slurm_%j.log
#SBATCH --error=logs/shot7m2/experiment2/slurm_%j.err

# Number of GPUs
GPUS=2

common_args="--dataset shot7m2 \
    --path_to_data_dir /scratch/izar/boesch/data/Shot7M2/train/train_dictionary_poses.npy\
    --batch_size 512 \
    --model hbehavemae \
    --input_size 400 1 72 \
    --stages 2 3 4 \
    --q_strides 2,1,4;2,1,6 \
    --mask_unit_attn True False False \
    --patch_kernel 2 1 3 \
    --init_embed_dim 96 \
    --init_num_heads 2 \
    --out_embed_dims 78 128 256 \
    --epochs 200 \
    --num_frames 400 \
    --decoding_strategy multi \
    --decoder_embed_dim 128 \
    --decoder_depth 1 \
    --decoder_num_heads 1 \
    --pin_mem \
    --num_workers 8 \
    --sliding_window 17 \
    --blr 1.6e-4 \
    --warmup_epochs 40 \
    --masking_strategy random \
    --mask_ratio 0.70 \
    --clip_grad 0.02 \
    --checkpoint_period 20 \
    --norm_loss False \
    --seed 0 \
    --output_dir /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/experiment2 \
    --log_dir logs/shot7m2/experiment2"

# Create log directories if they don't exist
mkdir -p logs/shot7m2

# activate behavemae conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavemae

# Run training
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --node_rank 0 --master_addr=127.0.0.1 --master_port=2999 \
    run_pretrain.py --distributed $common_args
