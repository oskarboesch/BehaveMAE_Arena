#!/bin/bash

#SBATCH --account=cs433
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=arena-ts2vec
#SBATCH --output=logs/ts2vec/experiment2/slurm_%j.log
#SBATCH --error=logs/ts2vec/experiment2/slurm_%j.err

common_args="--dataset arena \
    --train_data_path /scratch/izar/boesch/data/Arena_Data/shuffle-3_split-train.npz \
    --inference_data_path /scratch/izar/boesch/data/Arena_Data/shuffle-3_split-train.npz \
    --output_dir /scratch/izar/boesch/ts2vec/outputs/arena/experiment1 \
    --num_frames 900 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001 \
    --n_train_samples 10000 \
    --device 0 \
    --num_workers 8 \
    --output_dims 64 \
    --hidden_dims 64 \
    --depth 10 \
    --sliding_length 1 \
    --encoding_window 2 \
    --sliding_padding 50"

# Create log directories if they don't exist
mkdir -p logs/ts2vec/experiment2

# activate behavemae conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavemae
export PYTHONNOUSERSITE=1

# Run from project root
cd /home/boesch/BehaveMAE

# Run TS2Vec training + inference on Arena
python -u main_ts2vec.py $common_args
