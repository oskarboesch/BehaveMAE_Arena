#!/bin/bash

#SBATCH --account=cs433
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=shot7m2-ts2vec
#SBATCH --output=logs/ts2vec/experiment1/slurm_%j.log
#SBATCH --error=logs/ts2vec/experiment1/slurm_%j.err

common_args="--dataset shot7m2 \
    --train_data_path /scratch/izar/boesch/data/Shot7M2/train/train_dictionary_poses.npy \
    --inference_data_path /scratch/izar/boesch/data/Shot7M2/test/test_dictionary_poses.npy \
    --output_dir /scratch/izar/boesch/ts2vec/outputs/shot7m2/experiment1 \
    --num_frames 900 \
    --batch_size 8 \
    --epochs 50 \
    --lr 0.001 \
    --n_train_samples 100000 \
    --device 0 \
    --num_workers 8 \
    --output_dims 64 \
    --hidden_dims 64 \
    --depth 10 \
    --sliding_length 1 \
    --sliding_padding 50"

# Create log directories if they don't exist
mkdir -p logs/ts2vec/experiment1

# activate behavemae conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavemae
export PYTHONNOUSERSITE=1

# Run from project root
cd /home/boesch/BehaveMAE

# Run TS2Vec training + inference on SHOT7M2
python -u main_ts2vec.py $common_args
