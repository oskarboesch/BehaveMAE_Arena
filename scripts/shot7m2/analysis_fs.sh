#!/bin/bash

experiment=experiment1

python run_emb_analysis.py \
    --dataset shot7m2 \
    --path_to_emb_dir /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/${experiment} \
    --keypoints_path /scratch/izar/boesch/data/Shot7M2/test/test_dictionary_poses.npy \
    --output_dir /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/${experiment}/emb_analysis \
    --ndim_for_cluster 10 \
    --agg_method mean \
    --embed_type full_sequence \
    --kmeans_ks 2 3 6 \
    --gmm_ks 2 2 2 \
    --dbscan_eps_range 10 50 1 \
    --dbscan_eps 10 41 42 \
    --dbscan_min_samples 5 \
    --window_size_manifold 1 \
    --window_stride_manifold 1 \
    --window_size_metadata 1 \
    --window_stride_metadata 1 \
    --max_nan_ratio_per_window 0.2

cd hierAS-eval

nr_submissions=$(ls /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))