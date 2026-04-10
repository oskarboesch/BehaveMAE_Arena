#!/bin/bash

experiment=experiment1

python run_emb_analysis.py \
    --dataset shot7m2 \
    --path_to_emb_dir /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/${experiment} \
    --keypoints_path /scratch/izar/boesch/data/Shot7M2/test/test_dictionary_poses.npy \
    --output_dir /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/${experiment}/emb_analysis \
    --ndim_for_cluster 50 \
    --embed_type windowed \
    --agg_method mean \
    --gmm_ks 2 2 2 \
    --kmeans_ks 3 7 3 \
    --dbscan_eps_range 0.1 20 1.0 \
    --dbscan_eps 3.1 9.1 6.1 \
    --dbscan_min_samples 10 \
    --window_size_manifold 1 \
    --window_stride_manifold 1 \
    --window_size_metadata 1 \
    --window_stride_metadata 1 \
    --max_nan_ratio_per_window 0.2

cd hierAS-eval

nr_submissions=$(ls /scratch/izar/boesch/BehaveMAE/outputs/shot7m2/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))