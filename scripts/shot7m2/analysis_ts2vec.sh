#!/bin/bash

experiment=experiment1

python run_emb_analysis.py \
    --dataset shot7m2 \
    --path_to_emb_dir /scratch/izar/boesch/ts2vec/outputs/shot7m2/${experiment} \
    --keypoints_path /scratch/izar/boesch/data/Shot7M2/test/test_dictionary_poses.npy \
    --output_dir /scratch/izar/boesch/ts2vec/outputs/shot7m2/${experiment}/emb_analysis \
    --ndim_for_cluster 50 \
    --embed_type ts2vec \
    --agg_method mean \
    --gmm_ks 10 9 2 \
    --kmeans_ks 10 9 3 \
    --dbscan_eps_range 0.1 5 0.1 \
    --dbscan_eps 2.8 2.1 6.1 \
    --dbscan_min_samples 100 \
    --window_size_manifold 1 \
    --window_stride_manifold 1 \
    --window_size_metadata 1 \
    --window_stride_metadata 1 \
    --max_nan_ratio_per_window 0.2

cd hierAS-eval