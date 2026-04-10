#!/bin/bash

experiment=experiment1

python run_emb_analysis.py \
    --dataset arena \
    --path_to_emb_dir /scratch/izar/boesch/BehaveMAE/outputs/arena/${experiment} \
        --meta_data_path /scratch/izar/boesch/data/Arena_Data/hdp_meta.tsv \
    --syllable_labels_path /scratch/izar/boesch/data/Arena_Data/kpt-Moseq/results.h5 \
    --keypoints_path /scratch/izar/boesch/data/Arena_Data/shuffle-3_split-train.npz \
    --output_dir /scratch/izar/boesch/BehaveMAE/outputs/arena/${experiment}/emb_analysis \
    --min_vids_per_strain 30 \
    --ndim_for_cluster 50 \
    --embed_type full_sequence \
    --agg_method mean \
    --kmeans_ks 9 2 \
    --kmeans_k_range 2 40 \
    --gmm_ks_range 2 40 \
    --gmm_ks 8 2 \
    --pca_dims 50   \
    --dbscan_eps 0.5 1.8 \
    --window_size_manifold 1 \
    --window_stride_manifold 1 \
    --window_size_metadata 1 \
    --window_stride_metadata 1 \
    --n_windows_per_run_for_metadata 2 \
    --window_size_syllables 1 \
    --window_stride_syllables 1 \
    --max_n_samples_for_syllable_analysis 10000 \
    --base_window_size 100 \
    --base_window_stride 100 \
    --n_windows_per_run_base 10 \
    --max_nan_ratio_per_window 0.2

cd hierAS-eval

nr_submissions=$(ls /scratch/izar/boesch/BehaveMAE/outputs/arena/${experiment}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))