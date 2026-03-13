#!/usr/bin/env bash
set -euo pipefail

python main_classification.py \
  --dataset keypoints \
  --classifier_model_type logistic_regression \
  --window_size 1000 \
  --window_stride 1000 \
  --agg_method mean \
  --train_dataset_path /scratch/izar/boesch/data/Arena_Data/shuffle-3_split-train.npz \
  --test_dataset_path /scratch/izar/boesch/data/Arena_Data/shuffle-3_split-test.npz \
  --metadata_path /scratch/izar/boesch/data/Arena_Data/hdp_meta.tsv \
  --output_dir /scratch/izar/boesch/classification_results