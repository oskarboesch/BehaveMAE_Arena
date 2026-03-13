#!/usr/bin/env bash
set -euo pipefail

python main_classification.py \
  --dataset kp-moseq \
  --classifier_model_type logistic_regression \
  --window_size 1000 \
  --window_stride 1000 \
  --agg_method mean \
  --train_dataset_path /scratch/izar/boesch/data/Arena_Data/kpt-Moseq/results.h5\
  --metadata_path /scratch/izar/boesch/data/Arena_Data/hdp_meta.tsv \
  --output_dir /scratch/izar/boesch/classification_results