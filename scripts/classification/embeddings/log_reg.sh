#!/usr/bin/env bash
set -euo pipefail

python main_classification.py \
  --dataset embeddings \
  --classifier_model_type logistic_regression \
  --window_size 1000 \
  --window_stride 1000 \
  --embedding_stage stage2 \
  --agg_method mean \
  --train_dataset_path /scratch/izar/boesch/BehaveMAE/outputs/900_models/ofd_tailhip_20260226-205043/all_embeddings.h5 \
  --test_dataset_path /scratch/izar/boesch/BehaveMAE/outputs/900_models/ofd_tailhip_20260226-205043/test/all_embeddings.h5 \
  --metadata_path /scratch/izar/boesch/data/Arena_Data/hdp_meta.tsv \
  --output_dir /scratch/izar/boesch/classification_results