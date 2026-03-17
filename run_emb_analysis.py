# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import os
from pathlib import Path

from analysis.embedding_analysis import embedding_analysis, get_args_parser

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    embeddings_file = os.path.join(args.output_dir, "test_submission_combined.npy")
    if os.path.exists(embeddings_file):
        print("Embeddings files already exists - no inference needed. Skipping.")
        exit(0)

    embedding_analysis(args)
