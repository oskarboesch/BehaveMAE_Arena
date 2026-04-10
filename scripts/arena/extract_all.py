#!/usr/bin/env python3
# launch_test_all.py — run as: python launch_test_all.py [--dry-run]

import subprocess
import os
import sys
import yaml

dry_run = "--dry-run" in sys.argv

BASE_OUTPUT = "/scratch/izar/boesch/BehaveMAE/outputs/arena"
BASE_LOG    = "logs/arena"
DEFAULT_GPU_GRES = "gpu:1"

with open("scripts/arena/configs.yml") as f:
    config = yaml.safe_load(f)

BASE        = config["base"]
experiments = config["experiments"]

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --account={account}
#SBATCH --qos={qos}
{partition_line}#SBATCH --gres={gpu_gres}
{constraint_line}#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=test-arena-{name}
#SBATCH --output={log_dir}/test_slurm_%j.log
#SBATCH --error={log_dir}/test_slurm_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavemae

# --- extract embeddings ---
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_extract_emb.py \\
    --bootstrap \\
    --path_to_data_dir {path_to_data_dir} \\
    --dataset {dataset} \\
    --sampling_rate {sampling_rate} \\
    --batch_size 1 \\
    --model gen_hiera \\
    --input_size {input_size} \\
    --stages {stages} \\
    --q_strides "{q_strides}" \\
    --mask_unit_attn {mask_unit_attn} \\
    --patch_kernel {patch_kernel} \\
    --init_embed_dim {init_embed_dim} \\
    --init_num_heads {init_num_heads} \\
    --out_embed_dims {out_embed_dims} \\
    --window_size_embedding {window_size_embedding} \\
    --num_workers 8 \\
    --max_nan_frac {max_nan_frac} \\
{centeralign_extract}{pos_only_extract}{no_pos_extract}{subsample_keypoints_extract}    --output_dir {output_dir}

# --- run test ---
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run_test.py \\
    --path_to_data_dir {path_to_data_dir} \\
    --dataset {dataset} \\
    --sampling_rate {sampling_rate} \\
    --embedsum False \\
    --fast_inference True \\
    --batch_size {test_batch_size} \\
    --model gen_hiera \\
    --input_size {num_frames} {input_size} \\
    --stages {stages} \\
    --q_strides "{q_strides}" \\
    --mask_unit_attn {mask_unit_attn} \\
    --patch_kernel {patch_kernel} \\
    --init_embed_dim {init_embed_dim} \\
    --init_num_heads {init_num_heads} \\
    --out_embed_dims {out_embed_dims} \\
    --distributed \\
    --num_frames {num_frames} \\
    --num_workers 8 \\
    --max_nan_frac {max_nan_frac} \\
{centeralign_test}{pos_only_test}{no_pos_test}{subsample_keypoints_test}    --output_dir {output_dir}

# --- evaluate ---
cd hierAS-eval

nr_submissions=$(ls {output_dir}/test_submission_* 2>/dev/null | wc -l)
files=($(seq 0 $((nr_submissions - 1))))
"""

for exp in experiments:
    name      = exp["name"]
    overrides = {k: v for k, v in exp.items() if k != "name"}
    cfg       = {**BASE, **overrides}

    log_dir    = os.path.join(BASE_LOG, name)
    output_dir = os.path.join(BASE_OUTPUT, name)
    os.makedirs(log_dir, exist_ok=True)

    # YAML "~" maps to None; treat key presence as enabling the flag.
    has_centeralign = ("centeralign" in overrides) or bool(cfg.get("centeralign", False))
    has_pos_only = ("pos_only" in overrides) or bool(cfg.get("pos_only", False))
    has_no_pos = ("no_pos" in overrides) or bool(cfg.get("no_pos", False))
    has_subsample_keypoints = ("subsample_keypoints" in overrides) or bool(cfg.get("subsample_keypoints", False))
    centeralign_extract = "    --centeralign \\\n" if has_centeralign else ""
    centeralign_test = "    --centeralign \\\n" if has_centeralign else ""
    pos_only_extract = "    --pos_only \\\n" if has_pos_only else ""
    pos_only_test = "    --pos_only \\\n" if has_pos_only else ""
    no_pos_extract = "    --no_pos \\\n" if has_no_pos else ""
    no_pos_test = "    --no_pos \\\n" if has_no_pos else ""
    subsample_keypoints_extract = "    --subsample_keypoints \\\n" if has_subsample_keypoints else ""
    subsample_keypoints_test = "   --subsample_keypoints \\\n" if has_subsample_keypoints else ""
    partition_line = f"#SBATCH --partition={cfg['partition']}\n" if cfg.get("partition") else ""
    constraint_line = f"#SBATCH --constraint={cfg['constraint']}\n" if cfg.get("constraint") else ""

    script = SLURM_TEMPLATE.format(
        account=cfg.get("account", "cs433"),
        qos=cfg.get("qos", "normal"),
        partition_line=partition_line,
        constraint_line=constraint_line,
        gpu_gres=cfg.get("gpu_gres", DEFAULT_GPU_GRES),
        name=name,
        log_dir=log_dir,
        output_dir=output_dir,
        num_frames=cfg["num_frames"],
        sampling_rate=cfg.get("sampling_rate", 1),
        path_to_data_dir=cfg["path_to_data_dir"],
        dataset=cfg["dataset"],
        input_size=cfg["input_size"],
        stages=cfg["stages"],
        q_strides=cfg["q_strides"],
        mask_unit_attn=cfg["mask_unit_attn"],
        patch_kernel=cfg["patch_kernel"],
        init_embed_dim=cfg["init_embed_dim"],
        init_num_heads=cfg["init_num_heads"],
        out_embed_dims=cfg["out_embed_dims"],
        test_batch_size=cfg.get("test_batch_size", 64),
        centeralign_extract=centeralign_extract,
        centeralign_test=centeralign_test,
        pos_only_extract=pos_only_extract,
        pos_only_test=pos_only_test,
        no_pos_extract=no_pos_extract,
        no_pos_test=no_pos_test,
        subsample_keypoints_extract=subsample_keypoints_extract,
        subsample_keypoints_test=subsample_keypoints_test,
        max_nan_frac=cfg.get("max_nan_frac", 0.0),
        window_size_embedding=cfg.get("window_size_embedding", 1),
    )

    job_file = os.path.join(log_dir, "test_job.sh")
    with open(job_file, "w") as f:
        f.write(script)

    print(f"{'[DRY RUN] ' if dry_run else ''}Submitting test: {name}")
    if not dry_run:
        result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
        print(f"  {result.stdout.strip() or result.stderr.strip()}")
    else:
        print(f"  job script written to {job_file}")