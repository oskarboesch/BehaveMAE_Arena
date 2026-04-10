#!/usr/bin/env python3
import subprocess
import os
import sys
import yaml

dry_run = "--dry-run" in sys.argv

BASE_OUTPUT = "/scratch/izar/boesch/BehaveMAE/outputs/arena"
BASE_LOG    = "logs/arena"
GPUS        = 2

with open("scripts/arena/configs.yml") as f:
    config = yaml.safe_load(f)

BASE       = config["base"]
BASE_FLAGS = config["flags"]
experiments = config["experiments"]

SLURM_TEMPLATE = """\
#!/bin/bash
#SBATCH --account=cs433
#SBATCH --qos=normal
#SBATCH --gres=gpu:{gpus}
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=arena-{name}
#SBATCH --output={log_dir}/slurm_%j.log
#SBATCH --error={log_dir}/slurm_%j.err

GPUS={gpus}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate behavemae

OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \\
    --nproc_per_node=$GPUS \\
    --node_rank 0 \\
    --master_addr=127.0.0.1 \\
    --master_port=2999 \\
    run_pretrain.py \\
{args}
"""

for exp in experiments:
    name      = exp["name"]
    overrides = {k: v for k, v in exp.items() if k != "name"}
    cfg       = {**BASE, **overrides}

    log_dir    = os.path.join(BASE_LOG, name)
    output_dir = os.path.join(BASE_OUTPUT, name)
    os.makedirs(log_dir,    exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # None (yaml ~) = flag, string/number = key-value arg
    flags    = BASE_FLAGS + [k for k, v in cfg.items() if v is None]
    key_vals = {k: v for k, v in cfg.items() if v is not None}

    arg_lines = []
    for flag in flags:
        arg_lines.append(f"    --{flag}")
    for key, val in key_vals.items():
        if key == "q_strides":
            arg_lines.append(f"    --{key} \"{val}\"")
        else:
            arg_lines.append(f"    --{key} {val}")
    arg_lines.append(f"    --output_dir {output_dir}")
    arg_lines.append(f"    --log_dir {log_dir}")

    args_str = " \\\n".join(arg_lines)

    script = SLURM_TEMPLATE.format(
        gpus=GPUS, name=name, log_dir=log_dir, args=args_str,
    )

    job_file = os.path.join(log_dir, "job.sh")
    with open(job_file, "w") as f:
        f.write(script)

    print(f"{'[DRY RUN] ' if dry_run else ''}Submitting: {name}")
    if not dry_run:
        result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
        print(f"  {result.stdout.strip() or result.stderr.strip()}")
    else:
        print(f"  job script written to {job_file}")