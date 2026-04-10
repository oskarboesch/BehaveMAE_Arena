#!/usr/bin/env python3
import os
import shlex
import subprocess
import sys
import re

import yaml


def _format_arg_value(value):
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


def _build_sbatch_header(slurm_cfg, job_name, log_file, err_file):
    gpu_gres = str(slurm_cfg.get("gres", "")).strip()
    if not gpu_gres:
        raise ValueError(
            "Missing required 'slurm.gres' in analysis config. "
            "Set e.g. slurm.gres: gpu:1 to force GPU allocation."
        )
    if "gpu" not in gpu_gres.lower():
        raise ValueError(
            f"Invalid slurm.gres='{gpu_gres}'. This launcher requires a GPU gres (e.g. gpu:1)."
        )

    partition = slurm_cfg.get("partition")
    constraint = slurm_cfg.get("constraint")
    gpu_count = slurm_cfg.get("gpus_per_task")
    if gpu_count is None:
        match = re.search(r"gpu\s*:\s*(\d+)", gpu_gres, flags=re.IGNORECASE)
        gpu_count = int(match.group(1)) if match else 1

    lines = [
        "#!/bin/bash",
        f"#SBATCH --account={slurm_cfg.get('account', 'cs433')}",
        f"#SBATCH --qos={slurm_cfg.get('qos', 'normal')}",
        f"#SBATCH --gres={gpu_gres}",
        f"#SBATCH --gpus-per-task={gpu_count}",
        f"#SBATCH --time={slurm_cfg.get('time', '24:00:00')}",
        f"#SBATCH --mem={slurm_cfg.get('mem', '128G')}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={slurm_cfg.get('cpus_per_task', 8)}",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={log_file}",
        f"#SBATCH --error={err_file}",
    ]

    if partition:
        lines.insert(3, f"#SBATCH --partition={partition}")
    if constraint:
        lines.insert(4 if partition else 3, f"#SBATCH --constraint={constraint}")

    return "\n".join(lines)


def main():
    dry_run = "--dry-run" in sys.argv

    with open("scripts/arena/analysis_configs.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    slurm_cfg = config.get("slurm", {})
    require_runtime_gpu = bool(slurm_cfg.get("require_runtime_gpu", True))
    base_cfg = config["base"]
    base_flags = config.get("flags", [])
    experiments = config["experiments"]

    log_root = config.get("log_root", "logs/arena")

    reserved_keys = {
        "name",
        "path_to_emb_root",
        "analysis_subdir",
        "conda_env",
        "path_to_emb_dir",
        "output_dir",
    }

    for exp in experiments:
        name = exp["name"]
        overrides = {k: v for k, v in exp.items() if k != "name"}
        cfg = {**base_cfg, **overrides}

        path_to_emb_dir = cfg.get("path_to_emb_dir")
        if not path_to_emb_dir:
            path_to_emb_dir = os.path.join(cfg["path_to_emb_root"], name)

        output_dir = cfg.get("output_dir")
        if not output_dir:
            output_dir = os.path.join(path_to_emb_dir, cfg.get("analysis_subdir", "emb_analysis"))

        log_dir = os.path.join(log_root, name)
        os.makedirs(log_dir, exist_ok=True)

        job_name = f"arena-analysis-{name}"
        job_file = os.path.join(log_dir, "analysis_job.sh")
        log_file = os.path.join(log_dir, "analysis_slurm_%j.log")
        err_file = os.path.join(log_dir, "analysis_slurm_%j.err")

        flags = list(base_flags) + [k for k, v in cfg.items() if v is None and k not in reserved_keys]
        key_vals = {
            k: v
            for k, v in cfg.items()
            if v is not None and k not in reserved_keys
        }

        arg_lines = []
        for flag in flags:
            arg_lines.append(f"    --{flag}")
        for key, val in key_vals.items():
            rendered = _format_arg_value(val)
            arg_lines.append(f"    --{key} {rendered}")

        arg_lines.append(f"    --path_to_emb_dir {shlex.quote(path_to_emb_dir)}")
        arg_lines.append(f"    --output_dir {shlex.quote(output_dir)}")
        args_str = " \\\n".join(arg_lines)

        sbatch_header = _build_sbatch_header(slurm_cfg, job_name, log_file, err_file)
        conda_env = cfg.get("conda_env", "cuml_env")

        script = f"""{sbatch_header}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate {conda_env}
export PYTHONNOUSERSITE=1

if [ "{str(require_runtime_gpu).lower()}" = "true" ]; then
    if [ -z "${{CUDA_VISIBLE_DEVICES:-}}" ] || [ "${{CUDA_VISIBLE_DEVICES}}" = "NoDevFiles" ]; then
        echo "ERROR: No GPU assigned (CUDA_VISIBLE_DEVICES is empty)." >&2
        exit 42
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not available on this node." >&2
        exit 43
    fi
    nvidia-smi -L || {{ echo "ERROR: nvidia-smi cannot list GPUs." >&2; exit 44; }}
    python - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    print("ERROR: torch import failed: {{}}".format(exc), file=sys.stderr)
    raise SystemExit(45)
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print("ERROR: torch sees no CUDA device.", file=sys.stderr)
    raise SystemExit(46)
print("GPU check OK: torch sees {{}} CUDA device(s).".format(torch.cuda.device_count()))
PY
fi

cd /home/boesch/BehaveMAE
mkdir -p {shlex.quote(output_dir)}

python -u run_emb_analysis.py \\
{args_str}
"""

        with open(job_file, "w", encoding="utf-8") as f:
            f.write(script)

        print(f"{'[DRY RUN] ' if dry_run else ''}Submitting analysis: {name}")
        if dry_run:
            print(f"  job script written to {job_file}")
        else:
            result = subprocess.run(["sbatch", job_file], capture_output=True, text=True)
            print(f"  {result.stdout.strip() or result.stderr.strip()}")


if __name__ == "__main__":
    main()
