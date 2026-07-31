# Running Pharos on RIT Research Computing

> **Status: written, not yet executed.** These scripts were derived from a working
> session on the cluster but have not themselves been run there. Treat the first
> execution as a test, not a deployment, and fix this file when reality disagrees.

Pharos runs fine on an 8 GB consumer card for everything it does today. Two things
need the cluster:

1. **Models above ~8 B.** A 14 B at Q4 is around 9 GB and does not fit locally, so
   cross-size comparison is capped without it.
2. **The adapter experiment.** LoRA fine-tuning a 7 B needs roughly 16-24 GB. This
   is a VRAM ceiling, not a throughput problem, so no local optimisation reaches it.

## The one thing that breaks everything

**The GPU nodes are ARM64** (`aarch64`, arch string `linux-rhel9-neoverse_v2` --
NVIDIA Grace CPUs). An x86_64 download will not run and often fails in a way that
looks like a missing library rather than a wrong architecture. Every binary below is
the ARM build, deliberately.

## Access

```bash
ssh <username>@sporcsubmit.rc.rit.edu
sinfo -s                 # partitions; GPUs live in `grace` (gg-*, gh-*)
```

A GPU node interactively:

```bash
srun --partition=grace --gres=gpu:1 --mem=32G --time=8:00:00 --qos=qos_tier3 --pty bash --login
nvidia-smi               # gg-00 is an A100-PCIE-40GB
```

RC cancels jobs that hold idle GPUs. Request, run, release; do not park an
interactive session on a GPU while reading.

## Why this does not use spack's Python

The cluster's spack Python is **3.11.7**. Pharos pins `requires-python = ">=3.12"`,
so it will not install under it. Rather than fight the module system for a newer
interpreter, `setup-env.sh` lets `uv` fetch its own standalone aarch64 build. That
also makes the environment identical to a local checkout, which is the point of
pinning versions in the first place.

## Setup, once per account

```bash
bash cluster/setup-env.sh
```

Installs `uv` and an ARM64 Ollama into `~/.local/bin` and `~/bin`, and syncs the
project. Read it before running it; it writes to your home directory.

## Jobs

```bash
sbatch cluster/sweep.sbatch          # multi-model triage sweep across size classes
```

Logs land in `cluster/logs/<jobname>-<jobid>.out`. Results land in `results/`, which
is tracked, so bring them back with `scp` or `git`.

## What is deliberately not here

No adapter-training job yet. PyTorch aarch64 + CUDA on Grace is unverified from
here, and writing an `sbatch` file that asserts a working training stack would be a
prediction rather than a script. Verify the wheel situation on a node first, then
add it.
