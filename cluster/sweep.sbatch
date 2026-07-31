#!/usr/bin/env bash
#SBATCH --job-name=pharos-sweep
#SBATCH --partition=sporc-gpu
#SBATCH --account=fl-mlm
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --qos=qos_tier3
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# Multi-model triage sweep.
#
# Partition/arch verified against the live cluster 2026-07-31: GPUs are in
# sporc-gpu on x86_64 Skylake (a100) and Sapphire Rapids (h100) nodes. The
# `grace` partition and the aarch64 nodes referenced in older notes are gone.
#
# Why this job exists: every Pharos measurement to date used one model
# (qwen2.5:7b-instruct), so every finding is a single-model result. The local 8 GB
# card covers up to about 8 B, which is enough to vary family but not size class.
# The A100 is what makes 14 B and above reachable, which is the part that turns
# "we tried a model" into "we varied the model".
#
# Resource requests are sized from `sacct` on real runs, not guessed. This matters
# for queue time, not politeness: Slurm backfills, so a job claiming eight hours
# needs an eight-hour hole in the schedule while a job claiming ninety minutes drops
# into a gap that opens constantly. An early sweep asked for 8h and used 7 minutes.
# Re-check with:
#   sacct -j <id> --format=JobID,JobName,Elapsed,Timelimit,MaxRSS,ReqMem
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
# Pin numerical thread pools to the allocation. Without this, numpy/scikit-learn
# size their pools from the machine's CPU count while the Slurm cgroup only
# schedules --cpus-per-task of them. On a 96-core node with 8 cores allocated that
# is a 12x oversubscription: it does not error, does not warn, and simply crawls.
# A verify job sat at the same byte of output for twenty minutes because of this.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

export OLLAMA_CONTEXT_LENGTH=8192

mkdir -p cluster/logs results

echo ">>> node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Ollama, backgrounded, with an explicit wait rather than a fixed sleep.
# ---------------------------------------------------------------------------
echo ">>> starting ollama"
mkdir -p "$HOME/.ollama"
ollama serve > "$HOME/.ollama/server-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -sf http://localhost:11434/api/tags >/dev/null || { echo "ollama did not come up" >&2; exit 1; }
echo ">>> ollama up"

# Registry keys, not raw tags: pharos.models resolves them and records which key
# produced each artifact. 14b is the one the local card cannot reach.
MODELS=(qwen2.5-3b llama3.2-3b mistral-7b qwen2.5-7b llama3.1-8b qwen2.5-14b)

for key in "${MODELS[@]}"; do
  tag="$(uv run python -c "from pharos.models import resolve; print(resolve('$key').tag)")"
  # Normally already present via cluster/prefetch-models.sbatch, which downloads on
  # a CPU node so no GPU idles during a multi-gigabyte fetch. Pull here only as a
  # fallback when the sweep is run without that dependency.
  if ! ollama list 2>/dev/null | grep -q "^${tag}[[:space:]]"; then
    echo ">>> $tag absent, pulling (prefer running prefetch-models.sbatch first)"
    ollama pull "$tag" || { echo "!! pull failed for $tag, skipping" >&2; continue; }
  fi
  echo ">>> $key ($tag)"

  # Confirm the model actually landed on the GPU. A silent CPU offload costs ~8x
  # and looks identical in the output, so it gets checked rather than assumed.
  uv run python - <<PYEOF
import json, urllib.request
with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=10) as r:
    for m in json.load(r).get("models", []):
        share = 100 * m.get("size_vram", 0) / m["size"]
        print(f">>> {m['name']}: {share:.0f}% on GPU")
PYEOF

  echo ">>> triage sweep: $key"
  uv run python scripts/measure_triage_lift.py \
      --model "$key" --tasks 40 \
      --out "results/triage_lift-${key}.json" || echo "!! $key failed" >&2

  ollama stop "$tag" || true
done

echo ">>> sweep complete"
ls -la results/
