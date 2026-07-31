#!/usr/bin/env bash
#SBATCH --job-name=pharos-prefetch
#SBATCH --partition=tier3
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# Download the sweep's models. CPU partition, no --gres, on purpose.
#
# Pulling a 9 GB model is network and disk work. Doing it inside a GPU allocation
# would hold a scarce A100 idle for the duration, and RC cancels jobs that leave
# GPUs idle. So the download is its own CPU job and the sweep is chained after it:
#
#   PRE=$(sbatch --parsable cluster/prefetch-models.sbatch)
#   sbatch --dependency=afterok:$PRE cluster/sweep.sbatch
#
# It also cannot be done from the login node. `ollama pull` needs the daemon, and a
# daemon backgrounded over ssh dies with the session -- which killed a pull at 8.4
# of 9 GB once already.
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs "$HOME/.ollama"

echo ">>> node $(hostname)"

ollama serve > "$HOME/.ollama/prefetch-$SLURM_JOB_ID.log" 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://localhost:11434/api/tags >/dev/null || { echo "ollama did not start" >&2; exit 1; }
echo ">>> ollama up"

# Registry keys, resolved through pharos.models so the tags cannot drift from what
# the sweep will ask for.
for key in qwen2.5-3b llama3.2-3b mistral-7b qwen2.5-7b llama3.1-8b qwen2.5-14b; do
  tag="$(uv run python -c "from pharos.models import resolve; print(resolve('$key').tag)")"
  echo ">>> $key -> $tag"
  ollama pull "$tag" 2>&1 | tail -1 || echo "!! pull failed for $tag" >&2
done

echo
echo ">>> installed"
uv run python -m pharos.cli models
echo ">>> PREFETCH COMPLETE"
