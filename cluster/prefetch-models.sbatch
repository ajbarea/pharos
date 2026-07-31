#!/usr/bin/env bash
#SBATCH --job-name=pharos-prefetch
#SBATCH --partition=tier3
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
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
#
# Resource requests are sized from `sacct` on real runs, not guessed. This matters
# for queue time, not politeness: Slurm backfills, so a job claiming eight hours
# needs an eight-hour hole in the schedule while a job claiming ninety minutes drops
# into a gap that opens constantly. An early sweep asked for 8h and used 7 minutes.
# Re-check with:
#   sacct -j <id> --format=JobID,JobName,Elapsed,Timelimit,MaxRSS,ReqMem
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
# Hugging Face cache and credentials.
#
# HF_HOME points at the shared home filesystem rather than a node-local temp dir, so
# a model downloaded by one job is reused by the next instead of being refetched. The
# cache is already several gigabytes; refetching it per job wastes both wall-clock and
# somebody else's bandwidth.
#
# HF_TOKEN is passed through from the environment and never written here. Requests
# work without it, but unauthenticated ones are rate limited, which shows up as a job
# that stalls rather than fails. Read-only, fine-grained, and separate from any local
# token: see docs/cluster.md.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1

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
