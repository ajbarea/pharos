#!/usr/bin/env bash
#SBATCH --job-name=pharos-adapter
#SBATCH --partition=sporc-gpu
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# The adapter experiment: is the decision rule learnable by gradient?
#
# Finding 5 showed in-context examples close none of the gap to the rule-given
# ceiling, and said explicitly that this does not settle the question because
# in-context and gradient learning are different mechanisms. This runs the other one.
#
# Scoped honestly: it trains on Pharos's clean definitional labels, not on the
# accept/revise/reject streams the design actually describes, because those are build
# step 3 and do not exist. A negative result here is therefore strong evidence
# against the premise; a positive one is necessary but not sufficient for it.
#
# Resource requests are sized from `sacct` on real runs, not guessed. This matters
# for queue time, not politeness: Slurm backfills, so a job claiming eight hours
# needs an eight-hour hole in the schedule while a job claiming ninety minutes drops
# into a gap that opens constantly. An early sweep asked for 8h and used 7 minutes.
# Re-check with:
#   sacct -j <id> --format=JobID,JobName,Elapsed,Timelimit,MaxRSS,ReqMem
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# The compute nodes hide the system gcc behind /tools/bin/blindfold/gcc, which
# prints "hidden from view" and fails. Anything that JIT-compiles at runtime needs
# a real compiler named explicitly, so point CC at one that exists.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

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
mkdir -p cluster/logs results

echo ">>> node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL="${PHAROS_ADAPTER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
echo ">>> model $MODEL"

# The 3B is the deliberate choice, not a compromise. In the six-model sweep the 3B
# scored highest and the 14B did not beat it, so the interesting question is whether
# training moves the best available small model rather than whether scale substitutes
# for the rule.
uv run --extra train python scripts/train_adapter.py \
    --model "$MODEL" \
    --events 1200 \
    --eval-tasks 60 \
    --epochs 3 \
    --out "results/adapter_learnability.json"

echo
echo ">>> ADAPTER EXPERIMENT COMPLETE"
