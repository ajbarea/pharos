#!/usr/bin/env bash
#SBATCH --job-name=pharos-verify
#SBATCH --partition=tier3
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# Does Pharos actually work on this cluster?
#
# No GPU requested on purpose. Generation, the label lattice, and the gate are
# deterministic and involve no model calls, so the whole core can be verified on a
# CPU node. Asking for a GPU to run tests would hold a scarce resource idle, and RC
# cancels jobs that do that.
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


echo ">>> node $(hostname)  |  $(nproc) cpus  |  $(uname -m)"
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
mkdir -p cluster/logs

echo
echo ">>> provenance"
uv run python -c "from pharos.provenance import run_provenance; print(run_provenance())"

echo
echo ">>> lint and types"
# Install every group before checking anything.
#
# `ui` and `croissant` are non-default dependency-groups, so a bare `uv run` syncs
# neither. ty still type-checks web.py, which imports fastapi, and reported four
# unresolved-import errors on a node where the package simply was not installed --
# a failure of the environment, not the code. CI never saw it because CI runs
# `uv sync --all-groups`. Running the same command here is the point: a gate that
# checks something narrower than CI can pass while CI fails.
uv sync --all-groups

uv run ruff format --check .
uv run ruff check .
uv run ty check

echo
echo ">>> test suite with coverage floor"
uv run pytest --cov=pharos --cov=scripts --cov-branch --cov-report=term-missing --cov-fail-under=88

echo
echo ">>> the gate, on the seed list CI uses"
for seed in 1 7 101; do
  uv run python -m pharos.cli gate --seed "$seed" --events 400 2>&1 \
    | grep -E "surface baseline|permutation null|VERDICT"
done

echo
echo ">>> export a corpus and validate its Croissant metadata"
uv run python -m pharos.cli export --seed 7 --events 200 --out "$HOME/pharos-export" 2>&1 | tail -3
uv run pytest tests/test_croissant_validation.py -q 2>&1 | tail -3

echo
echo ">>> VERIFY COMPLETE"
