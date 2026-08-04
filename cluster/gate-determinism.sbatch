#!/usr/bin/env bash
#SBATCH --job-name=pharos-gate-determinism
#SBATCH --partition=tier3
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# The cluster half of the bit-identity claim.
#
# The README says the gate produces bit-identical surface baselines on a WSL laptop and
# on an RHEL 9 cluster node with a different CPU count, kernel and libc, and that the
# property is the claim while the values move whenever the generator does. That makes it
# a check that has to be redone after every generator change, and it was carrying an
# admission that it had not been: "last run on the previous corpus and owed a repeat".
#
# This exists so the repeat is one command rather than an improvisation. Run the same
# script on the laptop, sync this artifact back, and diff them:
#
#   sbatch cluster/gate-determinism.sbatch
#   uv run python scripts/measure_gate_determinism.py --out results/gate_determinism.json
#   uv run python scripts/measure_gate_determinism.py --compare \
#       results/gate_determinism.json results/gate_determinism-cluster.json
#
# `--compare` refuses when the two artifacts name different commits, because the
# generator defines the corpus and two commits measure two corpora: a difference would
# then say nothing about the machines, which is the only thing being asked.
#
# No GPU, deliberately. The gate is deterministic and model-free -- generation, the
# label lattice, the surface probes and the permutation null are all CPU. Holding an
# A100 to run it would idle a scarce resource, which RC cancels jobs for.
#
# Sized from the laptop run: seven seeds at 400 events take about two minutes there.
# Thirty minutes is generous for a slower node and still small enough to backfill.
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# Pin numerical thread pools to the allocation. numpy and scikit-learn size their pools
# from the machine's CPU count while the Slurm cgroup schedules only --cpus-per-task of
# them, which does not error and simply crawls. It matters more here than elsewhere:
# thread count changes reduction order in a parallel BLAS, and a different reduction
# order is exactly the way a floating-point result stops being bit-identical. Pinning it
# means a mismatch is a real finding rather than an artifact of how busy the node was.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs results

echo ">>> node $(hostname)  |  $(nproc) cpus  |  $(uname -m)  |  $(uname -r)"
echo ">>> commit $(git rev-parse --short=12 HEAD)  dirty=$(git status --porcelain --untracked-files=no | wc -l)"

uv run python scripts/measure_gate_determinism.py \
    --out results/gate_determinism-cluster.json

echo
echo ">>> done. Sync it back and compare:"
echo "    rsync -az sporc:~/ajsoftworks/pharos/results/gate_determinism-cluster.json results/"
echo "    uv run python scripts/measure_gate_determinism.py --compare \\"
echo "        results/gate_determinism.json results/gate_determinism-cluster.json"
