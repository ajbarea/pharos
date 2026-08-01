#!/usr/bin/env bash
#SBATCH --job-name=pharos-review-adapter
#SBATCH --partition=sporc-gpu
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# Does a learner trained on a reviewer's decisions learn the rule, or the reviewer?
#
# Finding 6 trained on the generator's ground truth and reached F1 1.000, which
# settled capability. Finding 8 then showed the review stream a reviewer produces
# can be systematically wrong -- a reviewer holding the wrong standard supplies
# targets matching the world on 0.575 or 0.425, both below the 0.625 majority floor
# -- and that no amount of carefulness repairs it. What neither measured is what a
# model does with those targets.
#
# Two outcomes, and they mean opposite things for the design:
#
#   - The adapter matches the TEACHER and not the world. Personalization works
#     exactly as specified, and its risk is that it works: a fleet learning from an
#     over-escalating analyst faithfully reproduces the over-escalation. The
#     mitigation is then annotator-reliability weighting (REALM, arXiv:2604.17289),
#     not more data.
#   - The adapter is pulled toward the world by its own prior. Then the base model's
#     inductive bias is doing work the design does not currently account for, and the
#     personalization claim needs restating.
#
# Both are publishable and they are distinguishable only by scoring the same decode
# twice, against the world and against the teacher, which `--reviewer` makes the
# script do.
#
# Reviewers are chosen to span target accuracy rather than to be a full grid: four
# runs at roughly forty minutes each fits one allocation, where eight would not.
#   by-the-book   1.000  the control; should reproduce finding 6
#   inattentive   0.875  random error, correct standard
#   two-of-three  0.575  systematic error, one step
#   any-one       0.425  systematic error, two steps
#
#   sbatch cluster/review-adapter.sbatch
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
# The compute nodes hide the system gcc behind /tools/bin/blindfold/gcc, which
# prints "hidden from view" and fails. Anything that JIT-compiles needs a real one.
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs results

echo ">>> node $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL="${PHAROS_ADAPTER_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REVIEWERS="${PHAROS_REVIEWERS:-by-the-book inattentive two-of-three any-one}"

# Evaluation size, and the corpus size that keeps the training set fixed while it
# grows. The trainer splits tasks into eval-first, train-rest, so holding
# EVENTS - EVAL constant at 1140 keeps every run's training data the same size as
# the first sweep's and makes the comparison across sizes clean.
#
# 600 is not arbitrary: scripts/measure_power.py puts the minimum detectable effect
# at 60 tasks at 0.233, which does not resolve the 0.083 gap between the careless
# teacher's adapter and that teacher. At 600 the minimum detectable is 0.070, which
# does.
EVAL_TASKS="${PHAROS_EVAL_TASKS:-600}"
EVENTS="${PHAROS_EVENTS:-$((1140 + EVAL_TASKS))}"
echo ">>> model $MODEL"
echo ">>> reviewers $REVIEWERS"
echo ">>> eval tasks $EVAL_TASKS from $EVENTS events (train held at $((EVENTS - EVAL_TASKS)))"

for reviewer in $REVIEWERS; do
  echo
  echo "============================================================"
  echo ">>> teacher: $reviewer"
  echo "============================================================"
  # --skip-base-eval on every run after the first: the base model does not change
  # between teachers, so re-measuring it four times spends roughly a quarter of the
  # allocation confirming a number this job already has. The first run keeps it so
  # the artifact set carries one honest baseline.
  extra=""
  if [ "$reviewer" != "$(echo "$REVIEWERS" | awk '{print $1}')" ]; then
    extra="--skip-base-eval"
  fi

  uv run --extra train python scripts/train_adapter.py \
      --model "$MODEL" \
      --events "$EVENTS" \
      --eval-tasks "$EVAL_TASKS" \
      --epochs 3 \
      --reviewer "$reviewer" \
      $extra \
      --out "results/review_adapter-${reviewer}.json"
done

echo
echo ">>> REVIEW-TAUGHT ADAPTER SWEEP COMPLETE"
echo ">>> artifacts:"
ls -la results/review_adapter-*.json
