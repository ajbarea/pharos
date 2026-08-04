#!/usr/bin/env bash
#SBATCH --job-name=pharos-review-adapter
#SBATCH --partition=sporc-gpu
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# %A_%a rather than %j, so an array task's log is named by the array job and the task
# index that produced it. With %j each task is named by its own internal job id, which
# is a different number from the `21444845_7` that squeue, sacct and every error message
# use -- array task 0 of the first run landed in a file called `...-21444846.out`, and
# mapping a log back to a teacher meant grepping the file for its own index. For a
# non-array submission %A_%a still resolves, to the job id and a bare `4294967294`.
#SBATCH --output=cluster/logs/%x-%A_%a.out
#SBATCH --error=cluster/logs/%x-%A_%a.err
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

# The four named teachers span target accuracy but they are four, and four adapters is
# four case studies rather than a fleet: federated-LoRA work runs 20 clients in its main
# experiments and extends to 50 and 100. Submitting this as an array runs one teacher
# per array task over the full standard-by-carefulness grid instead --- the same grid
# `measure_review_sweep.py` sweeps, so each teacher's target accuracy is known before an
# adapter is trained on it, and the 24 cells span 0.445 to 1.000.
#
#   sbatch cluster/review-adapter.sbatch                    # the committed four
#   sbatch --array=0-23%4 cluster/review-adapter.sbatch     # the 24-teacher fleet
#
# %4 because four concurrent A100s is what the allocation holds; the array serialises
# the rest rather than queueing 24 at once.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  GRID=()
  for threshold in 1 2 3; do
    for slip in 0 0.05 0.1 0.15 0.2 0.3 0.4 0.5; do
      GRID+=("t${threshold}s${slip}")
    done
  done
  REVIEWERS="${GRID[$SLURM_ARRAY_TASK_ID]}"
  echo ">>> array task $SLURM_ARRAY_TASK_ID of ${#GRID[@]} -> teacher $REVIEWERS"
fi

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

# Optional: evaluate on a DIFFERENT corpus instantiation entirely. Unset, training
# and evaluation split one corpus by event, which shows the rule generalises across
# events. Set, they share no corpus at all -- different vessels, different events,
# different renderings -- which is what contamination-resistance guidance asks of a
# benchmark and is a strictly stronger claim about what the adapter learned.
EVAL_SEED_ARG=""
SUFFIX=""
if [ -n "${PHAROS_EVAL_SEED:-}" ]; then
  EVAL_SEED_ARG="--eval-seed ${PHAROS_EVAL_SEED}"
  SUFFIX="-xseed${PHAROS_EVAL_SEED}"
  echo ">>> cross-corpus: evaluating on seed ${PHAROS_EVAL_SEED}"
fi
echo ">>> model $MODEL"
echo ">>> reviewers $REVIEWERS"
echo ">>> eval tasks $EVAL_TASKS from $EVENTS events (train held at $((EVENTS - EVAL_TASKS)))"

for reviewer in $REVIEWERS; do
  echo
  echo "============================================================"
  echo ">>> teacher: $reviewer"
  echo "============================================================"
  # --skip-base-eval on every run after the first: the base model does not change
  # between teachers, so re-measuring it once per teacher spends a large fraction of
  # the allocation confirming a number the job already has. One honest baseline is
  # kept so the artifact set carries it.
  #
  # "The first" has to mean the first *across the whole submission*, not the first in
  # this process. Under `--array` each task runs a single teacher, so every task saw
  # itself as first and re-ran the base evaluation: 24 tasks, 24 base evaluations at
  # 600 tasks apiece, of which 23 measure an unchanged model. Array task 0 keeps it
  # and the rest skip.
  extra=""
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    [ "$SLURM_ARRAY_TASK_ID" -ne 0 ] && extra="--skip-base-eval"
  elif [ "$reviewer" != "$(echo "$REVIEWERS" | awk '{print $1}')" ]; then
    extra="--skip-base-eval"
  fi

  uv run --extra train python scripts/train_adapter.py \
      --model "$MODEL" \
      --events "$EVENTS" \
      --eval-tasks "$EVAL_TASKS" \
      --epochs 3 \
      --reviewer "$reviewer" \
      $EVAL_SEED_ARG \
      $extra \
      --out "results/review_adapter-${reviewer}${SUFFIX}.json"
done

echo
echo ">>> REVIEW-TAUGHT ADAPTER SWEEP COMPLETE"
echo ">>> artifacts:"
ls -la results/review_adapter-*${SUFFIX}.json
