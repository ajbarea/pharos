#!/usr/bin/env bash
# Run the triage benchmark across every installed registry model.
#
# Every Pharos finding to date used one model, so every finding is a single-model
# result. This is the cheapest way to stop that being true: triage generates eight
# tokens per task, so a whole model costs a couple of minutes rather than an hour.
#
# Models are stopped between runs on purpose. Ollama sizes GPU offload once, at load
# time, and keeps that split for the life of the loaded model; loading a second model
# on top of a resident one can silently push most layers to CPU and cost roughly 8x
# with no error and no warning. So: load one, run, unload, and check the split each
# time rather than assume it.
#
#   scripts/sweep_models.sh [n_tasks]
set -euo pipefail

TASKS="${1:-40}"
cd "$(dirname "$0")/.."
mkdir -p results

installed() {
  curl -sf http://localhost:11434/api/tags \
    | python3 -c "import json,sys; print(' '.join(m['name'] for m in json.load(sys.stdin).get('models',[])))"
}

gpu_share() {
  curl -sf http://localhost:11434/api/ps | python3 -c "
import json, sys
for m in json.load(sys.stdin).get('models', []):
    total = m['size'] or 1
    print(f\"    {m['name']}: {100 * m.get('size_vram', 0) / total:.0f}% on GPU\")
" || true
}

PRESENT="$(installed)"
KEYS="$(uv run python -c "from pharos.models import REGISTRY; print(' '.join(REGISTRY))")"

echo "sweep: ${TASKS} triage tasks per model"
for key in $KEYS; do
  tag="$(uv run python -c "from pharos.models import resolve; print(resolve('$key').tag)")"
  if [[ " $PRESENT " != *" $tag "* ]]; then
    echo ">>> skip $key ($tag not pulled)"
    continue
  fi

  echo ">>> $key  ($tag)"
  uv run python scripts/measure_triage_lift.py \
      --model "$key" --tasks "$TASKS" \
      --out "results/triage_lift-${key}.json" >/dev/null 2>&1 \
    && gpu_share \
    || echo "    !! failed"

  ollama stop "$tag" >/dev/null 2>&1 || true
done

echo
echo ">>> sweep complete"
uv run python scripts/compare_models.py 2>/dev/null || ls -la results/triage_lift-*.json
