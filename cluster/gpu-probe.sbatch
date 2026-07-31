#!/usr/bin/env bash
#SBATCH --job-name=pharos-gpu-probe
#SBATCH --partition=sporc-gpu
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# Can the adapter experiment actually run here?
#
# This answers the one question that gates writing an adapter-training job at all:
# does a CUDA-capable torch see a GPU on an allocated node, under the Python this
# project pins. Everything else about that experiment is design work; this is the
# fact it rests on.
#
# Deliberately sbatch and not srun. GPU partitions here are contended, so an
# interactive srun sits in the queue, and binding a queued job to an ssh session
# with a timeout kills it before it ever runs. That mistake cost two attempts.
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/bin:$PATH"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p cluster/logs

echo ">>> node $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo ">>> torch sees the device?"
cd "$HOME/torch-probe"
uv run --python 3.13 python - <<'PYEOF'
import torch, peft, transformers, accelerate

print(f"torch        {torch.__version__}")
print(f"peft         {peft.__version__}")
print(f"transformers {transformers.__version__}")
print(f"accelerate   {accelerate.__version__}")
print(f"cuda avail   {torch.cuda.is_available()}")
print(f"devices      {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible: the adapter job cannot be written yet")

print(f"device 0     {torch.cuda.get_device_name(0)}")
free, total = torch.cuda.mem_get_info()
print(f"vram         {total / 1e9:.1f} GB total, {free / 1e9:.1f} GB free")

# Actually move work onto the device. `is_available()` returning True is not the
# same as a kernel running, and the difference is exactly the kind of thing that
# looks fine until a training loop starts.
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(f"bf16 matmul  {tuple(y.shape)} on {y.device} OK")

# A LoRA-shaped smoke test: does peft wrap a module and mark only the adapter
# trainable? This is the mechanism the whole personal/shared split depends on.
import torch.nn as nn
from peft import LoraConfig, get_peft_model

base = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).cuda()
wrapped = get_peft_model(base, LoraConfig(r=8, target_modules=["0", "2"]))
trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in wrapped.parameters())
print(f"lora         {trainable} trainable of {total_params} ({100 * trainable / total_params:.2f}%)")
assert 0 < trainable < total_params, "peft did not freeze the base"
print("VERDICT      the adapter stack runs on this hardware")
PYEOF

echo
echo ">>> GPU PROBE COMPLETE"
