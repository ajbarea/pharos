#!/usr/bin/env bash
#SBATCH --job-name=pharos-triton-diag
#SBATCH --partition=sporc-gpu
#SBATCH --account=fl-mlm
#SBATCH --qos=qos_tier3
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/logs/%x-%j.out
#SBATCH --error=cluster/logs/%x-%j.err
#
# Why did generation fail when the GPU probe passed?
#
# The adapter job died inside model.generate(): Triton tried to JIT-compile its CUDA
# shim, invoked /tools/bin/blindfold/gcc, and the link against libcuda.so.1 failed.
# The GPU probe never saw this because a bf16 matmul goes through cuBLAS; Triton only
# engages for the fused kernels generation uses. The probe was necessary and not
# sufficient, which is worth remembering about probes generally.
#
# This tests the candidate fixes in one allocation rather than discovering them one
# failed two-hour submission at a time.
set -euo pipefail

export PATH="$HOME/bin:$HOME/.local/bin:$PATH"

# torch sizes its intra-op pool from the machine's CPU count, not the cgroup's, so
# this job was the one script running numerical code without a cap on it.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
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

echo ">>> node $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader

echo
echo ">>> what compilers exist, and which does Triton reach for?"
for c in /usr/bin/gcc /usr/bin/cc /tools/bin/blindfold/gcc; do
  printf '  %-28s ' "$c"
  [ -x "$c" ] && "$c" --version 2>&1 | head -1 || echo "absent"
done
echo "  CC in env: ${CC:-<unset>}  CXX: ${CXX:-<unset>}"

echo
echo ">>> is libcuda.so.1 linkable?"
ldconfig -p 2>/dev/null | grep -m3 libcuda || echo "  not in ldconfig cache"
for d in /usr/lib64 /lib64 /usr/local/cuda/lib64/stubs; do
  [ -e "$d/libcuda.so.1" ] && echo "  found $d/libcuda.so.1"
done

echo
echo ">>> A: does a real gcc fix the Triton compile?"
export CC=/usr/bin/gcc
uv run --extra train python - <<'PYEOF' 2>&1 | tail -5
import os, torch
print("  CC =", os.environ.get("CC"))
try:
    import triton, triton.language as tl
    @triton.jit
    def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < n
        tl.store(out_ptr + off, tl.load(x_ptr + off, mask=m) + tl.load(y_ptr + off, mask=m), mask=m)
    x = torch.ones(1024, device="cuda"); y = torch.ones(1024, device="cuda")
    out = torch.empty_like(x)
    _add[(1,)](x, y, out, 1024, BLOCK=1024)
    torch.cuda.synchronize()
    print("  A: triton kernel compiled and ran ->", float(out[0]))
except Exception as exc:
    print(f"  A: FAILED {type(exc).__name__}: {str(exc)[:200]}")
PYEOF

echo
echo ">>> B: does eager attention avoid Triton entirely during generate()?"
uv run --extra train python - <<'PYEOF' 2>&1 | tail -6
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "Qwen/Qwen2.5-3B-Instruct"
for impl in ("eager", "sdpa"):
    try:
        tok = AutoTokenizer.from_pretrained(name)
        m = AutoModelForCausalLM.from_pretrained(
            name, dtype=torch.bfloat16, device_map="cuda", attn_implementation=impl
        )
        ids = tok("Decide: SIGNIFICANT or ROUTINE.", return_tensors="pt").to(m.device)
        out = m.generate(**ids, max_new_tokens=6, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
        print(f"  B[{impl}]: generate OK ->", repr(tok.decode(out[0][ids['input_ids'].shape[1]:])))
        del m; torch.cuda.empty_cache()
    except Exception as exc:
        print(f"  B[{impl}]: FAILED {type(exc).__name__}: {str(exc)[:200]}")
PYEOF

echo
echo ">>> DIAGNOSIS COMPLETE"
