# Running on a Cluster

While Pharos core generation and gating run on local workstations, cluster execution (e.g. Slurm HPC environments) is required for two workloads:

1. **Large Models (>8B Parameters)**: e.g. 14B at Q4 (~9 GB VRAM) exceeds 8 GB consumer GPUs.
2. **Adapter Fine-Tuning**: LoRA fine-tuning for 7B models requires 16–24 GB VRAM.

---

## Execution Environment Breakdown

Model-dependent numbers are **not** bit-reproducible across different CPU/GPU platforms (e.g., small numerics variations under temperature 0). The platform is part of the measurement provenance.

| Workload | Recommended Platform | Reason |
| :--- | :---: | :--- |
| **Generation, Gate, Tests, Docs** | 💻 **Local Workstation** | Offline, zero-model calls, 100% bit-identical |
| **Models up to ~8B** (Qwen 3B/7B, Llama 8B) | 💻 **Local Workstation** | 4.7 GB VRAM at Q4, ~16s/call, zero queue latency |
| **Models above 8B** (Qwen 14B) | ⚡ **HPC Cluster** | Exceeds 8 GB VRAM capacity |
| **LoRA Adapter Training** | ⚡ **HPC Cluster** | Requires 16–24 GB VRAM allocation |
| **Published Measurement Sets** | 🎯 **Single Unified Platform** | Prevents cross-platform variance from confounding model scores |

!!! note "Platform Consistency Rule"
    Never mix local and cluster runs within the same benchmark comparison table. Every committed result artifact in `results/` records its exact execution platform.

---

## Slurm Cluster Pipeline

```bash
# 1. One-time environment setup
bash cluster/setup-env.sh

# 2. Verify non-GPU codebase (CPU-only Slurm node)
sbatch cluster/verify.sbatch

# 3. Verify PyTorch & PEFT adapter setup on GPU
sbatch cluster/gpu-probe.sbatch

# 4. Prefetch model weights (CPU job) and chain GPU evaluation job
PRE=$(sbatch --parsable cluster/prefetch-models.sbatch)
sbatch --dependency=afterok:$PRE cluster/sweep.sbatch
```

!!! tip "CPU-Only Verification"
    `verify.sbatch` requests zero GPUs. Generation, lattice algebra, and gate checks run purely on CPU, keeping GPU nodes unblocked.

---

## High-Impact Slurm Traps

### ⚠️ Trap 1: Thread Oversubscription

Slurm cgroups restrict `--cpus-per-task` (e.g., 8 cores allocated on a 96-core node). By default, `numpy` and `scikit-learn` inspect total system cores (96), resulting in **$12\times$ thread oversubscription** and severe performance degradation.

```json
{"event": "run.context", "machine_cpus": 96, "usable_cpus": 8, "thread_limits": {"OMP_NUM_THREADS": null}, "oversubscription_risk": true}
```

!!! warning "Fixing Thread Mismatches"
    Every Pharos job automatically logs thread mismatch warnings on launch and exports `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK`.

### ⚠️ Trap 2: Session-Bound Interactive Jobs

Interactive `srun` sessions or backgrounded `ollama pull` commands terminate if the parent SSH connection times out.

!!! tip "Asynchronous Job Chaining"
    Use `sbatch` with `--dependency=afterok:<job_id>` to chain model downloads ahead of evaluation runs without occupying idle GPUs.

---

## Verified Cluster Achievements

* ⚠️ **Bit-Identical Gate Verification**: surface baselines matched bit-for-bit between WSL Linux laptops and RHEL 9 HPC cluster nodes on the pre-2026-08-03 corpus. The generator now derives a random stream per event, so the values are 0.6378, 0.6545 and 0.6604 on seeds 1, 7 and 101; these were recomputed on the laptop and the cluster half of the comparison is owed a repeat.
* ✅ **Gradient Learnability**: Validated on NVIDIA A100-PCIE-40GB (`torch 2.13.0+cu130`, `peft 0.20.0`). LoRA adapter fine-tuning moved triage F1 from **0.469 to 1.000**.

---

## Hugging Face Credentials & Caching

| Credential Scope | Type | Environment Variable / Location |
| :--- | :--- | :--- |
| **`pharos-local`** | Fine-grained **Read-Only** | Workstation (`hf auth login`) |
| **`pharos-cluster`** | Fine-grained **Read-Only** | Cluster `~/.bashrc` |

!!! note "Zero Credentials Needed for Core Run"
    All model checkpoints and evaluation corpora used by Pharos are public. Unauthenticated downloads work automatically with rate-limit warnings.

!!! info "Shared Filesystem Cache"
    All cluster job scripts export `HF_HOME` pointing to shared home storage rather than transient node-local scratch, avoiding redundant weight downloads across nodes.

