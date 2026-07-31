# Running Pharos on RIT Research Computing

Everything Pharos does day to day runs on an 8 GB consumer card. Two things do not,
and both are VRAM ceilings rather than throughput problems, so no local optimisation
reaches them:

1. **Models above ~8 B.** A 14 B at Q4 is around 9 GB and does not fit locally.
2. **Adapter training.** LoRA fine-tuning a 7 B needs roughly 16-24 GB.

## The cluster as it actually is

Surveyed from a node, not from documentation:

| | |
| --- | --- |
| Login | `ssh <username>@sporcsubmit.rc.rit.edu` |
| GPU partition | `sporc-gpu` |
| Nodes | `a100` (2-4 per node) on `skl-a-*`, 4x `h100` on `spr-a-02` |
| Architecture | **x86_64** -- standard CUDA wheels apply |
| Account | required: `fl-mlm`, `prdiscourse`, or `rc-onboard` |
| spack Python | 3.11.x, below this project's `>=3.12` pin |

```bash
sinfo -s                                     # partitions
sinfo -p sporc-gpu -o "%20n %6c %30G %8t"    # which nodes have which GPUs
```

Because spack's Python is too old, `setup-env.sh` lets `uv` fetch its own standalone
interpreter rather than fighting the module system. That also makes the cluster
environment identical to a local checkout, which is the point of pinning versions.

## Setup, once per account

```bash
bash cluster/setup-env.sh
```

Installs `uv` and a linux-amd64 Ollama into `~/.local/bin` and `~/bin`, syncs the
project, and warns if no Hugging Face credential is present. Read it before running
it; it writes to your home directory.

## Jobs

```bash
sbatch cluster/verify.sbatch          # lint, types, tests, gate. No GPU, deliberately
sbatch cluster/gpu-probe.sbatch       # does the training stack run on this node?
PRE=$(sbatch --parsable cluster/prefetch-models.sbatch)
sbatch --dependency=afterok:$PRE cluster/sweep.sbatch
sbatch cluster/adapter.sbatch         # LoRA fine-tune + evaluate
```

`verify.sbatch` requests **no GPU** on purpose. Generation, the lattice, and the gate
make no model calls, so the whole core is verifiable on a CPU node, and holding a
scarce GPU idle to run a test suite is antisocial.

`triton-diagnose.sbatch` exists for the compiler problem below and is not part of the
normal path.

Logs land in `cluster/logs/<jobname>-<jobid>.out`. Results land in `results/`, which
is tracked, so bring them back with `git` or `scp`.

## Syncing code

```bash
scripts/sync_cluster.sh              # to origin/main
scripts/sync_cluster.sh <ref>        # to a specific commit or tag
```

**This fetches commits; it does not copy files.** An earlier version rsynced the
working tree, which had an expensive flaw: rsync moves files but never touches git,
so the cluster's HEAD stayed where it was last cloned while the contents changed
underneath it. Every artifact produced there recorded `git_dirty: true` against a
stale commit, which means the code behind a measurement could not be reconstructed
from its own provenance stamp. The script now refuses to run from a dirty tree or a
HEAD that is not an ancestor of `origin/main`, because measuring code that matches no
commit is the failure it exists to prevent.

That rsync also cost the cluster's entire log history. `--delete` with no exclusion
for `cluster/logs/` read the absent local directory as an instruction to remove the
remote one. Logs written only on the cluster are gone; `results/` survived because it
exists on both sides. The general rule, which outlives this particular script: when a
directory is generated remotely and never locally, `--delete` will destroy it.

## Two traps worth knowing before you submit anything

### Thread oversubscription

numpy and scikit-learn size their pools from the machine's CPU count while a Slurm
cgroup only schedules `--cpus-per-task`. On a 96-core node with 8 allocated that is a
twelvefold oversubscription which does not error, does not warn, and simply crawls: a
verify job sat at the same byte of output for twenty minutes before this was found,
and the obvious readings -- hung network call, deadlock, undersized machine -- were
all wrong.

Every job script now exports `OMP_NUM_THREADS` and its siblings from
`SLURM_CPUS_PER_TASK`, and every run logs the mismatch itself rather than leaving you
to infer it from a stopwatch:

```json
{"event": "run.context", "machine_cpus": 96, "usable_cpus": 8,
 "thread_limits": {"OMP_NUM_THREADS": null}, "oversubscription_risk": true}
```

### The system compiler is hidden

`/tools/bin/blindfold/gcc` shadows gcc on the compute nodes and prints *"The system
install of this program, gcc, has been hidden from view"* before exiting non-zero.
Anything that JIT-compiles at runtime hits this. Triton does, when a fused attention
kernel is selected, which killed the first adapter run inside `model.generate()`
after the model had loaded and the data had been built.

Two defences, both applied:

- `export CC=/usr/bin/gcc`. A real gcc 11.5.0 is present; only the default name is
  shadowed.
- Pass `attn_implementation` explicitly instead of taking the library default. A
  diagnostic run confirmed `sdpa` and `eager` both generate without touching Triton at
  all, which is the better fix because it removes the dependency rather than
  satisfying it.

**The GPU probe did not catch this, and that is the lesson.** It ran a bf16 matmul,
which goes through cuBLAS; Triton only engages for the fused kernels generation uses.
A probe that exercises a different code path than the workload is necessary and not
sufficient, and it is worth asking of any probe what it is *not* touching.

## Hugging Face tokens and cache

Read-only, and never write: Pharos calls `from_pretrained` and `load_dataset` and has
no upload path anywhere. All six job scripts export the same `HF_HOME` on the shared
filesystem so a model downloaded by one job is reused by the next. Full policy, and
why CI needs no token at all, in [`docs/cluster.md`](../docs/cluster.md).

## What the cluster has confirmed

**The gate is reproducible across platforms.** Surface baselines of 0.6547, 0.6588,
and 0.6675 on seeds 1, 7, and 101 are bit-identical between a WSL laptop and an
RHEL 9 cluster node with a different CPU count, kernel, and libc. Generation and
gating make no model calls precisely so this would hold; it now holds as a
measurement rather than a design intention.

**The adapter stack runs, and the rule is gradient-learnable.** On an A100-PCIE-40GB
under this project's pins -- torch 2.13.0+cu130, peft 0.20.0, transformers 5.14.1 --
peft correctly freezes a wrapped base at 3.02% trainable, and a LoRA fine-tune of
Qwen2.5-3B-Instruct moves triage F1 from 0.469 to 1.000 on a held-out split. Read
that score with its caveat: it is saturated, and sixty flawless answers still bound
the true error rate only at 5%. See [`docs/findings.md`](../docs/findings.md).

**Model-dependent numbers are not bit-reproducible across machines.** Re-running five
models on cluster hardware changed 2 of 200 judgements at temperature zero with a
fixed seed. Platform is therefore part of a measurement's identity, and **a published
comparison must not straddle platforms** -- otherwise part of the spread between
models is the machine rather than the model. Every artifact records the platform it
came from so this is checkable rather than remembered.

## Etiquette

Research Computing cancels jobs that hold idle GPUs. Request, run, release. Do not
park an interactive session on a GPU while reading.
