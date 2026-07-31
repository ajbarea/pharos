# Running Pharos on RIT Research Computing

> **Status: executed and green, 2026-07-31.** `verify.sbatch` completed on
> `skl-a-34` in 19m17s: 160 tests, 96.60% coverage, lint and types clean, Croissant
> metadata validated, and the gate run on three seeds. `sweep.sbatch` has not been
> run yet.

**The gate reproduced bit-identically to a local run.** Seeds 1, 7, and 101 gave
surface baselines of 0.6547, 0.6588, and 0.6675 on the cluster, matching a WSL
machine with different CPU count, kernel, and libc to four decimal places. That is
what the offline-and-deterministic constraint was for, and it is now demonstrated
rather than claimed.

Pharos runs fine on an 8 GB consumer card for everything it does today. Two things
need the cluster:

1. **Models above ~8 B.** A 14 B at Q4 is around 9 GB and does not fit locally, so
   cross-size comparison is capped without it.
2. **The adapter experiment.** LoRA fine-tuning a 7 B needs roughly 16-24 GB. This
   is a VRAM ceiling, not a throughput problem, so no local optimisation reaches it.

## What the older notes get wrong

Surveyed live on 2026-07-31. Three things had changed since the notes this was
first written from, and all three matter:

- **There is no `grace` partition**, and no `gg-`/`gh-` nodes. GPUs are in
  **`sporc-gpu`**: `a100` (2-4 per node) on `skl-a-*`, and 4x `h100` on `spr-a-02`.
- **Those nodes are x86_64**, not aarch64. Standard CUDA wheels apply, which makes
  the adapter experiment considerably less risky than the ARM-era notes implied.
  Leftover aarch64 binaries in `~/bin` and `~/.local/bin` still passed
  `command -v` and failed only when run, so `setup-env.sh` tests execution.
- **Jobs need `--account`** (`fl-mlm`, `prdiscourse`, or `rc-onboard`).

The one thing that carried over unchanged: spack's Python is 3.11.x against this
project's `>=3.12` pin, so setup lets `uv` fetch its own interpreter.

## Pin your thread pools

Not optional. numpy and scikit-learn size their pools from the machine's CPU count
while the Slurm cgroup only schedules `--cpus-per-task`. On a 96-core node with 8
allocated that is a 12x oversubscription which does not error, does not warn, and
simply crawls: a verify job sat at the same byte of output for twenty minutes
before this was found. Both job scripts now export `OMP_NUM_THREADS` and friends
from `SLURM_CPUS_PER_TASK`, and every run logs `run.context` with an
`oversubscription_risk` flag when the numbers do not reconcile.

## Syncing code to the cluster

```bash
rsync -az --delete \
  --exclude='.venv/' --exclude='site/' --exclude='__pycache__/' \
  --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.coverage' \
  --exclude='export/' --exclude='.cache/' --exclude='adapter-out/' \
  --exclude='cluster/logs/' \
  ./ sporc:~/ajsoftworks/pharos/
```

**`--exclude='cluster/logs/'` is not optional.** Job output lives only on the
cluster; nothing writes it locally. Without that exclusion `--delete` reads the
absent local directory as an instruction to remove the remote one, and every log
from every completed job disappears. That happened once, and it destroyed the
sweep's logs while the results themselves survived only because `results/` exists
on both sides.

The same reasoning applies to anything else that is generated remotely and never
locally. When in doubt, drop `--delete`.

## Access

```bash
ssh <username>@sporcsubmit.rc.rit.edu
sinfo -s                 # partitions; GPUs are in `sporc-gpu`
sinfo -p sporc-gpu -o "%20n %6c %30G %8t"   # which nodes have which GPUs
```

A GPU node interactively:

```bash
srun --partition=sporc-gpu --account=fl-mlm --qos=qos_tier3 \
     --gres=gpu:a100:1 --cpus-per-task=8 --mem=48G --time=4:00:00 --pty bash --login
nvidia-smi               # a100 on skl-a-*, h100 on spr-a-02
```

RC cancels jobs that hold idle GPUs. Request, run, release; do not park an
interactive session on a GPU while reading.

## Why this does not use spack's Python

The cluster's spack Python is **3.11.x**. Pharos pins `requires-python = ">=3.12"`,
so it will not install under it. Rather than fight the module system for a newer
interpreter, `setup-env.sh` lets `uv` fetch its own standalone build. That
also makes the environment identical to a local checkout, which is the point of
pinning versions in the first place.

## Setup, once per account

```bash
bash cluster/setup-env.sh
```

Installs `uv` and a linux-amd64 Ollama into `~/.local/bin` and `~/bin`, and syncs
the project. It clears aarch64 leftovers from the ARM era first, since those pass
`command -v` and fail only when run. Read it before running it; it writes to your
home directory.

## Jobs

```bash
sbatch cluster/sweep.sbatch          # multi-model triage sweep across size classes
```

Logs land in `cluster/logs/<jobname>-<jobid>.out`. Results land in `results/`, which
is tracked, so bring them back with `scp` or `git`.

## What is deliberately not here

No adapter-training job yet. The CUDA wheel situation on `sporc-gpu` is unverified
from a node, and writing an `sbatch` file that asserts a working training stack
would be a prediction rather than a script. Verify it on a node first, then add it.
x86_64 makes this much lower-risk than the ARM-era notes implied, but lower-risk is
not verified.
