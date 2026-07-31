# Running on a cluster

Everything Pharos does today runs on a consumer card. Two things do not:

1. **Models above roughly 8 B.** A 14 B at Q4 is around 9 GB and does not fit in
   8 GB of VRAM, so cross-size comparison is capped without a bigger card.
2. **Adapter training.** LoRA fine-tuning a 7 B needs roughly 16-24 GB.

Both are VRAM ceilings rather than throughput problems, so no local optimisation
reaches them. The scripts live in `cluster/` in the repository; this page is the
short version and the two traps worth knowing before you submit anything.

## What runs where, and why it matters

Not a preference. Model-dependent numbers are **not** bit-reproducible across
machines -- re-running five models on cluster hardware changed 2 of 200 judgements at
temperature zero with a fixed seed -- while the corpus and the gate, which make no
model calls, reproduce exactly. So platform is part of a measurement's identity.

| Work | Where | Why |
| --- | --- | --- |
| Generation, gate, tests, docs | **Local** | No model calls, bit-identical anywhere, and instant |
| Models up to ~8B | **Local** | 7B at Q4 is 4.7 GB and runs at ~16 s/call. Zero queue |
| Models above 8B | **Cluster** | 14B at Q4 is ~9 GB and does not fit an 8 GB card |
| Adapter training | **Cluster** | Needs 16-24 GB. A VRAM ceiling, not a throughput one |
| Anything published as a set | **One platform, all of it** | See below |

**The rule that matters: a comparison must not straddle platforms.** The six-model
sweep was first run locally for five models, then re-run on the cluster once the 14B
became reachable. Had the published table mixed the local five with the cluster 14B,
part of the spread between models would have been the machine rather than the model.
Everything in `results/` is now from one platform for that reason, and each artifact
records the platform it came from so this is checkable rather than remembered.

Local remains the right default for iteration: the GPU partition here runs fully
allocated for hours at a time, so a two-minute job can wait ninety minutes for a
slot. Develop locally, publish from wherever the largest member of the set has to
run.

## The pipeline

```bash
bash cluster/setup-env.sh                        # once per account
sbatch cluster/verify.sbatch                     # lint, types, tests, gate. No GPU
sbatch cluster/gpu-probe.sbatch                  # does the adapter stack run here?
PRE=$(sbatch --parsable cluster/prefetch-models.sbatch)
sbatch --dependency=afterok:$PRE cluster/sweep.sbatch
```

`verify.sbatch` requests **no GPU**, deliberately. Generation, the lattice, and the
gate involve no model calls, so the whole core is verifiable on a CPU node, and
asking for a GPU to run a test suite holds a scarce resource idle.

## Trap one: thread oversubscription

numpy and scikit-learn size their thread pools from the machine's CPU count, while
a Slurm cgroup only schedules `--cpus-per-task`. On a 96-core node with 8 cores
allocated that is a twelve-fold oversubscription. It does not error, does not warn,
and simply crawls: a verify job sat at the same byte of output for twenty minutes
before this was found, and the obvious readings -- a hung network call, a deadlock,
a machine too small -- were all wrong.

Both job scripts now export `OMP_NUM_THREADS` and its siblings from
`SLURM_CPUS_PER_TASK`. More usefully, **every Pharos run logs the mismatch itself**:

```json
{"event": "run.context", "machine_cpus": 96, "usable_cpus": 8,
 "thread_limits": {"OMP_NUM_THREADS": null}, "oversubscription_risk": true}
```

`os.cpu_count()` reports the machine; `sched_getaffinity` reports what this process
may actually use. When they disagree and nothing has capped the pools, the run says
so on its first line instead of leaving you to infer it from a stopwatch.

## Trap two: anything that waits belongs in `sbatch`

An interactive `srun` for a contended GPU sits in the queue. Bind that to an SSH
session with a timeout and the timeout kills the job before it ever starts. The same
applies to a multi-gigabyte `ollama pull`: a daemon backgrounded over SSH dies with
the session, in one case at 8.4 of 9 GB.

Both mistakes were made here. `sbatch` exists for exactly this, and model downloads
now run as their own **CPU-only** job so no GPU idles through a fetch, with the
sweep chained behind it by `--dependency=afterok`.

## What the cluster has confirmed

**The gate is reproducible across platforms.** Surface baselines of 0.6547, 0.6588,
and 0.6675 on seeds 1, 7, and 101 are bit-identical between a WSL laptop and an
RHEL 9 cluster node with a different CPU count, kernel, and libc. Generation and
gating make no model calls precisely so that this would hold; it now holds as a
measurement rather than a design intention.

**The adapter stack runs.** On an A100-PCIE-40GB, under the Python this project
pins: torch 2.13.0+cu130, peft 0.20.0, transformers 5.14.1, a bf16 matmul executing
on the device, and peft correctly freezing a wrapped base at 3.02% trainable. That
last check matters more than it looks: `torch.cuda.is_available()` returning true is
not the same as a kernel running, and the freeze is the mechanism the personal and
shared adapter split depends on.

## Etiquette

Research Computing cancels jobs that hold idle GPUs. Request, run, release. Do not
park an interactive session on a GPU while reading.
