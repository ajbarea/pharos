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

Every job that runs numerical code exports `OMP_NUM_THREADS` and its siblings from
`SLURM_CPUS_PER_TASK`; the model-download job is exempt because it only fetches. More
usefully, **every Pharos run logs the mismatch itself**:

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

`sbatch` exists for exactly this. Model downloads run as their own **CPU-only** job
so no GPU idles through a fetch, with the sweep chained behind by
`--dependency=afterok`.

## What the cluster has confirmed

**The gate is reproducible across platforms.** Surface baselines of 0.6547, 0.6588,
and 0.6675 on seeds 1, 7, and 101 are bit-identical between a WSL laptop and an
RHEL 9 cluster node with a different CPU count, kernel, and libc. Generation and
gating make no model calls precisely so that this would hold; it now holds as a
measurement rather than a design intention.

**The adapter stack runs, and the rule is gradient-learnable.** On an A100-PCIE-40GB
under this project's pins -- torch 2.13.0+cu130, peft 0.20.0, transformers 5.14.1 --
peft freezes a wrapped base at 3.02% trainable, and a LoRA fine-tune of
Qwen2.5-3B-Instruct moves triage F1 from 0.469 to 1.000 on a held-out split. Read the
score with its caveat: it is saturated, and sixty flawless answers still bound the
true error rate only at 5%. See [Findings](findings.md).

The freeze check matters more than it looks. `torch.cuda.is_available()` returning
true is not the same as a kernel running, and the freeze is the mechanism the
personal and shared adapter split depends on.

## Hugging Face: tokens and the cache

**Read-only, and never write.** Pharos calls `from_pretrained` for models and
`load_dataset` for the external-validation corpora. It has no `push_to_hub` and no
upload path anywhere, so a write-scoped token would grant an authority nothing in
this repository exercises. Current Hub guidance is to use fine-grained tokens for
anything automated and to create one per usage, so that revoking access to a machine
does not revoke it everywhere:

| Token | Scope | Where it would live |
| --- | --- | --- |
| `pharos-local` | fine-grained, **read** | workstation, via `hf auth login` |
| `pharos-cluster` | fine-grained, **read** | cluster `~/.bashrc`, separate so it can be revoked alone |

**Neither exists today, and nothing currently needs one.** Verified 2026-08-01 on
both machines: no `~/.cache/huggingface/token`, no `HF_TOKEN` in the environment.
The adapter jobs run entirely from the model cache already on the cluster, and an
unauthenticated run prints one warning about rate limits and then proceeds. The
table above is therefore the shape to create *when* a token is needed -- a fresh
download of a model not in the cache, or a gated repository -- not a description of
current state, which is what it read as before this note.

CI needs no token at all. The only Hugging Face-adjacent test is Croissant
validation, which runs against the schema locally and touches no network.

Publishing the Pharos corpus to the Hub as a dataset is a plausible future step for
a resource paper. That would need write -- and a *third*, separate token created at
that point, not a widening of these two.

**Nothing is committed, and the tree is arranged so that stays true.** `.env`,
`*.token`, and `secrets/` are ignored. Credentials reach a job through the
environment: `sbatch` exports the submitting shell by default, so a token exported in
`~/.bashrc` is present inside the allocation without ever being written to a file
in the repository.

**One cache, on the shared filesystem.** Every job script exports the same
`HF_HOME`, pointing at the shared home rather than a node-local scratch that
disappears with the allocation. This was previously set in two of six scripts, which
meant some jobs quietly refetched weights another job had already downloaded. Current
usage on the cluster is about 6 GB of Hugging Face cache plus 26 GB of Ollama models
against a 1 TB home, so the constraint is bandwidth and queue time, not disk.

Everything Pharos downloads is public, so an unauthenticated run *works*. It is
rate limited, though, and a throttled download inside a GPU allocation spends the
allocation rather than failing fast -- which is why `setup-env.sh` warns when no
credential is present instead of waiting for a job to discover it.

## The full job reference

[`cluster/README.md`](https://github.com/ajbarea/pharos/blob/main/cluster/README.md)
sits next to the job scripts and carries the rest: every `sbatch` file and what it is
for, the survey of partitions and accounts, and the attested output of the last
verify run.

## Etiquette

Research Computing cancels jobs that hold idle GPUs. Request, run, release. Do not
park an interactive session on a GPU while reading.
