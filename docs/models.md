# Choosing a model

```bash
uv run python -m pharos.cli models
```

```text
KEY              TAG                                SIZE    VRAM    INSTALLED  VERIFIED
llama3.2-3b      llama3.2:3b-instruct-q4_K_M        3B      2.3     yes        yes
qwen2.5-3b       qwen2.5:3b-instruct                3B      2.0     yes        yes
qwen2.5-7b       qwen2.5:7b-instruct                7.6B    4.7     yes        yes
llama3.1-8b      llama3.1:8b-instruct-q4_K_M        8B      4.9     yes        yes
mistral-7b       mistral:7b-instruct                7B      4.1     yes        yes
qwen2.5-14b      qwen2.5:14b-instruct               14B     9.0     no         candidate
```

Any script or endpoint that takes `--model` accepts a registry key, a raw tag, or
nothing at all:

```bash
uv run python scripts/measure_triage_lift.py --model llama3.1-8b --tasks 40
uv run python scripts/measure_triage_lift.py --model some-model:70b   # passthrough
```

## What `verified` means, and why it is not decoration

**`verified` means the model has answered a Pharos triage task and returned a
parseable verdict.** `candidate` means nobody has run it. The distinction is
enforced by a test, which fails if the flag is set on a model that has not been
swept.

That test exists because "supported models" lists in research code are usually
aspirational, and a reader cannot tell which entries were ever executed. Pharos has
a specific reason to care: its first several findings were all measured on
`qwen2.5:7b-instruct` and nothing else, so every one of them was a single-model
result. A registry that blurred tested with untested would have hidden that
limitation rather than exposed it.

`INSTALLED` is read live from the Ollama daemon on every invocation, not asserted.
A stopped daemon reports everything as absent rather than raising.

## Why the registry cannot refuse a model

`resolve` wraps an unrecognised name as an ad-hoc, unverified spec and passes it
through to the backend. The registry is a convenience and an honesty record, never
a gate on what can be run. A model Pharos has never heard of still works; it simply
carries `family: unknown` and `verified: false` into the provenance of whatever it
produces.

## The one entry that needs more than 8 GB

`qwen2.5-14b` is deliberately listed and deliberately not installed. At roughly
9 GB it exceeds a consumer card, and it is the first thing to run on a cluster
node. Leaving it in the registry as a `candidate` records the intent without
pretending the measurement exists.

## Sweeping every installed model

```bash
scripts/sweep_models.sh 40        # 40 triage tasks per installed model
uv run python scripts/compare_models.py
```

The sweep stops each model between runs on purpose. Ollama sizes GPU offload once,
at load time, and keeps that split for the life of the loaded model, so loading a
second model on top of a resident one can silently push most layers to CPU at
roughly eight times the cost with no error and no warning. The script checks the
resident share per model rather than assuming it.

`compare_models.py` prints every score against the majority-class floor and the
surface baseline, never against 0.5. Ground truth here is content-defined, so a
probe reading nothing already scores well above chance, and a model that looks
respectable against 0.5 may be doing nothing at all. See
[Findings](findings.md) for what the sweep actually showed.
