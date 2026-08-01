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
qwen2.5-14b      qwen2.5:14b-instruct               14B     9.0     no         yes
```

Any script or endpoint that takes `--model` accepts a registry key, a raw tag, or
nothing at all:

```bash
uv run python scripts/measure_triage_lift.py --model llama3.1-8b --tasks 40
uv run python scripts/measure_triage_lift.py --model some-model:70b   # passthrough
```

## What `verified` means

**`verified` means the model has answered a Pharos triage task and returned a
parseable verdict.** `candidate` means nobody has run it. A test enforces the
distinction and fails if the flag is set on a model that has not been swept.

The flag exists because a "supported models" list cannot otherwise be told apart
from an aspirational one. Findings 1 to 3 were measured on `qwen2.5:7b-instruct`
alone, and the registry is what makes that visible rather than hidden.

`INSTALLED` is read live from the Ollama daemon on every invocation, not asserted.
A stopped daemon reports everything as absent rather than raising.

## Unknown models still run

`resolve` wraps an unrecognised name as an ad-hoc, unverified spec and passes it to
the backend. The registry records what has been tried; it never gates what can run.
A model Pharos has never heard of works, and carries `family: unknown` and
`verified: false` into the provenance of whatever it produces.

## The entry that needs more than 8 GB

`qwen2.5-14b` is listed but not installed locally. At roughly 9 GB it exceeds an
8 GB card, so it runs on a cluster A100. `INSTALLED` therefore reads `no` on a
workstation while `VERIFIED` reads `yes`: the model has answered Pharos tasks, just
not on this machine.

## Sweeping every installed model

```bash
scripts/sweep_models.sh 40        # 40 triage tasks per installed model
uv run python scripts/compare_models.py
```

The sweep stops each model between runs. Ollama sizes GPU offload once at load
time and keeps that split for the life of the loaded model, so loading a second
model on top of a resident one can push most layers to CPU with no error and no
warning. Measured cost of that split here: roughly eight times slower. The script
checks the resident share per model rather than assuming it.

`compare_models.py` prints every score against the majority-class floor and the
surface baseline, never against 0.5. Ground truth here is content-defined, so a
probe reading nothing already scores well above chance, and a model that looks
respectable against 0.5 may be doing nothing at all. See
[Findings](findings.md) for what the sweep actually showed.
