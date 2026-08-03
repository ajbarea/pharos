# Model Selection & Registry

Inspect installed and verified models in the Pharos model registry:

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

---

## Model Selection Syntax

Any script or endpoint accepting `--model` accepts a registry key, raw tag, or passthrough:

```bash
# Using a registry key
uv run python scripts/measure_triage_lift.py --model llama3.1-8b --tasks 40

# Passthrough for unlisted / large models
uv run python scripts/measure_triage_lift.py --model some-model:70b
```

---

## Registry Verification Mechanics

| Status Flag | Meaning | Impact |
| :--- | :--- | :--- |
| **`VERIFIED: yes`** | Model has executed Pharos triage tasks and returned parseable outputs | Enforced by test suite to prevent speculative claims |
| **`VERIFIED: no`** | Model spec is registered as a candidate but has not been swept | Flagged if quoted as evidence of capability |
| **`INSTALLED`** | Evaluated live against the local Ollama daemon on each CLI invocation | Reports `no` if daemon is stopped or model is remote |

!!! note "What `VERIFIED` Enforces"
    The `VERIFIED` flag distinguishes models with empirical measurement history from aspirational model specs.

!!! info "Unregistered & Passthrough Models"
    `resolve` wraps unrecognized model names as ad-hoc specs (`family: unknown`, `verified: false`). The registry tracks history; it **never blocks execution**.

---

## Off-Node & Cluster Models

!!! warning "VRAM Hardware Constraints"
    `qwen2.5-14b` (~9 GB VRAM at Q4) exceeds 8 GB consumer GPUs and is executed on cluster nodes (e.g. NVIDIA A100). On a local workstation, `INSTALLED` reports `no` while `VERIFIED` reports `yes`.

---

## Multi-Model Benchmarking

Run full model sweeps across installed models:

```bash
# Run 40 triage tasks per installed model
scripts/sweep_models.sh 40

# Aggregate and compare results
uv run python scripts/compare_models.py
```

!!! danger "GPU Offload & Thread Safety"
    Ollama sizes GPU offload once, at load time, and keeps that split for the life of the loaded model, so loading a second model on top of a resident one can silently push most layers to CPU (up to **$8\times$ slowdown**) with no error and no warning. `sweep_models.sh` stops each model between runs and *reports* the resulting split per model rather than assuming it. It does not enforce a share: the 3B-8B models sit at 100%, and `qwen2.5-14b` at 64% on an 8 GB card, which is the number to read before trusting a latency.

