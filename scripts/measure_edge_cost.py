#!/usr/bin/env python3
"""What a triage agent costs on the hardware it is supposed to run on.

The application this testbed serves is edge triage: nodes "typically operating on the
scale of a laptop or mobile device" that "locally triage data before transmission
according to its expected intelligence value". Every other measurement here asks
whether the agent is *correct*. None asks whether it fits.

That gap was easy to miss because it had already been quietly closed and never
reported. Findings 1 to 3 and 5 were all measured on an 8 GB consumer GPU, which is
laptop-class hardware, so the accuracy results are edge results already. They were
simply never described in those terms, and no cost was attached to them.

Three costs, because they bind at different times:

  footprint   what has to be resident on the node to answer at all
  latency     what one triage decision costs once it is resident
  sync        what crosses the network per personalization round

The third is the one specific to this design rather than to on-device inference in
general. A fleet ships adapters, not models: the base is identical everywhere, so a
round costs the adapter and nothing else. That is computed from the parameter counts
the training runs recorded rather than measured from a file, which is exact and does
not depend on which serialization a checkpoint happened to use.

Latency needs Ollama serving the model and is skipped without it, so the payload and
footprint arithmetic stays runnable anywhere.

    uv run python scripts/measure_edge_cost.py --out results/edge_cost.json
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pharos.attribute import DEFAULT_ENDPOINT, generate_text
from pharos.generate import GeneratorConfig, generate
from pharos.models import DEFAULT_ENDPOINT_TAGS
from pharos.provenance import run_provenance
from pharos.tasks import build_triage_tasks
from pharos.validity import check_sample_size

SEED = 7
EVENTS = 60

#: Tasks to time. Enough for a median and a p95 without turning a cost measurement
#: into an accuracy sweep; correctness is measured elsewhere and is not re-asked here.
TIMED_TASKS = 20

#: The decode a triage answer actually needs. One word from a closed set, so the
#: latency reported is the latency of a decision rather than of an essay.
TRIAGE_PREDICT = 8

#: Bytes per parameter for the dtype an adapter ships in. bf16 is what the training
#: runs used; fp32 is carried because a checkpoint saved without care doubles the
#: payload and the difference is worth seeing rather than assuming.
DTYPE_BYTES = {"bf16": 2, "fp32": 4}

MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SyncCost:
    """What one personalization round puts on the wire, per node."""

    trainable_params: int
    total_params: int
    bf16_mib: float
    fp32_mib: float
    source: str

    @property
    def trainable_share(self) -> float:
        return self.trainable_params / self.total_params if self.total_params else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "trainable_share": round(self.trainable_share, 6),
            "bf16_mib": round(self.bf16_mib, 2),
            "fp32_mib": round(self.fp32_mib, 2),
            "source": self.source,
        }


def _cost(trainable: int, total: int, source: str) -> SyncCost:
    return SyncCost(
        trainable_params=trainable,
        total_params=total,
        bf16_mib=trainable * DTYPE_BYTES["bf16"] / MIB,
        fp32_mib=trainable * DTYPE_BYTES["fp32"] / MIB,
        source=source,
    )


def sync_cost(
    results: Path, *, trainable: int | None = None, total: int | None = None
) -> SyncCost | None:
    """Adapter payload, from whichever training artifact recorded parameter counts.

    Returns None rather than guessing when nothing carries the counts. A payload
    figure derived from the rank and the architecture would look identical to a
    measured one and be wrong the moment a target-module list changed.

    The explicit override exists because the counts were printed to a job log and not
    persisted until `train_adapter.py` was fixed to record them. Supplying them names
    the log as the source in the artifact instead of backfilling a `results/` file by
    hand, which would have made a transcribed number indistinguishable from a measured
    one. Runs after that fix read straight from the artifact and need no override.
    """
    if trainable and total:
        return _cost(trainable, total, "supplied on the command line")
    for path in sorted(results.glob("review_adapter-*.json")) + sorted(
        results.glob("adapter_learnability.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        lora = payload.get("lora") or {}
        if lora.get("trainable_params") and lora.get("total_params"):
            return _cost(
                int(lora["trainable_params"]),
                int(lora["total_params"]),
                f"results/{path.name}",
            )
    return None


def footprint(endpoint: str = DEFAULT_ENDPOINT_TAGS) -> list[dict[str, object]]:
    """On-disk size of each installed model, as the node has to carry it.

    Read from the server's own inventory rather than from this repository's model
    registry: the registry knows which models the sweep *may* use, and the question
    here is what is actually resident on this machine.
    """
    try:
        with urllib.request.urlopen(endpoint, timeout=5.0) as fh:  # noqa: S310
            models = json.load(fh).get("models", [])
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return []
    sized = sorted(
        ((str(m.get("name")), int(m["size"])) for m in models if isinstance(m.get("size"), int)),
        key=lambda pair: -pair[1],
    )
    return [{"model": name, "resident_mib": round(size / MIB, 1)} for name, size in sized]


def evict(model: str, *, endpoint: str) -> bool:
    """Unload `model` from VRAM so the next call genuinely pays for loading it.

    Without this the reported cold start silently depends on whether anything touched
    the model in the last few minutes: Ollama keeps weights resident after a request,
    so a second run of this script measured 0.6s and called it "the weights reaching
    VRAM" when they had never left. Forcing the eviction makes the label true by
    construction rather than by luck, which is the same defect finding 9 found in a
    probe that repeated one prompt.
    """
    body = json.dumps({"model": model, "keep_alive": 0, "prompt": "", "stream": False}).encode()
    request = urllib.request.Request(  # noqa: S310 -- fixed local Ollama endpoint
        endpoint, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0):  # noqa: S310
            pass
    except (urllib.error.URLError, TimeoutError):
        return False
    # The unload is asynchronous; without a pause the next call can still find the
    # weights resident and the cold start is understated again.
    time.sleep(2.0)
    return True


def timed_decisions(model: str, *, endpoint: str, tasks, limit: int) -> list[float]:
    """Seconds per triage decision, one call per task.

    One call per task rather than repeats of one prompt. Finding 9 established that a
    repeated prompt measures a warm cache rather than the workload, and that applies to
    latency at least as much as it did to accuracy.
    """
    seconds = []
    for task in tasks[:limit]:
        started = time.perf_counter()
        generate_text(task.prompt, endpoint=endpoint, model=model, num_predict=TRIAGE_PREDICT)
        seconds.append(time.perf_counter() - started)
    return seconds


@dataclass(frozen=True, slots=True)
class Latency:
    """Cold start and warm steady state, kept apart because they are different costs.

    The first call pays for loading several gigabytes of weights into VRAM; every call
    after it does not. Folding the two together produces a p95 that is really "the
    first call" and a median that silently excludes the largest cost in the
    measurement. Both matter at the edge and they matter at different times: a node
    that sleeps between watches pays cold start every time it wakes, while one held
    resident pays it once.
    """

    cold_start_s: float
    warm_median_s: float
    warm_p95_s: float
    warm_min_s: float
    n_warm: int

    @property
    def decisions_per_hour(self) -> float:
        return 3600.0 / self.warm_median_s if self.warm_median_s else 0.0

    @property
    def cold_start_ratio(self) -> float:
        """How many warm decisions one wake-up costs."""
        return self.cold_start_s / self.warm_median_s if self.warm_median_s else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "cold_start_s": round(self.cold_start_s, 3),
            "warm_median_s": round(self.warm_median_s, 3),
            "warm_p95_s": round(self.warm_p95_s, 3),
            "warm_min_s": round(self.warm_min_s, 3),
            "n_warm": self.n_warm,
            "decisions_per_hour": round(self.decisions_per_hour, 1),
            "cold_start_in_warm_decisions": round(self.cold_start_ratio, 1),
        }


def split_latency(seconds: Sequence[float]) -> Latency:
    """Separate the first call from the rest."""
    if len(seconds) < 2:
        raise SystemExit("need at least two timed calls to separate cold start from warm")
    cold, warm = seconds[0], sorted(seconds[1:])
    return Latency(
        cold_start_s=cold,
        warm_median_s=statistics.median(warm),
        warm_p95_s=warm[min(len(warm) - 1, int(0.95 * len(warm)))],
        warm_min_s=warm[0],
        n_warm=len(warm),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--tasks", type=int, default=TIMED_TASKS)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--trainable", type=int, help="adapter trainable params, if not in an artifact"
    )
    parser.add_argument("--total", type=int, help="base model total params, if not in an artifact")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    results = Path(__file__).resolve().parents[1] / "results"
    report: dict[str, object] = {"provenance": run_provenance(seed=SEED)}

    sync = sync_cost(results, trainable=args.trainable, total=args.total)
    print("sync cost, per node per personalization round:")
    if sync is None:
        print("  no training artifact records parameter counts; run the adapter sweep first")
    else:
        print(
            f"  adapter is {sync.trainable_params:,} of {sync.total_params:,} parameters "
            f"({sync.trainable_share:.3%})"
        )
        print(f"  payload {sync.bf16_mib:.1f} MiB at bf16, {sync.fp32_mib:.1f} MiB at fp32")
        print(f"  counts from {sync.source}")
        print("  the base model never moves, so a round costs the adapter and nothing else")
        report["sync"] = sync.as_dict()

    print("\nresident footprint of installed models:")
    resident = footprint()
    for row in resident:
        print(f"  {row['model']:<24} {row['resident_mib']:>9.1f} MiB")
    report["footprint"] = resident

    if args.skip_latency:
        print("\nlatency skipped")
        report["latency"] = None
    else:
        tasks = build_triage_tasks(generate(GeneratorConfig(seed=SEED, n_events=EVENTS)))
        print(f"\ntiming {args.tasks} triage decisions on {args.model}:")
        if not evict(args.model, endpoint=args.endpoint):
            print("  WARNING: could not evict the model; cold start below is not cold")
        seconds = timed_decisions(args.model, endpoint=args.endpoint, tasks=tasks, limit=args.tasks)
        latency = split_latency(seconds)
        print(f"  cold start {latency.cold_start_s:.1f}s, which is the weights reaching VRAM")
        print(
            f"  warm: median {latency.warm_median_s:.2f}s  p95 {latency.warm_p95_s:.2f}s  "
            f"min {latency.warm_min_s:.2f}s  over {latency.n_warm} calls"
        )
        print(f"  one resident node sustains about {latency.decisions_per_hour:,.0f} per hour")
        print(f"  one wake-up costs what {latency.cold_start_ratio:,.0f} warm decisions cost")
        report["latency"] = {
            "model": args.model,
            "num_predict": TRIAGE_PREDICT,
            **latency.as_dict(),
        }
        report["validity"] = check_sample_size(latency.n_warm, label="edge latency").as_dict()

    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
