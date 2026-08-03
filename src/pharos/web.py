"""A minimal explorer, so Pharos can be understood without reading Pharos.

The corpus, the lattice, and the gate are the three things a newcomer has to
grasp, and all three are currently only reachable through Python. This serves them
over HTTP behind one self-contained page: pick a seed, look at labelled reports,
ask whether one label dominates another, and run a triage task against a model you
choose from a dropdown.

Two constraints shape it.

**No network at render time.** The page inlines its own CSS and JavaScript and
loads nothing from a CDN, because a testbed that promises to run offline should not
have a front door that fails without internet.

**The API is the honest surface.** Every endpoint returns the same objects the
Python API produces, so the page is a client rather than a reimplementation. If the
UI shows a label, that label came from `pharos.labels`, not from a formatting
routine written twice.

FastAPI lives in the optional `ui` dependency group, so the core install stays at
numpy, scikit-learn, and OpenTelemetry.

    uv sync --group ui
    uv run python -m pharos.cli serve
"""

from pathlib import Path
from typing import Any

from pharos import models
from pharos.analyst import (
    DEFAULT_CEILING,
    DEFAULT_ENSEMBLE,
    KEEP_COMPARTMENTS,
    Proposal,
    evidence_shown,
)
from pharos.attribute import DEFAULT_ENDPOINT, generate_text
from pharos.disclosure import admit
from pharos.export import corpus_row
from pharos.gate import run_gate
from pharos.generate import GeneratorConfig, generate
from pharos.labels import Capacity, Compartment, Label, Sensitivity, declassify, join
from pharos.tasks import build_triage_tasks
from pharos.world import SIGNIFICANT_PATTERN

STATIC = Path(__file__).parent / "static"

#: Bounded so a curious click cannot start a minutes-long job. The explorer is for
#: understanding the mechanism; the CLI is for measuring it.
MAX_EVENTS = 400


def _label_payload(label: Label) -> dict[str, Any]:
    return {
        "sensitivity": label.sensitivity.name,
        "compartments": sorted(str(c) for c in label.compartments),
        "capacity": label.capacity.name,
        "rendered": f"{label.sensitivity.name}"
        f"[{','.join(sorted(str(c) for c in label.compartments))}]",
    }


def _parse_label(payload: dict[str, Any]) -> Label:
    """Build a Label from the page's JSON, failing closed on anything unrecognised."""
    try:
        sensitivity = Sensitivity[payload.get("sensitivity", "OPEN")]
        capacity = Capacity[payload.get("capacity", "FREETEXT")]
        compartments = frozenset(Compartment[c] for c in payload.get("compartments", []))
    except KeyError as exc:
        raise ValueError(f"unknown label component: {exc}") from exc
    return Label(sensitivity, compartments, capacity)


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse

    app = FastAPI(
        title="Pharos explorer",
        description="Poke at the corpus, the label lattice, and the gate.",
    )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/models")
    def api_models() -> dict[str, Any]:
        """The model selector's contents, with live installation status."""
        return {"default": models.DEFAULT_KEY, "models": models.catalog()}

    @app.get("/api/vocabulary")
    def api_vocabulary() -> dict[str, Any]:
        """Everything the label pickers need, taken from the enums themselves."""
        return {
            "sensitivity": [s.name for s in Sensitivity],
            "compartments": [str(c) for c in Compartment],
            "capacity": [c.name for c in Capacity],
        }

    @app.get("/api/corpus")
    def api_corpus(seed: int = 7, events: int = 40, limit: int = 12) -> dict[str, Any]:
        """A small corpus, its label histogram, export rows, and a sample of reports."""
        if events > MAX_EVENTS:
            raise HTTPException(400, f"events must be <= {MAX_EVENTS}")
        reports = generate(GeneratorConfig(seed=seed, n_events=events))
        histogram: dict[str, int] = {}
        for report in reports:
            key = _label_payload(report.label)["rendered"]
            histogram[key] = histogram.get(key, 0) + 1
        return {
            "seed": seed,
            "n_events": events,
            "n_reports": len(reports),
            "label_histogram": dict(sorted(histogram.items())),
            "export_rows": [corpus_row(r) for r in reports],
            "reports": [
                {
                    "report_id": r.report_id,
                    "event_id": r.event_id,
                    "report_type": str(r.report_type),
                    "center_id": r.center.center_id,
                    "is_plant": r.is_plant,
                    "text": r.text,
                    "label": _label_payload(r.label),
                }
                for r in reports[:limit]
            ],
        }

    @app.post("/api/dominance")
    def api_dominance(payload: dict[str, Any]) -> dict[str, Any]:
        """Does the holder label dominate the object label, and does it the other way?

        Both directions are returned because the interesting case is when neither
        holds, and a one-way answer hides it.
        """
        try:
            holder = _parse_label(payload["holder"])
            item = _parse_label(payload["item"])
        except (KeyError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        forward = holder.dominates(item)
        backward = item.dominates(holder)
        return {
            "holder": _label_payload(holder),
            "item": _label_payload(item),
            "holder_may_read_item": forward,
            "item_may_read_holder": backward,
            "incomparable": not forward and not backward,
            "join": _label_payload(join([holder, item], capacity=item.capacity)),
        }

    @app.get("/api/gate")
    def api_gate(seed: int = 7, events: int = 120, null_trials: int = 8) -> dict[str, Any]:
        """The gate on a small corpus. Smaller than the published runs, and says so."""
        if events > MAX_EVENTS:
            raise HTTPException(400, f"events must be <= {MAX_EVENTS}")
        reports = generate(GeneratorConfig(seed=seed, n_events=events))
        result = run_gate(reports, null_trials=null_trials)
        return {
            "seed": seed,
            "events": events,
            "surface_baseline": round(result.surface_baseline, 4),
            "null_mean": round(result.null_mean, 4) if result.null_mean is not None else None,
            "null_sd": round(result.null_sd, 4) if result.null_sd is not None else None,
            "significant": result.leak_is_significant,
            "per_probe_auc": {k: round(v, 4) for k, v in result.per_probe_auc.items()},
            "note": (
                "Fewer events and fewer null trials than the published runs, so this "
                "is noisier than the numbers in the paper. Use the CLI to reproduce those."
            ),
        }

    @app.post("/api/triage")
    def api_triage(payload: dict[str, Any]) -> dict[str, Any]:
        """Run one triage task against the selected model."""
        seed = int(payload.get("seed", 7))
        index = int(payload.get("index", 0))
        spec = models.resolve(payload.get("model"))

        reports = generate(GeneratorConfig(seed=seed, n_events=40))
        tasks = build_triage_tasks(reports)
        if not tasks:
            raise HTTPException(500, "no tasks generated")
        task = tasks[index % len(tasks)]

        try:
            answer = generate_text(
                task.prompt,
                endpoint=payload.get("endpoint", DEFAULT_ENDPOINT),
                model=spec.tag,
            )
        except Exception as exc:  # the backend is out of our control
            raise HTTPException(
                502,
                f"model call failed for {spec.tag}: {exc}. Is Ollama running, "
                f"and has this tag been pulled?",
            ) from exc

        upper = answer.upper()
        said_significant = "SIGNIFICANT" in upper
        said_routine = "ROUTINE" in upper
        verdict = None if said_significant == said_routine else said_significant

        # TriageTask.label already computes this join at ENUM capacity. Recomputing
        # it here would be a second implementation of a governance decision, free to
        # drift from the one every measurement uses.
        governed = task.label
        return {
            "task_id": task.task_id,
            "model": spec.as_dict(),
            "n_sources": len(task.sources),
            "truth_significant": task.significant,
            "model_verdict": verdict,
            "correct": None if verdict is None else verdict == task.significant,
            "raw": answer.strip()[:600],
            "governed_label": _label_payload(governed),
            "prompt": task.prompt,
        }

    @app.get("/api/review")
    def api_review(seed: int = 7, index: int = 0, verdict: bool = True) -> dict[str, Any]:
        """What every reviewer in the default grid does with one proposed verdict.

        Takes the verdict as a parameter rather than calling a model, so the page can
        show both cases side by side without waiting on a backend. The interesting
        cell is a reviewer who objects and whose own correction is still not
        releasable: the objection is real, and acting on it changes nothing.
        """
        reports = generate(GeneratorConfig(seed=seed, n_events=40))
        tasks = build_triage_tasks(reports)
        if not tasks:
            raise HTTPException(500, "no tasks generated")
        task = tasks[index % len(tasks)]

        release = declassify(task.label, KEEP_COMPARTMENTS)
        proposal = Proposal(task.task_id, verdict, release)
        rows = []
        for policy in DEFAULT_ENSEMBLE:
            decision = policy.review(task, proposal, seed=seed)
            corrected = decision.corrected_release
            rows.append(
                {
                    "analyst": policy.name,
                    "escalation_threshold": policy.escalation_threshold,
                    "escalates": policy.escalates,
                    "action": str(decision.action),
                    "grounds": sorted(str(g) for g in decision.grounds),
                    "reasons": sorted(str(r) for r in decision.reasons),
                    "corrected_verdict": decision.corrected_verdict,
                    "corrected_release": None if corrected is None else _label_payload(corrected),
                    "correction_releasable": (
                        None if corrected is None else DEFAULT_CEILING.dominates(corrected)
                    ),
                    # What the reviewer's own correction would meet at the ceiling.
                    # A correction that is merely blocked and one that nobody may
                    # ever authorise look identical without this.
                    "correction_disposition": (
                        None
                        if corrected is None
                        else str(policy.judge_release(corrected).disposition)
                    ),
                }
            )

        return {
            "task_id": task.task_id,
            "seed": seed,
            "truth_significant": task.significant,
            "proposed_verdict": verdict,
            "evidence_shown": sorted(evidence_shown(task)),
            "evidence_needed": len(SIGNIFICANT_PATTERN),
            "proposed_release": _label_payload(release),
            "ceiling": _label_payload(DEFAULT_CEILING),
            "proposal_releasable": DEFAULT_CEILING.dominates(release),
            "proposal_decision": admit(release, DEFAULT_CEILING).as_dict(),
            "reviewers": rows,
            "note": (
                "Reviewers are decision procedures with named parameters, not "
                "simulated people. Each row bounds a mechanism and estimates no "
                "population."
            ),
        }

    return app


def serve(host: str = "127.0.0.1", port: int = 8080) -> int:
    try:
        import uvicorn
    except ImportError:
        print("The explorer needs the ui group: uv sync --group ui")  # noqa: T201
        return 1
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
    return 0
