"""Command line entry: generate a corpus, gate it, report the verdict.

Exits non-zero when a corpus is unusable, so this can gate continuous
integration rather than merely informing a human who may not be reading.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from pharos.croissant import croissant_record, to_json
from pharos.export import write_corpus
from pharos.generate import GeneratorConfig, generate
from pharos.manifest import build_manifest
from pharos.models import DEFAULT_KEY, catalog


def _report(manifest) -> None:
    gate = manifest.gate
    print("=" * 68)
    print(f"pharos {manifest.pharos_version}  seed={manifest.config.seed}")
    print(f"corpus            {manifest.n_reports} reports over {manifest.n_events} events")
    print(f"plant share       {manifest.plant_share:.3f}")
    print(f"label cells       {len(manifest.label_histogram)}")
    for cell, count in manifest.label_histogram.items():
        print(f"  {cell:34} {count}")
    print(f"gate split        train {gate.n_train} / test {gate.n_test}")
    print(f"gate held out     {', '.join(gate.held_out_centers)}")
    for probe, auc in gate.per_probe_auc.items():
        print(f"  probe {probe:22} AUC {auc:.4f}")
    print(
        f"surface baseline  {gate.surface_baseline:.4f}  (ceiling {manifest.max_surface_baseline})"
    )
    if gate.null_mean is not None:
        print(
            f"permutation null  {gate.null_mean:.4f} +/- {gate.null_sd:.4f}  "
            f"p95 {gate.null_p95:.4f}  over {gate.null_trials} trials"
        )
        print(f"leak vs null      z = {gate.null_z:+.2f}  significant: {gate.leak_is_significant}")
    print(f"strict band       {gate.band[0]} to {gate.band[1]}  (met: {gate.passed})")
    print(f"VERDICT           {'USABLE' if manifest.usable else 'NOT USABLE'}")
    print()
    print("The surface baseline is what a model reaches while reading nothing, so any")
    print("triage score has to be reported against it. It is expected to exceed chance:")
    print("plants carry the significant facts more often by construction, so the fact")
    print("mix differs and shape carries some information. What matters is that the")
    print("baseline is measured, exceeds the gate's own null, and leaves room above it.")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pharos", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate_cmd = sub.add_parser("gate", help="Generate a corpus and run the shortcut gate")
    gate_cmd.add_argument("--seed", type=int, default=7)
    gate_cmd.add_argument("--events", type=int, default=400)
    gate_cmd.add_argument("--plant-rate", type=float, default=0.3)
    gate_cmd.add_argument("--out", type=Path, help="Write the manifest as JSON here")

    export_cmd = sub.add_parser(
        "export",
        help="Write a corpus, its manifest, and its Croissant metadata to a directory",
    )
    export_cmd.add_argument("--seed", type=int, default=7)
    export_cmd.add_argument("--events", type=int, default=400)
    export_cmd.add_argument("--plant-rate", type=float, default=0.3)
    export_cmd.add_argument("--out", type=Path, default=Path("export"))

    models_cmd = sub.add_parser("models", help="List selectable models and what is installed")
    models_cmd.add_argument("--json", action="store_true", help="Emit the catalog as JSON")

    serve_cmd = sub.add_parser("serve", help="Run the explorer web UI")
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)

    if args.command == "models":
        return _models(as_json=args.json)
    if args.command == "serve":
        from pharos.web import serve

        return serve(host=args.host, port=args.port)

    config = GeneratorConfig(seed=args.seed, n_events=args.events, plant_rate=args.plant_rate)

    if args.command == "export":
        return _export(config, args.out)

    manifest = build_manifest(config)
    _report(manifest)
    if args.out:
        args.out.write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if manifest.usable else 1


def _models(*, as_json: bool) -> int:
    """List the registry, annotated with what the local backend actually has.

    `verified` means the model has answered a Pharos task, not that it looks
    suitable. The distinction is the point: every published Pharos number so far
    came from one model, and a list that blurs tested with untested would hide that.
    """
    rows = catalog()
    if as_json:
        print(json.dumps({"default": DEFAULT_KEY, "models": rows}, indent=2))
        return 0

    print(f"{'KEY':<16} {'TAG':<34} {'SIZE':<7} {'VRAM':<7} {'INSTALLED':<10} VERIFIED")
    for row in rows:
        print(
            f"{row['key']:<16} {row['tag']:<34} {row['parameters']:<7} "
            f"{row['approx_vram_gb']:<7} {'yes' if row['installed'] else 'no':<10} "
            f"{'yes' if row['verified'] else 'candidate'}"
        )
    print()
    print("verified = has answered a Pharos triage task. candidate = never run yet.")
    print(f"default  = {DEFAULT_KEY}")
    print("pull a candidate with:  ollama pull <tag>")
    return 0


def _export(config: GeneratorConfig, out: Path) -> int:
    """Write the corpus, the manifest, and the Croissant record as one artifact.

    Refuses to write an unusable corpus. An export is what gets cited, and a
    citable artifact whose own gate rejected it is worse than no artifact.
    """
    reports = generate(config)
    manifest = build_manifest(config)
    _report(manifest)
    if not manifest.usable:
        print("refusing to export: this corpus did not pass its own gate")
        return 1

    corpus_path = out / "corpus.jsonl"
    size, digest = write_corpus(reports, corpus_path)
    record = croissant_record(
        manifest,
        sha256=digest,
        date_published=datetime.now(UTC).date().isoformat(),
    )
    (out / "croissant.json").write_text(to_json(record), encoding="utf-8")
    (out / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

    print(f"wrote {corpus_path}  ({size} bytes, sha256 {digest[:16]}...)")
    print(f"wrote {out / 'croissant.json'}")
    print(f"wrote {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
