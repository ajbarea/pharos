"""Command line entry: generate a corpus, gate it, report the verdict.

Exits non-zero when a corpus is unusable, so this can gate continuous
integration rather than merely informing a human who may not be reading.
"""

import argparse
import sys
from pathlib import Path

from pharos.generate import GeneratorConfig
from pharos.manifest import build_manifest


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
    print(f"gate AUC          {gate.auc:.4f}  band {gate.band[0]} to {gate.band[1]}")
    print(f"VERDICT           {'USABLE' if manifest.usable else 'NOT USABLE'}")
    if not manifest.usable:
        print()
        print("A corpus that fails the gate carries a surface tell: something about")
        print("report shape predicts plant membership without reading the text. A fleet")
        print("trained on it would federate the tell to every deployment.")
    print("=" * 68)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pharos", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gate_cmd = sub.add_parser("gate", help="Generate a corpus and run the shortcut gate")
    gate_cmd.add_argument("--seed", type=int, default=7)
    gate_cmd.add_argument("--events", type=int, default=400)
    gate_cmd.add_argument("--plant-rate", type=float, default=0.3)
    gate_cmd.add_argument("--out", type=Path, help="Write the manifest as JSON here")
    args = parser.parse_args(argv)

    config = GeneratorConfig(seed=args.seed, n_events=args.events, plant_rate=args.plant_rate)
    manifest = build_manifest(config)
    _report(manifest)
    if args.out:
        args.out.write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if manifest.usable else 1


if __name__ == "__main__":
    sys.exit(main())
