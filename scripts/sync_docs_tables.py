#!/usr/bin/env python3
"""Regenerate the tables in `docs/` that restate a committed artifact.

Most numbers on the docs site are prose written once beside a measurement, and prose
is allowed to be prose. A few are different: they restate an artifact row for row, and
those rot silently the moment the artifact moves. The claim table in `findings.md` did
exactly that twice in one day. It was updated by hand when the power analysis learned
to price a claim against a known constant, then went stale again hours later when two
of its claims were rerun at a larger size, and in between it stated a verdict that
contradicted another section of the same page.

So the fix is the one the paper's tables already use: generate the block, and give CI
a way to fail when the file disagrees with the artifact behind it.

Blocks are delimited by HTML comments, which render as nothing:

    <!-- BEGIN GENERATED: power-claims -->
    ...
    <!-- END GENERATED: power-claims -->

Anything outside a marked block is left alone, so the surrounding argument stays
hand-written where it belongs.

    uv run python scripts/sync_docs_tables.py           # rewrite the blocks
    uv run python scripts/sync_docs_tables.py --check   # fail if any is stale
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"


def _fail(message: str) -> NoReturn:
    """Always raises. Typed `NoReturn` so callers narrow correctly after it."""
    print(f"sync_docs_tables: {message}", file=sys.stderr)
    raise SystemExit(2)


def power_claims() -> str:
    """The claim table: every headline claim against the size it was measured at.

    Verdict wording is derived rather than typed, because the distinction between
    "unresolved at a size we could buy" and "unresolved at any size" is the one a
    reader acts on, and it was previously maintained by hand.
    """
    path = RESULTS / "power.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make power` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    claims = payload.get("claims") or []
    if not claims:
        _fail("power.json carries no claims; refusing to emit an empty table")

    lines = [
        "| Finding | n | Gap it rests on | vs | Verdict | Claim |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for claim in claims:
        against = "constant" if claim["against_constant"] else "condition"
        if claim["resolved"]:
            verdict = "**resolved**"
        elif claim["n_needed"]:
            verdict = f"unresolved (needs n≥{claim['n_needed']})"
        else:
            verdict = "unresolved (needs n>2000)"
        lines.append(
            f"| {claim['finding']} | {claim['n']} | {claim['effect']:.3f} | "
            f"{against} | {verdict} | {claim['description']} |"
        )
    resolved = sum(1 for c in claims if c["resolved"])
    lines.append("")
    lines.append(f"**{resolved} of {len(claims)}** resolve at the size they were run.")
    return "\n".join(lines)


#: Artifacts whose *script* computes validity but whose committed artifact predates
#: that change. A rerun, not a code change, and hours of GPU rather than minutes of
#: editing -- which is why they are listed apart from the exemptions below.
AWAITING_RERUN = {
    "learnability": "measure_rule_learnability.py now records it per shot count",
    "label_fidelity": "measure_label_fidelity.py now records it over the scored turns",
    "decode_stability": "measure_decode_stability.py now records it over the repeated passes",
    "adapter_learnability": "train_adapter.py now records it per evaluation pass",
    **{
        f"review_adapter-{t}{x}": "train_adapter.py now records it per evaluation pass"
        for t in ("by-the-book", "inattentive", "two-of-three", "any-one")
        for x in ("", "-xseed101")
    },
}

#: Artifacts with no sampling-validity question to answer, and why. An exemption is a
#: claim, so each carries its reason and appears on the page rather than being filtered
#: out silently.
NO_SAMPLING_QUESTION = {
    "power": "prices hypothetical evaluation sizes; simulates outcomes rather than measuring any",
    "federation_eligibility": "deterministic over the label lattice; nothing is sampled",
    "external_gate_validation": "carries its own permutation-null statistics per corpus",
    "triage_lift": "superseded by the per-model triage_lift-* artifacts, which are assessed",
    "fleet_sensitivity": "a sweep over a nuisance parameter; reports invariants, samples nothing",
    "teacher_fleet": "aggregates assessed adapter artifacts; adds no measurement of its own",
    "adapter_replication": (
        "compares assessed adapter artifacts against their own replicates; the question is "
        "whether two runs agree, which no sampling flag answers"
    ),
    "gate_determinism": (
        "reports the gate's surface baseline at full precision on one machine; the result "
        "is the comparison against another machine, not the number"
    ),
    "gate_determinism-cluster": "the second machine of that comparison",
    "fl_benchmarks": (
        "sizes the problem rather than settling it, is quoted nowhere in the manuscript, "
        "and reports a bootstrap interval per condition instead of a flag"
    ),
}


def measurement_health() -> str:
    """Every artifact's validity assessment, published rather than left in a field.

    `pharos.validity` marks a measurement unquotable when it trips a condition that
    makes a score misleading: too few samples, a class floor the score does not clear,
    degenerate predictions, unparsed answers. The flag was computed, warned about on
    the console, written into some artifacts, and then read by nothing. A guard that
    refuses to let prose quote a flagged number would be wrong, because quoting one as
    evidence of *failure* is exactly what the flag licenses and is what finding 3b
    correctly does. So the enforcement is publication instead: the caveats appear on
    the page, generated from the artifacts, and cannot drift away from them.

    Artifacts with no validity block are listed too. An unassessed measurement is a
    different problem from a flagged one and the two should not look alike.
    """
    rows: list[tuple[str, str, str]] = []
    unassessed: list[str] = []
    # `AWAITING_RERUN` is hand-maintained and goes stale in one direction: a rerun
    # lands and the entry saying it has not is left behind, which once listed
    # `decode_stability` as both quotable at n=30 and awaiting the rerun that produced
    # the 30. Derive the pending list from the artifacts; the registry keeps only the
    # reasons, which cannot be derived.
    now_assessed: set[str] = set()
    for path in sorted(RESULTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validity = payload.get("validity")
        which_pass = ""
        if not isinstance(validity, dict):
            # Adapter artifacts carry the assessment per evaluation pass rather than
            # at the top level, so look one level down before calling it unassessed.
            nested = [
                (name, v["validity"])
                for name, v in payload.items()
                if isinstance(v, dict) and isinstance(v.get("validity"), dict)
            ]
            # And one level down inside a *list*, which is where a per-condition
            # measurement puts it: `learnability` assesses each shot count separately
            # and has no single top-level verdict. Only the dict case was handled, so
            # every such artifact was reported as carrying no assessment at all while
            # its rows each carried one. `learnability` was masked by an
            # `AWAITING_RERUN` entry that has since been satisfied; its replication had
            # no such cover and showed up as an unassessed gap on the page.
            for name, value in payload.items():
                if not isinstance(value, list):
                    continue
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        continue
                    row_validity = item.get("validity")
                    if isinstance(row_validity, dict):
                        nested.append((f"{name}[{index}]", row_validity))
            if not nested:
                if path.stem not in NO_SAMPLING_QUESTION and path.stem not in AWAITING_RERUN:
                    unassessed.append(path.stem)
                continue
            # The worst pass decides the flag, but it has to say WHICH pass. Collapsing
            # them anonymously reported `review_adapter-by-the-book` as unquotable on
            # the strength of its untrained baseline, while the adapter that artifact
            # exists to report scored 0.995 and was perfectly quotable. A flag that
            # names the wrong thing is the same defect as a flag on the wrong number.
            name, validity = min(nested, key=lambda item: bool(item[1].get("quotable", True)))
            which_pass = f"**{name}**: "
        concerns = validity.get("concerns") or []
        mark = "yes" if validity.get("quotable") else "**no**"
        note = f"{which_pass}{'; '.join(str(c) for c in concerns)}" if concerns else "-"
        rows.append((path.stem, f"{validity.get('n', '?')}", f"{mark} | {note}"))
        now_assessed.add(path.stem)

    # The other seven builders refuse an absent artifact; this one globbed and would
    # have published a header, a separator and "**0 of 0** assessed artifacts are
    # flagged" over an empty `results/`, which `--check` would then call current. It
    # is the same defect as a guard that passes because it matched nothing, and it was
    # found by asserting the refusal these builders all document.
    if not rows:
        _fail("no assessed artifact in results/; refusing to emit an empty health table")

    lines = [
        "| Artifact | n | Quotable | Why not |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| `{name}` | {n} | {rest} |" for name, n, rest in rows]
    flagged = sum(1 for _, _, rest in rows if rest.startswith("**no**"))
    lines.append("")
    lines.append(
        f"**{flagged} of {len(rows)}** assessed artifacts are flagged. A flagged number "
        "may still be quoted as evidence that something *failed*, which is what the "
        "flag asserts; it may not be quoted as evidence of capability."
    )
    if unassessed:
        lines.append("")
        lines.append(
            "**Carrying no validity assessment, which is a gap rather than a pass:** "
            + ", ".join(f"`{name}`" for name in sorted(unassessed))
            + "."
        )
    present = {p.stem for p in RESULTS.glob("*.json")}
    pending = sorted(
        (n, w) for n, w in AWAITING_RERUN.items() if n in present and n not in now_assessed
    )
    if pending:
        lines.append("")
        lines.append(
            f"Assessed by their script but not yet in the committed artifact "
            f"({len(pending)} of these), which needs a rerun rather than an edit:"
        )
        lines.append("")
        for name, why in pending:
            lines.append(f"- `{name}` -- {why}")
    exempt = sorted(NO_SAMPLING_QUESTION.items())
    if exempt:
        lines.append("")
        lines.append("Exempt, because there is no sampling question to answer:")
        lines.append("")
        for name, why in exempt:
            missing = "" if name in present else " *(artifact absent)*"
            lines.append(f"- `{name}` -- {why}{missing}")
    return "\n".join(lines)


def teacher_fleet() -> str:
    """The 24-teacher grid behind finding 10, plus the split that changes sign.

    Twenty-four rows of six numbers is the largest hand-maintained table this page could
    have carried, and it was one: pasted from a shell one-liner while the section was
    being written. Every rerun of the grid would have silently invalidated all of it.

    The ceiling column is the reason the table needs more than the obvious four columns.
    Fidelity falls from 1.000 to 0.473 as the slip rate rises, which reads as
    inheritance decaying and is not: a teacher slipping at rate `s` disagrees with its
    own rule that often, so `1 - s` is the best score any model can post against its
    labels. The dagger marks rows the validity check refuses, and it is load-bearing --
    the row with the table's highest fidelity is one of them.
    """
    path = RESULTS / "teacher_fleet.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make teacher-fleet` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        _fail("teacher_fleet.json carries no rows; refusing to emit an empty table")
    summary = payload["summary"]

    lines = [
        "| Teacher | Targets vs world | Adapter vs **world** | "
        "Adapter vs **teacher** | Ceiling `1-s` | Inherited |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        dagger = "" if row["quotable"] else " †"
        gap = row["adapter_agrees_with_world"] - row["teacher_agrees_with_world"]
        lines.append(
            f"| `{row['reviewer']}`{dagger} | {row['teacher_agrees_with_world']:.3f} | "
            f"{row['adapter_agrees_with_world']:.3f} | "
            f"{row['adapter_agrees_with_teacher']:.3f} | "
            f"{1 - row['slip_rate']:.3f} | {gap:+.3f} |"
        )

    refused = len(rows) - summary["quotable"]
    lines += [
        "",
        f"† marks the {refused} adapters the validity check refuses, for accuracy "
        "beneath the majority floor or for recall bought with more false positives "
        "than true ones.",
        "",
        "| Threshold | n | Median inherited | Beat their teacher | Quotable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for block in summary["by_threshold"]:
        lines.append(
            f"| {block['threshold']} | {block['n']} | {block['median_gap']:+.4f} | "
            f"{block['adapters_better_than_their_teacher']} | {block['quotable']} |"
        )
    return "\n".join(lines)


def secure_readership() -> str:
    """What the aggregate discloses, one row per source-join it was read off.

    Generated rather than typed because the interesting column is the comparison
    between the headcount and the adversary's prior, and a corpus change moves both.
    An earlier hand-written version of this table showed thirteen rows for fourteen
    joins, because it was keyed on the compartment set and two joins carrying the same
    compartments at different sensitivities collapsed into one.
    """
    path = RESULTS / "secure_reliability.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make secure` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    readership = payload.get("readership") or {}
    headcounts = readership.get("headcounts") or {}
    if not headcounts:
        _fail("secure_reliability.json carries no readership; refusing to emit an empty table")

    truth = readership["truth"]
    prior = readership["prior_expectation"]
    lines = [
        "| Sensitivity \\| compartments | Read off the aggregate | True | Adversary's prior |",
        "| --- | --- | --- | --- |",
    ]
    for key in sorted(headcounts, key=lambda k: (-headcounts[k], k)):
        observed = headcounts[key]
        exact = "**" if observed == truth[key] else ""
        lines.append(
            f"| {key.replace('|', '\\|')} | {exact}{observed}{exact} | "
            f"{truth[key]} | {prior[key]:.2f} |"
        )
    lines += [
        "",
        f"{readership['exact_headcounts']} of {readership['labels_probed']} joins yield an "
        "exactly correct headcount. Bold marks the exact ones; a row that stopped being "
        "exact would lose its bold here rather than in a sentence nobody reran.",
    ]
    return "\n".join(lines)


def _authority_payload() -> dict[str, Any]:
    path = RESULTS / "authority_anchors.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make authority` first")
    return json.loads(path.read_text(encoding="utf-8"))


def authority_grid() -> str:
    """Finding 19's agreement grid, one row per composition and one column per budget.

    This was typed by hand, at eight of the fifteen budgets the sweep actually runs,
    and it was wrong: making the anchor draw nested moved every cell and the table did
    not move with them. A grid transcribed from a run is the most stale-prone shape in
    the document -- nothing about editing prose forces a reader to check seventy-five
    numbers -- so it is read off the artifact instead.
    """
    payload = _authority_payload()
    grid = payload.get("grid") or []
    if not grid:
        _fail("authority_anchors.json carries no grid; refusing to emit an empty table")

    counts = payload["anchor_counts"]
    fleet = payload["fleet"]
    cells = {(row["n_wrong"], row["anchors"]): row for row in grid}
    compositions = sorted({row["n_wrong"] for row in grid})

    lines = [
        f"| Wrong of {fleet} | " + " | ".join(str(c) for c in counts) + " |",
        "| --- " * (len(counts) + 1) + "|",
    ]
    for n_wrong in compositions:
        rendered = []
        repaired_yet = False
        for count in counts:
            row = cells.get((n_wrong, count))
            if row is None:
                rendered.append("—")
                continue
            # Bold marks the crossing itself, not every cell above it: the finding is
            # where the threshold falls, and bolding the whole tail hides it.
            first = row["repaired"] and not repaired_yet
            repaired_yet = repaired_yet or row["repaired"]
            rendered.append(f"**{row['agreement']:.3f}**" if first else f"{row['agreement']:.3f}")
        lines.append(f"| {n_wrong} | " + " | ".join(rendered) + " |")

    lines += [
        "",
        f"Agreement over unanchored tasks only, at anchor seed {payload['grid_anchor_seed']}. "
        f"Bold marks each composition's first budget reaching "
        f"{payload['repaired_threshold']:.2f}; that crossing moves with the draw, which "
        "is what the table below reports.",
    ]
    return "\n".join(lines)


def authority_price() -> str:
    """The audited-items threshold per fleet composition, from finding 19's artifact.

    Five numbers that are the entire practical content of the finding, and exactly the
    shape that goes stale: they move whenever the corpus, the anchor draw or the
    repaired threshold moves, and none of those changes touches this page.

    Reported as a median over draws with its range, because the single-draw version of
    this table was not reproducible: re-deriving the anchor draw moved the five-wrong
    price from 5 items to 30 and turned the nine-wrong price from 180 into never. The
    spread column is the finding those numbers were hiding.
    """
    payload = _authority_payload()
    spreads = payload.get("anchors_needed_spread") or {}
    if not spreads:
        _fail("authority_anchors.json carries no thresholds; refusing to emit an empty table")

    total = payload["events"]
    fleet = payload["fleet"]
    swept = max(payload["anchor_counts"])
    draws = len(payload["anchor_seeds"])
    lines = [
        f"| Wrong of {fleet} | Audited items needed (median) | Share of the round | "
        f"Range over {draws} draws | Draws that reached it |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(spreads, key=int):
        spread = spreads[key]
        reached = f"{spread['reached']} of {spread['seeds']}"
        span = (
            (
                f"{spread['lowest']}–{spread['highest']}"
                if spread["lowest"] != spread["highest"]
                else str(spread["lowest"])
            )
            if spread["reached"]
            else "—"
        )
        median = spread["median"]
        if median is None:
            lines.append(f"| {key} | not reached within {swept} | — | {span} | {reached} |")
            continue
        lines.append(f"| {key} | {median} | {median / total:.1%} | {span} | {reached} |")
    lines += [
        "",
        f"Threshold for 'repaired' is agreement ≥ {payload['repaired_threshold']:.2f} on "
        f"unanchored tasks, over a corpus of {total}. The median is taken over {draws} "
        "anchor draws with draws that never repair ordered last, so a composition most "
        "draws fail to repair reports no number rather than the sweep's upper edge.",
    ]
    return "\n".join(lines)


def audit_policy() -> str:
    """Finding 20: budget to repair, by selection policy.

    The oracle column is rendered apart from the deployable ones with an explicit
    marker, because a table that lists it inline invites a reader to quote a number no
    deployment can have. Finding 12 had to learn that distinction the hard way about a
    different oracle.
    """
    path = RESULTS / "audit_policy.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make audit` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds") or {}
    if not thresholds:
        _fail("audit_policy.json carries no thresholds; refusing to emit an empty table")

    deployable = list(payload["deployable"])
    order = [*deployable, "oracle"]
    compositions = sorted({int(n) for d in thresholds.values() for n in d})
    best = payload["best_deployable"]

    header = " | ".join(f"`{p}`" + (" †" if p == "oracle" else "") for p in order)
    lines = [
        f"| Wrong of {payload['fleet']} | {header} |",
        "| --- |" + " --- |" * len(order),
    ]
    # The baseline's cell carries its spread inline. Every other policy in this table is
    # deterministic given the aggregate, so a bare number there is exact; a bare number
    # in the `uniform` column would be one draw of a quantity that varies by an order of
    # magnitude, and the margin a policy is credited with is measured against it.
    spread = payload.get("uniform_spread") or {}
    for n in compositions:
        cells = []
        for name in order:
            value = thresholds.get(name, {}).get(str(n))
            if value is None:
                cells.append("not reached")
            elif name == best:
                cells.append(f"**{value}**")
            else:
                cells.append(str(value))
            if name == "uniform" and str(n) in spread:
                draw = spread[str(n)]
                if draw["reached"]:
                    cells[-1] += f" ({draw['lowest']}–{draw['highest']})"
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    pool = payload["auditable_pool"]
    draws = len(payload.get("uniform_seeds") or [])
    lines += [
        "",
        f"Items an authority must rule on to repair the estimate, out of **{pool} "
        f"auditable** tasks ({payload['events']} in the corpus). Lower is better; bold "
        f"is the winning deployable policy. † `oracle` reads ground truth and is a "
        f"bound rather than a method.",
    ]
    if draws:
        lines += [
            "",
            f"`uniform` is a draw, not a rule: its cell is the median over {draws} draws "
            "with the full range in brackets. The other policies select from the "
            "aggregate and have no draw to vary.",
        ]
    return "\n".join(lines)


#: Which reviewer each finding-10 adapter was trained by, and what that reviewer's
#: standard is. Prose about the design rather than a measured quantity, so it stays
#: here; the agreement numbers beside it are all read from the artifacts.
TEACHERS = {
    "by-the-book": "3 of 3",
    "inattentive": "3 of 3, slips 15%",
    "two-of-three": "2 of 3",
    "any-one": "1 of 3",
}


def teacher_inheritance(suffix: str = "") -> str:
    """Finding 10: what an adapter learns when its teacher is systematically wrong.

    Generated because every row of the hand-typed version had drifted, and one of the
    drifts carried the finding's headline. The page said the two systematically-wrong
    teachers "hand over their error rate to within 0.002 and 0.001". The artifacts say
    -0.0076 and +0.0202 -- an order of magnitude larger, and the claim that inheritance
    is an *identity* rests on exactly that gap being small.

    The conclusion survives at the true numbers, because two points of slack across a
    600-task evaluation is still inheritance rather than dilution. But "to within 0.002"
    was a stronger sentence than anything measured supported, and it was published.
    """
    rows = []
    for teacher, standard in TEACHERS.items():
        path = RESULTS / f"review_adapter-{teacher}{suffix}.json"
        if not path.exists():
            _fail(f"{path.relative_to(ROOT)} is missing; finding 10 needs every teacher")
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = payload["teacher"]["train_target_agreement"]
        world = payload["adapter"]["accuracy"]
        against_teacher = payload["adapter_vs_teacher"]["accuracy"]
        inherited = world - targets
        rows.append(
            f"| {teacher} | {standard} | {targets:.4f} | {world:.4f} | "
            f"{against_teacher:.4f} | {inherited:+.4f} |"
        )

    first = json.loads(
        (RESULTS / f"review_adapter-by-the-book{suffix}.json").read_text(encoding="utf-8")
    )
    data = first["data"]
    return "\n".join(
        [
            "| Teacher | Standard | Targets matching world | Adapter vs **world** | "
            "Adapter vs **teacher** | Inherited error |",
            "| --- | --- | --- | --- | --- | --- |",
            *rows,
            "",
            f"Accuracy on {data['n_eval']} held-out tasks, {data['n_train']} training tuples "
            f"per adapter. The last column is the adapter's agreement with the world minus "
            f"its teacher's: near zero means the adapter reproduced its teacher's error rate "
            f"rather than diluting it.",
        ]
    )


def teacher_inheritance_xseed() -> str:
    """The same four teachers, evaluated on a corpus sharing no events."""
    return teacher_inheritance(suffix="-xseed101")


def linkage_controls() -> str:
    """Finding 11: what each mitigation costs and what it leaves recoverable.

    Generated because every cell of the hand-typed version had drifted. Contributions
    were published as 7,053 and 1,000 where the artifact says 6606 and 400, and the
    k=25, k=50, rarity and subsample rows were each wrong in two or three columns. The
    argument -- that nothing closes this channel without taking most of the data with
    it -- survived every one of those errors, which is exactly why nobody re-read the
    numbers under it.
    """
    path = RESULTS / "fleet_linkage.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make linkage` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    controls = payload.get("controls") or {}
    if not controls:
        _fail("fleet_linkage.json carries no controls; refusing to emit an empty table")

    labels = {
        "none": "no mitigation",
        "k_anonymity_k10": "k-anonymity, k=10",
        "k_anonymity_k25": "k-anonymity, k=25",
        "k_anonymity_k50": "k-anonymity, k=50",
        "k_anonymity_k100": "k-anonymity, k=100",
        "rarity_suppression_keep0.75": "suppress rarest 25%",
        "rarity_suppression_keep0.5": "suppress rarest 50%",
        "rarity_suppression_keep0.25": "suppress rarest 75%",
        "subsample_p0.5": "subsample p=0.5",
        "subsample_p0.2": "subsample p=0.2",
        "subsample_p0.05": "subsample p=0.05",
        "pooled": "pool every analyst",
    }
    # A control the artifact grows and this table cannot name would otherwise be
    # dropped silently, which is how a mitigation goes unpublished.
    unnamed = set(controls) - set(labels)
    if unnamed:
        _fail(f"fleet_linkage.json has controls this table cannot name: {sorted(unnamed)}")

    lines = [
        "| Mitigation | Recovered | Mean anonymity set | Volume kept | Analysts silenced |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in labels.items():
        row = controls.get(key)
        if row is None:
            continue
        recovered = row["recovery"]
        cell = f"**{recovered:.3f}**" if recovered == 0.0 else f"{recovered:.3f}"
        lines.append(
            f"| {label} | {cell} | {row['mean_anonymity_set']:.2f} | "
            f"{row['retained_volume']:.1%} | {row['silenced_analysts']} |"
        )

    prior = payload["prior"]
    lines += [
        "",
        f"Bold marks a mitigation that drives recovery to zero. The guessing prior is "
        f"{prior:.3f}, so a row at or below it has closed *this* attack; read the volume "
        f"and silenced columns for what that cost. Contributions under the two rulings "
        f"are {payload['drop_compartments']['contributions']:,} when compartments are "
        f"shed and {payload['keep']['contributions']:,} when they are kept.",
    ]
    return "\n".join(lines)


def privacy_budget() -> str:
    """Finding 15: what participation noise buys, and the budget it spends doing it."""
    path = RESULTS / "privacy_budget.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make budget` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in (payload.get("rows") or []) if (r.get("budget") or {}).get("fabricate")]
    if not rows:
        _fail("privacy_budget.json carries no fabricating rows; refusing an empty table")

    lines = [
        "| Keep | Fabricate | Recovered | Label noise | epsilon per indicator | epsilon composed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        budget = row["budget"]
        lines.append(
            f"| {budget['keep']:.1f} | {budget['fabricate']:.1f} | {row['recovery']:.3f} | "
            f"{row['label_noise']:.4f} | {budget['epsilon_per_indicator']:.2f} | "
            f"{budget['epsilon_effective']:.1f} |"
        )

    baseline = payload["baseline_recovery"]
    lines += [
        "",
        f"Participation noise against a baseline recovery of {baseline:.3f} over "
        f"{payload['fleet']} analysts. `epsilon composed` is the better of basic and "
        f"advanced composition across {rows[0]['budget']['indicators']} indicators at "
        f"delta = {rows[0]['budget']['delta']:g}. Read the label-noise column beside it: "
        f"the setting that suppresses the attack corrupts most of the labels, and an "
        f"epsilon in the hundreds is a budget in name rather than a guarantee.",
    ]
    return "\n".join(lines)


def label_fidelity() -> str:
    """Finding 1: what leave-one-out attribution recovers, and what it mislabels.

    Generated because this headline has now drifted twice. It was first written at
    n=8, corrected to n=24 on 2026-07-30, and was still quoting the n=24 figures on
    2026-08-06 after the corpus had been re-measured at n=40 -- with the corrected
    numbers appearing in the prose two paragraphs below, so the page stated both.
    A sentence that has been wrong twice for the same reason should stop being a
    sentence.
    """
    path = RESULTS / "label_fidelity.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make results` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcomes = payload.get("outcomes") or {}
    if not outcomes:
        _fail("label_fidelity.json carries no outcomes; refusing to emit an empty summary")

    exact = outcomes.get("exact", 0)
    leak = outcomes.get("leak", 0)
    creep = outcomes.get("creep", 0)
    turns = exact + leak + creep
    wrong = leak + creep
    calls = sum(row.get("calls", 0) for row in payload.get("rows") or [])

    return "\n".join(
        [
            "| Turns | Source recall | Source precision | Exact label | Leak | Creep |",
            "| --- | --- | --- | --- | --- | --- |",
            f"| {turns} | **{payload['source_recall_mean']:.3f}** | "
            f"**{payload['source_precision_mean']:.3f}** | {exact} | {leak} | {creep} |",
            "",
            f"Eight-source summarization turns under exact leave-one-out, "
            f"{calls} model calls. A wrong governed label on **{wrong} of {turns} turns "
            f"({wrong / turns:.0%})**: {leak} leak and {creep} creep. Recall is what the "
            f"ablation misses; precision is what it wrongly blames.",
        ]
    )


def difficulty_confound() -> str:
    """Finding 17: estimated difficulty by true overlap, per fleet composition.

    Hand-typed until 2026-08-05, and every cell of it had drifted. The item counts
    were wrong, every difficulty value was wrong, and the prose read a peak off the
    wrong column: it said the random-slip row "peaks at overlap 0, furthest from the
    boundary", where the artifact peaks at overlap 1. That last one is why this is
    generated now rather than corrected in place. A wrong digit is a wrong digit; a
    wrong *peak* is the argument the finding makes.
    """
    path = RESULTS / "difficulty_confound.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make difficulty` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    if not rows:
        _fail("difficulty_confound.json carries no rows; refusing to emit an empty table")

    overlaps: list[str] = sorted((str(k) for k in rows[0]["difficulty_by_overlap"]), key=int)
    boundary = payload.get("near_boundary_overlap", "2")
    converged = set(payload.get("converged_rows") or [])

    def head(o: str) -> str:
        return f"**ovl={o}**" if o == boundary else f"ovl={o}"

    lines = [
        "| Fleet | " + " | ".join(head(o) for o in overlaps) + " | Spread | Converged |",
        "| --- |" + " --- |" * (len(overlaps) + 2),
    ]
    for row in rows:
        by_overlap = row["difficulty_by_overlap"]
        peak = max(by_overlap, key=lambda o: by_overlap[o])
        cells = []
        for o in overlaps:
            value = f"{by_overlap[o]:.3f}"
            cells.append(
                f"**{value}**" if o == peak and len(set(by_overlap.values())) > 1 else value
            )
        iters = row.get("iterations")
        settled = "yes" if row["composition"] in converged else "**no**"
        lines.append(
            f"| {row['composition']} | "
            + " | ".join(cells)
            + f" | {row['difficulty_spread']:.3f} | {settled}, {iters} iter |"
        )

    items = payload.get("items_by_overlap") or {}
    lines += [
        "",
        "| Signature facts present | " + " | ".join(head(o) for o in overlaps) + " |",
        "| --- |" + " --- |" * len(overlaps),
        "| Items | " + " | ".join(str(items.get(o, "--")) for o in overlaps) + " |",
        "",
        "Estimated item difficulty by true signature overlap, under fleets differing only "
        "in composition. **Bold** in each row marks that row's own peak, read from the "
        "artifact rather than asserted in prose.",
    ]
    return "\n".join(lines)


def blind_spot() -> str:
    """Finding 21: where the audit policy stops working.

    Both halves in one block, because either alone misleads. The thresholds say the
    policy failed; the hit rates say why, and that the oracle did not.
    """
    path = RESULTS / "blind_spot.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make blindspot` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds") or {}
    hits = payload.get("audit_hit_rate") or {}
    if not thresholds or not hits:
        _fail("blind_spot.json is incomplete; refusing to emit a partial table")

    order = [p for p in thresholds if p != "oracle"] + ["oracle"]
    shares = sorted({int(n) for d in thresholds.values() for n in d})
    swept = max(payload["budgets"])

    lines = [
        "| Blind of {} | ".format(payload["fleet"]) + " | ".join(f"`{p}`" for p in order) + " |",
        "| --- |" + " --- |" * len(order),
    ]
    for n in shares:
        cells = []
        for name in order:
            value = thresholds.get(name, {}).get(str(n))
            cells.append("--" if value is None else str(value))
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Audited items needed to repair; `--` is not reached within {swept}. "
        "Below: the share of a 20-item audit landing on a genuinely corrupted task.",
        "",
        f"| Blind of {payload['fleet']} | " + " | ".join(f"`{p}`" for p in order) + " |",
        "| --- |" + " --- |" * len(order),
    ]
    for n in shares:
        row = hits.get(str(n), {})
        lines.append(f"| {n} | " + " | ".join(f"{row.get(name, 0.0):.2f}" for name in order) + " |")
    return "\n".join(lines)


def channel_bias() -> str:
    """Finding 22: detection z per channel, across the blind-spot sweep."""
    path = RESULTS / "channel_bias.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make channel-bias` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    sweep = payload.get("sweep") or []
    if not sweep:
        _fail("channel_bias.json carries no sweep; refusing to emit an empty table")

    channels = [d["channel"] for d in sweep[0]["detections"]]
    blind = payload["blind_channel"]
    levels = payload.get("noise_levels") or [0.0]

    def score(hit: dict[str, Any] | None) -> str:
        """The p-value, with the effect beside it where the channel is detected.

        A permutation p-value has no undefined case to print around: a null with no
        spread returns 1.0, which is a real answer.
        """
        if hit is None:
            return "--"
        p = hit["p_value"]
        shown = f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"
        return f"**{shown}** ({hit['delta']:+.3f})" if hit["detected"] else shown

    lines: list[str] = []
    controls = {entry["slip_rate"]: entry["detections"] for entry in payload["threshold_control"]}

    for slip in levels:
        rows = [r for r in sweep if r["slip_rate"] == slip]
        if not rows:
            _fail(f"channel_bias.json has no sweep rows at slip {slip}; refusing a partial table")
        lines += [
            f"**Verdict noise {slip:.2f}**",
            "",
            "| Blind of {} | ".format(payload["fleet"])
            + " | ".join(f"`{c}`" + (" ‡" if c == blind else "") for c in channels)
            + " |",
            "| --- |" + " --- |" * len(channels),
        ]
        for row in rows:
            cells = [
                score(next((d for d in row["detections"] if d["channel"] == channel), None))
                for channel in channels
            ]
            lines.append(f"| {row['n_blind']} | " + " | ".join(cells) + " |")
        lines.append(
            "| threshold control | "
            + " | ".join(
                score(next((d for d in controls.get(slip, []) if d["channel"] == c), None))
                for c in channels
            )
            + " |"
        )
        lines.append("")

    lines += [
        f"One-sided permutation p-values against a within-stratum null over "
        f"{payload['permutations']} permutations, computed as (b+1)/(m+1) so the floor "
        f"is {1 / (payload['permutations'] + 1):.1e} and never zero. Bold is a detection "
        f"at p ≤ {payload['alpha']}, with the stratified gap beside it. ‡ is the channel "
        f"the fleet actually discounts.",
        "",
        "Read the **gap** for extent and the p-value for detection. The gap is linear "
        "in the blind share; the p-value saturates at its floor once an effect is "
        "comfortably significant and cannot distinguish shares above that point. "
        "p = 1.0000 on a noiseless fleet is not a missing value: with every analyst "
        "deterministic, each task's rate is fixed by its evidence stratum, every "
        "permutation returns the observed gap, and b = m exactly.",
    ]
    return "\n".join(lines)


def governance_sensitivity() -> str:
    """Finding 24: where each governance finding's crossing lands as the fleet grows."""
    path = RESULTS / "governance_sensitivity.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make governance-sensitivity` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    cliff = payload.get("cliff") or []
    if not cliff:
        _fail("governance_sensitivity.json carries no cliff scan; refusing an empty table")

    bracket = payload["cliff_bracket"]
    lines = [
        "| Fleet | Bare majority | Breaking share (median) | Breaking share (range) | "
        "Crossing at the majority | Survives a bare majority |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in cliff:
        span = row["breaking_share_range"]
        rng = f"{span[0]:.3f} – {span[1]:.3f}" if span else "--"
        median = f"{row['breaking_share_median']:.3f}" if row["breaking_share_median"] else "--"
        at_majority = f"{row['draws_where_the_cliff_is_at_the_majority']}/{row['draws']}"
        survives = f"{row['draws_surviving_a_bare_majority']}/{row['draws']}"
        lines.append(
            f"| {row['fleet']} | {row['majority']} | {median} | {rng} "
            f"| {at_majority} | {survives} |"
        )

    span = bracket["breaking_share_range"]
    depths = bracket["post_cliff_agreement"]
    draws = len(payload.get("draws") or cliff[0]["draw_seeds"])
    lines += [
        "",
        f"Every composition from 1 to the fleet size, one EM fit each, over {draws} corpus "
        f"draws, with recovery at agreement ≥ {payload['repaired_threshold']}. The last two "
        "columns are counts of draws, not verdicts, and that is the finding: the crossing "
        f"is a distribution spanning **{span[0]:.3f} – {span[1]:.3f}** rather than a "
        "constant, and it becomes *less* predictable as the fleet grows. At five and nine "
        "analysts every draw breaks at the same composition; at fifteen and twenty-five the "
        "same fleet survives a bare majority on some corpora and not others.",
        "",
        f"Agreement past the crossing takes {len(depths)} distinct values across draws "
        f"({min(depths):.4f} – {max(depths):.4f}). Within a single draw it is constant at "
        "every composition above the crossing, so the failure still has no gradient — a "
        "fleet is identified or it is not — but the level it falls to is a property of the "
        "corpus.",
    ]
    return "\n".join(lines)


def estimator_initialization() -> str:
    """Finding 25: whether a different start escapes the cliff, per composition."""
    path = RESULTS / "estimator_initialization.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make estimator-initialization` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    priced = [row for row in payload.get("rows", []) if row["draws_broken"]]
    if not priced:
        _fail("estimator_initialization.json prices no broken composition; refusing an empty table")

    lines = [
        "| Fleet | Wrong | Draws broken | Draws with an escape | Restarts recovering | "
        "Median log-likelihood gap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in priced:
        marker = " *(crossing)*" if row["is_majority"] else ""
        lines.append(
            f"| {row['fleet']} | {row['n_wrong']}{marker} "
            f"| {row['draws_broken']}/{row['draws']} "
            f"| {row['draws_with_an_escape']} "
            f"| {row['restart_recovery_rate']:.3f} "
            f"| {row['likelihood_gap_median']:+.1f} |"
        )

    lines += [
        "",
        f"Every composition the published start gets wrong, over {len(payload['draws'])} corpus "
        f"draws and {payload['restarts']} random restarts each, plus an uninformative start, an "
        "adversarial one, and the ground truth. An *escape* is a start that both fits strictly "
        "better and recovers the truth.",
        "",
        "The gap is signed and the sign is the finding: it is the truth's log-likelihood minus "
        "the published answer's, so **negative means the wrong answer is the better fit**. It is "
        "negative everywhere past the crossing and grows with the share, which is why no "
        "likelihood-guided initialisation helps there. Exactly zero means the truth is not a "
        "fixed point at all — seeded there, EM leaves.",
    ]
    return "\n".join(lines)


def corpus_sensitivity() -> str:
    """Finding 26: whether findings 20-23 survive the corpus draw, claim by claim."""
    path = RESULTS / "corpus_sensitivity.json"
    if not path.exists():
        _fail(f"{path.relative_to(ROOT)} is missing; run `make corpus-sensitivity` first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    audit = payload.get("audit") or []
    blind = payload.get("blind") or []
    if not audit or not blind:
        _fail("corpus_sensitivity.json carries no swept draws; refusing an empty table")

    lines = [
        "| Claim | Finding | Holds in | Of draws |",
        "| --- | --- | --- | --- |",
    ]
    by_comp = {e["n_wrong"]: e for e in payload["by_composition"]}
    for n_wrong in sorted(by_comp):
        entry = by_comp[n_wrong]
        lines.append(
            f"| `margin` ties the oracle bound at {n_wrong} of 9 | 20 "
            f"| **{entry['ties']}** | {entry['draws']} |"
        )
    for n_wrong in sorted(by_comp):
        entry = by_comp[n_wrong]
        lines.append(
            f"| `margin` beats a uniform draw at {n_wrong} of 9 | 20 "
            f"| **{entry['beats']}** | {entry['draws']} |"
        )

    hosted = len(blind)
    ties = sum(1 for r in blind if r["channel_ties_oracle"])
    perfect = sum(1 for r in blind if r["channel_hit_rate"] == 1.0)
    chance = sum(1 for r in blind if r["disagreement_policies_at_chance"])
    norepair = sum(1 for r in blind if r["channel_corrected"] == 0 and r["oracle_corrected"] == 0)
    lines += [
        f"| Disagreement policies sit at chance at unanimity | 21 | **{chance}** | {hosted} |",
        f"| Provenance ties the oracle bound | 23 | **{ties}** | {hosted} |",
        f"| Provenance finds *every* corrupted item | 23 | **{perfect}** | {hosted} |",
        f"| No policy repairs an unanchored label | 23 | **{norepair}** | {hosted} |",
    ]
    if payload.get("channel"):
        detected = sum(
            1
            for r in payload["channel"]
            if r["blinded_detected_at_every_noise_level"] and r["no_other_channel_detected"]
        )
        lines.append(
            f"| Blinded channel detected, controls silent | 22 | **{detected}** "
            f"| {len(payload['channel'])} |"
        )

    pool = payload["auditable_pool_range"]
    hit = sorted(r["channel_hit_rate"] for r in blind)
    lines += [
        "",
        f"Fleet of {payload['fleet']}, {payload['draws_attempted']} corpus draws, every "
        "denominator stated. Finding 21's experiment needs a blind channel orthogonal to "
        f"item difficulty and refuses to run where they are entangled, so it is "
        f"constructible on **{hosted} of {payload['draws_attempted']}** draws; a draw that "
        "cannot host the negative control says nothing about the finding and is excluded "
        "rather than counted against it.",
        "",
        f"The auditable pool an audit budget is a fraction of ranges **{pool[0]} to "
        f"{pool[1]}**, not the 97 the script documented. Provenance recovers "
        f"**{hit[0]:.2f} to {hit[-1]:.2f}** of corrupted items against every "
        "disagreement-reading policy's 0.25 or less, so the *advantage* is robust even "
        "where the 1.00 is not.",
    ]
    return "\n".join(lines)


#: Block name to builder. A block present in a doc but absent here is an error rather
#: than a no-op: a marker with nothing behind it is how a table quietly stops updating.
BLOCKS = {
    "governance-sensitivity": governance_sensitivity,
    "estimator-initialization": estimator_initialization,
    "corpus-sensitivity": corpus_sensitivity,
    "power-claims": power_claims,
    "measurement-health": measurement_health,
    "teacher-fleet": teacher_fleet,
    "secure-readership": secure_readership,
    "authority-grid": authority_grid,
    "authority-price": authority_price,
    "audit-policy": audit_policy,
    "blind-spot": blind_spot,
    "difficulty-confound": difficulty_confound,
    "label-fidelity": label_fidelity,
    "teacher-inheritance": teacher_inheritance,
    "teacher-inheritance-xseed": teacher_inheritance_xseed,
    "linkage-controls": linkage_controls,
    "privacy-budget": privacy_budget,
    "channel-bias": channel_bias,
}

#: A BEGIN marker on its own. Used to catch pairs the full pattern cannot match --
#: adjacent markers with no body, a mismatched name, a missing END -- because those
#: are silently skipped by `_MARKER` and a block that is never rendered looks exactly
#: like a block that is up to date. This guard exists because that happened: the
#: measurement-health block was added with its markers on consecutive lines, produced
#: no output, and `--check` reported everything current.
_OPENER = re.compile(r"<!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->")

_MARKER = re.compile(
    r"(?P<open><!-- BEGIN GENERATED: (?P<name>[a-z0-9-]+) -->\n)"
    r"(?P<body>.*?)"
    r"(?P<close>\n<!-- END GENERATED: (?P=name) -->)",
    re.DOTALL,
)


def render(text: str) -> tuple[str, list[str]]:
    """Rewrite every marked block. Returns the new text and the names it rewrote."""
    seen: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        builder = BLOCKS.get(name)
        if builder is None:
            _fail(f"no builder registered for generated block {name!r}")
        seen.append(name)
        return f"{match.group('open')}{builder()}{match.group('close')}"

    return _MARKER.sub(replace, text), seen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any generated block differs from its artifact",
    )
    args = parser.parse_args()

    stale: list[str] = []
    malformed: list[str] = []
    checked = 0
    for path in sorted(DOCS.rglob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated, names = render(original)
        # Every BEGIN marker must have been rendered. One that was not is a marker the
        # full pattern could not match, which renders nothing and reports nothing.
        malformed.extend(
            f"{path.relative_to(ROOT)} ({m.group('name')})"
            for m in _OPENER.finditer(original)
            if m.group("name") not in names
        )
        checked += len(names)
        if not names:
            continue
        if args.check:
            if updated != original:
                stale.append(f"{path.relative_to(ROOT)} ({', '.join(names)})")
        elif updated != original:
            path.write_text(updated, encoding="utf-8")
            print(f"updated {path.relative_to(ROOT)}: {', '.join(names)}")

    # A guard that verified nothing must not report success. This exact failure --
    # a check passing because it silently matched no files -- has happened twice in
    # this project's tooling, so it is asserted rather than assumed.
    if checked == 0:
        _fail("found no generated blocks in docs/; the markers are missing or renamed")

    if malformed:
        print(
            "generated blocks whose markers did not match and rendered nothing: "
            + "; ".join(malformed),
            file=sys.stderr,
        )
        print(
            "check the BEGIN/END pair: same name, END on its own line, a body between them",
            file=sys.stderr,
        )
        return 1

    if stale:
        print("stale generated blocks: " + "; ".join(stale), file=sys.stderr)
        print("run `uv run python scripts/sync_docs_tables.py` to refresh", file=sys.stderr)
        return 1
    print(f"sync_docs_tables: {checked} generated block(s) current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
