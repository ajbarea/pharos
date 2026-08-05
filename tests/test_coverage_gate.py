"""The coverage floor has to be able to fail.

This file exists because it could not. `pytest-cov` prints "FAIL Required test
coverage of N% not reached" from a raw `total < fail_under` comparison, but decides
the process exit status with `should_fail_under(total, fail_under, precision)`, which
compares `round(total, precision)`. At coverage's default precision of 0 the two
disagree for any total in [N-0.5, N): the run prints FAIL and exits 0. This suite sat
at 91.74% against a floor of 92% and `make ci` reported success -- locally and in
GitHub Actions, which runs the same command.

The tests below pin the two things that keep them in agreement: the precision that
arms the check, and the floor being the same number everywhere it is written down.
"""

import re
import tomllib
from pathlib import Path

import pytest
from coverage.results import should_fail_under

ROOT = Path(__file__).resolve().parent.parent
FLOOR_PATTERN = re.compile(r"--cov-fail-under=(\d+(?:\.\d+)?)")


def configured_precision() -> int:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["coverage"]["report"]["precision"]


def declared_floors() -> dict[str, list[float]]:
    """Every place the floor is written, keyed by file."""
    sources = [ROOT / "Makefile", ROOT / ".github" / "workflows" / "ci.yml"]
    return {
        source.name: [float(m) for m in FLOOR_PATTERN.findall(source.read_text(encoding="utf-8"))]
        for source in sources
        if source.exists()
    }


def test_the_floor_is_the_same_number_everywhere_it_is_written():
    floors = declared_floors()
    assert floors, "no --cov-fail-under found; the gate moved and this test went blind"
    distinct = {floor for values in floors.values() for floor in values}
    assert len(distinct) == 1, f"the coverage floor disagrees across files: {floors}"


def test_precision_makes_the_printed_verdict_and_the_exit_status_agree():
    """The regression itself: a total half a point under the floor must fail."""
    precision = configured_precision()
    (floor,) = {floor for values in declared_floors().values() for floor in values}

    just_under = floor - 0.26  # inside the band that rounds up at precision 0
    assert just_under < floor, "constructed total is not actually under the floor"
    assert should_fail_under(just_under, floor, precision), (
        f"a total of {just_under} is below the floor of {floor} but would exit 0 at "
        f"precision {precision}; the gate is disarmed"
    )


def test_the_default_precision_is_what_broke_it():
    """Kept so the reason for the setting survives the setting.

    If a future coverage release stops rounding, this test fails and the precision
    line can be reconsidered on evidence rather than removed on a guess.
    """
    (floor,) = {floor for values in declared_floors().values() for floor in values}
    assert not should_fail_under(floor - 0.26, floor, 0)


@pytest.mark.parametrize("offset", [-50.0, -0.26, -0.01, 0.0, +0.01, +8.0])
def test_the_verdict_tracks_the_raw_comparison_on_both_sides(offset):
    """No band anywhere below the floor may pass, and nothing at or above it may fail."""
    (floor,) = {floor for values in declared_floors().values() for floor in values}
    total = floor + offset
    assert should_fail_under(total, floor, configured_precision()) == (total < floor)
