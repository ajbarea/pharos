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
def test_the_verdict_tracks_the_raw_comparison_outside_the_rounding_band(offset):
    """Outside one rounding step of the floor, exit status and raw comparison agree."""
    (floor,) = {floor for values in declared_floors().values() for floor in values}
    total = floor + offset
    assert should_fail_under(total, floor, configured_precision()) == (total < floor)


def test_the_rounding_band_is_real_and_is_the_documented_behaviour():
    """There *is* a band below the floor that passes, and it is not a defect.

    pytest-cov rounds the total to the configured precision before comparing, so with
    `precision = 2` and a floor of 92 every total in [91.995, 92.0) rounds to 92.00 and
    exits zero. An earlier version of the test above asserted that "no band anywhere
    below the floor may pass", sampled only offsets outside that band, and so reported
    coverage of a property the code does not have.

    The band is upstream's deliberate design rather than an oversight: the fail-under
    check was changed in pytest-cov 7.0.0 to use the configured precision precisely so
    that the exit status agrees with the number `coverage report` prints. A gate that
    printed 92.00% and then failed would be the worse bug, and was the one `precision`
    was added here to fix.

    What is worth pinning is that the band is exactly one rounding step wide and that
    nothing below it passes, so the gate cannot drift further open without this
    failing.
    """
    (floor,) = {floor for values in declared_floors().values() for floor in values}
    precision = configured_precision()
    step = 10.0**-precision

    # Inside the band: rounds up to the floor, and passes. This is the documented
    # behaviour, asserted rather than assumed.
    assert not should_fail_under(floor - step / 2.5, floor, precision)

    # Immediately below the band: no longer rounds to the floor, and fails.
    assert should_fail_under(floor - step, floor, precision)

    # The band cannot be wider than one rounding step in the other direction either.
    assert should_fail_under(floor - step * 2, floor, precision)


def test_the_floor_is_not_reachable_by_rounding_from_a_whole_point_below():
    """The gate's real guarantee, stated in terms a reader can act on.

    Whatever the rounding band does at the fourth decimal place, a run that is a
    tenth of a point short must fail. That is the property the floor exists to
    provide, and it is what the coverage number in the README is worth.
    """
    (floor,) = {floor for values in declared_floors().values() for floor in values}
    for shortfall in (0.1, 0.5, 1.0, 5.0):
        assert should_fail_under(floor - shortfall, floor, configured_precision()), (
            f"a run {shortfall} points below the floor passed the gate"
        )
