"""The governance kit: fleets, the aggregator's view, selection, abstention, and shape.

Findings 18 through 29 are measurements over a small set of shared objects -- a fleet of
simulated analysts, the aggregate a server actually holds, a selection rule, a decision to
withhold, and a statistic over vote dispersion. Those objects were defined inside the
measurement scripts and imported between them: thirty-four names crossed script
boundaries, two scripts were each imported by seven others, and one import reached past a
leading underscore. The scripts had become a library without becoming importable.

This package is that library. The scripts keep what is genuinely theirs -- grids, budgets,
ladders, artifact schemas, and the exit-code protocol a sweep matches on -- and a test
asserts that no script imports another.

The re-exports below are the supported surface. Import from the submodules for anything
else, and expect the submodule layout to be the stable thing rather than this list.
"""

from pharos.governance.abstention import (
    HALVED,
    REPORT_BUDGET,
    Cell,
    beats_every_draw,
    first_budget_halving,
    score,
)
from pharos.governance.fleet import (
    BLIND,
    BLIND_RUNGS,
    CHANNEL_ENTANGLEMENT_SLACK,
    MASK_SEED,
    REFUSED_EXIT,
    RUNGS,
    WRONG_THRESHOLD,
    ChannelCheck,
    ChannelUnusableError,
    assert_channel_usable,
    blind_fleet,
    contributions_for,
    fleet_of,
    ladder,
    majority,
)
from pharos.governance.policy import (
    DEPLOYABLE,
    POLICIES,
    POLICY_SEED,
    UNIFORM_SEEDS,
    Policy,
    choose_anchors,
    select,
)
from pharos.governance.shape import (
    ALPHA,
    MIN_STRATUM,
    NULL_DRAWS,
    Dispersion,
    dispersion,
)
from pharos.governance.view import ServerObservation, fleet_view, observe

__all__ = [
    "ALPHA",
    "BLIND",
    "BLIND_RUNGS",
    "CHANNEL_ENTANGLEMENT_SLACK",
    "DEPLOYABLE",
    "HALVED",
    "MASK_SEED",
    "MIN_STRATUM",
    "NULL_DRAWS",
    "POLICIES",
    "POLICY_SEED",
    "REFUSED_EXIT",
    "REPORT_BUDGET",
    "RUNGS",
    "UNIFORM_SEEDS",
    "WRONG_THRESHOLD",
    "Cell",
    "ChannelCheck",
    "ChannelUnusableError",
    "Dispersion",
    "Policy",
    "ServerObservation",
    "assert_channel_usable",
    "beats_every_draw",
    "blind_fleet",
    "choose_anchors",
    "contributions_for",
    "dispersion",
    "first_budget_halving",
    "fleet_of",
    "fleet_view",
    "ladder",
    "majority",
    "observe",
    "score",
    "select",
]
