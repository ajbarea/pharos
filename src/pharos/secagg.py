"""What a server may learn when it is only allowed to learn a sum.

Finding 11 found the leak this module exists to close: a stream of individually
approved contributions names the analyst who sent it, at 0.820 for the most highly
cleared, from an attack that reads no content. Findings 12 and 13 then closed both
ways of recovering per-contributor reliability *after* the fact. What was left open,
and is stated as the open problem in the README, is the other direction: compute the
estimate **under** aggregation, so the per-analyst stream the attack reads never
exists.

That is a protocol question, and the protocol is secure aggregation (Bonawitz et al.,
CCS 2017): clients mask their vectors with pairwise terms that cancel in the sum, so
the server recovers the total and no individual input. This module supplies the sum
and, more importantly, supplies the server's *view* of it as a value with no
per-client field to read. A privacy claim enforced by prose is a claim; a privacy
claim enforced by the only type the estimator is handed is a property.

**What is simulated and what is not.** The masking is real and its cancellation is
checked, because the aggregate has to be exactly right for the estimator built on it
to mean anything. Key agreement, threshold secret sharing for dropout recovery, and
authentication are *not* implemented: they change who can run the protocol and what
happens when a client vanishes mid-round, and neither changes what the server learns
from a completed round. What the server learns is what is being measured, so that is
what is built.

**The arithmetic is integer on purpose, and it earns its cost twice.** Real secure
aggregation works in a finite ring because masks must cancel exactly, and floating
point cancellation is not exact. The second reason is local: this repository has
already measured its own float non-determinism across machines, where the gate's
surface baseline disagreed at the fifth decimal between an AVX-512 Xeon and an AVX2
Ryzen because the BLAS picked a different reduction order. A fixed-point sum modulo
2^64 has no reduction order to pick. The aggregate is bit-identical on any machine
that can add.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING

from pharos.telemetry import get_logger, record_routine

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Ring the masked shares live in. Masks cancel modulo this and nowhere else.
MODULUS = 1 << 64

#: Fixed-point denominator. The estimator sums per-analyst log-likelihoods, which sit
#: in roughly [-40, 0], so 2^24 leaves ~6e-8 of resolution per term and the widest
#: fleet measured here cannot come within thirty bits of the ring.
SCALE = 1 << 24

#: Smallest cohort the aggregator will reveal a sum over. A server that can run the
#: protocol with one participant has not aggregated anything -- the "sum" is that
#: participant's input -- so the floor is the property, not a tuning knob. Three
#: rather than two because a two-client sum plus one client's own input is the other
#: client's input.
MIN_PARTICIPANTS = 3


class CohortTooSmallError(ValueError):
    """Raised when a sum would be revealed over too few participants.

    A distinct type because the caller that hits this is a fleet sweep at a small
    size, and it needs to tell "this configuration cannot be aggregated" apart from
    "this configuration aggregated to a bad number".
    """


class IncompleteCohortError(ValueError):
    """Raised when fewer clients submitted than the round declared.

    Distinct from `CohortTooSmallError`: that one means the round was too small to be
    an aggregate at all, this one means the round was the right size and somebody
    dropped, so the masks have not cancelled. The remedies differ -- widen the cohort
    versus re-run or implement dropout recovery -- and one exception type for both
    would send a caller to the wrong one.
    """


#: Largest magnitude `encode` will accept. The ring is 2^64 and values are scaled by
#: 2^24, so a single term must stay under 2^39 to be representable and the sum under
#: 2^63. Enforced because the failure is silent: 2^40 encodes to 0.0 and 40 terms of
#: 1.4e10 sum to a confidently wrong negative number.
MAX_MAGNITUDE = (MODULUS // 2) // SCALE


def encode(value: float) -> int:
    """One real to one ring element, two's complement for negatives."""
    if not -MAX_MAGNITUDE < value < MAX_MAGNITUDE:
        raise ValueError(
            f"{value} exceeds the +/-{MAX_MAGNITUDE} the fixed-point ring can hold; "
            "encoding it would wrap silently"
        )
    return round(value * SCALE) % MODULUS


def decode(element: int) -> float:
    """One ring element back to one real, reading the top half as negative."""
    centered = element % MODULUS
    if centered >= MODULUS // 2:
        centered -= MODULUS
    return centered / SCALE


@lru_cache(maxsize=4096)
def _mask_between(seed: int, lower: int, upper: int, length: int) -> tuple[int, ...]:
    """The shared mask for one unordered pair of clients.

    Both clients derive the same vector from the same seed, and only the sign they
    apply it with differs. Keyed by the pair so a client's mask against one peer says
    nothing about its mask against another, and seeded from a string because
    `random.Random` hashes those with SHA-512 rather than through `hash()`, which is
    the difference between a derivation that replays on another interpreter and one
    that only replays within a process.

    Cached because both members of a pair ask for the same vector inside one round,
    and re-deriving it is half the arithmetic in the protocol. The cache is keyed on
    the seed, and callers advance the seed per round, so this shares work within a
    round without ever reusing a mask across one.
    """
    rng = random.Random(f"pharos.secagg:{seed}:{lower}:{upper}")  # noqa: S311
    return tuple(rng.getrandbits(64) for _ in range(length))


@dataclass(frozen=True, slots=True)
class ServerView:
    """Everything the aggregator is permitted to know after a round.

    There is deliberately no per-client field. This is the whole point of the module:
    an estimator handed one of these cannot accidentally read an individual
    contribution, because there is no attribute that holds one. `participants` is a
    count and not a roster.

    **The residual channel is not a field here, and that is deliberate.** A total is
    not silent about its addends: a coordinate nobody contributed to sums to exactly
    zero, and in this testbed an analyst contributes only to the tasks their clearance
    lets them read, so an all-zero coordinate can disclose that a task was unreadable
    fleet-wide. But that inference is only sound when the encoded terms are known to
    be strictly signed -- with mixed signs a coordinate can sum to zero *because* its
    contributors cancelled. Offering it as a `support` property would have let a
    caller read "nobody contributed" off a number that does not say so. It is
    therefore derived where the sign argument can actually be made, in finding 18,
    against an encoding whose every term is a negative log-likelihood.
    """

    total: tuple[float, ...]
    participants: int


@dataclass
class SecureAggregator:
    """Collects masked shares and reveals only their sum.

    Clients submit through `submit`, which masks the vector against every peer before
    it is added to the running total, so no unmasked input is ever held. `reveal`
    refuses below `min_participants`.

    The aggregator is a mutable accumulator rather than a function over a list of
    vectors because that is the shape the property needs: a list of every client's
    vector, held in one place, is exactly the object secure aggregation exists to
    prevent. Handing this object around cannot leak what it never holds.
    """

    length: int
    seed: int
    min_participants: int = MIN_PARTICIPANTS
    _total: list[int] = field(init=False, repr=False)
    _submitted: set[int] = field(init=False, repr=False)
    _cohort: int | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # The floor is a property of the protocol, and a keyword that can be set to 1
        # is not a floor. `secure_sum(..., min_participants=1)` used to return one
        # client's vector verbatim, inside the type whose entire job is to not be one
        # client's vector.
        if self.min_participants < MIN_PARTICIPANTS:
            raise ValueError(
                f"min_participants={self.min_participants} is below the "
                f"{MIN_PARTICIPANTS} floor; a sum over fewer is not an aggregate"
            )
        self._total = [0] * self.length
        self._submitted = set()
        self._cohort: int | None = None

    def submit(self, client_index: int, vector: Sequence[float], *, cohort: int) -> None:
        """Add one client's masked share to the running total.

        `cohort` is how many clients are in this round, which each client needs to
        derive its mask against every peer. A client masks with `+m` against every
        higher-indexed peer and `-m` against every lower-indexed one, so the pair
        terms cancel once all of them have arrived.
        """
        if len(vector) != self.length:
            raise ValueError(f"expected {self.length} coordinates, got {len(vector)}")
        if client_index in self._submitted:
            raise ValueError(f"client {client_index} submitted twice")
        if cohort < self.min_participants:
            raise CohortTooSmallError(f"cohort={cohort} is below the {self.min_participants} floor")
        if not 0 <= client_index < cohort:
            raise ValueError(f"client_index {client_index} is outside a cohort of {cohort}")
        if self._cohort is None:
            self._cohort = cohort
        elif cohort != self._cohort:
            raise ValueError(
                f"cohort changed from {self._cohort} to {cohort} mid-round; "
                "masks are derived per cohort and would not cancel"
            )

        share = [encode(v) for v in vector]
        for peer in range(cohort):
            if peer == client_index:
                continue
            lower, upper = min(client_index, peer), max(client_index, peer)
            mask = _mask_between(self.seed, lower, upper, self.length)
            sign = 1 if client_index < peer else -1
            for i, m in enumerate(mask):
                share[i] = (share[i] + sign * m) % MODULUS

        for i, s in enumerate(share):
            self._total[i] = (self._total[i] + s) % MODULUS
        self._submitted.add(client_index)

    def reveal(self) -> ServerView:
        """The sum, once every declared client has contributed to make it one.

        The participant floor is not the only thing that has to hold. Masks are derived
        against a declared cohort, so if fewer than that arrive the pair terms do not
        cancel and the total is noise wearing a number's shape -- three clients
        submitting under `cohort=5` used to return `(-267720907167.4, ...)` for a true
        `(3.0, 3.0)`, pass the floor, and hand back a confident `ServerView`. The module
        says dropout recovery is not implemented; that has to mean *refused*, not
        *silently wrong*.
        """
        participants = len(self._submitted)
        if participants < self.min_participants:
            raise CohortTooSmallError(
                f"{participants} participants is below the {self.min_participants} floor; "
                "a sum over fewer is an individual contribution wearing a total's name"
            )
        if self._cohort is not None and participants != self._cohort:
            raise IncompleteCohortError(
                f"{participants} of {self._cohort} declared clients submitted; "
                "the pairwise masks have not cancelled and the total is meaningless. "
                "Dropout recovery is not implemented"
            )
        record_routine("secagg.reveal", float(participants), coordinates=self.length)
        return ServerView(tuple(decode(t) for t in self._total), participants)


def secure_sum(
    vectors: Sequence[Sequence[float]],
    *,
    seed: int,
    min_participants: int = MIN_PARTICIPANTS,
) -> ServerView:
    """One round of masked aggregation over every client's vector.

    A convenience over `SecureAggregator` for callers that already hold the round's
    vectors, such as a simulation. It masks and discards each one rather than summing
    them directly, so the cancellation stays on the measured path instead of being
    a claim about a path nothing exercises.
    """
    if not vectors:
        raise CohortTooSmallError("no participants")
    length = len(vectors[0])
    if any(len(v) != length for v in vectors):
        raise ValueError("all client vectors must have the same length")

    aggregator = SecureAggregator(length=length, seed=seed, min_participants=min_participants)
    for index, vector in enumerate(vectors):
        aggregator.submit(index, vector, cohort=len(vectors))
    view = aggregator.reveal()
    get_logger().debug(
        "secagg.round_complete",
        extra={
            "event": "secagg.round_complete",
            "participants": view.participants,
            "coordinates": length,
        },
    )
    return view
