"""The ETHICS §4 disclosure floor, in one place.

`docs/ETHICS.md` §4 is categorical: do not publish any cell computed from fewer than 20 dreams or
fewer than 5 distinct users. `scripts/check_release_cells.py` enforces that on tracked files after
the fact. This module is the other half — the floor as a *constructive* constraint, so an artifact
can be built compliant rather than audited and then patched.

The animations motivated it. A video that sweeps "the mean arc over the first n reports" necessarily
renders small-n frames, and a mean over two reports is close to publishing the two trajectories
themselves. Flooring the sweep is the fix, and it has to respect both halves of the rule: 20 reports
is not enough if they came from three people.

    from psychohistory.utils.disclosure import min_prefix
    start = min_prefix(users)      # smallest prefix meeting both floors
"""
from __future__ import annotations

from collections.abc import Sequence

__all__ = ["MIN_REPORTS", "MIN_USERS", "min_prefix", "n_users_in"]

MIN_REPORTS = 20
MIN_USERS = 5


def n_users_in(users: Sequence) -> int:
    return len(set(users))


def min_prefix(users: Sequence, min_reports: int = MIN_REPORTS,
               min_users: int = MIN_USERS) -> int:
    """Length of the shortest prefix of `users` meeting both disclosure floors.

    `users` is the contributor identifier of each item in the order the artifact will accumulate
    them, so the returned length is the first point at which a running aggregate may be displayed.

    Raises if no prefix qualifies, which is the correct outcome: it means the whole sample is below
    the floor and nothing derived from it may be published.
    """
    seen: set = set()
    for i, u in enumerate(users, start=1):
        seen.add(u)
        if i >= min_reports and len(seen) >= min_users:
            return i
    raise ValueError(
        f"no prefix of {len(users)} items reaches {min_reports} items from {min_users} distinct "
        f"contributors (whole sample has {len(seen)}); nothing derived from it may be published")
