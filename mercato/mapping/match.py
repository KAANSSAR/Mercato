"""One-to-one matching between two lists using the Hungarian algorithm."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

AUTO = 85.0     # accept without review
REVIEW = 70.0   # plausible, write to the review CSV


@dataclass
class Match:
    left: dict
    right: dict | None
    score: float
    status: str  # auto, review, unmatched


def assign(left: list[dict], right: list[dict], score: Callable[[dict, dict], float],
           auto: float = AUTO, review: float = REVIEW) -> list[Match]:
    """Best one-to-one pairing maximising total score. Each left item gets at most one right."""
    if not left:
        return []
    if not right:
        return [Match(item, None, 0.0, "unmatched") for item in left]
    s = np.array([[score(a, b) for b in right] for a in left], dtype=float)
    rows, cols = linear_sum_assignment(-s)
    paired = dict(zip(rows, cols))
    out = []
    for i, item in enumerate(left):
        j = paired.get(i)
        sc = float(s[i, j]) if j is not None else 0.0
        if j is None or sc < review:
            out.append(Match(item, None, sc, "unmatched"))
        else:
            out.append(Match(item, right[j], sc, "auto" if sc >= auto else "review"))
    return out
