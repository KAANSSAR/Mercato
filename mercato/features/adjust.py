"""Fair-comparison maths: per 90, shrinkage, possession, opposition, form, fees.

Tuning constants (K, ALPHA, half-life) are placeholders until fitted.
"""

import math

MIN_MINUTES = 900     # minutes before a player is fully rated
K = 900               # shrinkage strength, in minutes (to tune)
ELO_BASE = 1700
ALPHA = 1.0           # to fit: regress per-match output on opponent Elo
FORM_HALF_LIFE = 5    # matches


def per90(stat: float, minutes: float) -> float:
    if minutes <= 0:
        return 0.0
    return stat / minutes * 90


def shrink(player_rate: float, minutes: float, position_avg: float, k: float = K) -> float:
    """Pull small samples towards the position average."""
    return (minutes * player_rate + k * position_avg) / (minutes + k)


def padj(stat: float, team_possession: float) -> float:
    """StatsBomb-style possession adjustment to a 50% baseline. Possession as 0 to 1."""
    if not 0 <= team_possession < 1:
        raise ValueError("team_possession must be in [0, 1)")
    return stat * 0.5 / (1 - team_possession)


def match_weight(opponent_elo: float, alpha: float = ALPHA) -> float:
    return (opponent_elo / ELO_BASE) ** alpha


def form_lambda(half_life: float = FORM_HALF_LIFE) -> float:
    """EWMA weight giving the stated half-life in matches."""
    return 1 - 0.5 ** (1 / half_life)


def ewma_form(results: list[float], half_life: float = FORM_HALF_LIFE) -> float:
    """Recency-weighted average, results oldest first (points or Elo-weighted points).

    Same decay as form_t = lam * result_t + (1 - lam) * form_(t-1), but normalised
    so the first match isn't over-weighted as the seed.
    """
    if not results:
        return 0.0
    n = len(results)
    weights = [0.5 ** ((n - 1 - i) / half_life) for i in range(n)]
    return sum(w * r for w, r in zip(weights, results)) / sum(weights)


def inflate_fee(fee: float, index_year: float, index_now: float) -> float:
    return fee * index_now / index_year


def log_fee(fee: float) -> float:
    return math.log(fee)
