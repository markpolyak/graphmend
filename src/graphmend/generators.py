"""Programmatic action generators with an exactly known proposal law.

``CompetentWithSlips``: with probability ``1 - eps`` it draws ``m`` distinct edits
uniformly at random and proposes the one with the lowest true residual after the
edit; otherwise (a slip) it proposes a uniformly random edit. Because the law is
explicit, the probability that it proposes a good edit in a given state is exact
(``capability``), which is what checking assumptions about the generator needs.
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .env import ConstraintRepairEnv, Edit


def integer_candidates(env: ConstraintRepairEnv) -> list[tuple[Edit, float]]:
    """``[(edit, psi_after)]`` for every single edit, integer residual."""
    base = env.psi()
    return [(ed, base + env._delta_psi(ed)) for ed in list(env._candidate_edits())]


class CompetentWithSlips:
    def __init__(self, m: int = 10, eps: float = 0.1, rng: random.Random | None = None) -> None:
        self.m, self.eps = m, eps
        self.rng = rng or random.Random()

    def propose(self, candidates: Sequence[tuple[Edit, float]]) -> tuple[Edit, float]:
        """Pick one ``(edit, psi_after)`` from ``candidates`` (see module docstring)."""
        if self.rng.random() < 1 - self.eps:
            pick = self.rng.sample(list(candidates), min(self.m, len(candidates)))
            return min(pick, key=lambda t: t[1])
        return self.rng.choice(list(candidates))

    def capability(self, values: Sequence[float], threshold: float) -> float:
        """Exact probability that the proposed edit has ``psi_after <= threshold``,
        given the residuals ``values`` after every candidate edit."""
        n = len(values)
        k = sum(1 for v in values if v <= threshold + 1e-12)
        best = 1.0 if n - k < self.m else 1 - math.comb(n - k, self.m) / math.comb(n, self.m)
        return (1 - self.eps) * best + self.eps * k / n
