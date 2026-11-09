"""Weighted residuals and the competent-with-slips generator."""

from __future__ import annotations

import math
import random

from graphmend import (STANDARD, CompetentWithSlips, WeightedResidual,
                       generate_instance_at_depth, integer_candidates)


def _env():
    env, _ = generate_instance_at_depth(3, 8, coupling=0.0, n_perturb=6, **STANDARD)
    return env


def test_weighted_candidates_are_exact() -> None:
    for weights in ("uniform", "geometric"):
        env = _env()
        wr = WeightedResidual(env, random.Random(1), weights=weights)
        for ed, after in wr.candidates():
            env._apply_edit(ed)
            assert math.isclose(wr.psi(), after, abs_tol=1e-9)
            env._undo_edit(ed)


def test_weighted_residual_vanishes_at_solution() -> None:
    env = _env()
    wr = WeightedResidual(env, random.Random(2), weights="geometric")
    assert wr.psi() > 0
    for a in env.solution_actions():
        env.apply(a)
    assert wr.psi() == 0


def test_capability_matches_simulation() -> None:
    env = _env()
    cands = integer_candidates(env)
    gen = CompetentWithSlips(m=10, eps=0.1, rng=random.Random(0))
    thr = env.psi() - 1
    exact = gen.capability([v for _, v in cands], thr)
    n = 20000
    hits = sum(gen.propose(cands)[1] <= thr for _ in range(n))
    assert abs(hits / n - exact) < 4 * math.sqrt(exact * (1 - exact) / n) + 1e-9
