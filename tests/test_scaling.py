"""The size-dependence data (data/scaling.json) are reproduced for small instances."""

from __future__ import annotations

import json
import math
import os

import pytest

from graphmend import N_PERTURB, analyse, generate_instance_at_depth

ROWS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "scaling.json")))["rows"]
# Cheap instances only (depth <= 8, n <= 10); scripts/enumerate_scaling.py checks the rest.
SAMPLE = [r for r in ROWS if r["d"] <= 8 and r["n"] <= 10 and r["i"] < 3]


@pytest.mark.parametrize("row", SAMPLE, ids=lambda r: f"n{r['n']}-d{r['d']}-i{r['i']}")
def test_scaling_matches_published_row(row: dict) -> None:
    n, d, i = row["n"], row["d"], row["i"]
    env, _ = generate_instance_at_depth(8_000_000 + 100_000 * n + 1000 * d + i, d, coupling=0.0,
                                        n_perturb=N_PERTURB[d], n_nodes=n, n_labels=3,
                                        n_constraints=round(3.25 * n))
    got = analyse(env, cap=500_000)
    assert (got["R"], got["n_edits"], got["viable0"]) == (row["R"], row["edits"], row["viable0"])
    assert math.isclose(got["trap"], row["trap"], abs_tol=1e-12)
    assert math.isclose(got["walk"], row["walk"], abs_tol=1e-12)
