"""The analysis reproduces the published data (data/structure.json) exactly."""

from __future__ import annotations

import json
import math
import os

import pytest

from graphmend import N_PERTURB, STANDARD, analyse, generate_instance_at_depth

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "structure.json")
ROWS = json.load(open(DATA))["rows"]
KEYS = ("psi0", "n_edits", "k0", "plateau0", "dead0", "R", "dead", "viable0", "capped")

# A few instances from every cell of depth <= 8 (depth 12 is slow; run the full
# enumeration script to check it), including ones with traps and dead states.
SAMPLE = [r for r in ROWS if r["depth"] <= 8 and r["i"] < 4] + [
    r for r in ROWS if r["depth"] == 4 and r["coupling"] == 0.0 and r["i"] == 130]


@pytest.mark.parametrize("row", SAMPLE, ids=lambda r: f"d{r['depth']}-c{r['coupling']}-i{r['i']}")
def test_analysis_matches_published_row(row: dict) -> None:
    env, _ = generate_instance_at_depth(row["seed"], row["depth"], coupling=row["coupling"],
                                        n_perturb=N_PERTURB[row["depth"]], **STANDARD)
    got = analyse(env)
    for k in KEYS:
        assert got[k] == row[k], k
    for k in ("trap", "walk"):
        assert math.isclose(got[k], row[k], abs_tol=1e-12), k
    if row["esc"] is None:
        assert got["esc"] is None
    else:
        assert math.isclose(got["esc"], row["esc"], abs_tol=1e-12)


def test_illustrative_instance() -> None:
    # The descent graph drawn in the paper: |R| = 12, two dead states, a quarter
    # of R are traps, the blind descent gets stuck with probability 1/2.
    env, _ = generate_instance_at_depth(7400130, 4, coupling=0.0, n_perturb=3, **STANDARD)
    got = analyse(env)
    assert (got["R"], got["dead"], got["k0"]) == (12, 2, 3)
    assert got["trap"] == 0.25 and got["walk"] == 0.5
