"""plateau_reachability reproduces the published data (data/plateau.json) exactly."""

from __future__ import annotations

import json
import math
import os

import pytest

from graphmend import N_PERTURB, STANDARD, analyse, generate_instance_at_depth, plateau_reachability

HERE = os.path.dirname(__file__)
ROWS = json.load(open(os.path.join(HERE, "..", "data", "plateau.json")))["rows"]
BASE = {(r["depth"], r["coupling"], r["i"]): r
        for r in json.load(open(os.path.join(HERE, "..", "data", "structure.json")))["rows"]}

# A few instances from every cell of depth <= 8, plus every one of them whose s0 is a
# trap (depth 12 is slow; run scripts/enumerate_plateau.py to check it).
SAMPLE = [r for r in ROWS if r["depth"] <= 8 and (r["i"] < 3 or not r["solv"][0])]


def _env(row: dict):
    env, _ = generate_instance_at_depth(row["seed"], row["depth"], coupling=row["coupling"],
                                        n_perturb=N_PERTURB[row["depth"]], **STANDARD)
    return env


@pytest.mark.parametrize("row", SAMPLE, ids=lambda r: f"d{r['depth']}-c{r['coupling']}-i{r['i']}")
def test_plateau_matches_published_row(row: dict) -> None:
    got = plateau_reachability(_env(row), kmax=2)
    assert got["solv"] == row["solv"]
    assert (got["R"], got["dead"]) == (row["R"], row["dead"])
    if row["esc_viable"] is None:
        assert got["esc_viable"] is None
    else:
        assert math.isclose(got["esc_viable"], row["esc_viable"], abs_tol=1e-12)


def test_consistent_with_structure_data() -> None:
    for r in ROWS:
        b = BASE[(r["depth"], r["coupling"], r["i"])]
        assert (r["R"], r["dead"], r["solv"][0]) == (b["R"], b["dead"], b["viable0"])
        assert r["solv"] == sorted(r["solv"])  # more plateau edits never hurt
        if r["esc_viable"] is not None and b["esc"] is not None:
            assert r["esc_viable"] <= b["esc"] + 1e-12  # into viable implies into D != {}


def test_state_restored() -> None:
    env = _env(SAMPLE[0])
    before = analyse(env)
    plateau_reachability(env)
    assert analyse(env) == before
