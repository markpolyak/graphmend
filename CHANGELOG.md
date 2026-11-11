# Changelog

## 1.1.0 (2026-11-11)

- `graphmend.analysis.plateau_reachability`: is the goal reachable when at most k
  plateau edits (edits that keep ψ) are allowed, each taken in a dead state; share of
  dead states with a plateau edit into a viable state.
- `scripts/enumerate_plateau.py` and `data/plateau.json`: these quantities for the
  2400 instances of the standard grid; `tests/test_plateau.py`.
- Docstrings: a dead state is a state of R (goal states are not called dead).
- The environment, the instance generator and `analyse` are unchanged: results of
  1.0.0 (`data/structure.json`) are reproduced exactly.

## 1.0.0 (2026-11-09)

First release.
