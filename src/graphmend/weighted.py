"""Real-valued residuals on top of a GraphMend instance.

``WeightedResidual`` replaces the count of violated constraints by the sum of their
weights. Two weightings are provided:

  "uniform"    weights drawn from U(0.5, 1.5): each repair removes roughly a
               constant amount of psi (additive progress);
  "geometric"  weights 0.75**r for a random ranking r of the constraints: repairing
               the heaviest violation shrinks psi by a roughly constant factor
               (multiplicative progress).

``measure`` adds Gaussian measurement noise with a given standard deviation.
"""

from __future__ import annotations

import random

from .env import ConstraintRepairEnv, Edit


class WeightedResidual:
    def __init__(
        self,
        env: ConstraintRepairEnv,
        rng: random.Random,
        weights: str = "uniform",
        base: float = 0.75,
    ) -> None:
        self.env = env
        n = len(env.constraints)
        if weights == "uniform":
            self.w = [rng.uniform(0.5, 1.5) for _ in range(n)]
        elif weights == "geometric":
            order = list(range(n))
            rng.shuffle(order)
            self.w = [base ** order[c] for c in range(n)]
        else:
            raise ValueError(f"unknown weighting {weights!r}")

    def psi(self) -> float:
        """Exact weighted residual of the current state."""
        e = self.env
        return sum(w for w, c in zip(self.w, e.constraints) if not c.satisfied(e))

    def measure(self, rng: random.Random, sd: float) -> float:
        """The residual observed with additive Gaussian noise N(0, sd**2)."""
        return self.psi() + (rng.gauss(0.0, sd) if sd > 0 else 0.0)

    def candidates(self) -> list[tuple[Edit, float]]:
        """``[(edit, psi_after)]`` for every single edit, computed from the
        constraints affected by the edited variable only."""
        e, out, base = self.env, [], self.psi()
        for ed in list(e._candidate_edits()):
            aff = e._var_index.get(e._edit_var(ed), ())
            before = sum(self.w[ci] for ci in aff if not e.constraints[ci].satisfied(e))
            e._apply_edit(ed)
            after = sum(self.w[ci] for ci in aff if not e.constraints[ci].satisfied(e))
            e._undo_edit(ed)
            out.append((ed, base - before + after))
        return out
