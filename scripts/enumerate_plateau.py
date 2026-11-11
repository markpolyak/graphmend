#!/usr/bin/env python3
"""Plateau edits in dead states over the standard GraphMend grid (since 1.1.0).

For every (depth, coupling, i) of scripts/enumerate_structure.py the instance is
generated and graphmend.analysis.plateau_reachability is computed with kmax = 2:
whether the goal is reachable with at most 0, 1, 2 plateau edits taken in dead
states, and the share of dead states with a plateau edit into a viable state.
About 10 minutes on 10 cores. Output: data/plateau.json.

    python scripts/enumerate_plateau.py
    N=20 DEPTHS=2,4 OUT=/tmp/small.json python scripts/enumerate_plateau.py
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

from graphmend import N_PERTURB, STANDARD, generate_instance_at_depth, plateau_reachability

HERE = os.path.dirname(os.path.abspath(__file__))
KMAX = 2


def seed_of(depth: int, coupling: float, i: int) -> int:
    return 7_000_000 + 100_000 * depth + 10_000 * round(coupling * 10) + i


def run(args: tuple[int, float, int]) -> dict:
    depth, coupling, i = args
    t0 = time.perf_counter()
    seed = seed_of(depth, coupling, i)
    env, _ = generate_instance_at_depth(seed, depth, coupling=coupling,
                                        n_perturb=N_PERTURB[depth], **STANDARD)
    row = {"depth": depth, "coupling": coupling, "i": i, "seed": seed}
    row.update(plateau_reachability(env, KMAX))
    row["sec"] = time.perf_counter() - t0
    return row


def main() -> None:
    depths = [int(x) for x in os.environ.get("DEPTHS", "2,4,8,12").split(",")]
    couplings = [float(x) for x in os.environ.get("COUPLINGS", "0,0.4,0.8").split(",")]
    n = int(os.environ.get("N", 200))
    jobs = [(d, c, i) for d in depths for c in couplings for i in range(n)]
    t = time.time()
    with Pool(int(os.environ.get("PROCS", os.cpu_count() or 1))) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    out = os.environ.get("OUT", os.path.join(HERE, "..", "data", "plateau.json"))
    json.dump({"instance": STANDARD, "n_perturb": {str(k): v for k, v in N_PERTURB.items()},
               "kmax": KMAX, "rows": rows}, open(out, "w"), indent=1)
    print(f"{len(rows)} instances in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
