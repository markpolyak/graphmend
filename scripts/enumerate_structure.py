#!/usr/bin/env python3
"""Exhaustive structural analysis of the standard GraphMend grid.

For every (depth, coupling, i) the instance is generated deterministically and its
monotone-descent graph is enumerated completely (graphmend.analysis.analyse).
Default grid: depths 2, 4, 8, 12; coupling 0.0, 0.4, 0.8; 200 instances per cell
(2400 instances, a few minutes on 10 cores). Output: data/structure.json.

    python scripts/enumerate_structure.py
    N=20 DEPTHS=2,4 OUT=/tmp/small.json python scripts/enumerate_structure.py
"""
from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool

from graphmend import N_PERTURB, STANDARD, analyse, generate_instance_at_depth

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = int(os.environ.get("CAP", 300_000))


def seed_of(depth: int, coupling: float, i: int) -> int:
    return 7_000_000 + 100_000 * depth + 10_000 * round(coupling * 10) + i


def run(args: tuple[int, float, int]) -> dict:
    depth, coupling, i = args
    t0 = time.perf_counter()
    seed = seed_of(depth, coupling, i)
    env, _ = generate_instance_at_depth(seed, depth, coupling=coupling,
                                        n_perturb=N_PERTURB[depth], **STANDARD)
    row = {"depth": depth, "coupling": coupling, "i": i, "seed": seed}
    row.update(analyse(env, cap=CAP))
    row["minedits"] = env.instance.min_edits_to_solution if env.instance else None
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
    out = os.environ.get("OUT", os.path.join(HERE, "..", "data", "structure.json"))
    json.dump({"instance": STANDARD, "n_perturb": {str(k): v for k, v in N_PERTURB.items()},
               "cap": CAP, "rows": rows}, open(out, "w"), indent=1)
    print(f"{len(rows)} instances in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
