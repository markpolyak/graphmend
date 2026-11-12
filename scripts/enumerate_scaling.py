#!/usr/bin/env python3
"""Dependence on the size of the graph (since 1.2.0).

The analysis of scripts/enumerate_structure.py repeated for n = 6, 8, 10, 12, 16 nodes
(3 labels, round(3.25 n) constraints, i.e. 26 for n = 8), coupling 0, depths 4, 8 and
12, 30 instances per cell. Instance (n, d, i) uses base seed
8_000_000 + 100_000 n + 1000 d + i. Depth 12 is not run for n = 16: a single instance
needs more than LIMIT_MB there.

Large graphs need a lot of memory, so every instance runs in its own process, at most
PROCS at a time, and a process whose resident memory exceeds LIMIT_MB is killed and
recorded as such ("killed_mb"). The defaults (2 processes, 3000 MB) keep the run under
about 6 GB. About 40 minutes on an Apple M2 Pro. Output: data/scaling.json; for each
finished instance, ``rss_mb`` is the peak resident memory of its process.

    python scripts/enumerate_scaling.py
    NODES=6,8 DEPTHS=4,8 N=5 OUT=/tmp/small.json python scripts/enumerate_scaling.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROCS = int(os.environ.get("PROCS", 2))
LIMIT_MB = int(os.environ.get("LIMIT_MB", 3000))
CAP = 500_000

CHILD = r'''
import json, resource, sys, time
from graphmend import N_PERTURB, analyse, generate_instance_at_depth
n, d, i, cap = map(int, sys.argv[1:5])
t0 = time.perf_counter()
env, _ = generate_instance_at_depth(8_000_000 + 100_000 * n + 1000 * d + i, d, coupling=0.0,
                                    n_perturb=N_PERTURB[d], n_nodes=n, n_labels=3,
                                    n_constraints=round(3.25 * n))
r = analyse(env, cap=cap)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux
print(json.dumps({"n": n, "d": d, "i": i, "R": r["R"], "capped": r["capped"],
                  "edits": r["n_edits"], "trap": r["trap"], "walk": r["walk"],
                  "viable0": r["viable0"], "sec": time.perf_counter() - t0, "rss_mb": rss_mb}))
'''


def rss_mb(pid: int) -> float:
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return int(out.strip()) / 1024 if out.strip() else 0.0


def main() -> None:
    nodes = [int(x) for x in os.environ.get("NODES", "6,8,10,12,16").split(",")]
    depths = [int(x) for x in os.environ.get("DEPTHS", "4,8,12").split(",")]
    n_inst = int(os.environ.get("N", 30))
    jobs = [(n, d, i) for d in depths for n in nodes for i in range(n_inst)
            if not (n == 16 and d == 12)]
    rows, running, t = [], {}, time.time()
    while jobs or running:
        while jobs and len(running) < PROCS:
            n, d, i = jobs.pop(0)
            p = subprocess.Popen([sys.executable, "-c", CHILD, str(n), str(d), str(i), str(CAP)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            running[p] = ((n, d, i), time.time())
        time.sleep(1)
        for p in list(running):
            (n, d, i), t0 = running[p]
            if p.poll() is None:
                m = rss_mb(p.pid)
                if m > LIMIT_MB:
                    p.kill(); p.wait()
                    rows.append({"n": n, "d": d, "i": i, "killed_mb": round(m), "sec": time.time() - t0})
                    del running[p]
                continue
            out, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"instance {(n, d, i)} failed:\n{err}")
            rows.append(json.loads(out))
            del running[p]
    rows.sort(key=lambda r: (r["n"], r["d"], r["i"]))
    out = os.environ.get("OUT", os.path.join(HERE, "..", "data", "scaling.json"))
    json.dump({"limit_mb": LIMIT_MB, "procs": PROCS, "cap": CAP, "rows": rows}, open(out, "w"), indent=1)
    print(f"{len(rows)} instances in {time.time() - t:.0f}s -> {out}")


if __name__ == "__main__":
    main()
