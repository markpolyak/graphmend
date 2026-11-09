#!/usr/bin/env python3
"""Summary table of data/structure.json (Table 1 of the structural-properties paper).

    python scripts/summarize.py [path/to/structure.json]
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "structure.json")
rows = json.load(open(path))["rows"]
assert not any(r["capped"] for r in rows), "some descent graphs were capped"

print("| depth | c | median |R| | s0 is a trap, % | trap-free, % | traps in R, % "
      "| blind descent stuck, % | plateau escape, % |")
print("|--:|--:|--:|--:|--:|--:|--:|--:|")
for d in sorted({r["depth"] for r in rows}):
    for c in sorted({r["coupling"] for r in rows}):
        R = [r for r in rows if r["depth"] == d and r["coupling"] == c]
        esc = [r["esc"] for r in R if r["esc"] is not None]
        print(f"| {d} | {c:.1f} | {st.median(r['R'] for r in R):g} "
              f"| {100 * sum(not r['viable0'] for r in R) / len(R):.1f} "
              f"| {100 * sum(r['trap'] == 0 for r in R) / len(R):.1f} "
              f"| {100 * st.mean(r['trap'] for r in R):.1f} "
              f"| {100 * st.mean(r['walk'] for r in R):.1f} "
              f"| {100 * st.mean(esc):.0f} |")
print(f"\n{len(rows)} instances; largest |R| = {max(r['R'] for r in rows)}; "
      f"dead start states: {sum(r['dead0'] for r in rows)}")
