#!/usr/bin/env python3
"""Figures from data/structure.json and one re-enumerated instance (needs matplotlib).

  fig_descent.png    monotone-descent graph of one small instance (depth 4, c = 0)
  fig_structure.png  mean trap share and blind-descent failure vs depth
  fig_spread.png     per-instance spread of blind-descent failure at c = 0

    python scripts/figures.py [out_dir] [--lang ru]
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from graphmend import (N_PERTURB, STANDARD, descent_graph,  # noqa: E402
                       generate_instance_at_depth)
from graphmend.analysis import viability_and_blind_descent  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = args[0] if args else os.path.join(HERE, "..", "figures")
RU = "--lang" in sys.argv and sys.argv[sys.argv.index("--lang") + 1] == "ru"
os.makedirs(OUT, exist_ok=True)
T = {
    "goal": ("goal, $\\psi=0$", "цель, $\\psi=0$"), "viable": ("viable", "жизнеспособное"),
    "trap": ("trap", "ловушка"), "dead": ("dead", "мёртвое"),
    "trap_title": ("Trap share in $R$", "Доля ловушек в $R$"),
    "walk_title": ("Blind descent stuck", "Застревание слепого спуска"),
    "depth": ("depth $\\psi(s_0)$", "глубина $\\psi(s_0)$"),
    "spread_x": ("blind descent stuck on the instance, %",
                 "вероятность застревания слепого спуска на экземпляре, %"),
    "spread_y": ("instances, %", "доля экземпляров, %"),
    "depth_n": ("depth {}", "глубина {}"),
}
L = {k: v[1] if RU else v[0] for k, v in T.items()}
dec = (lambda x: f"{x:.1f}".replace(".", "{,}")) if RU else (lambda x: f"{x:.1f}")

rows = json.load(open(os.path.join(HERE, "..", "data", "structure.json")))["rows"]
DEPTHS = sorted({r["depth"] for r in rows}); COUPL = sorted({r["coupling"] for r in rows})
COL = ["#2a78d6", "#eb6834", "#1baf7a"]; MK = ["o", "s", "^"]; LS = ["-", "--", ":"]
INK = "#3a3a37"; GRID = "#e6e5df"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 9, "mathtext.fontset": "stix", "axes.edgecolor": "#8a8a85",
                     "axes.linewidth": 0.6})


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300)
    plt.close(fig)
    print("wrote", p)


# --- monotone-descent graph of one small instance -------------------------------
r = next(r for r in rows if r["depth"] == 4 and r["coupling"] == 0.0 and r["i"] == 130)
env, _ = generate_instance_at_depth(r["seed"], 4, coupling=0.0, n_perturb=N_PERTURB[4], **STANDARD)
s0 = (tuple(env.labels), frozenset(env.edges))
kids, psi, _, _ = descent_graph(env)
viable, _ = viability_and_blind_descent(kids, psi)
for k in list(psi):
    kids.setdefault(k, [])
    viable.setdefault(k, psi[k] == 0)
levels = sorted({psi[k] for k in kids}, reverse=True)
pos = {}
for lv in levels:
    layer = [k for k in kids if psi[k] == lv]
    par = {k: [pos[p][0] for p in pos if k in kids[p]] for k in layer}
    layer.sort(key=lambda k: (st.mean(par[k]) if par[k] else 0, not viable[k]))
    for j, k in enumerate(layer):
        pos[k] = (j - (len(layer) - 1) / 2, lv)


def kind(k):
    if psi[k] == 0:
        return "goal"
    if viable[k]:
        return "viable"
    return "dead" if not kids[k] else "trap"


COLORS = {"goal": "#1baf7a", "viable": "#2a78d6", "trap": "#f2b134", "dead": "#d23c3c"}
fig, ax = plt.subplots(figsize=(6.3, 2.6))
for k, ch in kids.items():
    for c in ch:
        (x0, y0), (x1, y1) = pos[k], pos[c]
        rad = 0.0 if y0 - y1 == 1 else (0.3 if x1 >= x0 else -0.3)
        ax.annotate("", (x1, y1 + 0.13), (x0, y0 - 0.13),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#9a9a95", shrinkA=0, shrinkB=0,
                                    mutation_scale=7, connectionstyle=f"arc3,rad={rad}"))
for t, fc in COLORS.items():
    ks = [k for k in kids if kind(k) == t]
    ax.scatter([pos[k][0] for k in ks], [pos[k][1] for k in ks], s=95, c=fc, edgecolors="white",
               linewidths=0.8, zorder=3, label=L[t])
ax.annotate("$s_0$", pos[s0], xytext=(9, 4), textcoords="offset points", fontsize=10, color=INK)
ax.set_yticks(levels); ax.set_ylabel("$\\psi$")
ax.set_xticks([]); ax.set_ylim(-0.4, max(levels) + 0.4)
for s_ in ("top", "right", "bottom"):
    ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.2)
fig.tight_layout(pad=0.3)
save(fig, "fig_descent.png")

# --- trap share and blind-descent failure vs depth ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
for ax, key, title in ((axes[0], "trap", L["trap_title"]), (axes[1], "walk", L["walk_title"])):
    for i, c in enumerate(COUPL):
        y = [100 * st.mean(r[key] for r in rows if r["depth"] == d and r["coupling"] == c)
             for d in DEPTHS]
        ax.plot(DEPTHS, y, color=COL[i], marker=MK[i], ls=LS[i], lw=1.3, ms=5,
                label=f"$c={dec(c)}$", markeredgecolor="white", markeredgewidth=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(DEPTHS); ax.set_xlabel(L["depth"])
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
axes[0].set_ylabel("%")
axes[0].legend(frameon=False, fontsize=8, loc="upper left", handlelength=2.2)
fig.tight_layout(pad=0.3, w_pad=1.0)
save(fig, "fig_structure.png")

# --- per-instance spread at c = 0 (empirical distribution function) -------------
fig, ax = plt.subplots(figsize=(6.3, 2.3))
SEQ = ["#8fb8e8", "#2a78d6", "#123a6b"]
for i, d in enumerate(DEPTHS[1:]):
    w = sorted(100 * r["walk"] for r in rows if r["depth"] == d and r["coupling"] == 0.0)
    ax.step(w, [100 * (j + 1) / len(w) for j in range(len(w))], where="post", color=SEQ[i],
            ls=LS[i], lw=1.3, label=L["depth_n"].format(d))
ax.set_xlabel(L["spread_x"]); ax.set_ylabel(L["spread_y"])
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(pad=0.3)
save(fig, "fig_spread.png")
