"""Exact analysis of one GraphMend instance by exhaustive enumeration.

For an instance with integer psi, the monotone-descent graph from s0 is built:
its vertices are the states reachable from s0 by edits that strictly decrease psi
("improving" edits), its arcs are those edits. R is the set of its states with
psi > 0. Definitions:

  dead      a state with no improving edit;
  viable    a state of R from which psi == 0 is reachable by improving edits;
  trap      a state of R that is not viable (every dead state is a trap);
  blind     the blind descent: from every state, take an improving edit chosen
            uniformly at random; its probability of getting stuck before psi == 0
            is computed exactly by dynamic programming over the acyclic graph;
  plateau   an edit that keeps psi; ``esc`` is the share of dead states from which
            one plateau edit leads to a state with an improving edit.

``analyse`` returns these quantities for one environment; ``scripts/enumerate.py``
runs it over the standard grid of instances.
"""

from __future__ import annotations

from typing import Hashable

from .env import ConstraintRepairEnv

StateKey = tuple[tuple[int, ...], frozenset[tuple[int, int]]]


def state_key(env: ConstraintRepairEnv) -> StateKey:
    """A hashable key of the current state (labels and edge set)."""
    return (tuple(env.labels), frozenset(env.edges))


def set_state(env: ConstraintRepairEnv, k: StateKey) -> None:
    """Put ``env`` into the state with key ``k``."""
    env.labels = list(k[0])
    env.edges = set(k[1])
    env.outdeg = [0] * env.n_nodes
    for (u, _v) in env.edges:
        env.outdeg[u] += 1


def is_dead(env: ConstraintRepairEnv) -> bool:
    """True if no edit strictly decreases psi in the current state."""
    return not any(env._delta_psi(ed) < 0 for ed in list(env._candidate_edits()))


def descent_graph(
    env: ConstraintRepairEnv, cap: int | None = None
) -> tuple[dict[Hashable, list[Hashable]], dict[Hashable, int], dict[Hashable, int], bool]:
    """Enumerate the monotone-descent graph from the current state of ``env``.

    Returns ``(kids, psi, plateau, capped)``: ``kids[k]`` lists the states reached
    from state ``k`` by improving edits (expanded only for states with psi > 0, so
    the keys of ``kids`` are exactly R); ``psi`` maps every visited state, including
    goal states, to its residual; ``plateau[k]`` counts edits that keep psi at ``k``;
    ``capped`` is True if enumeration stopped after ``cap`` states. The state of
    ``env`` is left at an arbitrary visited state.
    """
    s0 = state_key(env)
    kids: dict[Hashable, list[Hashable]] = {}
    psi: dict[Hashable, int] = {s0: env.psi()}
    plateau: dict[Hashable, int] = {}
    stack, capped = [s0], False
    while stack:
        k = stack.pop()
        if k in kids:
            continue
        set_state(env, k)  # type: ignore[arg-type]
        p = psi[k]
        ch, pl = [], 0
        for ed in list(env._candidate_edits()):
            d = env._delta_psi(ed)
            if d < 0:
                env._apply_edit(ed)
                kk = state_key(env)
                env._undo_edit(ed)
                psi[kk] = p + d
                ch.append(kk)
                if p + d > 0 and kk not in kids:
                    stack.append(kk)
            elif d == 0:
                pl += 1
        kids[k] = ch
        plateau[k] = pl
        if cap is not None and len(kids) > cap:
            capped = True
            break
    return kids, psi, plateau, capped


def viability_and_blind_descent(
    kids: dict[Hashable, list[Hashable]], psi: dict[Hashable, int]
) -> tuple[dict[Hashable, bool], dict[Hashable, float]]:
    """Viability of every state of R and the probability that the blind descent
    started there gets stuck, by one pass over R in increasing psi (the graph is
    acyclic because every arc strictly decreases psi)."""
    viable: dict[Hashable, bool] = {}
    stuck: dict[Hashable, float] = {}
    for k in sorted(kids, key=lambda k: psi[k]):
        ch = kids[k]
        if not ch:
            viable[k], stuck[k] = False, 1.0
            continue
        viable[k] = any(psi[c] == 0 or viable.get(c, False) for c in ch)
        stuck[k] = sum(0.0 if psi[c] == 0 else stuck.get(c, 1.0) for c in ch) / len(ch)
    return viable, stuck


def plateau_escape(env: ConstraintRepairEnv, dead: list[Hashable]) -> int:
    """Number of dead states from which one plateau edit leads to a state that has
    an improving edit."""
    esc = 0
    for k in dead:
        set_state(env, k)  # type: ignore[arg-type]
        ok = False
        for ed in list(env._candidate_edits()):
            if env._delta_psi(ed) == 0:
                env._apply_edit(ed)
                if any(env._delta_psi(e2) < 0 for e2 in list(env._candidate_edits())):
                    ok = True
                env._undo_edit(ed)
                if ok:
                    break
        esc += ok
    return esc


def analyse(env: ConstraintRepairEnv, cap: int | None = 300_000) -> dict:
    """All exact per-instance quantities, from the current state of ``env`` as s0.

    Keys: ``psi0``, ``n_edits``, ``k0`` (improving edits at s0), ``plateau0``,
    ``dead0``, ``R`` (|R|), ``dead`` (dead states in R), ``viable0``, ``trap``
    (share of traps in R), ``esc`` (share of dead states with a plateau escape, or
    None if there are none), ``walk`` (probability that the blind descent from s0
    gets stuck), ``capped``. The state of ``env`` is restored to s0 on return.
    """
    s0 = state_key(env)
    psi0 = env.psi()
    kids, psi, plateau, capped = descent_graph(env, cap)
    viable, stuck = viability_and_blind_descent(kids, psi)
    dead = [k for k in kids if not kids[k]]
    esc = plateau_escape(env, dead)
    n_edits = sum(1 for _ in env._candidate_edits())
    set_state(env, s0)
    return {
        "psi0": psi0, "n_edits": n_edits, "k0": len(kids[s0]), "plateau0": plateau[s0],
        "dead0": not kids[s0], "R": len(kids), "dead": len(dead),
        "viable0": viable[s0], "trap": sum(not viable[k] for k in kids) / len(kids),
        "esc": esc / len(dead) if dead else None, "walk": stuck[s0], "capped": capped,
    }
