"""The environment contract: solvability, exact restore, determinism, exact deltas."""

from __future__ import annotations

import random

from graphmend import STANDARD, generate_instance, generate_instance_at_depth


def test_generated_instance_is_broken_and_solvable() -> None:
    for seed in range(30):
        env = generate_instance(seed=seed, n_constraints=24, n_perturb=6)
        inst = env.instance
        assert inst is not None
        assert inst.initial_psi >= 1
        assert env.psi() == inst.initial_psi
        for action in env.solution_actions():
            env.apply(action)
        assert env.psi() == 0


def test_snapshot_restore_is_exact() -> None:
    env = generate_instance(seed=3, n_constraints=24, n_perturb=8)
    psi0, labels0, edges0 = env.psi(), list(env.labels), set(env.edges)
    sid = env.snapshot()
    for action in ("add_edge 0 1", "remove_edge 2 3", "relabel 4 A", "add_edge 5 6"):
        env.apply(action)
    env.restore(sid)
    assert (env.psi(), env.labels, env.edges) == (psi0, labels0, edges0)


def test_generation_is_deterministic() -> None:
    a = generate_instance(seed=11, n_constraints=20, coupling=0.5)
    b = generate_instance(seed=11, n_constraints=20, coupling=0.5)
    assert a.instance == b.instance


def test_delta_psi_is_exact_and_edit_count_is_standard() -> None:
    env, _ = generate_instance_at_depth(7, 8, coupling=0.4, n_perturb=6, **STANDARD)
    edits = list(env._candidate_edits())
    assert len(edits) == 72  # 8*7 edge edits + 8*2 relabels
    base = env.psi()
    for ed in edits:
        d = env._delta_psi(ed)
        env._apply_edit(ed)
        assert env.psi() == base + d
        env._undo_edit(ed)
    assert env.psi() == base


def test_guaranteed_actions_have_the_claimed_effect() -> None:
    rng = random.Random(0)
    for seed in range(20):
        env = generate_instance(seed=seed, n_constraints=24, n_perturb=6)
        p = env.psi()
        for want, check in (("descent", lambda q: q < p), ("plateau", lambda q: q == p)):
            a = getattr(env, f"guaranteed_{want}")(rng)
            if a is None:
                continue
            sid = env.snapshot()
            env.apply(a)
            assert check(env.psi())
            env.restore(sid)


def test_depth_scan_hits_target() -> None:
    for depth, npert in ((2, 2), (4, 3), (8, 6)):
        env, _ = generate_instance_at_depth(100 + depth, depth, coupling=0.0,
                                            n_perturb=npert, **STANDARD)
        assert env.psi() == depth
