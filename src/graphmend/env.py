"""GraphMend: a synthetic constraint-repair environment with an exact residual.

The state is a directed graph without self-loops whose nodes carry labels, plus a
fixed set of declarative constraints. The residual (progress measure) is::

    psi = number of violated constraints

which is exact, cheap and integer. Every instance is generated from a hidden
reference solution, so a state with psi == 0 always exists. Actions are local
edits written as short strings, so a language model can emit them directly::

    add_edge U V        # add directed edge U -> V
    remove_edge U V     # remove directed edge U -> V
    relabel U X         # set node U's label to X (a letter A, B, ...)

Snapshot/restore is an exact deep copy keyed by id.

Difficulty is controlled by the instance depth (initial psi, via ``n_perturb`` and
``generate_instance_at_depth``) and by COUPLING: how concentrated constraints are
on a few shared "hub" nodes, so that repairing one constraint is more likely to
disturb another.

For programmatic action generators the environment can return a random action
with a guaranteed effect on psi (``guaranteed_descent``, ``guaranteed_plateau``,
``guaranteed_junk``) by enumerating the finite edit space.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, NamedTuple

# An action is a compact string (see the module docstring).
Action = str
# The residual value; integer in the base environment.
Psi = int


@dataclass(frozen=True)
class Obs:
    """What was observed after applying an action."""

    summary: str
    info: Mapping[str, Any] = field(default_factory=dict)


# A variable token: ("edge", u, v) or ("label", u). Heterogeneous, so typed loosely.
Var = tuple[object, ...]


class Edit(NamedTuple):
    """One local edit the brute-force search enumerates and apply()/undo work on.

    ``op`` in {"add", "del", "lbl"}. For edge ops ``x`` is the target node v; for
    ``lbl`` ``x`` is the new label and ``old`` the previous label (kept for undo).
    """

    op: str
    u: int
    x: int
    old: int = -1


def _label_char(idx: int) -> str:
    return chr(ord("A") + idx)


def _label_idx(ch: str) -> int:
    return ord(ch) - ord("A")


# --------------------------------------------------------------------------- #
# Constraints. Each exposes satisfied(env) and variables(); variables are opaque
# hashable tokens used for the coupling metric and the affected-constraint index.
# --------------------------------------------------------------------------- #
class Constraint:
    kind: str = "constraint"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:  # pragma: no cover
        raise NotImplementedError

    def variables(self) -> frozenset[Var]:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover
        raise NotImplementedError


@dataclass(frozen=True)
class RequiredEdge(Constraint):
    u: int
    v: int
    kind: str = "required_edge"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:
        return (self.u, self.v) in env.edges

    def variables(self) -> frozenset[Var]:
        return frozenset({("edge", self.u, self.v)})

    def describe(self) -> str:
        return f"edge {self.u}->{self.v} required"


@dataclass(frozen=True)
class ForbiddenEdge(Constraint):
    u: int
    v: int
    kind: str = "forbidden_edge"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:
        return (self.u, self.v) not in env.edges

    def variables(self) -> frozenset[Var]:
        return frozenset({("edge", self.u, self.v)})

    def describe(self) -> str:
        return f"edge {self.u}->{self.v} forbidden"


@dataclass(frozen=True)
class RequiredLabel(Constraint):
    u: int
    label: int
    kind: str = "required_label"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:
        return env.labels[self.u] == self.label

    def variables(self) -> frozenset[Var]:
        return frozenset({("label", self.u)})

    def describe(self) -> str:
        return f"node {self.u} must be labelled {_label_char(self.label)}"


@dataclass(frozen=True)
class OutDegreeAtMost(Constraint):
    u: int
    k: int
    n_nodes: int  # needed so variables() is self-contained
    kind: str = "out_degree_at_most"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:
        return env.outdeg[self.u] <= self.k

    def variables(self) -> frozenset[Var]:
        # References every out-edge of u: an edit to any u->w edge can change it.
        return frozenset(
            ("edge", self.u, w) for w in range(self.n_nodes) if w != self.u
        )

    def describe(self) -> str:
        return f"out-degree of node {self.u} <= {self.k}"


@dataclass(frozen=True)
class LabelImpliesEdge(Constraint):
    """If node u has label ``label`` then edge u->v must exist. Couples a label
    variable and an edge variable -- a primary source of coupling."""

    u: int
    label: int
    v: int
    kind: str = "label_implies_edge"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:
        return env.labels[self.u] != self.label or (self.u, self.v) in env.edges

    def variables(self) -> frozenset[Var]:
        return frozenset({("label", self.u), ("edge", self.u, self.v)})

    def describe(self) -> str:
        return (
            f"if node {self.u} is {_label_char(self.label)} "
            f"then edge {self.u}->{self.v} required"
        )


@dataclass(frozen=True)
class ForbiddenTwoPath(Constraint):
    """Forbids the directed 2-path u->v->w (a local acyclicity/path rule).
    Couples two edge variables."""

    u: int
    v: int
    w: int
    kind: str = "forbidden_two_path"

    def satisfied(self, env: "ConstraintRepairEnv") -> bool:
        return not ((self.u, self.v) in env.edges and (self.v, self.w) in env.edges)

    def variables(self) -> frozenset[Var]:
        return frozenset({("edge", self.u, self.v), ("edge", self.v, self.w)})

    def describe(self) -> str:
        return f"path {self.u}->{self.v}->{self.w} forbidden"


# --------------------------------------------------------------------------- #
# The environment.
# --------------------------------------------------------------------------- #
@dataclass
class Instance:
    """Metadata for a generated instance (the 'known solution' bundle)."""

    n_nodes: int
    n_labels: int
    solution_labels: list[int]
    solution_edges: frozenset[tuple[int, int]]
    initial_labels: list[int]
    initial_edges: frozenset[tuple[int, int]]
    constraints: tuple[Constraint, ...]
    initial_psi: int
    min_edits_to_solution: int  # Hamming distance initial -> solution
    realized_coupling: float  # measured fraction of constraint pairs sharing a var
    seed: int


class ConstraintRepairEnv:
    """A directed labelled graph with declarative constraints."""

    def __init__(
        self,
        n_nodes: int,
        n_labels: int,
        constraints: tuple[Constraint, ...],
        labels: list[int],
        edges: set[tuple[int, int]],
        instance: Instance | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_labels = n_labels
        self.constraints = constraints
        self.labels = list(labels)
        self.edges = set(edges)
        self.instance = instance
        self.outdeg = [0] * n_nodes
        for (u, _v) in self.edges:
            self.outdeg[u] += 1
        # Affected-constraint index: variable -> indices of constraints touching it.
        self._var_index: dict[Var, list[int]] = {}
        for ci, c in enumerate(self.constraints):
            for var in c.variables():
                self._var_index.setdefault(var, []).append(ci)
        self._snaps: dict[str, tuple[list[int], set[tuple[int, int]]]] = {}
        self._next_sid = 0

    # ---- psi: number of violated constraints ----
    def psi(self) -> int:
        return sum(1 for c in self.constraints if not c.satisfied(self))

    # ---- environment interface: snapshot / restore / apply ----
    def snapshot(self) -> str:
        sid = str(self._next_sid)
        self._next_sid += 1
        # Deep copy keyed by id: restore is exact.
        self._snaps[sid] = (list(self.labels), set(self.edges))
        return sid

    def restore(self, sid: str) -> None:
        labels, edges = self._snaps[sid]
        self.labels = list(labels)
        self.edges = set(edges)
        self.outdeg = [0] * self.n_nodes
        for (u, _v) in self.edges:
            self.outdeg[u] += 1

    def apply(self, action: Action) -> Obs:
        ok = self._apply_action_string(action)
        return Obs(
            summary=f"psi={self.psi()} after '{action}'",
            info={"action": action, "parsed_ok": ok},
        )

    def state_summary(self) -> str:
        lines = [
            f"Graph: {self.n_nodes} nodes, "
            f"labels {[_label_char(l) for l in self.labels]}"
        ]
        edge_str = ", ".join(f"{u}->{v}" for (u, v) in sorted(self.edges))
        lines.append(f"Edges: {edge_str if edge_str else '(none)'}")
        viol = [c.describe() for c in self.constraints if not c.satisfied(self)]
        lines.append(f"Violated constraints ({len(viol)}):")
        lines.extend(f"  - {d}" for d in viol)
        return "\n".join(lines)

    # ---- action string parsing / application ----
    def _apply_action_string(self, action: Action) -> bool:
        parts = action.split()
        try:
            if parts[0] == "add_edge":
                u, v = int(parts[1]), int(parts[2])
                if u == v or not self._valid_node(u) or not self._valid_node(v):
                    return False
                if (u, v) not in self.edges:
                    self.edges.add((u, v))
                    self.outdeg[u] += 1
                return True
            if parts[0] == "remove_edge":
                u, v = int(parts[1]), int(parts[2])
                if (u, v) in self.edges:
                    self.edges.discard((u, v))
                    self.outdeg[u] -= 1
                return True
            if parts[0] == "relabel":
                u = int(parts[1])
                new = _label_idx(parts[2])
                if not self._valid_node(u) or not (0 <= new < self.n_labels):
                    return False
                self.labels[u] = new
                return True
        except (IndexError, ValueError):
            return False
        return False

    def _valid_node(self, u: int) -> bool:
        return 0 <= u < self.n_nodes

    # ---- internal edit primitives (used by the brute-force search) ----
    def _apply_edit(self, edit: Edit) -> None:
        if edit.op == "add":
            self.edges.add((edit.u, edit.x))
            self.outdeg[edit.u] += 1
        elif edit.op == "del":
            self.edges.discard((edit.u, edit.x))
            self.outdeg[edit.u] -= 1
        else:  # "lbl"
            self.labels[edit.u] = edit.x

    def _undo_edit(self, edit: Edit) -> None:
        if edit.op == "add":
            self.edges.discard((edit.u, edit.x))
            self.outdeg[edit.u] -= 1
        elif edit.op == "del":
            self.edges.add((edit.u, edit.x))
            self.outdeg[edit.u] += 1
        else:  # "lbl"
            self.labels[edit.u] = edit.old

    def _edit_var(self, edit: Edit) -> Var:
        if edit.op in ("add", "del"):
            return ("edge", edit.u, edit.x)
        return ("label", edit.u)

    def _delta_psi(self, edit: Edit) -> int:
        """Exact change in psi if ``edit`` were applied. Only re-checks the
        constraints touching the edited variable (the affected-constraint index),
        which is why psi and its deltas stay cheap regardless of instance size."""
        affected = self._var_index.get(self._edit_var(edit), ())
        before = sum(1 for ci in affected if not self.constraints[ci].satisfied(self))
        self._apply_edit(edit)
        after = sum(1 for ci in affected if not self.constraints[ci].satisfied(self))
        self._undo_edit(edit)
        return after - before

    def _candidate_edits(self) -> Iterator[Edit]:
        """Every single local edit available from the current state."""
        for u in range(self.n_nodes):
            for v in range(self.n_nodes):
                if u == v:
                    continue
                if (u, v) in self.edges:
                    yield Edit("del", u, v)
                else:
                    yield Edit("add", u, v)
        for u in range(self.n_nodes):
            cur = self.labels[u]
            for lab in range(self.n_labels):
                if lab != cur:
                    yield Edit("lbl", u, lab, cur)

    def _is_corrective(self, edit: Edit) -> bool:
        """Whether the edit moves a variable toward the known solution value.
        Preferring corrective descents/plateaus keeps instances solvable while
        still letting coupling create plateaus and stalls."""
        if self.instance is None:
            return False
        if edit.op == "add":
            return (edit.u, edit.x) in self.instance.solution_edges
        if edit.op == "del":
            return (edit.u, edit.x) not in self.instance.solution_edges
        return edit.x == self.instance.solution_labels[edit.u]

    def _edit_to_action(self, edit: Edit) -> Action:
        if edit.op == "add":
            return f"add_edge {edit.u} {edit.x}"
        if edit.op == "del":
            return f"remove_edge {edit.u} {edit.x}"
        return f"relabel {edit.u} {_label_char(edit.x)}"

    # ---- guaranteed-effect actions for programmatic generators ----
    def _search(self, want: str, rng: random.Random) -> Action | None:
        """Return a random edit whose psi-delta matches ``want`` in
        {"descent" (<0), "plateau" (==0), "junk" (>0)}, preferring corrective
        edits for descent/plateau so the known solution stays reachable."""
        preferred: list[Edit] = []
        fallback: list[Edit] = []
        for edit in self._candidate_edits():
            d = self._delta_psi(edit)
            match = (
                (want == "descent" and d < 0)
                or (want == "plateau" and d == 0)
                or (want == "junk" and d > 0)
            )
            if not match:
                continue
            if want in ("descent", "plateau") and self._is_corrective(edit):
                preferred.append(edit)
            else:
                fallback.append(edit)
        pool = preferred if preferred else fallback
        if not pool:
            return None
        return self._edit_to_action(rng.choice(pool))

    def guaranteed_descent(self, rng: random.Random) -> Action | None:
        return self._search("descent", rng)

    def guaranteed_plateau(self, rng: random.Random) -> Action | None:
        return self._search("plateau", rng)

    def guaranteed_junk(self, rng: random.Random) -> Action:
        action = self._search("junk", rng)
        if action is not None:
            return action
        # No psi-increasing edit exists (state is nearly all-satisfied): return a
        # syntactically valid but non-descending edit so the proposal is rejected.
        for edit in self._candidate_edits():
            if self._delta_psi(edit) >= 0:
                return self._edit_to_action(edit)
        # Degenerate: only descending edits remain; emit one (will be accepted).
        return self._edit_to_action(next(iter(self._candidate_edits())))

    # ---- reference: the known solution path ----
    def solution_actions(self) -> list[Action]:
        """A concrete edit sequence from the current state to psi == 0 (sets every
        variable to its solution value). Not necessarily monotone in psi -- it is
        the *existence* proof of a solution, used in tests and demos."""
        assert self.instance is not None
        acts: list[Action] = []
        for u in range(self.n_nodes):
            if self.labels[u] != self.instance.solution_labels[u]:
                acts.append(
                    f"relabel {u} {_label_char(self.instance.solution_labels[u])}"
                )
        for (u, v) in sorted(self.instance.solution_edges - self.edges):
            acts.append(f"add_edge {u} {v}")
        for (u, v) in sorted(self.edges - self.instance.solution_edges):
            acts.append(f"remove_edge {u} {v}")
        return acts


class ConstraintRepairMacroVar:
    """psi = number of violated constraints, as a callable-object interface."""

    def value(self, env: "ConstraintRepairEnv") -> Psi:
        assert isinstance(env, ConstraintRepairEnv)
        return env.psi()


# --------------------------------------------------------------------------- #
# Instance generator: random SOLVABLE instances with a known solution.
# --------------------------------------------------------------------------- #
def _realized_coupling(constraints: tuple[Constraint, ...]) -> float:
    """Fraction of unordered constraint pairs sharing at least one variable."""
    varsets = [c.variables() for c in constraints]
    n = len(varsets)
    if n < 2:
        return 0.0
    shared = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if varsets[i] & varsets[j]:
                shared += 1
    return shared / total if total else 0.0


def generate_instance(
    seed: int,
    n_nodes: int = 8,
    n_labels: int = 3,
    n_constraints: int = 20,
    edge_density: float = 0.25,
    coupling: float = 0.4,
    n_perturb: int = 6,
) -> ConstraintRepairEnv:
    """Generate a random solvable constraint-repair instance.

    A random solution assignment is drawn first; every constraint is built to be
    satisfied by it (so psi(solution) == 0 by construction). The initial state
    is the solution with ``n_perturb`` chosen
    constraints deliberately violated, so initial psi ~ n_perturb regardless of
    coupling. ``coupling`` in [0, 1] concentrates constraints on fewer hub nodes,
    raising the fraction of constraint pairs that share a variable (the realized
    coupling fraction, measured and stored on the instance).

    Determinism: fully determined by ``seed`` and the knobs.
    """
    rng = random.Random(seed)

    # --- solution assignment ---
    sol_labels = [rng.randrange(n_labels) for _ in range(n_nodes)]
    sol_edges: set[tuple[int, int]] = set()
    for u in range(n_nodes):
        for v in range(n_nodes):
            if u != v and rng.random() < edge_density:
                sol_edges.add((u, v))
    sol_outdeg = [0] * n_nodes
    for (u, _v) in sol_edges:
        sol_outdeg[u] += 1

    # A constraints-less env over the solution state, used to validate that each
    # generated constraint is satisfied by the solution.
    sol_env = ConstraintRepairEnv(n_nodes, n_labels, (), sol_labels, sol_edges)

    # Coupling concentrates constraints on a shrinking set of HUB nodes. Edge
    # constraints route their source node to a hub with probability ``coupling``,
    # and every hub carries a (tight) out-degree constraint that shares a variable
    # with each edge sourced at that hub -- the bridge that makes many constraint
    # pairs conflict. Higher coupling -> fewer hubs -> more edges per hub ->
    # fixing one constraint more often breaks another (the out-degree cap and the
    # required edges on a hub fight over the same out-edges). Constraint COUNT is
    # preserved because non-hub picks still supply variety. At coupling==0 sources
    # are uniform and hubs many, so conflicts are rare.
    n_hubs = max(1, round(n_nodes * (1.0 - coupling)))
    hubs = sorted(rng.sample(range(n_nodes), min(n_hubs, n_nodes)))

    def route_source(default: int) -> int:
        return rng.choice(hubs) if rng.random() < coupling else default

    sol_by_source: dict[int, list[int]] = {}
    for (u, v) in sol_edges:
        sol_by_source.setdefault(u, []).append(v)
    nonsol_by_source: dict[int, list[int]] = {}
    for u in range(n_nodes):
        outs = [v for v in range(n_nodes) if v != u and (u, v) not in sol_edges]
        if outs:
            nonsol_by_source[u] = outs

    # --- constraints, each satisfied by the solution ---
    constraints: list[Constraint] = []
    # One tight out-degree constraint per hub (the coupling bridge).
    for u in hubs:
        constraints.append(OutDegreeAtMost(u, sol_outdeg[u], n_nodes))

    kinds = [
        "required_edge",
        "forbidden_edge",
        "required_label",
        "label_implies_edge",
        "forbidden_two_path",
    ]
    guard = 0
    while len(constraints) < n_constraints and guard < n_constraints * 400:
        guard += 1
        kind = rng.choice(kinds)
        c: Constraint | None = None
        if kind == "required_edge" and sol_by_source:
            u = route_source(rng.choice(list(sol_by_source)))
            if u in sol_by_source:
                c = RequiredEdge(u, rng.choice(sol_by_source[u]))
        elif kind == "forbidden_edge" and nonsol_by_source:
            u = route_source(rng.choice(list(nonsol_by_source)))
            if u in nonsol_by_source:
                c = ForbiddenEdge(u, rng.choice(nonsol_by_source[u]))
        elif kind == "required_label":
            u = route_source(rng.randrange(n_nodes))
            c = RequiredLabel(u, sol_labels[u])
        elif kind == "label_implies_edge" and sol_by_source:
            u = route_source(rng.choice(list(sol_by_source)))
            if u in sol_by_source:
                c = LabelImpliesEdge(u, sol_labels[u], rng.choice(sol_by_source[u]))
        elif kind == "forbidden_two_path":
            mid = route_source(rng.randrange(n_nodes))
            u, w = rng.randrange(n_nodes), rng.randrange(n_nodes)
            if len({u, mid, w}) == 3 and not (
                (u, mid) in sol_edges and (mid, w) in sol_edges
            ):
                c = ForbiddenTwoPath(u, mid, w)
        if c is not None and c not in constraints and c.satisfied(sol_env):
            constraints.append(c)

    constraints_t = tuple(constraints)

    # --- perturb the solution into the initial state ---
    # Targeted perturbation: violate ``n_perturb`` randomly chosen constraints by
    # editing one of each one's variables. This fixes the initial violation count
    # (roughly n_perturb, up to coupling-induced overlap) INDEPENDENTLY of the
    # coupling knob, so coupling isolates *repair difficulty* -- with shared
    # variables, undoing one violation tends to (re)break another, which is what
    # creates plateaus and dead ends. Untargeted random flips would instead
    # entangle initial psi with coupling and confound the ablation.
    init_labels = list(sol_labels)
    init_edges = set(sol_edges)

    def violate(c: Constraint) -> None:
        if isinstance(c, RequiredEdge):
            init_edges.discard((c.u, c.v))
        elif isinstance(c, ForbiddenEdge):
            init_edges.add((c.u, c.v))
        elif isinstance(c, RequiredLabel):
            init_labels[c.u] = (c.label + 1) % n_labels
        elif isinstance(c, OutDegreeAtMost):
            # Add out-edges from u until the cap is exceeded.
            for w in range(n_nodes):
                if init_edges and sum(1 for (x, _y) in init_edges if x == c.u) > c.k:
                    break
                if w != c.u and (c.u, w) not in init_edges:
                    init_edges.add((c.u, w))
        elif isinstance(c, LabelImpliesEdge):
            init_labels[c.u] = c.label
            init_edges.discard((c.u, c.v))
        elif isinstance(c, ForbiddenTwoPath):
            init_edges.add((c.u, c.v))
            init_edges.add((c.v, c.w))

    targets = rng.sample(constraints, min(n_perturb, len(constraints)))
    for c in targets:
        violate(c)

    def psi_of(labels: list[int], edges: set[tuple[int, int]]) -> int:
        return ConstraintRepairEnv(
            n_nodes, n_labels, constraints_t, labels, edges
        ).psi()

    # Guard against the rare case where overlaps cancel to psi == 0.
    if psi_of(init_labels, init_edges) == 0 and constraints:
        violate(rng.choice(constraints))

    initial_psi = psi_of(init_labels, init_edges)
    min_edits = sum(1 for u in range(n_nodes) if init_labels[u] != sol_labels[u])
    min_edits += len(set(sol_edges) ^ set(init_edges))

    instance = Instance(
        n_nodes=n_nodes,
        n_labels=n_labels,
        solution_labels=sol_labels,
        solution_edges=frozenset(sol_edges),
        initial_labels=list(init_labels),
        initial_edges=frozenset(init_edges),
        constraints=constraints_t,
        initial_psi=initial_psi,
        min_edits_to_solution=min_edits,
        realized_coupling=_realized_coupling(constraints_t),
        seed=seed,
    )
    return ConstraintRepairEnv(
        n_nodes, n_labels, constraints_t, init_labels, init_edges, instance=instance
    )


def generate_instance_at_depth(
    base_seed: int,
    target_psi: int,
    *,
    max_scan: int = 1000,
    n_nodes: int = 8,
    n_labels: int = 3,
    n_constraints: int = 20,
    edge_density: float = 0.25,
    coupling: float = 0.4,
    n_perturb: int = 6,
) -> tuple[ConstraintRepairEnv, int]:
    """Deterministically scan seeds until an instance has initial_psi == target_psi.

    Fixes the initial depth psi(s0) so that difficulty-by-count is held constant
    across, e.g., a coupling sweep (only the constraint structure varies). Scans
    ``s = base_seed*max_scan + attempt`` for attempt in [0, max_scan).
    Returns (env, chosen_seed).
    """
    for attempt in range(max_scan):
        s = base_seed * max_scan + attempt
        env = generate_instance(
            seed=s,
            n_nodes=n_nodes,
            n_labels=n_labels,
            n_constraints=n_constraints,
            edge_density=edge_density,
            coupling=coupling,
            n_perturb=n_perturb,
        )
        assert env.instance is not None
        if env.instance.initial_psi == target_psi:
            return env, s
    raise RuntimeError(
        f"no instance with initial_psi={target_psi} within {max_scan} seeds "
        f"from base {base_seed} (coupling={coupling})"
    )
