# GraphMend

A synthetic test bed for closed loops in which actions are proposed by an external
stochastic generator, such as a language model, and a controller applies each
action, measures the result and keeps or reverts the change. GraphMend is built so
that the assumptions behind theoretical guarantees for such loops can be checked
**exactly**, instance by instance, rather than estimated from run statistics.

- **State:** a directed graph without self-loops on 8 nodes, each node labelled with
  one of 3 labels.
- **Actions:** add or remove one arc, or relabel one node — exactly 72 actions in
  every state, written as short strings (`add_edge U V`, `remove_edge U V`,
  `relabel U X`).
- **Goal:** satisfy 26 declarative constraints of six kinds (required arc, forbidden
  arc, required label, upper bound on the number of outgoing arcs, "label implies
  arc", forbidden two-arc path through a node).
- **Residual:** ψ = number of violated constraints — integer, exact and cheap.
  Every instance is generated from a hidden reference solution, so ψ = 0 is always
  reachable and ψ = 0 means the task is really solved.
- **Exact restore:** snapshot/restore is a deep copy.
- **Difficulty knobs:** instance depth ψ(s₀) and coupling c ∈ [0, 1] (how strongly
  constraints are concentrated on a few shared nodes).

GraphMend was used under the label E1 in
M. Polyak, *Contraction Contracts: Enforced Progress Guarantees for Black-Box Agent
Loops*, preprint, 2026, DOI: [10.5281/zenodo.22173004](https://doi.org/10.5281/zenodo.22173004).

## Install

```bash
pip install -e .                 # the package (standard library only)
pip install -e ".[test,figures]" # + pytest and matplotlib
```

Python 3.10+.

## Quick start

```python
from graphmend import STANDARD, N_PERTURB, generate_instance_at_depth, analyse

env, seed = generate_instance_at_depth(7400130, 4, coupling=0.0,
                                       n_perturb=N_PERTURB[4], **STANDARD)
print(env.state_summary())       # text an LLM can read
sid = env.snapshot()
env.apply("add_edge 0 1")        # an action an LLM can write
print(env.psi())
env.restore(sid)                 # exact

print(analyse(env))              # exact structural analysis of this instance
# {'psi0': 4, 'n_edits': 72, 'k0': 3, ..., 'R': 12, 'dead': 2,
#  'viable0': True, 'trap': 0.25, 'esc': 1.0, 'walk': 0.5, 'capped': False}
```

## What is computed exactly

For integer ψ, an edit is *improving* if it strictly decreases ψ; a state is *dead*
if it has no improving edit. The *monotone-descent graph* of an instance consists of
the states reachable from s₀ by improving edits; R is its set of states with ψ > 0.
A state of R is *viable* if ψ = 0 is reachable from it by improving edits and a
*trap* otherwise. `graphmend.analysis` computes, for every instance:

- whether a given state is dead (`is_dead`);
- R, viability of every state of R and the share of traps (`descent_graph`,
  `analyse`);
- the probability that the *blind descent* — a random walk taking a uniformly
  chosen improving edit at every step — gets stuck before ψ = 0 (exact dynamic
  programming over the acyclic graph);
- the share of dead states from which one *plateau* edit (keeping ψ) leads to a
  state with an improving edit.

Additional tools:

- `WeightedResidual` — real-valued residuals (uniform or geometric constraint
  weights) and additive Gaussian measurement noise;
- `CompetentWithSlips` — a programmatic action generator with an exactly known
  proposal law and exact `capability` (probability of proposing a good edit);
- `env.guaranteed_descent / guaranteed_plateau / guaranteed_junk` — a random edit
  with the requested effect on ψ, for scripted generators. Descent and plateau
  edits that move the state toward the hidden reference solution are preferred
  when they exist, so these queries use knowledge of the solution that a language
  model does not have; they are **not** the blind descent (use
  `graphmend.analysis` for that).

## Reproducing the published structural data

```bash
python scripts/enumerate_structure.py   # 2400 instances, ~2 min on 10 cores of an Apple M2 Pro -> data/structure.json
python scripts/summarize.py             # summary table
python scripts/figures.py figures       # figures (add --lang ru for Russian labels)
pytest                                  # includes an exact check against data/structure.json
```

Standard grid: depths 2, 4, 8, 12 (n_perturb = 2, 3, 6, 10); coupling 0.0, 0.4,
0.8; 200 instances per cell. The instance for (depth d, coupling c, index i) is
`generate_instance_at_depth(7_000_000 + 100_000*d + 10_000*round(10*c) + i, d, ...)`.
`data/structure.json` holds one record per instance.

## Citation

See `CITATION.cff`. [![DOI](https://zenodo.org/badge/1451433924.svg)](https://doi.org/10.5281/zenodo.24085091)

## License

MIT.
