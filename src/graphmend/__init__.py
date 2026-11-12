"""GraphMend: a test bed with an exactly computable residual for agent control loops."""

from __future__ import annotations

from .analysis import analyse, descent_graph, is_dead, plateau_reachability, state_key, set_state
from .env import (
    Action,
    ConstraintRepairEnv,
    ConstraintRepairMacroVar,
    Edit,
    Instance,
    Obs,
    generate_instance,
    generate_instance_at_depth,
)
from .generators import CompetentWithSlips, integer_candidates
from .weighted import WeightedResidual

# The standard configuration used in the papers.
STANDARD = {"n_nodes": 8, "n_labels": 3, "n_constraints": 26}
# n_perturb used to reach each standard depth.
N_PERTURB = {2: 2, 4: 3, 8: 6, 12: 10}

__version__ = "1.2.0"

__all__ = [
    "Action", "CompetentWithSlips", "ConstraintRepairEnv", "ConstraintRepairMacroVar",
    "Edit", "Instance", "N_PERTURB", "Obs", "STANDARD", "WeightedResidual", "analyse",
    "descent_graph", "generate_instance", "generate_instance_at_depth",
    "integer_candidates", "is_dead", "plateau_reachability", "set_state", "state_key",
]
