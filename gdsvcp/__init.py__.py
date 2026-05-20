"""
gdsvcp – Geometric Edit Distance for Single-Vertex Crease Patterns under L².
"""

from gdsvcp.core import (
    weight_vector,
    compute_kappa,
    validate_svcp,
    PositivityViolation,
    ClippingInfeasible,
)
from gdsvcp.l2 import l2_minimiser, dG_l2
from gdsvcp.clipping import clipped_project
from gdsvcp.norms import l1_minimisers, linfty_minimiser

__all__ = [
    "weight_vector",
    "compute_kappa",
    "validate_svcp",
    "PositivityViolation",
    "ClippingInfeasible",
    "l2_minimiser",
    "dG_l2",
    "clipped_project",
    "l1_minimisers",
    "linfty_minimiser",
]