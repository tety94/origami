"""
gdsvcp – Geometric Edit Distance for Single-Vertex Crease Patterns.

Implements the results of:
  "How Far Is a Single-Vertex Crease Pattern from Flat Foldability?
   Minimal Angular Corrections under Three Norms"

Quick reference
---------------
    compute_kappa(alpha)          Kawasaki deficit κ  (radians)
    dG_l2(alpha)                  L² distance |κ|√(2/n)
    l2_minimiser(alpha)           unconstrained L² minimiser δ*
    clipped_project(alpha)        constrained L² minimiser (Algorithm 4.1)
    linfty_minimiser(alpha)       unique L∞ minimiser (= L² minimiser)
    l1_minimisers(alpha)          all n² extreme L¹ minimisers
    l1_feasible_minimisers(alpha) feasibility-filtered L¹ minimisers
    positivity_check(alpha)       exact / sufficient positivity report

All functions expect angles in **radians**.
"""

from gdsvcp.core import (
    weight_vector,
    compute_kappa,
    validate_svcp,
    PositivityViolation,
    ClippingInfeasible,
)
from gdsvcp.l2 import l2_minimiser, dG_l2, positivity_check
from gdsvcp.clipping import clipped_project
from gdsvcp.norms import (
    linfty_minimiser,
    l1_minimisers,
    l1_feasible_minimisers,
    enumerate_l1_minimiser_polytope,
)

__all__ = [
    # core
    "weight_vector",
    "compute_kappa",
    "validate_svcp",
    "PositivityViolation",
    "ClippingInfeasible",
    # L²
    "l2_minimiser",
    "dG_l2",
    "positivity_check",
    # constrained L²
    "clipped_project",
    # L∞ and L¹
    "linfty_minimiser",
    "l1_minimisers",
    "l1_feasible_minimisers",
    "enumerate_l1_minimiser_polytope",
]