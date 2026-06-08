"""
cdmsvcp — Edit Distance to Full Flat-Foldability at a Single Vertex
====================================================================
(Kawasaki + Maekawa simultaneously)

Public API
----------
LSVCP                  data class for a labelled single-vertex crease pattern
LabelledOp             record for one edit operation (insertion or flip)
M, V                   label constants ('M', 'V')
mountain_count         number of mountain creases
valley_count           number of valley creases
signed_delta           δ = M − V
maekawa_deficit        ν = |δ| − 2  (even m only)
is_fully_flat_foldable κ = 0 and ν = 0 test
edit_distance          d_MV((C,μ), F_MV) — exact formula
repair                 Algorithm FullRepair — optimal repair
lsvcp_insert           primitive labelled crease insertion
lsvcp_flip             primitive label flip

Re-exported from cdsvcp (geometry layer)
-----------------------------------------
SVCP                   single-vertex crease pattern (angles only)
kawasaki_deficit       κ(C) = S_odd − π  (even m)
kawasaki_residual      K(C) = S_odd − π  (all m)
is_flat_foldable       Kawasaki–Justin test (geometry only)
insert                 raw crease insertion (geometry only)
"""

from .lsvcp import (
    LSVCP,
    LabelledOp,
    M,
    V,
    mountain_count,
    valley_count,
    signed_delta,
    maekawa_deficit,
    is_fully_flat_foldable,
    edit_distance,
    repair,
    full_repair,
    lsvcp_insert,
    lsvcp_flip,
)

from cdsvcp import (
    SVCP,
    kawasaki_deficit,
    kawasaki_residual,
    is_flat_foldable,
    insert,
)

__version__ = "0.1.0"
__all__ = [
    # LSVCP layer
    "LSVCP",
    "LabelledOp",
    "M",
    "V",
    "mountain_count",
    "valley_count",
    "signed_delta",
    "maekawa_deficit",
    "is_fully_flat_foldable",
    "edit_distance",
    "repair",
    "full_repair",
    "lsvcp_insert",
    "lsvcp_flip",
    # Geometry layer (re-exported from cdsvcp)
    "SVCP",
    "kawasaki_deficit",
    "kawasaki_residual",
    "is_flat_foldable",
    "insert",
]
