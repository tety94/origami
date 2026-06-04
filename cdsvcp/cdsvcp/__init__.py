"""
cdsvcp — Combinatorial Edit Distance for Single-Vertex Crease Patterns
=======================================================================

Public API
----------
SVCP              data class for a crease pattern
ModCOp            named record for one Mod-C operation
kawasaki_deficit  κ(C)  (even m only)
kawasaki_residual K(C)  (all m ≥ 2)
is_flat_foldable  Kawasaki–Justin test
edit_distance     dC(C, F) ∈ {0, 1, 2}
repair            optimal repair (dC operations)
two_insert        Algorithm TwoInsert for even inputs with κ ≠ 0
single_insert     single insertion for odd inputs
insert            primitive crease insertion
delete            primitive crease deletion
"""

from .svcp import (
    SVCP,
    ModCOp,
    kawasaki_deficit,
    kawasaki_residual,
    is_flat_foldable,
    edit_distance,
    repair,
    two_insert,
    single_insert,
    insert,
    delete,
    sodd,
    seven,
)

__version__ = "0.1.0"
__all__ = [
    "SVCP",
    "ModCOp",
    "kawasaki_deficit",
    "kawasaki_residual",
    "is_flat_foldable",
    "edit_distance",
    "repair",
    "two_insert",
    "single_insert",
    "insert",
    "delete",
    "sodd",
    "seven",
]
