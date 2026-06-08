"""
tests/test_cdmsvcp.py
=====================
Unit tests for the cdmsvcp package.

Covers:
  - LSVCP data class (construction, validation)
  - mountain_count / valley_count / signed_delta
  - maekawa_deficit
  - is_fully_flat_foldable
  - edit_distance — all cases A, B, C, D, E
  - lsvcp_insert / lsvcp_flip primitives
  - full_repair / repair — all cases
  - CLI smoke tests

All worked examples are taken directly from the LaTeX paper,
Sections 3.1–3.5 (Cases A–E).
"""

from __future__ import annotations

import pytest
from fractions import Fraction

from cdsvcp import SVCP

from cdmsvcp import (
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
    kawasaki_deficit,
    is_flat_foldable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deg(x) -> Fraction:
    """Convert integer degrees to fraction-of-π."""
    return Fraction(x, 180)


def make_svcp(*degrees) -> SVCP:
    """Build SVCP from degree values (must sum to 360)."""
    return SVCP(tuple(deg(d) for d in degrees))


def make(*degrees_and_labels) -> LSVCP:
    """
    Build LSVCP.  Call as make(deg1, deg2, …, label1, label2, …)
    where first half are ints (degrees) and second half are 'M'/'V'.
    """
    # Split evenly
    n = len(degrees_and_labels) // 2
    degrees = degrees_and_labels[:n]
    labels  = degrees_and_labels[n:]
    return LSVCP(make_svcp(*degrees), tuple(labels))


def lsvcp(degrees, labels) -> LSVCP:
    """Build LSVCP from a list of degrees and list of labels."""
    return LSVCP(make_svcp(*degrees), tuple(labels))


# ---------------------------------------------------------------------------
# LSVCP validation
# ---------------------------------------------------------------------------

class TestLSVCPValidation:
    def test_valid_4(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert len(lc) == 4

    def test_invalid_label_count(self):
        with pytest.raises(ValueError, match="labels length"):
            LSVCP(make_svcp(90, 90, 90, 90), (M, M, M))

    def test_invalid_label_value(self):
        with pytest.raises(ValueError, match="Invalid label"):
            LSVCP(make_svcp(90, 90, 90, 90), (M, M, M, 'X'))

    def test_repr(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert "MMMV" in repr(lc)

    def test_angles_property(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert lc.angles == lc.svcp.angles


# ---------------------------------------------------------------------------
# Count / delta / deficit helpers
# ---------------------------------------------------------------------------

class TestCounts:
    def test_mountain_count(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert mountain_count(lc) == 3

    def test_valley_count(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert valley_count(lc) == 1

    def test_signed_delta_positive(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert signed_delta(lc) == 2

    def test_signed_delta_zero(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, V, V])
        assert signed_delta(lc) == 0

    def test_signed_delta_negative(self):
        lc = lsvcp([90, 90, 90, 90], [M, V, V, V])
        assert signed_delta(lc) == -2

    def test_maekawa_deficit_zero(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert maekawa_deficit(lc) == 0

    def test_maekawa_deficit_minus2(self):
        # δ = 0 → ν = −2
        lc = lsvcp([90, 90, 90, 90], [M, M, V, V])
        assert maekawa_deficit(lc) == -2

    def test_maekawa_deficit_plus2(self):
        # δ = 4 → ν = 2
        lc = lsvcp([90, 90, 90, 90], [M, M, M, M])
        assert maekawa_deficit(lc) == 2

    def test_maekawa_deficit_odd_raises(self):
        lc = lsvcp([120, 120, 120], [M, M, V])
        with pytest.raises(ValueError, match="even crease count"):
            maekawa_deficit(lc)


# ---------------------------------------------------------------------------
# is_fully_flat_foldable
# ---------------------------------------------------------------------------

class TestFullyFlatFoldable:
    def test_already_foldable(self):
        # κ=0, ν=0
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert is_fully_flat_foldable(lc)

    def test_kappa_wrong(self):
        # κ ≠ 0
        lc = lsvcp([120, 80, 100, 60], [M, M, M, V])
        assert not is_fully_flat_foldable(lc)

    def test_maekawa_wrong(self):
        # κ=0, ν≠0
        lc = lsvcp([90, 90, 90, 90], [M, M, M, M])
        assert not is_fully_flat_foldable(lc)

    def test_odd_m(self):
        lc = lsvcp([120, 120, 120], [M, M, V])
        assert not is_fully_flat_foldable(lc)


# ---------------------------------------------------------------------------
# edit_distance — all cases
# ---------------------------------------------------------------------------

class TestEditDistance:
    # Case A
    def test_case_A(self):
        # Paper Example A: κ=0, ν=0 → d=0
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        assert edit_distance(lc) == 0

    # Case B: ν = 2 (δ = 4)
    def test_case_B_nu2(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, M])
        assert maekawa_deficit(lc) == 2
        assert edit_distance(lc) == 1    # |ν|/2 = 1

    # Case B: ν = −2 (δ = 0)
    def test_case_B_nu_minus2(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, V, V])
        assert maekawa_deficit(lc) == -2
        assert edit_distance(lc) == 1    # |ν|/2 = 1

    # Case B: ν = 4 (ten-sector example from paper)
    def test_case_B_nu4(self):
        a = deg(36)
        angles = (a,) * 10
        labels = (M,) * 8 + (V,) * 2   # δ = 6, ν = 4
        lc = LSVCP(SVCP(angles), labels)
        assert maekawa_deficit(lc) == 4
        assert edit_distance(lc) == 2    # |ν|/2 = 2

    # Case C: κ≠0, ν=0
    def test_case_C_nu0(self):
        lc = lsvcp([120, 80, 100, 60], [M, M, M, V])
        assert kawasaki_deficit(lc.svcp) != 0
        assert maekawa_deficit(lc) == 0
        assert edit_distance(lc) == 2

    # Case C: κ≠0, ν=2
    def test_case_C_nu2(self):
        lc = lsvcp([120, 80, 100, 60], [M, M, M, M])
        assert edit_distance(lc) == 2

    # Case C: κ≠0, ν=−2
    def test_case_C_nu_minus2(self):
        lc = lsvcp([120, 80, 100, 60], [M, V, M, V])
        assert edit_distance(lc) == 2

    # Case D: κ≠0, ν=4
    def test_case_D_nu4(self):
        lc = lsvcp([80, 60, 70, 50, 60, 40], [M, M, M, M, M, M])
        nu = maekawa_deficit(lc)
        assert nu == 4
        assert edit_distance(lc) == nu // 2 + 1   # = 3

    # Case D: κ≠0, ν=6
    def test_case_D_nu6(self):
        lc = lsvcp([60, 40, 50, 40, 60, 40, 50, 20], [M]*8)
        nu = maekawa_deficit(lc)
        assert nu == 6
        assert edit_distance(lc) == nu // 2 + 1   # = 4

    # Case E: |δ|=1
    def test_case_E_delta1(self):
        lc = lsvcp([120, 120, 120], [M, M, V])
        assert abs(signed_delta(lc)) == 1
        assert edit_distance(lc) == 1

    # Case E: |δ|=3
    def test_case_E_delta3(self):
        lc = lsvcp([120, 120, 120], [M, M, M])
        assert abs(signed_delta(lc)) == 3
        assert edit_distance(lc) == 1   # (3−1)/2 = 1

    # Case E: |δ|=5
    def test_case_E_delta5(self):
        lc = lsvcp([72, 72, 72, 72, 72], [M, M, M, M, M])
        assert abs(signed_delta(lc)) == 5
        assert edit_distance(lc) == 2   # (5−1)/2 = 2


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

class TestPrimitives:
    def test_lsvcp_insert_increases_m(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        lc2 = lsvcp_insert(lc, 1, deg(45), V)
        assert len(lc2) == 5

    def test_lsvcp_insert_label_assigned(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        lc2 = lsvcp_insert(lc, 1, deg(45), V)
        # New crease is at position 2 (the inserted s piece)
        assert lc2.labels[1] == V

    def test_lsvcp_flip_toggles_M_to_V(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        lc2 = lsvcp_flip(lc, 1)
        assert lc2.labels[0] == V
        assert lc2.labels[1:] == lc.labels[1:]

    def test_lsvcp_flip_toggles_V_to_M(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        lc2 = lsvcp_flip(lc, 4)
        assert lc2.labels[3] == M

    def test_lsvcp_flip_preserves_angles(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        lc2 = lsvcp_flip(lc, 2)
        assert lc2.svcp == lc.svcp

    def test_lsvcp_flip_out_of_range(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, V])
        with pytest.raises(IndexError):
            lsvcp_flip(lc, 0)
        with pytest.raises(IndexError):
            lsvcp_flip(lc, 5)


# ---------------------------------------------------------------------------
# full_repair / repair — correctness for all cases
# ---------------------------------------------------------------------------

class TestRepair:
    def _check(self, lc: LSVCP):
        """Run repair, verify result is fully flat-foldable, ops count matches."""
        d = edit_distance(lc)
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star), \
            f"Repaired LSVCP is not fully flat-foldable: {lc_star}"
        assert len(ops) == d, \
            f"Expected {d} ops, got {len(ops)}: {ops}"

    # Case A
    def test_repair_A(self):
        self._check(lsvcp([90, 90, 90, 90], [M, M, M, V]))

    # Case B: ν = 2
    def test_repair_B_nu2(self):
        self._check(lsvcp([90, 90, 90, 90], [M, M, M, M]))

    # Case B: ν = −2
    def test_repair_B_nu_minus2(self):
        self._check(lsvcp([90, 90, 90, 90], [M, M, V, V]))

    # Case B: ν = 4
    def test_repair_B_nu4(self):
        a = deg(36)
        lc = LSVCP(SVCP((a,) * 10), (M,) * 8 + (V,) * 2)
        self._check(lc)

    # Case B with δ < 0 (global swap)
    def test_repair_B_negative_delta(self):
        lc = lsvcp([90, 90, 90, 90], [V, V, V, V])
        self._check(lc)

    # Case C: κ≠0, ν=0
    def test_repair_C_nu0(self):
        self._check(lsvcp([120, 80, 100, 60], [M, M, M, V]))

    # Case C: κ≠0, ν=2
    def test_repair_C_nu2(self):
        self._check(lsvcp([120, 80, 100, 60], [M, M, M, M]))

    # Case C: κ≠0, ν=−2
    def test_repair_C_nu_minus2(self):
        self._check(lsvcp([120, 80, 100, 60], [M, V, M, V]))

    # Case D: ν=4
    def test_repair_D_nu4(self):
        self._check(lsvcp([80, 60, 70, 50, 60, 40], [M, M, M, M, M, M]))

    # Case D: ν=6
    def test_repair_D_nu6(self):
        self._check(lsvcp([60, 40, 50, 40, 60, 40, 50, 20], [M]*8))

    # Case E: |δ|=1
    def test_repair_E_delta1(self):
        self._check(lsvcp([120, 120, 120], [M, M, V]))

    # Case E: |δ|=3
    def test_repair_E_delta3(self):
        self._check(lsvcp([120, 120, 120], [M, M, M]))

    # Case E: |δ|=5
    def test_repair_E_delta5(self):
        self._check(lsvcp([72, 72, 72, 72, 72], [M, M, M, M, M]))

    # Verify op count is tight (not more than necessary)
    def test_repair_B_nu2_exactly_1_op(self):
        lc = lsvcp([90, 90, 90, 90], [M, M, M, M])
        _, ops = repair(lc)
        assert len(ops) == 1
        assert ops[0].kind == 'flip'

    def test_repair_C_nu0_exactly_2_ops(self):
        lc = lsvcp([120, 80, 100, 60], [M, M, M, V])
        _, ops = repair(lc)
        assert len(ops) == 2
        assert all(op.kind == 'insert' for op in ops)

    def test_repair_D_nu4_exactly_3_ops(self):
        lc = lsvcp([80, 60, 70, 50, 60, 40], [M, M, M, M, M, M])
        _, ops = repair(lc)
        assert len(ops) == 3
        insert_ops = [op for op in ops if op.kind == 'insert']
        flip_ops   = [op for op in ops if op.kind == 'flip']
        assert len(insert_ops) == 2
        assert len(flip_ops) == 1


# ---------------------------------------------------------------------------
# Specific paper examples: check angles and labels of repaired result
# ---------------------------------------------------------------------------

class TestPaperExamples:
    """
    Reproduce the exact worked examples from Sections 3.1–3.5 of the paper.
    We verify that the repaired LSVCP is in F_MV, and spot-check key values.
    """

    def test_example_B_nu2(self):
        """
        Example B (ν=2): C=(90°,90°,90°,90°), μ=(M,M,M,M).
        One flip → (M,M,M,V) or any permutation with δ=2.
        """
        lc = lsvcp([90, 90, 90, 90], [M, M, M, M])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 1
        assert ops[0].kind == 'flip'
        # δ* = 2
        from cdmsvcp import signed_delta as sd
        assert sd(lc_star) in (2, -2)

    def test_example_B_nu_minus2(self):
        """
        Example B (ν=−2): C=(90°,90°,90°,90°), μ=(M,M,V,V).
        One flip → δ = ±2.
        """
        lc = lsvcp([90, 90, 90, 90], [M, M, V, V])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 1

    def test_example_C_nu0(self):
        """
        Example C (ν=0): C=(120°,80°,100°,60°), μ=(M,M,M,V).
        κ = 40°, two insertions.
        """
        lc = lsvcp([120, 80, 100, 60], [M, M, M, V])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 2

    def test_example_C_nu2(self):
        """
        Example C (ν=2): C=(120°,80°,100°,60°), μ=(M,M,M,M).
        Two V,V insertions → δ* = 2.
        """
        lc = lsvcp([120, 80, 100, 60], [M, M, M, M])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 2

    def test_example_C_nu_minus2(self):
        """
        Example C (ν=−2): C=(120°,80°,100°,60°), μ=(M,V,M,V).
        Two M,M insertions → δ* = 2.
        """
        lc = lsvcp([120, 80, 100, 60], [M, V, M, V])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 2

    def test_example_D_nu4(self):
        """
        Example D (ν=4): C=(80°,60°,70°,50°,60°,40°), all M.
        κ=30°, distance=3 (2 ins + 1 flip).
        """
        lc = lsvcp([80, 60, 70, 50, 60, 40], [M]*6)
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 3

    def test_example_E_delta1(self):
        """
        Example E (|δ|=1): C=(120°,120°,120°), μ=(M,M,V).
        1 insertion (label M).
        """
        lc = lsvcp([120, 120, 120], [M, M, V])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 1
        assert ops[0].kind == 'insert'
        assert ops[0].label == M

    def test_example_E_delta3(self):
        """
        Example E (|δ|=3): C=(120°,120°,120°), μ=(M,M,M).
        1 insertion (label V).
        """
        lc = lsvcp([120, 120, 120], [M, M, M])
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 1
        assert ops[0].kind == 'insert'
        assert ops[0].label == V

    def test_example_E_delta5(self):
        """
        Example E (|δ|=5): C=(72°×5), μ=(M,M,M,M,M).
        1 insertion (label V) + 1 flip. Total 2.
        """
        lc = lsvcp([72, 72, 72, 72, 72], [M]*5)
        lc_star, ops = repair(lc)
        assert is_fully_flat_foldable(lc_star)
        assert len(ops) == 2
        assert ops[0].kind == 'insert'
        assert ops[0].label == V
        assert ops[1].kind == 'flip'


# ---------------------------------------------------------------------------
# Parity invariant (Proposition 1 of paper)
# ---------------------------------------------------------------------------

class TestParityInvariant:
    """m + δ is always even; this is preserved by every operation."""

    def _check_parity(self, lc: LSVCP):
        m = len(lc)
        d = signed_delta(lc)
        return (m + d) % 2 == 0

    def test_parity_even_m(self):
        for labels in [[M,M,M,V],[M,M,V,V],[M,V,V,V],[M,M,M,M]]:
            lc = lsvcp([90,90,90,90], labels)
            assert self._check_parity(lc)

    def test_parity_odd_m(self):
        for labels in [[M,M,V],[M,M,M]]:
            lc = lsvcp([120,120,120], labels)
            assert self._check_parity(lc)

    def test_parity_preserved_by_flip(self):
        lc = lsvcp([90,90,90,90],[M,M,M,M])
        lc2 = lsvcp_flip(lc, 1)
        assert self._check_parity(lc2)

    def test_parity_preserved_by_insert(self):
        lc = lsvcp([90,90,90,90],[M,M,M,V])
        lc2 = lsvcp_insert(lc, 1, deg(45), V)
        assert self._check_parity(lc2)


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def _run(self, args):
        from cdmsvcp.cli import main
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            try:
                main(args)
            except SystemExit:
                pass
        return f.getvalue()

    def test_cli_case_A(self):
        out = self._run(["1/2", "1/2", "1/2", "1/2", "--", "M", "M", "M", "V"])
        assert "d_MV = 0" in out

    def test_cli_case_B_nu2(self):
        out = self._run(["1/2", "1/2", "1/2", "1/2", "--", "M", "M", "M", "M"])
        assert "d_MV = 1" in out

    def test_cli_case_C(self):
        out = self._run(["2/3", "4/9", "5/9", "1/3", "--", "M", "M", "M", "V"])
        assert "d_MV = 2" in out

    def test_cli_repair_flag(self):
        out = self._run([
            "2/3", "4/9", "5/9", "1/3", "--",
            "M", "M", "M", "V", "--repair"
        ])
        assert "Repaired LSVCP" in out

    def test_cli_verbose_flag(self):
        out = self._run([
            "1/2", "1/2", "1/2", "1/2", "--",
            "M", "M", "M", "V", "--verbose"
        ])
        assert "κ" in out

    def test_cli_odd_m(self):
        out = self._run(["2/3", "2/3", "2/3", "--", "M", "M", "V"])
        assert "d_MV = 1" in out

    def test_cli_missing_labels(self):
        out = self._run(["1/2", "1/2", "1/2", "1/2"])
        # Should print an error or usage
        # (exits with error, output captured as empty or error message)

    def test_cli_invalid_angle_sum(self):
        out = self._run(["1/2", "1/2", "1/2", "--", "M", "M", "M"])
        # SVCP validation should catch this
