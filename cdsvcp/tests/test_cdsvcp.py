"""
tests/test_cdsvcp.py
====================
Unit tests for the cdsvcp package.

Covers all cases of Algorithm TwoInsert:
  Case 1  : n = 1
  Case 2  : n ≥ 2, M > κ  (with both primary and fallback sub-cases)
  Case 3  : n ≥ 3, M = κ
  Case 2b : n ≥ 3, M < κ

Also tests:
  - Kawasaki deficit / residual
  - edit_distance (all three values 0, 1, 2)
  - single_insert (odd inputs, dC = 1)
  - repair convenience wrapper
  - primitive insert / delete
  - SVCP validation
  - CLI smoke tests

All examples are taken directly from the paper (Examples 2.1–2.4).
"""

from __future__ import annotations

import pytest
from fractions import Fraction

from cdsvcp import (
    SVCP,
    ModCOp,
    delete,
    edit_distance,
    insert,
    is_flat_foldable,
    kawasaki_deficit,
    kawasaki_residual,
    repair,
    single_insert,
    sodd,
    seven,
    two_insert,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deg(x) -> Fraction:
    """Convert degrees (int or float) to fraction-of-π."""
    return Fraction(x, 180)


def _make(*degrees) -> SVCP:
    """Build an SVCP from a list of degree values (must sum to 360)."""
    return SVCP(tuple(deg(d) for d in degrees))


# ---------------------------------------------------------------------------
# SVCP validation
# ---------------------------------------------------------------------------

class TestSVCPValidation:
    def test_valid_2(self):
        C = SVCP((Fraction(1), Fraction(1)))
        assert len(C) == 2

    def test_valid_4_foldable(self):
        C = _make(90, 90, 90, 90)
        assert is_flat_foldable(C)

    def test_invalid_sum(self):
        with pytest.raises(ValueError, match="sum"):
            SVCP((Fraction(1, 2), Fraction(1, 2)))

    def test_invalid_zero_angle(self):
        with pytest.raises(ValueError, match="positive"):
            SVCP((Fraction(0), Fraction(2)))

    def test_invalid_m_1(self):
        with pytest.raises(ValueError, match="at least 2"):
            SVCP((Fraction(2),))


# ---------------------------------------------------------------------------
# sodd / seven / kawasaki helpers
# ---------------------------------------------------------------------------

class TestKawasakiHelpers:
    def test_sodd_seven_sum(self):
        C = _make(120, 80, 100, 60)
        assert sodd(C) + seven(C) == Fraction(2)

    def test_kawasaki_residual_odd(self):
        C = _make(120, 100, 140)
        Kres = kawasaki_residual(C)
        assert Kres == deg(80)

    def test_kawasaki_deficit_even(self):
        C = _make(120, 80, 100, 60)
        assert kawasaki_deficit(C) == deg(40)

    def test_kawasaki_deficit_odd_raises(self):
        C = _make(120, 100, 140)
        with pytest.raises(ValueError):
            kawasaki_deficit(C)

    def test_is_flat_foldable_true(self):
        C = _make(90, 90, 90, 90)
        assert is_flat_foldable(C)

    def test_is_flat_foldable_false_even(self):
        C = _make(120, 80, 100, 60)
        assert not is_flat_foldable(C)

    def test_is_flat_foldable_false_odd(self):
        C = _make(120, 100, 140)
        assert not is_flat_foldable(C)


# ---------------------------------------------------------------------------
# edit_distance
# ---------------------------------------------------------------------------

class TestEditDistance:
    def test_d0(self):
        C = _make(90, 90, 90, 90)
        assert edit_distance(C) == 0

    def test_d1_odd(self):
        C = _make(120, 100, 140)
        assert edit_distance(C) == 1

    def test_d1_odd_5creases(self):
        C = _make(80, 70, 60, 90, 60)
        assert edit_distance(C) == 1

    def test_d2_even(self):
        C = _make(120, 80, 100, 60)
        assert edit_distance(C) == 2

    def test_d2_n1(self):
        C = _make(200, 160)      # n=1, kap = 20°
        assert edit_distance(C) == 2


# ---------------------------------------------------------------------------
# Primitive operations
# ---------------------------------------------------------------------------

class TestPrimitiveOps:
    def test_insert_basic(self):
        C = _make(180, 180)
        C2 = insert(C, 1, deg(90))
        assert len(C2) == 3
        assert C2.angles[0] == deg(90)
        assert C2.angles[1] == deg(90)
        assert C2.angles[2] == deg(180)

    def test_insert_preserves_sum(self):
        C = _make(120, 80, 100, 60)
        C2 = insert(C, 1, deg(40))
        assert sum(C2.angles) == Fraction(2)

    def test_insert_bad_param_zero(self):
        C = _make(90, 90, 90, 90)
        with pytest.raises(ValueError):
            insert(C, 1, Fraction(0))

    def test_insert_bad_param_too_large(self):
        C = _make(90, 90, 90, 90)
        with pytest.raises(ValueError):
            insert(C, 1, deg(90))   # s must be strictly less than α_1 = 90°

    def test_delete_basic(self):
        C = _make(120, 100, 140)
        C2 = delete(C, 1)
        assert len(C2) == 2
        assert C2.angles[0] == deg(220)   # 120+100

    def test_delete_wraparound(self):
        C = _make(120, 100, 140)
        C2 = delete(C, 3)    # merge α_3=140 with cyclic successor α_1=120
        assert len(C2) == 2
        assert deg(260) in C2.angles   # 140+120

    def test_delete_forbidden_m2(self):
        C = SVCP((Fraction(1), Fraction(1)))
        with pytest.raises(ValueError, match="prohibited"):
            delete(C, 1)

    def test_delete_preserves_sum(self):
        C = _make(90, 90, 90, 30, 60)
        C2 = delete(C, 2)
        assert sum(C2.angles) == Fraction(2)


# ---------------------------------------------------------------------------
# single_insert  (dC = 1, odd inputs)  — Example 2.1 of the paper
# ---------------------------------------------------------------------------

class TestSingleInsert:
    def test_paper_example_d1(self):
        # C = (120°, 100°, 140°),  Kres = 80°
        # j*=1, s*=40°  →  C* = (80°, 40°, 100°, 140°)
        C = _make(120, 100, 140)
        C_star, op = single_insert(C)
        assert is_flat_foldable(C_star)
        assert len(C_star) == 4
        assert op.kind == 'insert'

    def test_single_insert_odd_5(self):
        C = _make(80, 70, 60, 90, 60)
        C_star, op = single_insert(C)
        assert is_flat_foldable(C_star)

    def test_single_insert_kres_zero(self):
        # K=0 on an odd pattern means feasible j* ∈ {2,…,m-1}
        # e.g. (π/3, π/3, π/3, π/3, π*2/3) – Kres = 0
        # Build manually: need odd m and Sodd = π
        # Try (60°, 60°, 60°, 120°, 60°):
        #   Sodd = 60+60+60 = 180° = π  →  Kres = 0
        C = _make(60, 60, 60, 120, 60)
        assert kawasaki_residual(C) == 0
        C_star, op = single_insert(C)
        assert is_flat_foldable(C_star)
        assert op.pos not in (1, 5)   # must be in {2,3,4}

    def test_single_insert_raises_for_even(self):
        C = _make(90, 90, 90, 90)
        with pytest.raises(ValueError):
            single_insert(C)


# ---------------------------------------------------------------------------
# two_insert — all four cases of Algorithm TwoInsert
# ---------------------------------------------------------------------------

class TestTwoInsert:

    # --- Case 1: n = 1 ---
    def test_case1_n1(self):
        # C = (200°, 160°), kap = 20°
        C = _make(200, 160)
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)
        assert len(C_star) == 4
        assert all(o.kind == 'insert' for o in ops)

    def test_case1_n1_paper_values(self):
        # C = (π+κ, π−κ) with κ=π/6  →  angles = (7π/6, 5π/6)?
        # No: angles must sum to 2π. Use (210°, 150°), kap=30°.
        C = _make(210, 150)
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)
        assert len(ops) == 2

    # --- Case 2: n ≥ 2, M > κ (primary) — Example 2.2 ---
    def test_case2_paper_example(self):
        # C = (120°, 80°, 100°, 60°), kap=40°, M=120°>40°
        C = _make(120, 80, 100, 60)
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)
        assert len(C_star) == 6
        assert all(o.kind == 'insert' for o in ops)

    def test_case2_larger_pattern(self):
        # n=3, M > kap
        C = _make(100, 50, 80, 40, 70, 20)
        assert edit_distance(C) == 2
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)

    def test_case2_fallback(self):
        # Construct a case where t*=Δ−s*<0, i.e. 2*α_q ≤ α_p − κ
        # Need α_p > κ and α_q very small.
        # E.g. α_1=150°, α_2=5°, α_3=150°, α_4=55°  → sum=360°
        # Sodd=300°, kap=120°; M=150°>120°; q=2, α_q=5°; 2*5=10 ≤ 30=150-120 → fallback
        C = _make(150, 5, 150, 55)
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)

    # --- Case 3: n ≥ 3, M = κ — Example 2.3 ---
    def test_case3_paper_example(self):
        # C = (90°,40°,90°,30°,90°,20°), kap=90°, M=κ
        C = _make(90, 40, 90, 30, 90, 20)
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)
        assert ops[0].kind == 'delete'
        assert ops[1].kind == 'insert'

    def test_case3_n3_uniform(self):
        # All odd angles = κ = π/(n-1) = π/2 = 90°, n=3
        # Sodd = 3*90=270, kap=90, even sum=90°
        # Need: 3 odd angles each 90°, 3 even angles summing to 90°
        C = _make(90, 30, 90, 30, 90, 30)
        kap = kawasaki_deficit(C)
        M = max(C.angles[i] for i in range(0, 6, 2))
        assert M == kap
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)

    # --- Case 2b: n ≥ 3, M < κ — Example 2.4 ---
    def test_case2b_paper_example(self):
        # C = (100°,30°,100°,20°,100°,10°), kap=120°, M=100°<120°
        C = _make(100, 30, 100, 20, 100, 10)
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)
        assert all(o.kind == 'insert' for o in ops)
        assert len(C_star) == 8

    def test_case2b_n4(self):
        # n=4, all odd angles = 60°, kap = 60°*4 - 180° = 60°
        # But M=60°=kap → case 3! Let's make M < kap:
        # Need Sodd > π and every odd < kap.
        # Try n=4: want kap=90°, so Sodd=270°.
        # 4 odd angles each < 90° summing to 270°: e.g. 70,70,70,60 = 270.
        # 4 even angles summing to 90°: e.g. 30,20,25,15 = 90.
        C = _make(70, 30, 70, 20, 70, 25, 60, 15)
        kap = kawasaki_deficit(C)
        M = max(C.angles[i] for i in range(0, 8, 2))
        assert M < kap
        C_star, ops = two_insert(C)
        assert is_flat_foldable(C_star)

    # --- two_insert validation errors ---
    def test_raises_odd_input(self):
        C = _make(120, 100, 140)
        with pytest.raises(ValueError, match="even"):
            two_insert(C)

    def test_raises_already_foldable(self):
        C = _make(90, 90, 90, 90)
        with pytest.raises(ValueError, match="already flat-foldable"):
            two_insert(C)


# ---------------------------------------------------------------------------
# repair  (convenience wrapper)
# ---------------------------------------------------------------------------

class TestRepair:
    def test_repair_d0(self):
        C = _make(90, 90, 90, 90)
        C_star, ops = repair(C)
        assert C_star == C
        assert ops == []

    def test_repair_d1(self):
        C = _make(120, 100, 140)
        C_star, ops = repair(C)
        assert is_flat_foldable(C_star)
        assert len(ops) == 1

    def test_repair_d2(self):
        C = _make(120, 80, 100, 60)
        C_star, ops = repair(C)
        assert is_flat_foldable(C_star)
        assert len(ops) == 2

    def test_repair_all_cases_foldable(self):
        patterns = [
            _make(200, 160),                        # n=1
            _make(120, 80, 100, 60),                # Case 2
            _make(90, 40, 90, 30, 90, 20),          # Case 3
            _make(100, 30, 100, 20, 100, 10),       # Case 2b
            _make(120, 100, 140),                   # odd, d=1
        ]
        for C in patterns:
            C_star, ops = repair(C)
            assert is_flat_foldable(C_star), f"Not foldable after repair: {C}"
            assert len(ops) == edit_distance(C)


# ---------------------------------------------------------------------------
# Regression: κ range  (Remark 2.3)
# ---------------------------------------------------------------------------

class TestKapRange:
    def test_kap_in_range(self):
        import random
        random.seed(42)
        for _ in range(200):
            n = random.randint(1, 6)
            m = 2 * n
            # generate random positive angles summing to 2π via Dirichlet-like
            cuts = sorted(random.uniform(0, 1) for _ in range(m - 1))
            raw = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, m-1)] + [1 - cuts[-1]]
            fracs = tuple(Fraction(int(r * 1800), 1800) for r in raw)
            total = sum(fracs)
            # adjust last to make sum exactly 2
            fracs = fracs[:-1] + (fracs[-1] + Fraction(2) - total,)
            if any(f <= 0 for f in fracs):
                continue
            C = SVCP(fracs)
            kap = kawasaki_deficit(C)
            assert -Fraction(1) < kap < Fraction(1), f"κ out of range: {kap}"


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    def test_cli_d0(self, capsys):
        from cdsvcp.cli import main
        main(["1/2", "1/2", "1/2", "1/2"])
        out = capsys.readouterr().out
        assert "dC(C, F) = 0" in out

    def test_cli_d1(self, capsys):
        from cdsvcp.cli import main
        main([str(Fraction(2, 3)), str(Fraction(5, 9)), str(Fraction(7, 9))])
        out = capsys.readouterr().out
        assert "dC(C, F) = 1" in out

    def test_cli_d2_with_repair(self, capsys):
        from cdsvcp.cli import main
        main([str(Fraction(2, 3)), str(Fraction(4, 9)),
              str(Fraction(5, 9)), str(Fraction(1, 3)),
              "--repair"])
        out = capsys.readouterr().out
        assert "dC(C, F) = 2" in out
        assert "Repaired pattern" in out

    def test_cli_verbose(self, capsys):
        from cdsvcp.cli import main
        main(["1/2", "1/2", "1/2", "1/2", "--verbose"])
        out = capsys.readouterr().out
        assert "S_odd" in out

    def test_cli_invalid_sum(self, capsys):
        from cdsvcp.cli import main
        with pytest.raises(SystemExit):
            main(["1/2", "1/2"])   # sum = 1 ≠ 2
