"""Spec 003d M4's v5 reproduction tool, the parts that decide a match without the tester's data: the tester's
checksum, its first-bin correction, SRK's floor-then-clamp, and which cohorts are in scope. Fast; solves nothing.
"""
import numpy as np

import v5_reproduce_003d as R


def test_the_checksum_is_the_testers():
    # expected value from srbench/library/build.py:cohort_digest on the same arrays (computed read-only, 2026-10-12)
    assert R.cohort_digest(np.array([0.0, 0.5, 1.0, 2.5]), np.array([1, 1, 0, 1])) == \
        "e0971d6132dcc96f4c5833c1e929643a142f658418f3f85e493bacf31649fc88"


def test_the_first_bin_correction_moves_only_the_clamped_age():
    assert list(R.correct_first_bin(np.array([0.25, 0.5, 0.75]), 0.5)) == [0.0, 0.5, 0.75]
    assert list(R.correct_first_bin(np.array([0.25]), None)) == [0.25]


def test_srk_floors_then_clamps_then_the_correction_puts_first_bin_deaths_at_zero():
    tau = np.linspace(0.0, 4.0, 4001)
    S = np.exp(-tau)
    spec = {"seed": 3, "n": 500, "L_phys": 1.0, "delta": 0.25, "censor_frac": 0.2}
    ages, events = R.materialize_srk(spec, tau, S)
    assert np.all(np.diff(ages) >= 0) and ages.size == 500
    assert np.allclose(ages / 0.25, np.round(ages / 0.25))          # every age on the delta grid, none at delta/2
    assert (ages == 0.0).any() and int((events == 0).sum()) > 0
    assert np.array_equal(R.materialize_srk(spec, tau, S)[0], ages)  # deterministic given the seed


def test_heterogeneous_cohorts_are_out_of_scope():
    assert R.homogeneous({"het_param": None, "het_var": 0.0})
    assert R.homogeneous({"het_param": "xc", "het_var": 0.0})
    assert not R.homogeneous({"het_param": "xc", "het_var": 0.2})
