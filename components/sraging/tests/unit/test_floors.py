"""FLOORED_DEATHS (T036, revised at review T064 as T072; spec 002 FR-026 to FR-028; research R17): the candidate
rule, the refinement, four categories, the message.

The expected sets follow from the rules and were checked against an independent discretisation: the refined
terms of the early deaths agree with a graded-start mesh (tests/unit/engine/test_graded_mesh.py's
`graded_start`) to within 3 nats, far inside the categories' margins (an impossible death's refined mass is
below 1e-12, i.e. its term is below -27.6 nats; these are below -45).
"""
import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging import floors
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy, graded_start
from sraging.warnings import evaluate

T = np.array([0.4, 0.9, 2.0, 5.0, 30.0, 41.0, 50.0, 52.0, 60.0, 61.0, 70.0, 75.0, 90.0, 400.0])
E = np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1])
COHORT = Cohort.from_arrays(T, E, delta=1.0)
QUIET = SRParams(eta=4e-4, beta=0.02, epsilon=4e-3, kappa=0.05)
HEAD = SRParams(eta=1e-3, beta=0.1, epsilon=4e-3, kappa=0.05)
CASES = {
    # theta-hat -> {row (1-based): category}
    "quiet": (QUIET, {14: "horizon"}),
    "head deaths": (HEAD, {1: "impossible", 2: "impossible", 14: "horizon"}),
}


def categories(rep):
    return {d["row"]: d["category"] for d in rep["deaths"]}


@pytest.mark.parametrize("name", list(CASES))
def test_listing_and_categories_match_the_expected_set(name):
    p, expected = CASES[name]
    rep = floors.report(COHORT, p)
    assert categories(rep) == expected
    assert rep["counts"] == {k: sum(v == k for v in expected.values()) for k in floors.CATEGORIES}
    assert set(rep["grad_v1_at_theta_hat"]) == {"eta", "beta", "epsilon", "kappa"}


def test_the_refined_terms_agree_with_a_graded_start():
    """The refinement is a convergence, not a guess: an independent graded-start mesh gives the same early-death
    terms to within 3 nats, while the fit's grid is 15-26 nats off."""
    rep = floors.report(COHORT, HEAD)
    g = rep["grid"]
    sub = Cohort.from_arrays(T[:2], np.ones(2, int), delta=1.0)
    ev = Evaluator(sub, numerics="v1", grid=GridPolicy("fixed", N=g["N"], tau_max=0.1, L=g["L"],
                                                       nodes=graded_start(1e-11, 1.02, g["mesh"]["dt"] / 16, 0.1)))
    ref = ev.terms(HEAD)[0]
    got = [d for d in rep["deaths"] if d["row"] in (1, 2)]
    for d, r in zip(got, ref):
        assert d["converged"] and abs(d["term_refined"] - r) < 3.0 and abs(d["term_v1"] - r) > 10.0


def test_an_ulp_move_of_v033_does_not_change_the_listing():
    """N7: the age-0 style death is listed through its corrected mass (< 1e-12), not only through v033's
    rounding (-700 on one side of an ulp, -30.5 on the other)."""
    rep = floors.report(COHORT, HEAD)
    d2 = next(d for d in rep["deaths"] if d["row"] == 2)
    assert d2["term_v033"] > -700 + 1 and abs(d2["term_v1"] - d2["term_v033"]) < floors.LIST_NATS


def test_without_refinement_or_with_a_tiny_threshold_the_other_categories_appear(monkeypatch):
    """The decision tree's other branches on the same deaths: with the impossible threshold far below their
    refined mass they are 'grid' (the refined term moved > 1 nat); with no refinement as well, 'cancellation'
    for the death v033 floored."""
    monkeypatch.setattr(floors, "IMPOSSIBLE_MASS", 1e-200)
    monkeypatch.setattr(floors, "CANDIDATE_MASS", 1e-200)
    assert categories(floors.report(COHORT, HEAD)) == {1: "grid", 14: "horizon"}
    monkeypatch.setattr(floors, "REFINE_MAX", 0)
    assert categories(floors.report(COHORT, HEAD)) == {1: "cancellation", 14: "horizon"}


def test_the_message_shows_three_values_per_death_and_what_to_do():
    rep = floors.report(COHORT, HEAD)
    msg = floors.message(rep)
    assert "2 impossible (age 0.4: -700.0 in this fit, -31.4 corrected, " in msg and "1 horizon (age 400:" in msg
    assert "v0.3.3's arithmetic (this fit)" in msg and "reference numerics" not in msg
    assert "essentially zero probability" in msg and "a longer horizon" in msg
    assert "refit without the deaths before a cut-off age you choose and state" in msg


def test_the_check_fires_through_evaluate_only_when_deaths_are_listed():
    rep = floors.report(COHORT, HEAD)
    ws, checks = evaluate(COHORT, (), {}, floor_report=rep)
    fd = next(c for c in checks if c.code == "FLOORED_DEATHS")
    assert fd.ran and fd.fired and [w.code for w in ws if w.code == "FLOORED_DEATHS"] == ["FLOORED_DEATHS"]
    _, checks = evaluate(COHORT, (), {}, floor_report={"deaths": []})
    fd = next(c for c in checks if c.code == "FLOORED_DEATHS")
    assert fd.ran and not fd.fired


@pytest.mark.parametrize("plant", ["floor", "impossible_mass", "no_candidate_mass"])
def test_a_planted_rule_change_is_caught(monkeypatch, plant):
    """Constitution V: the comparison with the expected set fails when the rule is perturbed."""
    if plant == "floor":
        monkeypatch.setattr(floors, "_FLOOR", -800.0)          # floored v033 terms no longer candidates by the floor
        monkeypatch.setattr(floors, "CANDIDATE_MASS", 0.0)     # ... nor by their mass
    elif plant == "impossible_mass":
        monkeypatch.setattr(floors, "IMPOSSIBLE_MASS", 1e-60)  # the refined head deaths are no longer impossible
    else:
        monkeypatch.setattr(floors, "CANDIDATE_MASS", 0.0)     # the ulp-robust candidate rule is off
    p, expected = CASES["head deaths"]
    assert categories(floors.report(COHORT, p)) != expected
