"""Spec 003c P1's walker check, the parts that decide a verdict without fitting: the seed's cohort list, the misses
recomputed from rows, the refit match, the deviance against the walk's reference, and the verdict rule
(docs/planning/spec003c_seed.md, "P1's rule, fixed before any refit"). Fast; fits nothing.
"""
import os
import re

import walker_check_003c as WC


def test_the_tool_checks_the_eleven_misses_the_seed_names_and_the_pilot_is_one_of_them():
    seed = open(os.path.join(WC.REPO, WC.SEED_DOC)).read()
    rule = seed[seed.index("## P1's rule"):]
    named = set(re.findall(r"(?:censor|hazard)-k[0-9.]+-p\d\d-c\d-s\d", rule))
    assert named == set(WC.MISSES) and len(WC.MISSES) == 11
    assert {c for c, n in WC.MISSES.items() if n == "profile_jax2"} == {c for c in named if c.startswith("censor-")}
    assert WC.PILOT in WC.MISSES and WC.PILOT in rule
    assert (WC.THR, WC.MATCH_DEX, WC.N_SOBOL, WC.BF_SEED) == (3.8415, 1e-4, 32, 20261005)


def _row(cid, cal, arm, lo, hi, shape="closed", kind=None, status="ok"):
    return {"cohort_id": cid, "calibrator": cal, "arm": arm, "status": status,
            "truth": {"eta": 2.0, "epsilon": 1.0},
            "claims": [{"parameter": "eta_over_eps", "shape": shape, "lower": lo, "upper": hi, "point": 1.5,
                        "label_info": {"kind": kind} if kind else None}]}


def test_misses_are_closed_unboxed_eta_eps_claims_excluding_the_truth_on_the_two_arms_only():
    rows = [_row("censor-a", "profile_jax2", "censor", 1.0, 1.9),                    # miss (truth 2 above)
            _row("censor-b", "profile_jax2", "censor", 1.0, 2.1),                    # hit
            _row("censor-c", "profile_jax2", "censor", 1.0, 1.9, kind="box_limited"),
            _row("censor-d", "profile_jax2", "censor", 1.0, 1.9, shape="open"),
            _row("censor-e", "profile_jax2", "censor", 1.0, 1.9, status="failed"),
            _row("pure-a", "profile_jax2", "pure", 1.0, 1.9),                        # another arm
            _row("hazard-a", "profile_jax2_h", "hazard", 2.1, 3.0)]                   # miss (truth 2 below)
    assert WC.misses_from_rows(rows) == {"censor-a": "profile_jax2", "hazard-a": "profile_jax2_h"}


def test_a_refit_counts_only_when_both_ends_are_within_the_tolerance():
    row = {"lower": 1.0, "upper": 2.0}
    assert WC.ends_match({"lower": 1.0, "upper": 2.0 * 10 ** 0.9e-4}, row)["match"]
    assert not WC.ends_match({"lower": 1.0, "upper": 2.0 * 10 ** 1.1e-4}, row)["match"]
    assert not WC.ends_match({"lower": 1.0 * 10 ** -2e-4, "upper": 2.0}, row)["match"]
    assert not WC.ends_match({"lower": None, "upper": 2.0}, row)["match"]


def test_the_deviance_is_never_negative_a_higher_value_at_the_truth_raises_the_reference():
    assert WC.deviance(-100.0, -102.0) == 4.0
    assert WC.deviance(-100.0, -99.0) == 0.0


def test_the_verdict_follows_the_seed():
    T = WC.THR
    assert WC.verdict(False, 0.1, True, None, 0) == ("UNDECIDED_REFIT_DIFFERS", None)
    assert WC.verdict(True, T - 0.01, True, None, 0) == ("SHORT", "A")
    assert WC.verdict(True, T - 0.01, False, T + 1, 5) == ("NOT_SHORT", "B")       # A set aside: B decides
    assert WC.verdict(True, T + 0.5, True, T - 0.01, 3) == ("SHORT", "B")
    assert WC.verdict(True, T + 0.5, True, T, 3) == ("NOT_SHORT", "B")              # at the threshold: not short
    assert WC.verdict(True, None, False, None, 0) == ("UNDECIDED", "B")


def test_the_branch_is_counted_over_the_censored_misses():
    def res(arm, v):
        return {"cohort_id": f"{arm}-x", "verdict": v}
    assert WC.summary([res("censor", "SHORT"), res("censor", "NOT_SHORT")])["branch"] == "walker_short"
    assert WC.summary([res("censor", "NOT_SHORT"), res("hazard", "SHORT")])["branch"] == "misses_real"
    assert WC.summary([res("censor", "NOT_SHORT"), res("censor", "UNDECIDED")])["branch"] == "undecided_to_naveh"


def test_step_b_runs_outside_the_rule_only_when_asked_and_step_a_still_decides_first():
    T = WC.THR
    assert not WC.run_step_b(False, False, True, T + 1)            # refit differs: the rule skips step B
    assert WC.run_step_b(False, True, True, T + 1)                 # outside the rule, on Naveh's request
    assert WC.run_step_b(True, False, True, T + 1)
    assert not WC.run_step_b(True, False, True, T - 0.01)          # step A decided SHORT
    assert not WC.run_step_b(False, True, True, T - 0.01)
    assert WC.run_step_b(True, False, False, T - 0.01)             # A set aside: B decides
    # the rule's verdict is unchanged by the outside-rule step B
    assert WC.verdict(False, T + 1, True, T + 0.5, 4) == ("UNDECIDED_REFIT_DIFFERS", None)
    assert WC.verdict(True, T + 1, True, T + 0.5, 4) == ("NOT_SHORT", "B")
