"""Spec 004's rules, as pure functions (src/sraging/sampler_rules.py; contracts/register-and-panel.md §3). The
foundational part (tasks T006): the verdict vocabulary, `require_assessable` (gate C2: nothing assessable is never a
pass) and `summarise` (constitution V: WARN is not PASS). The rules HR-1 to HR-11 join with their stories, each a
planted case beside its twin."""
import math

import numpy as np
import pytest

from sraging import sampler_rules as R


# ------------------------------------------------------------------------------------------------- Verdict.status
@pytest.mark.parametrize("status", ["pass", "fail", "incomplete", "capped", "reported"])
def test_a_verdict_takes_the_five_statuses(status):
    reasons = ("no assessable cell",) if status == "incomplete" else ()
    v = R.Verdict("HR-6", "clamp", status, incomplete_reasons=reasons)
    assert v.status == status and v.as_dict()["status"] == status


@pytest.mark.parametrize("status", ["PASS", "warn", "passed", "ok", "", None])
def test_a_verdict_refuses_any_other_status(status):
    with pytest.raises(ValueError, match="status"):
        R.Verdict("HR-6", "clamp", status)


def test_an_incomplete_verdict_says_why_and_a_pass_has_nothing_unassessed():
    with pytest.raises(ValueError, match="reasons"):
        R.Verdict("HR-1", None, "incomplete")                                   # incomplete without a reason
    with pytest.raises(ValueError, match="gate C2"):
        R.Verdict("HR-1", None, "pass", incomplete_reasons=("stratum human-10k: no valid cell at sd <= 0.3",))
    R.Verdict("HR-1", None, "incomplete", incomplete_reasons=("stratum human-10k: no valid cell at sd <= 0.3",))
    R.Verdict("HR-1", None, "pass")                                             # twins


def test_a_verdict_names_a_known_wall_or_none():
    with pytest.raises(ValueError, match="wall"):
        R.Verdict("HR-6", "mirror", "pass")
    assert [R.Verdict("HR-6", w, "pass").wall for w in ("clamp", "reflect", None)] == ["clamp", "reflect", None]


def test_a_verdict_carries_its_denominators_and_cpu_against_its_cap():
    v = R.Verdict("HR-3", None, "capped", n_valid=17, n_unassessable=0, statistic={"max_abs_dll": 0.01},
                  crit=0.05, cpu_spent=10.0, cap=10.0)
    d = v.as_dict()
    assert {k: d[k] for k in ("rule", "wall", "n_valid", "n_unassessable", "crit", "cpu_spent", "cap")} == \
        {"rule": "HR-3", "wall": None, "n_valid": 17, "n_unassessable": 0, "crit": 0.05, "cpu_spent": 10.0,
         "cap": 10.0}


# --------------------------------------------------------------------------------------------- require_assessable
CELL = {"id": "syn-sharp/c1/sd0.2", "assessable": True}
PLANT = {"id": "plant M=11", "plant": True, "informative": True}


def test_an_empty_set_is_incomplete():
    reasons = R.require_assessable([])
    assert reasons and "nothing to assess" in reasons[0]


def test_any_cell_flagged_unassessable_is_incomplete_and_named():
    bad = {"id": "human-100k/c3/sd0.5", "assessable": False, "reason": "b(dt/2) > crit/10 at m_max"}
    reasons = R.require_assessable([CELL, bad, PLANT])
    assert len(reasons) == 1 and "human-100k/c3/sd0.5" in reasons[0] and "crit/10 at m_max" in reasons[0]


def test_a_plant_flagged_not_informative_is_incomplete_and_named():
    weak = {"id": "plant M=5", "plant": True, "informative": False, "reason": "jump 3e-6 < 10 tol"}
    reasons = R.require_assessable([CELL, weak])
    assert len(reasons) == 1 and "plant M=5" in reasons[0] and "not informative" in reasons[0]


def test_all_assessable_and_informative_gives_no_reasons():
    assert R.require_assessable([CELL, dict(CELL, id="b"), PLANT]) == ()                        # the twin
    assert R.require_assessable([{"id": "c"}]) == ()                    # unflagged is assessable


def test_every_problem_is_listed_not_only_the_first():
    cells = [dict(CELL, id="a", assessable=False), dict(CELL, id="b", assessable=False),
             dict(PLANT, id="p", informative=False)]
    assert len(R.require_assessable(cells)) == 3


# ------------------------------------------------------------------------------------------------------ summarise
def V(status, rule="HR-1", wall=None):
    return R.Verdict(rule, wall, status, incomplete_reasons=("x",) if status == "incomplete" else ())


@pytest.mark.parametrize("bad", ["incomplete", "capped", "fail"])
def test_no_green_line_while_any_verdict_is_incomplete_capped_or_failed(bad):
    s = R.summarise([V("pass"), V("pass", "HR-2"), V(bad, "HR-3")])
    assert not s["green"] and R.GREEN not in s["line"] and "NOT A PASS" in s["line"]
    assert s["PASS"] == 2 and s["WARN"] + s["FAIL"] == 1


def test_every_pass_prints_the_green_line():
    s = R.summarise([V("pass"), V("pass", "HR-6", "clamp"), V("pass", "HR-6", "reflect")])    # the twin
    assert s["green"] and s["line"].startswith(R.GREEN) and (s["PASS"], s["WARN"], s["FAIL"]) == (3, 0, 0)


def test_pass_warn_and_fail_are_counted_separately():
    s = R.summarise([V("pass"), V("fail", "HR-2"), V("incomplete", "HR-3"), V("capped", "HR-4"),
                     V("reported", "HR-5")])
    assert (s["PASS"], s["WARN"], s["FAIL"], s["REPORTED"]) == (1, 2, 1, 1)
    assert s["by_status"] == {"pass": 1, "fail": 1, "incomplete": 1, "capped": 1, "reported": 1}
    assert "PASS 1" in s["line"] and "WARN 2" in s["line"] and "FAIL 1" in s["line"]


def test_nothing_or_only_measurements_is_not_green():
    assert not R.summarise([])["green"]
    assert not R.summarise([V("reported")])["green"]
    assert R.summarise([V("pass"), V("reported", "HR-5")])["green"]       # a measurement beside a pass is not a WARN


# ------------------------------------------------------------------------------------------------------------ HR-1
# tasks T014; contracts/register-and-panel.md §3: PASS iff 0 false passes (check passes and |l - ref| >= 0.1) among
# valid cells, and the M = 11 plant's check fires on every valid cell where |l(11) - ref| > 0.1. Incomplete (gate C2)
# if a stratum has no valid cell at sd <= 0.3, or no cell has M = 11 more than 0.1 off.
def hr1_cell(stratum, sd=0.2, err=0.01, err11=0.5, check11=0.3, verdict="converged", valid=True, **kw):
    ref = -1000.0
    return {"id": f"{stratum}/c0/sd{sd}", "cohort_id": f"{stratum}-c0", "stratum": stratum, "sd": sd,
            "ll": ref + err, "verdict": verdict, "M": kw.get("M", 25), "abs_dll": kw.get("abs_dll", 0.01),
            "ll_ref": ref, "ref_valid": valid, "ll_11": ref + err11, "check_11": check11}


def hr1_panel(over=None):
    """One sd 0.2 and one sd 0.5 cell per stratum; `over` replaces row i by hr1_cell(its stratum, **over[i])."""
    rows = [hr1_cell(s) for s in R.HR1_STRATA]
    rows += [hr1_cell(s, sd=0.5, err11=0.05, check11=0.01) for s in R.HR1_STRATA]     # M = 11 is close here
    for i, kw in (over or {}).items():
        rows[i] = hr1_cell(rows[i]["stratum"], **kw)
    return rows


def test_hr1_the_strata_are_the_panels_nine():
    assert R.HR1_STRATA == ("syn-broad", "syn-mid", "syn-sharp", "real-lo-r", "real-hi-r", "human-10k", "human-25k",
                            "human-100k", "realfull")


def test_hr1_a_false_pass_at_012_fails_and_its_twin_at_008_passes():
    v = R.hr1_verdict(hr1_panel({0: {"err": 0.12}}))
    assert v.status == "fail" and v.statistic["false_passes"] == ["syn-broad/c0/sd0.2"]
    t = R.hr1_verdict(hr1_panel({0: {"err": 0.08}}))                                       # the twin
    assert t.status == "pass" and t.statistic["false_passes"] == [] and t.incomplete_reasons == ()
    assert t.n_valid == 18 and t.crit == 0.1


def test_hr1_an_unconverged_cell_is_not_a_false_pass_and_is_reported():
    v = R.hr1_verdict(hr1_panel({3: {"err": 0.4, "verdict": "unconverged_at_cap", "M": 192, "abs_dll": 0.07}}))
    assert v.status == "pass" and v.statistic["unconverged"] == [{"id": "real-lo-r/c0/sd0.2", "M": 192,
                                                                   "abs_dll": 0.07}]


def test_hr1_an_invalid_reference_is_excluded_and_counted():
    v = R.hr1_verdict(hr1_panel({9: {"sd": 0.5, "err": 3.0, "valid": False}}))
    assert v.status == "pass" and v.statistic["n_reference_unconverged"] == 1 and v.n_valid == 17


def test_hr1_the_plant_missing_one_cell_fails():
    v = R.hr1_verdict(hr1_panel({2: {"err11": 0.4, "check11": 0.02}}))
    assert v.status == "fail" and v.statistic["plant_misses"] == ["syn-sharp/c0/sd0.2"]
    t = R.hr1_verdict(hr1_panel({2: {"err11": 0.4, "check11": 0.06}}))                     # the twin: it fires
    assert t.status == "pass" and t.statistic["plant_misses"] == []


def test_hr1_a_stratum_with_no_valid_cell_at_sd_up_to_03_is_incomplete():
    rows = hr1_panel({7: {"valid": False}})                       # human-100k's sd 0.2 cell; its 0.5 cell stays
    v = R.hr1_verdict(rows)
    assert v.status == "incomplete" and any("human-100k" in r and "sd <= 0.3" in r for r in v.incomplete_reasons)
    gone = [r for r in hr1_panel() if r["stratum"] != "realfull"]                    # a stratum with no row at all
    assert R.hr1_verdict(gone).status == "incomplete"
    assert R.hr1_verdict(hr1_panel()).status == "pass"                                           # the twin


def test_hr1_no_cell_with_m11_more_than_01_off_is_incomplete():
    rows = [dict(r, ll_11=r["ll_ref"] + 0.05, check_11=0.01) for r in hr1_panel()]
    v = R.hr1_verdict(rows)
    assert v.status == "incomplete" and any("plant" in r and "not informative" in r for r in v.incomplete_reasons)


def test_hr1_a_failure_is_not_hidden_by_an_incomplete_input():
    rows = hr1_panel({0: {"err": 0.2}, 7: {"valid": False}})
    v = R.hr1_verdict(rows)
    assert v.status == "fail" and v.incomplete_reasons            # the reasons are kept beside the fail


def test_hr1_reports_the_clopper_pearson_bound():
    v = R.hr1_verdict(hr1_panel())
    n = v.statistic["n_checked_valid"]
    assert n == 18 and v.statistic["cp_upper_95"] == pytest.approx(1 - 0.05 ** (1 / n), rel=1e-12)
    assert R.hr1_verdict([]).status == "incomplete"


# ------------------------------------------------------------------------------------------------------------ HR-2
# PASS iff every |CD - grad| <= tol and the plant's jump > 10 tol is detected; incomplete if neither the M = 5 plant
# nor the M = 3 fallback is informative (gate C2).
def hr2_cells(diff=1e-6):
    return [{"id": f"c{i}/sd{sd:.3f}", "cohort_id": f"c{i}", "sd": sd, "cd": 12.5 + diff, "grad": 12.5}
            for i in range(6) for sd in (0.30, 0.35, 0.40)]


def plant(M, jump, detected=True):
    return {"M": M, "jump": jump, "cd": 12.5 + (jump / 2e-4 if detected else 0.0), "grad": 12.5}


def test_hr2_a_difference_above_the_tolerance_fails_and_its_twin_passes():
    tol = 1e-4
    v = R.hr2_verdict(hr2_cells(2e-4), [plant(5, 0.01)], tol=tol)
    assert v.status == "fail" and len(v.statistic["over_tol"]) == 18 and v.crit == tol
    t = R.hr2_verdict(hr2_cells(5e-5), [plant(5, 0.01)], tol=tol)                             # the twin
    assert t.status == "pass" and t.statistic["plant"]["M"] == 5 and t.n_valid == 18


def test_hr2_an_undetected_informative_plant_fails():
    v = R.hr2_verdict(hr2_cells(5e-5), [plant(5, 0.01, detected=False)], tol=1e-4)
    assert v.status == "fail" and v.statistic["plant"]["detected"] is False


def test_hr2_the_m3_fallback_is_used_when_m5_is_not_informative():
    v = R.hr2_verdict(hr2_cells(5e-5), [plant(5, 5e-4), plant(3, 0.02)], tol=1e-4)
    assert v.status == "pass" and v.statistic["plant"]["M"] == 3


def test_hr2_neither_plant_informative_is_incomplete():
    v = R.hr2_verdict(hr2_cells(5e-5), [plant(5, 5e-4), plant(3, 9e-4)], tol=1e-4)
    assert v.status == "incomplete" and any("M = 5" in r and "M = 3" in r for r in v.incomplete_reasons)
    assert R.hr2_verdict(hr2_cells(5e-5), [plant(5, 5e-4)], tol=1e-4).status == "incomplete"     # no fallback run
    assert R.hr2_verdict([], [plant(5, 0.01)], tol=1e-4).status == "incomplete"                  # nothing to assess


def test_hr2_refuses_a_tolerance_that_is_not_a_positive_number():
    for bad in (0.0, -1e-4, float("nan"), None):
        with pytest.raises(ValueError, match="tol"):
            R.hr2_verdict(hr2_cells(), [plant(5, 0.01)], tol=bad)


# ------------------------------------------------------------------------------------------------------------ HR-3
# tasks T030: PASS iff every |l_adaptive - l_full| < 0.05 at a common M; the CPU saving is claimed iff the upper 95 %
# bound of the mean paired log ratio is < 0; the never-raise plant must exceed 0.05 on the sharp cells, else incomplete;
# a pair lacking a checked final value is incomplete; past the cap, capped.
def hr3_pairs(dll=0.01, ratio=0.7, n=24, checked=True):
    rng = np.random.default_rng(5)
    return [{"id": f"c{i // 2}/sd{0.2 if i % 2 == 0 else 0.3}", "sharp": i < 8, "M_common": 57,
             "ll_adaptive": -1000.0 + dll, "ll_full": -1000.0, "cpu_adaptive": 100.0 * ratio * math.exp(0.05 *
                                                                                                    rng.standard_normal()),
             "cpu_full": 100.0, "checked": checked} for i in range(n)]


def hr3_plants(dll=0.5):
    return [{"id": f"c{i // 2}/sd{0.2 if i % 2 == 0 else 0.3}", "sharp": i < 8, "ll_plant": -1000.0 - (dll if i < 8 else
                                                                                                         0.001),
             "ll_full": -1000.0} for i in range(24)]


def test_hr3_one_pair_at_006_fails_and_its_twin_at_004_passes():
    pairs = hr3_pairs()
    pairs[5]["ll_adaptive"] = -1000.0 + 0.06
    v = R.hr3_verdict(pairs, hr3_plants(), cpu_spent=4.0)
    assert v.status == "fail" and v.statistic["over"] == [pairs[5]["id"]]
    pairs[5]["ll_adaptive"] = -1000.0 + 0.04
    t = R.hr3_verdict(pairs, hr3_plants(), cpu_spent=4.0)                                        # the twin
    assert t.status == "pass" and t.n_valid == 24 and t.cpu_spent == 4.0 and t.cap == 10.0


def test_hr3_the_cpu_saving_is_claimed_only_when_the_upper_bound_is_below_zero():
    v = R.hr3_verdict(hr3_pairs(ratio=0.7), hr3_plants(), cpu_spent=4.0)
    s = v.statistic["saving"]
    assert s["claimed"] and s["upper_95_log_ratio"] < 0 and s["mean_log_ratio"] == pytest.approx(math.log(0.7), abs=0.03)
    t = R.hr3_verdict(hr3_pairs(ratio=1.0), hr3_plants(), cpu_spent=4.0)                        # the twin
    assert t.status == "pass" and not t.statistic["saving"]["claimed"] and t.statistic["saving"]["upper_95_log_ratio"] >= 0


def test_hr3_a_never_raise_plant_not_exceeding_005_on_the_sharp_cells_is_incomplete():
    v = R.hr3_verdict(hr3_pairs(), hr3_plants(dll=0.03), cpu_spent=4.0)
    assert v.status == "incomplete" and any("never-raise" in r for r in v.incomplete_reasons)
    plants = hr3_plants()
    plants[3]["ll_plant"] = -1000.0 - 0.02                                         # one sharp cell it does not reach
    assert R.hr3_verdict(hr3_pairs(), plants, cpu_spent=4.0).status == "incomplete"
    assert R.hr3_verdict(hr3_pairs(), [p for p in hr3_plants() if not p["sharp"]], cpu_spent=4.0).status == "incomplete"


def test_hr3_a_pair_lacking_a_checked_final_value_is_incomplete():
    pairs = hr3_pairs()
    pairs[7]["checked"] = False
    v = R.hr3_verdict(pairs, hr3_plants(), cpu_spent=4.0)
    assert v.status == "incomplete" and any(pairs[7]["id"] in r for r in v.incomplete_reasons)
    assert R.hr3_verdict([], hr3_plants(), cpu_spent=0.0).status == "incomplete"


def test_hr3_past_ten_cpu_hours_is_capped():
    assert R.hr3_verdict(hr3_pairs(), hr3_plants(), cpu_spent=10.0).status == "capped"
    assert R.hr3_verdict(hr3_pairs(), hr3_plants(), cpu_spent=9.9).status == "pass"              # the twin
    assert R.HR3_CAP == 10.0


# ------------------------------------------------------------------------------------------------------------ HR-4
# PASS iff every byte is equal for W 1..15 x M in {21, 31, 64}, value and gradient, and the remainder plant differs
# wherever it drops a node; a wall-time speedup is claimed iff the paired wall ratio's upper bound is < 1 at W.
def hr4_rows():
    return [{"W": W, "M": M, "kind": kind, "equal": True} for W in range(1, 16) for M in (21, 31, 64)
            for kind in ("value", "grad")]


def hr4_plants(differs=True):
    return [{"W": W, "M": M, "dropped": M % W, "differs": bool(M % W) and differs} for W in range(1, 16)
            for M in (21, 31, 64)]


def test_hr4_one_byte_differing_fails_and_its_twin_passes():
    rows = hr4_rows()
    rows[40]["equal"] = False
    v = R.hr4_verdict(rows, hr4_plants())
    assert v.status == "fail" and v.statistic["unequal"] == [{"W": rows[40]["W"], "M": rows[40]["M"],
                                                               "kind": rows[40]["kind"]}]
    t = R.hr4_verdict(hr4_rows(), hr4_plants())                                                  # the twin
    assert t.status == "pass" and t.n_valid == 90


def test_hr4_the_remainder_plant_must_differ_where_it_drops_a_node():
    v = R.hr4_verdict(hr4_rows(), hr4_plants(differs=False))
    assert v.status == "fail" and v.statistic["plant_same"]
    assert R.hr4_verdict(hr4_rows(), [p for p in hr4_plants() if not p["dropped"]]).status == "incomplete"


def test_hr4_a_speedup_is_claimed_only_when_the_paired_wall_ratio_bound_is_below_one():
    walls = [{"W": 5, "M": M, "rep": i, "wall_pool": 0.3 + 0.01 * i, "wall_serial": 1.0} for M in (21, 31, 64)
             for i in range(3)]
    v = R.hr4_verdict(hr4_rows(), hr4_plants(), walls=walls, stated_W=5)
    assert v.statistic["speedup"]["claimed"] and v.statistic["speedup"]["upper_95_ratio"] < 1
    slow = [dict(w, wall_pool=1.0 + 0.05 * w["rep"]) for w in walls]
    assert not R.hr4_verdict(hr4_rows(), hr4_plants(), walls=slow, stated_W=5).statistic["speedup"]["claimed"]


# ------------------------------------------------------------------------------------------------------------ HR-5
# OD-5 (a): PASS iff >= 95 % of cells have measured / stated in [0.5, 2] and the median ratio is in [0.8, 1.25]; the
# M-blind predictor must fail the rule (else the plant is not informative: incomplete).
def hr5_cells(n_out=0, median=1.0, n=160):
    rows = []
    for i in range(n):
        r = 3.0 if i < n_out else median * (1.0 + 0.001 * ((i % 7) - 3))
        rows.append({"id": f"cell{i}", "M": 38, "stated_cpu_s": 10.0, "measured_cpu_s": 10.0 * r,
                     "blind_stated_cpu_s": 10.0 * 11 / 38})
    return rows


def test_hr5_nine_of_160_outside_fails_and_eight_passes():
    v = R.hr5_verdict(hr5_cells(n_out=9))
    assert v.status == "fail" and v.statistic["n_outside"] == 9 and v.n_valid == 160
    t = R.hr5_verdict(hr5_cells(n_out=8))                                                        # the twin
    assert t.status == "pass" and t.statistic["share_inside"] == 152 / 160


def test_hr5_a_median_ratio_of_13_fails_and_11_passes():
    assert R.hr5_verdict(hr5_cells(median=1.3)).status == "fail"
    assert R.hr5_verdict(hr5_cells(median=1.1)).status == "pass"                                # the twin


def test_hr5_an_m_blind_predictor_that_passes_makes_the_rule_incomplete():
    rows = [dict(r, blind_stated_cpu_s=r["stated_cpu_s"]) for r in hr5_cells()]
    v = R.hr5_verdict(rows)
    assert v.status == "incomplete" and any("M-blind" in r for r in v.incomplete_reasons)
    assert R.hr5_verdict(hr5_cells()).statistic["plant"]["status"] == "fail"                     # the twin
    assert R.hr5_verdict([]).status == "incomplete"


# ------------------------------------------------------------------------------------------------------------ HR-8
# tasks T063: PASS iff every sample's bytes equal across W, and the n-mod-W plant differs where it drops someone.
def hr8_rows():
    return [{"W": W, "n": 25_013, "equal": True} for W in (1, 7, 15)]


def test_hr8_one_byte_differing_fails_and_its_twin_passes():
    rows = hr8_rows()
    rows[1]["equal"] = False
    plants = [{"W": W, "dropped": 25_013 % W, "differs": bool(25_013 % W)} for W in (7, 15)]
    v = R.hr8_verdict(rows, plants)
    assert v.status == "fail" and v.statistic["unequal"] == [7]
    assert R.hr8_verdict(hr8_rows(), plants).status == "pass"                                    # the twin


def test_hr8s_plant_must_differ():
    same = [{"W": 7, "dropped": 2, "differs": False}]
    assert R.hr8_verdict(hr8_rows(), same).status == "fail"
    assert R.hr8_verdict(hr8_rows(), [{"W": 7, "dropped": 0, "differs": False}]).status == "incomplete"


# ------------------------------------------------------------------------------------------------------------ HR-9
# gate C3, like for like: the plant SRK(dt, m = 16) against SRK(dt/16, m = 1) must reject at alpha 0.05; the twin
# sraging(dt/16) against SRK(dt/16, 1) and the fix sraging(dt/16) against FP must not reject at alpha 0.025 each.
def hr9(plant_p=1e-6, twin_p=0.4, fix_p=0.3):
    return {"plant": {"D": 0.03, "p_value": plant_p, "n": 25_000}, "twin": {"D": 0.006, "p_value": twin_p, "n": 25_000},
            "fix": {"D": 0.005, "p_value": fix_p, "n": 25_000}}


def test_hr9_a_plant_that_does_not_reject_fails():
    v = R.hr9_verdict(hr9(plant_p=0.2))
    assert v.status == "fail" and "plant" in v.statistic["failed"]


@pytest.mark.parametrize("arm", ["twin", "fix"])
def test_hr9_a_must_pass_arm_rejecting_at_0025_fails(arm):
    v = R.hr9_verdict(hr9(**{f"{arm}_p": 0.01}))
    assert v.status == "fail" and arm in v.statistic["failed"]
    assert R.hr9_verdict(hr9(**{f"{arm}_p": 0.03})).status == "pass"                             # 0.03 >= 0.025


def test_hr9_all_as_required_passes():
    v = R.hr9_verdict(hr9())
    assert v.status == "pass" and v.crit == {"plant": 0.05, "twin": 0.025, "fix": 0.025}
    assert R.hr9_verdict({"plant": hr9()["plant"]}).status == "incomplete"                       # an arm not run


# ------------------------------------------------------------------------------------------------------------ HR-6
# tasks T076 (gate C1, C2): judged on the dt/2 rung, whose bias is b(dt/2) = D / (2^p - 1), p clipped to [1/2, 1]; a
# cell is assessable iff b(dt/2) <= crit/10 at m <= m_max; per-wall families (clamp G = 60, reflect G = 8); the
# eta(1 + delta) plant must reject; past the cap, capped.
N4 = 25_000


def crit6(wall):
    return {"clamp": 1.973, "reflect": 1.698}[wall] / math.sqrt(N4)


def hr6_cell(wall="clamp", ks_frac=0.5, bias_frac=0.5, p_hat=0.5, m=8, m_max=16, cid="c"):
    c = crit6(wall)
    D = bias_frac * (c / 10) * (2 ** min(max(p_hat, 0.5), 1.0) - 1)       # so that b(dt/2) = bias_frac crit/10
    return {"id": cid, "wall": wall, "n": N4, "D": D, "ks_D": ks_frac * c, "p_hat": p_hat, "m": m, "m_max": m_max}


def hr6_plants(reject=True):
    return [{"id": f"plant{i}", "ks_D": (2.0 if reject else 0.5) * crit6("clamp"), "n": N4} for i in (1, 2)]


def test_hr6_the_families_are_per_wall():
    assert R.HR6_CRIT == {"clamp": 1.973, "reflect": 1.698} and R.HR6_G == {"clamp": 60, "reflect": 8}
    v = R.hr6_verdict([hr6_cell("reflect")], [], wall="reflect", cpu_spent=1.0)
    assert v.wall == "reflect" and v.crit == pytest.approx(1.698 / math.sqrt(N4)) and v.status == "pass"


def test_hr6_a_cell_whose_judged_bias_is_above_crit_over_10_is_unassessable():
    v = R.hr6_verdict([hr6_cell(bias_frac=1.1)], hr6_plants(), wall="clamp", cpu_spent=1.0)
    assert v.status == "incomplete" and v.n_unassessable == 1
    t = R.hr6_verdict([hr6_cell(bias_frac=0.9)], hr6_plants(), wall="clamp", cpu_spent=1.0)            # the twin
    assert t.status == "pass" and t.n_valid == 1


def test_hr6_p_hat_is_clipped_to_a_half_and_one():
    c = hr6_cell(bias_frac=0.9, p_hat=1.0)
    for p in (0.2, 3.0):                                     # below 1/2 reads 1/2, above 1 reads 1
        cell = dict(c, p_hat=p)
        b = R.hr6_judged_bias(cell)
        assert b == pytest.approx(c["D"] / (2 ** min(max(p, 0.5), 1.0) - 1))


def test_hr6_a_clamp_cell_above_crit_fails_and_its_twin_passes():
    v = R.hr6_verdict([hr6_cell(ks_frac=1.05), hr6_cell(cid="d")], hr6_plants(), wall="clamp", cpu_spent=1.0)
    assert v.status == "fail" and v.statistic["failing"] == ["c"]
    assert R.hr6_verdict([hr6_cell(ks_frac=0.95)], hr6_plants(), wall="clamp", cpu_spent=1.0).status == "pass"


def test_hr6_a_plant_cell_not_rejecting_fails():
    v = R.hr6_verdict([hr6_cell()], hr6_plants(reject=False), wall="clamp", cpu_spent=1.0)
    assert v.status == "fail" and v.statistic["plant_not_rejecting"] == ["plant1", "plant2"]


def test_hr6_m_above_m_max_is_incomplete():
    v = R.hr6_verdict([hr6_cell(m=32, m_max=16)], hr6_plants(), wall="clamp", cpu_spent=1.0)
    assert v.status == "incomplete" and any("m_max" in r for r in v.incomplete_reasons)


def test_hr6_past_35_cpu_hours_is_capped():
    assert R.hr6_verdict([hr6_cell()], hr6_plants(), wall="clamp", cpu_spent=35.0).status == "capped"
    assert R.hr6_verdict([hr6_cell()], hr6_plants(), wall="clamp", cpu_spent=34.9).status == "pass"
    assert R.hr6_verdict([], hr6_plants(), wall="clamp", cpu_spent=0.0).status == "incomplete"


# ------------------------------------------------------------------------------------------------------------ HR-7
# tasks T083: a detection counts by any score component (Bonferroni over the K components); PASS iff the plant is
# detected and its twin is not; the twin detected -> FAIL; the plant undetected -> incomplete; at the cap (gate C5's
# third branch) -> capped, with the smallest detectable planted shift reported.
def arm(zs):
    return {"z": dict(zip(("eta", "beta", "epsilon", "kappa"), zs))}


def test_hr7_a_detection_by_any_component_counts_bonferroni_over_them():
    zc = R.hr7_zcrit(4, 0.05)
    from statistics import NormalDist
    assert zc == pytest.approx(NormalDist().inv_cdf(1 - 0.05 / 8))
    assert R.hr7_detected(arm([0.1, 0.2, zc + 0.01, 0.0]), zc) and not R.hr7_detected(arm([zc - 0.01] * 4), zc)


def test_hr7_the_plant_detected_and_its_twin_not_passes():
    v = R.hr7_verdict({"plant": arm([0.5, 6.0, 0.2, 0.1]), "twin": arm([0.5, 1.0, -0.3, 0.2])})
    assert v.status == "pass" and v.statistic["plant_detected"] and not v.statistic["twin_detected"]


def test_hr7_the_twin_detected_fails():
    v = R.hr7_verdict({"plant": arm([0.5, 6.0, 0.2, 0.1]), "twin": arm([3.5, 0.0, 0.0, 0.0])})
    assert v.status == "fail"


def test_hr7_the_plant_undetected_is_incomplete():
    v = R.hr7_verdict({"plant": arm([0.5, 1.0, 0.2, 0.1]), "twin": arm([0.5, 1.0, -0.3, 0.2])})
    assert v.status == "incomplete" and any("plant" in r for r in v.incomplete_reasons)


def test_hr7_at_the_cap_reads_capped_with_the_smallest_detectable_shift():
    v = R.hr7_verdict({"plant": arm([0.5, 6.0, 0.2, 0.1]), "twin": arm([0.5, 1.0, -0.3, 0.2])}, branch=3,
                      smallest_detectable=2.4)
    assert v.status == "capped" and v.statistic["smallest_detectable_nats_per_1e5"] == 2.4


def test_hr7_a_step_recheck_without_a_bias_is_incomplete_and_its_twin_passes():
    """A6 (Naveh 2026-11-04): s5-het re-checks S5's step by P2's coupled halving and cites that bias; a re-check that
    gives none leaves the claim without its evidence, so the verdict reads incomplete, never pass."""
    arms = {"plant": arm([0.5, 6.0, 0.2, 0.1]), "twin": arm([0.5, 1.0, -0.3, 0.2])}
    v = R.hr7_verdict(arms, recheck_error="ValueError: planted")
    assert v.status == "incomplete" and any("re-check of S5's step" in r for r in v.incomplete_reasons)
    assert R.hr7_verdict(arms, recheck_error=None).status == "pass"                 # twin: the re-check gave its bias
    bad_twin = {**arms, "twin": arm([3.5, 0.0, 0.0, 0.0])}
    assert R.hr7_verdict(bad_twin, recheck_error="x").status == "fail"              # a detected twin still fails



# ----------------------------------------------------------------------------------------------------------- HR-10
# tasks T093 (research R25; register §3): a verdict per wall; the clamp judges the censored D_c, reflect the full D,
# each on the dt/2 rung at crit = 1.797 / sqrt(n_alive) (Bonferroni, G = 16); a test is assessable iff its path bias
# D / (2^p - 1) <= crit/10 at m <= m_max; the deterministic checks hold; the eta(1 + delta) plant rejects and its
# delta = 0 twin does not; SRK's zero-for-dead recorder fails the NaN check.
NA = 10_000


def crit10():
    return 1.797 / math.sqrt(NA)


def hr10_test(wall="clamp", Dc_frac=0.5, D_frac=0.5, bias_frac=0.5, p_hat=0.5, m=8, m_max=16, tid="t", gerr_frac=0.1):
    c = crit10()
    return {"id": tid, "wall": wall, "n_alive": NA, "p0": 0.02 if wall == "clamp" else 0.0, "u_c": 0.06,
            "x_c": 0.1, "D_c": Dc_frac * c, "D": D_frac * c, "p_hat": p_hat, "m": m, "m_max": m_max,
            "D_bias": bias_frac * (c / 10) * (2 ** min(max(p_hat, 0.5), 1.0) - 1),
            "reference_grid_error": gerr_frac * c / 10}


def hr10_checks(nan_ok=True):
    return {"nan_iff_dead": {"holds": nan_ok}, "times_equal": {"holds": True}, "bytes_by_W": {"holds": True}}


def hr10_plants(plant_frac=2.0, twin_frac=0.5, srk_nan_holds=False):
    return [{"id": "eta_plant", "kind": "plant", "statistic": plant_frac * crit10(), "n_alive": NA},
            {"id": "eta_twin", "kind": "twin", "statistic": twin_frac * crit10(), "n_alive": NA},
            {"id": "srk_recorder", "kind": "srk_recorder", "nan_check": {"holds": srk_nan_holds}}]


def hr10(tests=None, checks=None, plants=None, wall="clamp", cpu=1.0):
    return R.hr10_verdict([hr10_test(wall)] if tests is None else tests, hr10_checks() if checks is None else checks,
                          hr10_plants() if plants is None else plants, wall=wall, cpu_spent=cpu)


def test_hr10_a_verdict_per_wall_at_g_16():
    assert R.HR10_CRIT == 1.797 and R.HR10_G == 16 and R.HR10_CAP == 6.0
    for wall in ("clamp", "reflect"):
        v = hr10(wall=wall)
        assert v.rule == "HR-10" and v.wall == wall and v.status == "pass"
    with pytest.raises(ValueError):
        hr10(wall="absorb")


def test_hr10_the_clamp_judges_d_c_and_reflect_the_full_d():
    lump = hr10_test("clamp", Dc_frac=0.9, D_frac=3.0)               # a lump: fails the full D, passes D_c
    assert hr10([lump]).status == "pass"
    assert hr10([hr10_test("clamp", Dc_frac=1.1, D_frac=1.1)]).status == "fail"
    assert hr10([hr10_test("reflect", Dc_frac=0.5, D_frac=1.1)], wall="reflect").status == "fail"
    assert hr10([hr10_test("reflect", Dc_frac=0.5, D_frac=0.9)], wall="reflect").status == "pass"      # the twin


def test_hr10_the_censoring_rule():
    assert R.hr10_censoring(0.002, eps_max=8e-3, dt=0.05, quantile=lambda u: 0.03) == pytest.approx(
        (0.01, max(0.03, 5 * math.sqrt(2 * 8e-3 * 0.05))))
    u, x = R.hr10_censoring(0.02, eps_max=1e-4, dt=0.01, quantile=lambda u: 2 * u)
    assert u == pytest.approx(0.06) and x == pytest.approx(0.12)


def test_hr10_the_eta_plant_not_rejecting_fails_and_its_twin_rejecting_fails():
    v = hr10(plants=hr10_plants(plant_frac=0.9))
    assert v.status == "fail" and v.statistic["plant_not_rejecting"] == ["eta_plant"]
    assert hr10(plants=hr10_plants(twin_frac=1.1)).status == "fail"
    assert hr10(plants=hr10_plants(plant_frac=1.1, twin_frac=0.9)).status == "pass"


def test_hr10_a_recorder_writing_zero_for_the_dead_fails_the_nan_check():
    times = np.array([5.0, 50.0, 500.0])
    ages = np.array([10.0, 100.0])
    good = np.array([[np.nan, np.nan], [0.3, np.nan], [0.1, 0.6]])
    srk = np.nan_to_num(good, nan=0.0)                                # SRK's getISR: zeros for the dead (D-411)
    ok, bad = R.hr10_nan_check(times, ages, good), R.hr10_nan_check(times, ages, srk)
    assert ok["holds"] and not bad["holds"] and bad["zeros_among_dead"] == 3
    v = hr10(checks={**hr10_checks(), "nan_iff_dead": bad})
    assert v.status == "fail" and v.statistic["checks_failing"] == ["nan_iff_dead"]
    # SRK's recorder must fail the NaN check; if it does not, the plant is not informative
    assert hr10(plants=hr10_plants(srk_nan_holds=True)).status == "incomplete"


def test_hr10_an_unassessable_test_reads_incomplete():
    v = hr10([hr10_test(bias_frac=1.1), hr10_test(tid="u")])
    assert v.status == "incomplete" and v.n_unassessable == 1
    assert hr10([hr10_test(m=32, m_max=16)]).status == "incomplete"
    assert hr10([hr10_test(gerr_frac=1.1)]).status == "incomplete"            # the reference's grid unconverged
    assert hr10([]).status == "incomplete"
    assert hr10(checks={"nan_iff_dead": {"holds": True}}).status == "incomplete"     # a check not run
    assert hr10(plants=hr10_plants()[:2]).status == "incomplete"                    # SRK's recorder not run


def test_hr10_past_the_cap_is_capped():
    assert hr10(cpu=6.0).status == "capped" and hr10(cpu=5.9).status == "pass"


# ----------------------------------------------------------------------------------------------------------- HR-11
# tasks T105 (research R27-R29; register §3): (i) the reductions' bytes under both walls, with the shared-stream plant
# breaking one; (ii) on the default clamp, each cell's dt/2 KS against the tensor reference at crit = 1.655 /
# sqrt(n) (G = 6), assessable iff its bias <= crit/10 at m <= m_max with a converged tensor reference; the composite
# cell within 1e-4; each informative dropped-spread plant (predicted sup|dS| >= 2 x 1.86 / sqrt(n)) rejecting;
# (iii) an above-budget statement reading "no FP reference", not clean.
N11 = 25_000


def crit11():
    return 1.655 / math.sqrt(N11)


def hr11_cell(D_frac=0.5, bias_frac=0.5, p_hat=0.5, m=8, m_max=16, cid="c", ref="tensor", verdict="converged"):
    c = crit11()
    return {"id": cid, "n": N11, "D": D_frac * c, "p_hat": p_hat, "m": m, "m_max": m_max,
            "D_bias": bias_frac * (c / 10) * (2 ** min(max(p_hat, 0.5), 1.0) - 1),
            "reference_kind": ref, "reference_verdict": verdict}


def hr11_reductions(equal=True, breaks=True):
    return [{"id": f"xc+eta0/{w}", "kind": "case", "wall": w, "equal": equal} for w in ("clamp", "reflect")] + \
           [{"id": "shared-stream", "kind": "plant", "breaks": breaks}]


def hr11_plants(D_frac=2.0, size_frac=3.0):
    return [{"id": "drop-eta", "n": N11, "D": D_frac * crit11(), "predicted_sup_dS": size_frac * 1.86 / math.sqrt(N11)}]


def hr11(reductions=None, cells=None, plants=None, composite=None, above=None, cpu=1.0):
    return R.hr11_verdict(hr11_reductions() if reductions is None else reductions,
                          [hr11_cell()] if cells is None else cells, hr11_plants() if plants is None else plants,
                          {"sup_dS": 5e-5} if composite is None else composite,
                          {"kind": "none", "clean": False} if above is None else above, cpu_spent=cpu)


def test_hr11_all_as_required_passes():
    assert R.HR11_CRIT == 1.655 and R.HR11_G == 6 and R.HR11_CAP == 4.0
    v = hr11()
    assert v.rule == "HR-11" and v.wall == "clamp" and v.status == "pass"
    assert v.crit == pytest.approx({"x": 1.655, "G": 6})


def test_hr11_a_reduction_byte_differing_fails_and_a_plant_not_breaking_is_incomplete():
    v = hr11(reductions=hr11_reductions(equal=False))
    assert v.status == "fail" and len(v.statistic["reductions_unequal"]) == 2
    assert hr11(reductions=hr11_reductions(breaks=False)).status == "incomplete"


def test_hr11_a_cell_above_crit_on_the_dt2_rung_fails():
    assert hr11(cells=[hr11_cell(D_frac=1.05), hr11_cell(cid="d")]).status == "fail"
    assert hr11(cells=[hr11_cell(D_frac=0.95)]).status == "pass"                                  # the twin


def test_hr11_an_unassessable_cell_is_incomplete():
    for cell in (hr11_cell(bias_frac=1.1), hr11_cell(m=32, m_max=16), hr11_cell(ref="none"),
                 hr11_cell(verdict="unconverged_at_cap")):
        v = hr11(cells=[cell, hr11_cell(cid="ok")])
        assert v.status == "incomplete" and v.n_unassessable == 1
    assert hr11(cells=[]).status == "incomplete"


def test_hr11_dropped_spread_plants():
    assert hr11(plants=hr11_plants(D_frac=0.9)).status == "fail"                   # informative, not rejecting
    v = hr11(plants=hr11_plants(size_frac=1.5))                                    # below 2 x the detection shift
    assert v.status == "incomplete" and any("informative" in r for r in v.incomplete_reasons)
    assert hr11(plants=[]).status == "incomplete"


def test_hr11_the_composite_cell_agrees_to_1e_4():
    assert hr11(composite={"sup_dS": 2e-4}).status == "fail"
    assert R.hr11_verdict(hr11_reductions(), [hr11_cell()], hr11_plants(), None, {"kind": "none", "clean": False},
                          cpu_spent=1.0).status == "incomplete"


def test_hr11_an_above_budget_statement_must_read_no_fp_reference():
    assert hr11(above={"kind": "tensor", "clean": True}).status == "fail"
    assert hr11(above={"kind": "none", "clean": True}).status == "fail"
    assert R.hr11_verdict(hr11_reductions(), [hr11_cell()], hr11_plants(), {"sup_dS": 0.0}, None,
                          cpu_spent=1.0).status == "incomplete"


def test_hr11_past_the_cap_is_capped():
    assert hr11(cpu=4.0).status == "capped" and hr11(cpu=3.9).status == "pass"


# --------------------------------------------- a cell that raised is an error row, read as incomplete (2026-10-30)
ERR = {"type": "ZeroDivisionError", "message": "float division by zero", "traceback": ["..."], "cpu_s": 3.0}


def test_raised_splits_error_rows_and_names_them():
    ok, why = R.raised([{"id": "a"}, {"id": "b", "error": ERR}])
    assert ok == [{"id": "a"}] and len(why) == 1 and "b" in why[0] and "ZeroDivisionError" in why[0]


def test_hr6_reads_a_raised_cell_or_plant_as_incomplete():
    v = R.hr6_verdict([hr6_cell(), {"id": "bad", "wall": "clamp", "n": N4, "error": ERR}], hr6_plants(),
                      wall="clamp", cpu_spent=1.0)
    assert v.status == "incomplete" and "bad" in v.statistic["unassessable"] and v.n_valid == 1
    p = R.hr6_verdict([hr6_cell()], hr6_plants()[:1] + [{"id": "plant2", "error": ERR}], wall="clamp", cpu_spent=1.0)
    assert p.status == "incomplete" and p.statistic["plant_not_rejecting"] == []
    assert R.hr6_verdict([hr6_cell()], hr6_plants(), wall="clamp", cpu_spent=1.0).status == "pass"      # the twin


def test_hr7_reads_raised_replicates_as_incomplete():
    arms = {"plant": arm([0.5, 6.0, 0.2, 0.1]), "twin": arm([0.5, 1.0, -0.3, 0.2])}
    v = R.hr7_verdict(arms, n_errors=2)
    assert v.status == "incomplete" and any("2 replicate" in r for r in v.incomplete_reasons)
    assert R.hr7_verdict(arms).status == "pass"                                                         # the twin
    assert R.hr7_verdict({"plant": arm([0.5, 6.0, 0.2, 0.1]), "twin": arm([3.5, 0, 0, 0])}, n_errors=2).status == "fail"


def test_hr10_reads_a_raised_configuration_or_plant_as_incomplete():
    v = hr10([hr10_test(), {"id": "cfg/raised", "error": ERR}])
    assert v.status == "incomplete" and "cfg/raised" in v.statistic["unassessable"]
    bad = [p if p["kind"] != "plant" else {**p, "error": ERR} for p in hr10_plants()]
    assert hr10(plants=bad).status == "incomplete"
    assert hr10().status == "pass"                                                                      # the twin


def test_hr11_reads_a_raised_cell_plant_or_part_as_incomplete():
    assert hr11(cells=[hr11_cell(), {"id": "bad", "n": N11, "error": ERR}]).status == "incomplete"
    assert hr11(plants=hr11_plants() + [{"id": "p2", "error": ERR}]).status == "incomplete"
    assert hr11(reductions=hr11_reductions() + [{"id": "reductions", "error": ERR}]).status == "incomplete"
    assert hr11(composite={"error": ERR}).status == "incomplete"
    assert hr11(above={"error": ERR}).status == "incomplete"
    assert hr11().status == "pass"                                                                      # the twin


# ----------------------------------- Naveh 2026-11-03 (ff794f5): the amendments A1-A3 of §3, each beside its twin
def test_a1_a_frozen_row_is_assessable_by_its_cells_pooled_bias_not_its_own_noisy_d():
    noisy = hr6_cell(bias_frac=1.5)                                   # its own D/(2^p - 1) is over crit/10
    assert R.hr6_verdict([noisy], hr6_plants(), wall="clamp", cpu_spent=1.0).status == "incomplete"    # the twin
    frozen = {**noisy, "frozen_b": 0.5 * crit6("clamp") / 10, "D_own": noisy["D"], "D_key_mean": noisy["D"]}
    v = R.hr6_verdict([frozen], hr6_plants(), wall="clamp", cpu_spent=1.0)
    assert v.status == "pass" and v.statistic["cells"][0]["frozen"]


def test_a1_a_borrowed_row_over_3x_its_cells_mean_per_key_d_is_unassessable():
    base = {**hr6_cell(), "frozen_b": 0.5 * crit6("clamp") / 10, "D_key_mean": 0.0015}
    over = {**base, "id": "b", "borrowed": True, "D_own": 3.01 * 0.0015}
    v = R.hr6_verdict([over], hr6_plants(), wall="clamp", cpu_spent=1.0)
    assert v.status == "incomplete" and any("3 x" in r for r in v.incomplete_reasons)
    assert R.hr6_verdict([{**over, "D_own": 2.99 * 0.0015}], hr6_plants(), wall="clamp",
                         cpu_spent=1.0).status == "pass"                                        # the twin, under 3x
    assert R.hr6_verdict([{**over, "borrowed": False}], hr6_plants(), wall="clamp",
                         cpu_spent=1.0).status == "pass"                                        # not borrowed: no guard


def test_a3_hr6s_crit_is_the_familys_actual_row_count():
    assert R.family_crit(60) == pytest.approx(1.973, abs=5e-4) and R.family_crit(8) == pytest.approx(1.698, abs=5e-4)
    v = R.hr6_verdict([hr6_cell()], hr6_plants(), wall="clamp", cpu_spent=1.0, G=46)
    assert v.crit == pytest.approx(R.family_crit(46) / math.sqrt(N4)) and v.statistic["G"] == 46
    tight = hr6_cell(ks_frac=1.0)                                     # at G 60's crit: over G 46's
    assert R.family_crit(46) < R.family_crit(60)
    assert R.hr6_verdict([{**tight, "ks_D": 1.9400 / math.sqrt(N4)}], hr6_plants(), wall="clamp", cpu_spent=1.0,
                         G=46).status == "fail"
    assert R.hr6_verdict([{**tight, "ks_D": 1.9400 / math.sqrt(N4)}], hr6_plants(), wall="clamp",
                         cpu_spent=1.0).status == "pass"                                       # the twin at G 60


def test_a2_hr10_gates_on_the_frozen_death_time_bias_and_reports_the_path_bias():
    t = hr10_test(bias_frac=4.0)                                      # the path bias alone would read unassessable
    assert hr10([t]).status == "incomplete"                                                     # the twin
    f = {**t, "frozen_b": 0.4 * crit10() / 10, "path_bias": t["D_bias"]}
    v = hr10([f])
    assert v.status == "pass" and v.statistic["tests"][0]["path_bias_reported"] == t["D_bias"]


def test_a1_hr11_checks_its_own_bar_at_the_frozen_step():
    bar11 = crit11() / 10
    ok = {**hr11_cell(bias_frac=5.0), "frozen_b": 0.00083 if 0.00083 <= bar11 else 0.9 * bar11, "borrowed": True,
          "D_own": 0.001, "D_key_mean": 0.0007}
    assert hr11(cells=[ok]).status == "pass"
    over = {**ok, "frozen_b": 1.01 * bar11}                            # over HR-11's own bar, though under S4's
    assert hr11(cells=[over]).status == "incomplete"
    assert hr11(cells=[{**ok, "D_own": 3.5 * 0.0007}]).status == "incomplete"                   # the guard
