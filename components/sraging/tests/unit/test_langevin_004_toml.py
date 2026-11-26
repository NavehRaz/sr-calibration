"""Freeze-L's file (tasks T115; data-model "freeze-L, langevin_004.toml"; Naveh's decisions of 2026-11-03, ff794f5):
it parses with every section; naveh_ok_l records Naveh's OK (T119, 2026-11-04); every [judged] step equals the frozen
step it cites,
m_max twice it; E. coli has left the clamp family with its limit stated; HR-6's G is the grid's actual row count per
family; HR-10's reflect-only clamp tests are the five named; the amendments A0-A5 are named; no named reference is left:
[s5]'s values are P2's result (p2_s5_power_004.json), and the values computed from code (the plants' delta, HR-11's
plant sizes, the above-budget node count) are filled, the cheap ones re-computed here."""
import math
import os
import sys
import tomllib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_004 as ACC  # noqa: E402
import p2_s5_power_004 as P2  # noqa: E402

LANG = os.path.join(REPO, "tests", "acceptance", "langevin_004.toml")


NAVEH_OK_L = ("2026-11-04: OK freeze-L (Recommended), with S5 at m 64 + re-check (A6) and HR-11 (iii) on the "
              "4-parameter cell (A7); spec.md Clarifications 2026-11-04")


@pytest.fixture(scope="module")
def lang():
    with open(LANG, "rb") as fh:
        return tomllib.load(fh)


def test_it_parses_with_every_section_and_waits_for_navehs_ok(lang):
    assert {"rules", "judged", "s4", "s5", "paths", "multi", "plants", "caps", "amendments", "limits",
            "naveh_ok_l"} <= set(lang)
    assert lang["naveh_ok_l"] == NAVEH_OK_L and "open" not in lang
    assert lang["judged"]["branch"] == "frozen from P1 at pooled n" and lang["judged"]["rung"] == "dt/2"
    assert {k: lang["caps"][k] for k in ("s4", "s5-hom", "s5-het", "paths", "multi", "plants", "hr8")} == {
        "s4": 35.0, "s5-hom": 1.5, "s5-het": 86.0, "paths": 6.0, "multi": 4.0, "plants": 3.0, "hr8": 0.5}
    assert set(lang["amendments"]) == {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9"}


def test_every_judged_step_is_the_frozen_step_it_cites(lang):
    fs = ACC.frozen_steps()
    assert fs.source == "p1-pooled+p1-pooled-b"
    steps = lang["judged"]["steps"]
    assert set(steps) == set(dict.keys(fs))
    for k, s in steps.items():
        f = dict.__getitem__(fs, k)
        assert s["met"] is bool(f["met"]) and lang["judged"]["p_hat"][k] == f["p_hat"]
        assert s["pooled_b"] == f["pooled_b"] and s["D_key_mean"] == pytest.approx(f["D_key_mean"], rel=1e-12)
        if f["met"]:
            assert (s["m"], s["dt"], s["m_max"]) == (f["m"], f["dt"], 2 * f["m"]) and s["pooled_b"] <= s["bar"]
        else:
            assert "m" not in s and "dt" not in s and s["pooled_b"] > s["bar"]


def test_ecoli_has_left_the_clamp_family_with_its_limit(lang):
    import s4_fp_langevin_004 as S4
    assert lang["limits"]["ecoli_clamp"] == "the default clamp is not shown to agree with FP for E. coli at any " \
                                            "affordable step"
    assert lang["s4"]["clamp_excluded"] == ["ecoli"] == lang["paths"]["clamp_excluded"]
    grid = S4.build_grid(lang["s4"], ACC.p1_steps())
    assert not any(r["preset"] == "ecoli" for r in grid["clamp"])
    assert sum(r["preset"] == "ecoli" for r in grid["reflect"]) == 2                   # reflect keeps both cells


def test_hr6s_g_is_the_grids_actual_row_count_per_family(lang):
    import s4_fp_langevin_004 as S4
    from sraging import sampler_rules as R
    grid = S4.build_grid(lang["s4"], ACC.p1_steps())
    assert lang["s4"]["G"] == {"clamp": len(grid["clamp"]), "reflect": len(grid["reflect"])} == {"clamp": 46,
                                                                                                 "reflect": 8}
    assert lang["s4"]["crit_x"] == {w: R.family_crit(g) for w, g in lang["s4"]["G"].items()}


def test_hr10s_five_named_clamp_tests_are_judged_on_reflect(lang):
    import json
    ro = {(r["cell"], i) for r in lang["paths"]["reflect_only"] for i in r["ages"]}
    assert ro == {("ecoli|hom|clamp", 0), ("ecoli|hom|clamp", 1), ("ecoli|het|clamp", 0), ("ecoli|het|clamp", 1),
                  ("Jack_Russell|het|clamp", 0)}
    p1 = json.load(open(os.path.join(REPO, "tests", "acceptance", "results", "p1_langevin_004.json")))
    for c in p1["cells"]:                       # exactly the clamp tests over crit at P1's m 64
        if c["wall"] != "clamp":
            continue
        f = max(c["ladder"] + c["extension"], key=lambda r: r["m"])
        for j, x in enumerate(f["paths"]):
            assert ((c["id"], j) in ro) == (x["ks_D_censored"] > 1.797 / math.sqrt(x["n_alive"]))


def test_the_decided_items_are_filled(lang):
    assert lang["s4"]["cvs"] == [0.1, 0.2, 0.3] and lang["s4"]["plant_cells"] == ["Jack_Russell/v0/clamp",
                                                                               "mice_F/xc/cv0.2/clamp"]
    assert set(lang["paths"]["ages"]) == {f"{p}|{h}" for p in ("mice_F", "Jack_Russell", "ecoli", "human")
                                          for h in ("hom", "het")}
    assert lang["paths"]["plant"]["preset"] == "mice_F" and lang["multi"]["preset"] == "mice_F"
    fs = ACC.frozen_steps()
    assert lang["plants"]["hr9"]["fix_dt"] == fs["mice_F|hom|clamp"]["dt"]                         # A4
    assert "step_size" in lang["plants"]["b2"] and "age" in lang["plants"]["d410"]


def _references(x, path=""):
    """Every {p2 = ...} or {t119 = ...} table left in the parsed file, by its path."""
    if isinstance(x, dict):
        here = [path] if {"p2", "t119"} & set(x) else []
        return here + [r for k, v in x.items() for r in _references(v, f"{path}.{k}")]
    if isinstance(x, list):
        return [r for i, v in enumerate(x) for r in _references(v, f"{path}[{i}]")]
    return []


# None left: the one open at T115 (HR-11 (iii)'s node count) closed with A7 (Naveh 2026-11-04).
OPEN_REFERENCES = []


def test_no_named_reference_is_left(lang):
    assert _references(lang) == OPEN_REFERENCES
    text = open(LANG).read()
    assert "p2 =" not in text and text.count("t119 =") == len(OPEN_REFERENCES) and "<wall>" not in text


def test_s5s_values_are_p2s_result(lang):
    import json
    p2 = json.load(open(os.path.join(REPO, "tests", "acceptance", "results", "p2_s5_power_004.json")))
    assert p2["status"] == "written" and p2["errors"] == []
    s5, wall = lang["s5"], p2["branch"]["wall"]
    h = p2["halving"][wall]
    assert s5["R"] == p2["R"]["R_score"] and s5["branch"] == p2["branch"]
    frozen = ACC.frozen_steps()[f"s5|het|{wall}"]
    assert frozen["dt"] == h["dt"] and frozen["m"] == 16                               # P2 ran at the frozen step
    # A6 (Naveh 2026-11-04): S5 at m 64, dt / 4, its bias re-measured by s5-het's re-check
    assert (s5["m"], s5["dt"]) == (64, frozen["dt"] / 4) and s5["recheck_at_step"] is True
    assert P2.A6_C2_CLAUSE in lang["amendments"]["A6"] and "m 64" in lang["amendments"]["A6"]
    assert s5["cpu_per_cohort_s"] == h["cpu_per_cohort_s"] * 4 and "step_bias_in_se" not in s5
    assert s5["step_bias_in_se_at_frozen_m"] == h["bias_in_se"] and s5["sd_per_replicate"] == p2["R"]["sd_per_replicate"]
    assert ACC.s5_het_projected_cpu_h(s5) <= P2.CAP_S5_HET                           # branch 1 holds at m 64
    assert s5["plant_sd"] == p2["plant"]["plant_sd"]
    assert s5["smallest_detectable_nats_per_1e5"] == p2["smallest_detectable_nats_per_1e5"]
    assert s5["spec"] == p2["config"]["spec"] and s5["key"] == p2["config"]["key"]


def test_the_values_computed_from_code_are_filled(lang):
    from sraging.engine.tensor_reference import BUDGET
    s4 = lang["s4"]
    assert list(s4["plant_delta"]) == s4["plant_cells"] == list(s4["plant_predicted_sup_dS"])
    for pid, d in s4["plant_delta"].items():
        assert 0 < d < 0.5 and s4["plant_predicted_sup_dS"][pid] == pytest.approx(s4["plant_target_sup_dS"], rel=0.01)
    assert 0 < lang["paths"]["plant"]["delta"] < 0.5
    assert [p["cell"] for p in lang["multi"]["plants"]] == [c["id"] for c in lang["multi"]["cells"]]
    assert all(p["predicted_sup_dS"] > 0 for p in lang["multi"]["plants"])
    ab = lang["multi"]["above_budget"]                                                 # A7: the 4-parameter cell
    assert lang["multi"]["budget"] == BUDGET and "node_count" not in ab
    assert (ab["members"], ab["n"], ab["t_max"]) == ({"xc": 0.2, "eta": 0.2, "beta": 0.2, "epsilon": 0.2}, 25000, 214.0)
    assert ab["reason"] == "4 heterogeneous parameters with a spread: the tensor reference takes d <= 3 (research R28)"


def test_the_cheap_computed_values_are_their_functions_now(lang):
    """S4's homogeneous plant and HR-10's plant (a few CPU-s each) re-computed by the functions LANG names."""
    import paths_004 as PA
    import presets_004 as PR
    import s4_fp_langevin_004 as S4
    assert lang["s4"]["plant_delta"]["Jack_Russell/v0/clamp"] == S4.plant_delta(PR.preset("Jack_Russell"))["delta"]
    pl = lang["paths"]["plant"]
    assert pl["delta"] == PA.plant_delta(PR.preset(pl["preset"]), pl["age"], lang["paths"]["n"])["delta"]


def test_the_rules_constants_are_the_rule_codes(lang):
    from sraging import sampler_rules as R
    r = lang["rules"]
    assert (r["HR-10"]["crit"], r["HR-10"]["G"], r["HR-10"]["cap"]) == (R.HR10_CRIT, R.HR10_G, R.HR10_CAP)
    assert (r["HR-11"]["crit"], r["HR-11"]["G"], r["HR-11"]["cap"]) == (R.HR11_CRIT, R.HR11_G, R.HR11_CAP)
    assert r["HR-9"]["alpha"] == R.HR9_ALPHA and r["HR-6"]["guard_x"] == R.GUARD_X == 3.0


def test_the_recording_plant_steps_finer_than_its_grid(lang):
    """A8 (Naveh 2026-11-04, T121): at m 1 sraging stepped at Delta t and every death fell on the Delta t grid, so D-408's
    plant could not be informative (plants_004.json at 252ad3a)."""
    rec = lang["plants"]["recording"]
    assert rec["m"] == 16 and rec["delta"] < rec["dt"] and rec["dt"] / rec["m"] < rec["delta"]
    assert "m = 16" in lang["amendments"]["A8"] and "D-408" in lang["amendments"]["A8"]


def test_multi_reruns_once_at_fresh_keys_with_the_composite_at_1e_4(lang):
    """A9 (Naveh 2026-11-05, T126): after the first run (xc+eta's D 0.01112 above crit 0.01047, diagnosed as chance),
    multi re-runs once at a fresh key base, clear of the first run's keys (40600-42600) and the diagnosis's (50700-50702),
    and its composite cross-checks the tensor reference at 1e-4."""
    m = lang["multi"]
    assert m["key"] == 60600 and m["composite"]["bar"] == 0.0001 == m["composite"]["agree"]
    used = {m["key"] + 100 + i for i in range(len(m["cells"]))} | {m["key"] + 1000 + i for i in range(len(m["plants"]))}
    used |= {m["key"], m["key"] + 2000}
    first = {40600, 42600} | set(range(40700, 40706)) | set(range(41600, 41606))
    assert not used & (first | {50700, 50701, 50702})
    assert "re-runs once" in lang["amendments"]["A9"] and "1e-4" in lang["amendments"]["A9"]
