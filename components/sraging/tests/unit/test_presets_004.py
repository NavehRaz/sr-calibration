"""Spec 004's Langevin presets (spec.md Clarifications, Naveh 2026-10-29; tasks T072): every hard-coded value in
tools/presets_004.py re-derived from SRK's own code in the oracle snapshot (getTheta(name, ExtH=True) at mode_overall,
get_config_params(name)), exactly; h_ext = exp(-ExtH) where SRK has an ExtH, else 0; P1's base step t_end / (nsteps -
1) / time_step_multiplier in the unit getTheta converts to. NOT APPLICABLE without a usable snapshot."""
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import presets_004 as PR  # noqa: E402

KEYS = ("eta", "beta", "epsilon", "xc", "kappa")              # getTheta's order


@pytest.fixture(scope="module")
def snapshot():
    import oracle_fp_piecewise as OP
    problems = OP.preflight() if os.path.isdir(OP.SNAP) else ["the snapshot is absent"]
    if problems:
        pytest.skip("NOT APPLICABLE: the oracle snapshot cannot be used here: " + "; ".join(problems))
    with OP.session() as s:
        yield {name: s.call("preset_theta", name=p["srk_name"]) for name, p in PR.PRESETS.items()}


@pytest.mark.parametrize("name", PR.NAMES)
def test_each_value_is_srks_own_exactly(snapshot, name):
    p, r = PR.PRESETS[name], snapshot[name]
    th = [float(x) for x in r["theta"]]
    assert [p["params"][k] for k in KEYS] == th[:5]
    ext = th[5]
    if math.isnan(ext):
        assert p["ext_h"] is None and p["params"]["h_ext"] == 0.0              # SRK: a NaN external_hazard kills none
    else:
        assert p["ext_h"] == ext and p["params"]["h_ext"] == math.exp(-ext)
    assert p["time_unit"] == str(r["time_unit"])
    st = p["srk_step"]
    assert (st["t_end"], st["nsteps"], st["time_step_multiplier"]) == (int(r["t_end"]), int(r["nsteps"]),
                                                                      int(r["time_step_multiplier"]))
    assert p["base_dt"] == float(int(r["t_end"])) / (int(r["nsteps"]) - 1) / int(r["time_step_multiplier"])


def test_the_decided_presets_their_units_and_base_steps():
    assert PR.NAMES == ("mice_F", "Jack_Russell", "ecoli", "human") and set(PR.PRESETS) == set(PR.NAMES)
    assert {n: PR.PRESETS[n]["srk_name"] for n in PR.NAMES} == {
        "mice_F": "mice_F", "Jack_Russell": "Jack_Russell_vetCompass", "ecoli": "ecoli", "human": "combined_human_M"}
    assert {n: PR.PRESETS[n]["time_unit"] for n in PR.NAMES} == {"mice_F": "weeks", "Jack_Russell": "years",
                                                                 "ecoli": "hours", "human": "years"}
    assert {n: round(PR.base_dt(n), 6) for n in PR.NAMES} == {"mice_F": 0.042809, "Jack_Russell": 0.0024,
                                                             "ecoli": 0.030006, "human": 0.003143}
    p = PR.preset("Jack_Russell")
    assert p.time_unit == "years" and p.h_ext == math.exp(-5.577) and p.xc == 1.0
    assert all(PR.preset(n).h_ext == 0.0 for n in ("mice_F", "ecoli", "human"))
    with pytest.raises(PR.PresetsUndecided):
        PR.preset("Labrador")


def test_the_decided_roles_and_s5_refused_by_name():
    """spec.md Clarifications, Session 2026-10-30 (4075cd5): Naveh's four roles; S5 waits for P1."""
    assert set(PR.ROLES) == {"paths", "p3", "p4", "hr8", "plants", "s5"} and PR.UNDECIDED_ROLES == ()
    assert PR.role("paths")["presets"] == list(PR.NAMES)
    p3 = PR.role("p3")
    assert (p3["preset"], p3["delta"], p3["key"], p3["ns"], p3["sd"]) == ("human", 1.0, 3003, [2_000, 20_000, 100_000],
                                                                          0.2)
    assert [c[1] for c in p3["configs"]] == ["eta", "xc"]
    assert (PR.role("p4")["preset"], PR.role("p4")["n"]) == ("human", 25_000)
    h = PR.role("hr8")
    assert (h["preset"], h["het"], h["n"], h["W"]) == ("mice_F", "hom", 25_013, [1, 7, 15])
    assert h["dt"] == {"step": ["mice_F|hom|clamp", "dt"]}                  # the frozen clamp step, by name
    pl = PR.role("plants")
    plants = [k for k in pl if isinstance(pl[k], dict) and "preset" in pl[k]]
    assert plants and all(pl[k]["preset"] == "mice_F" for k in plants)
    assert (pl["recording"]["dt"], pl["recording"]["delta"]) == (1.0, 0.25)
    assert pl["hr9"]["dt"] == {"p1": ["b1", "mice_F", "hom", "dt_for_target"]} and pl["hr9"]["m"] == 16
    assert pl["d411"]["age"] == {"p1": ["cell_ages", "mice_F|hom|clamp", 0]}
    assert pl["b2"]["step_size"] == 214 / 4999 and pl["d410"]["age"] == 10 * 214 / 4999        # 2026-11-03 item 11
    assert pl["hr9"]["fix_dt"] == {"step": ["mice_F|hom|clamp", "dt"]}                 # A4: the frozen step
    with pytest.raises(PR.PresetsUndecided, match=r"not decided \(Naveh\)"):
        PR.role("p9")


def test_s5s_role_is_score_het_002s_cohort_at_theta0():
    """spec.md Clarifications 2026-11-01 (8f0f269): pure-k0.01-p00-c0-s0, n 2,000, xc sd 0.2, gen-1's transform."""
    import json
    from sraging import generator as GEN
    import s5_gen1_004 as S5
    r = PR.role("s5")
    spec = GEN.CohortSpec(**r["spec"])
    p, het = S5.params_of(spec)
    th = json.load(open(os.path.join(REPO, "tests", "acceptance", "results", "score_het_002.json")))["theta0"]
    for k in ("eta", "beta", "epsilon", "kappa"):
        assert getattr(p, k) == pytest.approx(th[k], rel=1e-12)
    assert (spec.n, spec.het_param, spec.het_var, het.parameter, het.sd) == (2_000, "xc", 0.2, "xc", 0.2)
    assert r["dt_clamp"] == {"step": ["s5|het|clamp", "dt"]} and r["dt_reflect"] == {"step": ["s5|het|reflect", "dt"]}


def test_references_resolve_from_p1_and_the_frozen_steps_and_are_refused_without_them():
    p1 = {"b1": {"mice_F": {"hom": {"dt_for_target": 0.31, "target": 0.02}}},
          "cell_ages": {"mice_F|hom|clamp": [80.0, 110.0]}}
    steps = {"mice_F|hom|clamp": {"dt": 0.0107, "steps_per_individual": 2.0e4},
             "s5|het|clamp": {"dt": 0.002, "steps_per_individual": 5.0e4}}
    h = PR.resolve(PR.role("hr8"), p1, steps=steps)
    assert h["dt"] == 0.0107 and h["W"] == [1, 7, 15]
    pl = PR.resolve(PR.role("plants"), p1, steps=steps)
    assert pl["hr9"]["dt"] == 0.31 and pl["hr9"]["predicted_lag"] == 0.02 and pl["d411"]["age"] == 80.0
    assert pl["hr9"]["fix_dt"] == 0.0107                                     # A4: the frozen step
    s5 = PR.resolve(PR.role("s5"), steps=steps, only=("dt_clamp", "steps_per_individual"))
    assert s5["dt_clamp"] == 0.002 and s5["dt_reflect"] == {"step": ["s5|het|reflect", "dt"]}   # resolved when needed
    with pytest.raises(PR.PresetsUndecided, match="frozen"):
        PR.resolve(PR.role("hr8"), p1)
