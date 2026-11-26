"""Spec 004's acceptance frame (tools/acceptance_004.py; tasks T004-T005): SR, the stop rule of contracts/acceptance.md
(gate C4, Naveh 2026-10-29), its CPU accounting and the close's reading of `capped`; the phases' prerequisites; the
`--naveh-ok` argument. Each planted case beside its twin."""
import math
import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_004 as ACC  # noqa: E402

H = 3600.0
OK = "2026-11-02: Naveh: go to 40 for S4"


class Clock:
    """A fake process clock: the parent's CPU seconds, advanced by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# ------------------------------------------------------------------------------------------------- SR 1-2: refuse
def test_a_projection_above_the_cap_is_refused_naming_the_cap_and_one_below_starts(capsys):
    with pytest.raises(ACC.Refused, match=r"cap 35 CPU-h"):
        ACC.stop_rule("s4", 36.0, 35.0, None, source="p1_langevin_004.json")
    b = ACC.stop_rule("s4", 34.0, 35.0, None, source="p1_langevin_004.json")                     # twin
    assert b.cap_cpu_h == 35.0 and not b.capped and b.naveh_ok is None
    out = capsys.readouterr().out
    assert "projected 34 CPU-h (p1_langevin_004.json) against the cap 35 CPU-h" in out          # SR 1: printed first


def test_a_projection_that_is_not_a_number_is_refused():
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ACC.Refused, match="projection"):
            ACC.stop_rule("hr3", bad, 10.0, None)
    assert ACC.stop_rule("hr3", 0.0, 10.0, None).projected_cpu_h == 0.0                         # twin


# ------------------------------------------------------------------------------------- SR 3: stop at the cap
def _cells(budget, clock, n=5, worker_h=0.4, parent_h=0.0):
    ran = []

    def work(cell):
        clock.t += parent_h * H
        ran.append(cell)
        return {"cell": cell, "value": cell * 10}, worker_h * H
    return ACC.run_capped(budget, range(n), work), ran


def test_cpu_reaching_the_cap_mid_run_stops_it_capped_keeping_the_finished_cells():
    clock = Clock()
    b = ACC.stop_rule("hr3", 0.5, 1.0, None, clock=clock)
    rows, ran = _cells(b, clock)                       # 0.4 CPU-h of worker CPU per cell: 1.2 after the third
    assert ran == [0, 1, 2] and [r["cell"] for r in rows] == [0, 1, 2]
    assert b.capped and ACC.phase_status("pass", b) == "capped"
    rec = b.record()
    assert rec["capped"] and rec["cpu_spent_h"] == pytest.approx(1.2) and rec["cap_cpu_h"] == 1.0
    assert rec["projected_cpu_h"] == 0.5 and rec["naveh_ok"] is None


def test_a_run_inside_its_cap_finishes_every_cell_and_keeps_its_status():
    clock = Clock()
    b = ACC.stop_rule("hr3", 0.5, 2.5, None, clock=clock)
    rows, ran = _cells(b, clock)                       # twin: 2.0 CPU-h in all, under 2.5
    assert ran == [0, 1, 2, 3, 4] and len(rows) == 5
    assert not b.capped and ACC.phase_status("pass", b) == "pass"


def test_the_cpu_is_the_parents_process_time_plus_the_workers_reported_cpu():
    clock = Clock()
    clock.t = 100.0                                    # the parent's CPU before the phase is not the phase's
    b = ACC.stop_rule("hr3", 0.5, 1.0, None, clock=clock)
    rows, ran = _cells(b, clock, worker_h=0.125, parent_h=0.125)     # 0.25 CPU-h per cell: the cap after the fourth
    assert ran == [0, 1, 2, 3] and b.capped
    rec = b.record()
    assert rec["parent_cpu_s"] == 4 * 0.125 * H and rec["worker_cpu_s"] == 4 * 0.125 * H
    assert rec["cpu_spent_h"] == 1.0
    clock = Clock()
    b = ACC.stop_rule("hr3", 0.5, 1.0, None, clock=clock)
    rows, ran = _cells(b, clock, worker_h=0.125)       # twin: the same workers without the parent's CPU stay under it
    assert len(ran) == 5 and not b.capped and b.record()["cpu_spent_h"] == 5 * 0.125


# --------------------------------------------------------------- hr3's crash fix (Naveh 2026-10-30): error rows
HR3_CELLS = [("c-a", "syn-sharp", 0.2, True), ("c-b", "human-10k", 0.2, False), ("c-c", "syn-mid", 0.3, False)]


def _hr3_work(raise_on=(), clock=None):
    def work(cell):
        cid, stratum, sd, sharp = cell
        if clock is not None:
            clock.t += 7.0
        if cid in raise_on:
            raise ZeroDivisionError("float division by zero")
        pid = f"{cid}/sd{sd:g}"
        pair = {"id": pid, "kind": "max", "sharp": sharp, "ll_adaptive": -100.0, "ll_full": -100.01, "M_common": 17,
                "cpu_adaptive": 1.0, "cpu_full": 2.0, "checked": True}
        plant = {"id": pid, "sharp": sharp, "ll_plant": -101.0, "ll_full": -100.01}
        return {"pairs": [pair], "plant": plant, "cohort_id": cid, "stratum": stratum, "sd": sd, "cpu_s": 7.0}, 0.0
    return work


def _hr3(tmp_path, raise_on=()):
    from sraging import sampler_rules as R
    clock = Clock()
    b = ACC.stop_rule("hr3", 0.5, 10.0, None, clock=clock)
    out = ACC.hr3_run(b, HR3_CELLS, _hr3_work(raise_on, clock), progress_path=tmp_path / "hr3_progress.jsonl")
    pairs, plants = ACC.hr3_rows(out)
    return out, R.hr3_verdict(pairs, plants, cpu_spent=b.spent_cpu_h, cap=10.0)


def test_hr3_a_cell_that_raises_is_an_error_row_and_the_verdict_reads_incomplete(tmp_path):
    out, v = _hr3(tmp_path, raise_on=("c-b",))
    assert len(out) == 3 and [o.get("error") is None for o in out] == [True, False, True]
    err = out[1]
    assert err["error"]["type"] == "ZeroDivisionError" and "division by zero" in err["error"]["message"]
    assert err["error"]["traceback"] and err["cpu_s"] == 7.0 and err["cohort_id"] == "c-b" and err["plant"] is None
    (pair,) = err["pairs"]
    assert pair["id"] == "c-b/sd0.2" and pair["checked"] is False and pair["kind"] == "error"
    assert v.status == "incomplete" and any("c-b/sd0.2" in r for r in v.incomplete_reasons)
    assert v.n_unassessable == 1 and v.statistic["saving"]["n"] == 2          # the error row adds no CPU ratio
    _, twin = _hr3(tmp_path / "twin", raise_on=())                              # the twin: none raising
    assert twin.status == "pass"


def test_hr3_prints_one_progress_line_per_cell(tmp_path, capsys):
    _hr3(tmp_path, raise_on=("c-b",))
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[hr3] cell ")]
    assert len(lines) == 3
    assert "1/3" in lines[0] and "c-a/sd0.2" in lines[0] and "sd 0.2" in lines[0] and "ok" in lines[0]
    assert "2/3" in lines[1] and "error" in lines[1] and "ZeroDivisionError" in lines[1]
    assert "cell CPU 7 s" in lines[2] and "of 10 CPU-h" in lines[2]


def test_hr3_appends_one_json_line_per_cell(tmp_path):
    import json
    _hr3(tmp_path, raise_on=("c-b",))
    rows = [json.loads(x) for x in open(tmp_path / "hr3_progress.jsonl")]
    assert [r["index"] for r in rows] == [1, 2, 3] and [r["status"] for r in rows] == ["ok", "error", "ok"]
    assert rows[1]["row"]["error"]["type"] == "ZeroDivisionError"


def test_run_capped_without_the_opt_in_still_raises():
    clock = Clock()
    b = ACC.stop_rule("hr3", 0.5, 10.0, None, clock=clock)
    with pytest.raises(ZeroDivisionError):
        ACC.run_capped(b, HR3_CELLS, _hr3_work(("c-b",)))


# ------------------------------------------- every long phase: a raising cell is an error row (Naveh 2026-10-30)
def _robust(tmp_path, raise_on=(), phase="s4"):
    clock = Clock()
    b = ACC.stop_rule(phase, 0.5, 10.0, None, clock=clock)

    def work(c):
        clock.t += 5.0
        if c in raise_on:
            raise RuntimeError(f"cell {c} broke")
        return {"id": c, "value": 1}, 0.0
    return ACC.robust_run(phase, b, ["a", "b", "c"], work, path=tmp_path / "p.jsonl"), b


def test_robust_run_keeps_a_raising_cell_as_an_error_row_and_goes_on(tmp_path, capsys):
    import json
    rows, b = _robust(tmp_path, raise_on=("b",))
    assert [r["id"] for r in rows] == ["a", "b", "c"] and rows[1]["error"]["type"] == "RuntimeError"
    assert rows[1]["error"]["message"] == "cell b broke" and rows[1]["error"]["cpu_s"] == 5.0
    assert rows[1]["error"]["traceback"] and ACC.errors_of(rows) == [{"id": "b", **rows[1]["error"]}]
    out = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[s4] cell ")]
    assert len(out) == 3 and "2/3 b: error (RuntimeError" in out[1] and "of 10 CPU-h" in out[2]
    lines = [json.loads(x) for x in open(tmp_path / "p.jsonl")]
    assert [x["status"] for x in lines] == ["ok", "error", "ok"] and lines[0]["phase"] == "s4"
    twin, _ = _robust(tmp_path / "t")
    assert ACC.errors_of(twin) == []
    assert ACC.progress_path("p1-langevin").endswith(os.path.join(".cache", "p1_langevin_004_progress.jsonl"))


def test_each_phases_error_row_reads_incomplete():
    from sraging import sampler_rules as R
    err = {"type": "ValueError", "message": "x", "traceback": [], "cpu_s": 1.0}
    s4 = ACC.s4_error_row({"id": "mice_F/eta/cv0.1/clamp", "wall": "clamp", "n": 25_000}, err)
    assert R.hr6_verdict([s4], [{"id": "p", "ks_D": 1.0, "n": 25_000}], wall="clamp", cpu_spent=0.0).status == \
        "incomplete"
    pa = ACC.paths_error_row({"id": "mice_F/v0/sd0/clamp", "wall": "clamp"}, err)
    v = R.hr10_verdict(pa["tests"], {}, [], wall="clamp", cpu_spent=0.0)
    assert v.status == "incomplete" and v.n_unassessable == 1
    pl = ACC.plant_error_row("hr9", err)
    assert pl["status"] == "incomplete" and ACC.combined_status(["pass", pl["status"]]) == "incomplete"
    assert R.hr11_verdict([], [{"id": "xc+eta", "n": 25_000, "error": err}], [], None, None,
                          cpu_spent=0.0).status == "incomplete"


def test_p1s_result_leaves_out_a_raised_cell_and_names_it():
    import p1_langevin_004 as P1
    out = P1.result([{"id": "mice_F|hom|clamp", "preset": "mice_F", "error": {"type": "E", "message": "m"}}])
    assert out["steps"] == {} and out["cell_errors"] == [{"id": "mice_F|hom|clamp", "type": "E", "message": "m"}]


def test_s5s_arms_leave_out_a_raised_replicate(monkeypatch):
    import numpy as np
    import s5_gen1_004 as S5
    monkeypatch.setattr(S5, "law", lambda spec, **kw: {"P": None})
    monkeypatch.setattr(S5, "spec_gradient", lambda spec, base: None)
    monkeypatch.setattr(S5, "with_fields", lambda spec, **kw: spec)
    monkeypatch.setattr(S5, "horizon", lambda spec: 10.0)
    monkeypatch.setattr(S5, "counts", lambda a, e, base: None)
    rng = np.random.default_rng(5)
    monkeypatch.setattr(S5, "score", lambda c, d: rng.random(4))
    monkeypatch.setattr(S5, "ks_gen1", lambda t, tm, spec: (0.01, 0.5))
    reps = [{"ages": [1.0], "events": [1], "times": np.ones(3), "t_max": 10.0, "cpu_s": 1.0} for _ in range(3)]
    reps.insert(1, {"id": "replicate 1", "error": {"type": "E", "message": "m"}})
    out = S5.run(object(), 4, 0.1, "clamp", 0, 0.25, work=lambda R_: reps)
    assert out["R"] == 3 and out["errors"] == [{"id": "replicate 1", "type": "E", "message": "m"}]


def test_p2s_halving_keeps_a_raising_cohort_and_goes_on(monkeypatch, capsys):
    import numpy as np
    import p2_s5_power_004 as P2
    from sraging.langevin import sample as LS
    monkeypatch.setattr(P2.S5, "law", lambda spec, **kw: {})
    monkeypatch.setattr(P2.S5, "horizon", lambda spec: 1.0)
    monkeypatch.setattr(P2.S5, "spec_gradient", lambda spec, base: None)
    monkeypatch.setattr(P2.S5, "params_of", lambda spec: (None, None))
    calls = []

    def config(**kw):
        calls.append(kw["key"])
        if kw["key"] == 101:
            raise ZeroDivisionError("planted")
        return type("C", (), kw)()
    monkeypatch.setattr(LS, "LangevinConfig", config)
    monkeypatch.setattr(LS, "prepare", lambda cfg: {"arrays": None, "t_max": 1.0, "ref": None})
    monkeypatch.setattr(LS, "run_rung", lambda cfg, a, t, s: (np.ones(2), np.ones(2)))
    monkeypatch.setattr(P2.S5, "counts", lambda a, e, base: None)
    rng = np.random.default_rng(6)
    monkeypatch.setattr(P2.S5, "score", lambda c, d: rng.random(4))
    spec = type("S", (), {"n": 10, "delta": 1.0, "censor_frac": 0.0, "time_range": None})()
    h = P2.coupled_halving(spec, 0.1, "clamp", 100, cohorts=4)
    assert h["cohorts"] == 3 and [e["cohort"] for e in h["errors"]] == [1] and calls == [100, 101, 102, 103]
    assert sum("cohort 2/4: error (ZeroDivisionError" in x for x in capsys.readouterr().out.splitlines()) == 1


def _halving_on_a_toy_law(monkeypatch, times):
    """P2's halving with the real S5.record, counts and score on a toy law gen-1 would give at censor_frac 0 (death
    cells 0, 1, 2 days), its horizon 2.5 days; the rungs' raw times are `times`."""
    import numpy as np
    import p2_s5_power_004 as P2
    from sraging.langevin import sample as LS
    monkeypatch.setattr(P2.S5, "law", lambda spec, **kw: {"ages": np.array([0.0, 1.0, 2.0]),
                                                          "events": np.array([1, 1, 1])})
    monkeypatch.setattr(P2.S5, "horizon", lambda spec: 2.5)
    monkeypatch.setattr(P2.S5, "spec_gradient", lambda spec, base: np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0],
                                                                            [0, 0, 1.0, 0]]))
    monkeypatch.setattr(P2.S5, "params_of", lambda spec: (None, None))
    cfgs = []

    def config(**kw):
        cfgs.append(kw)
        return type("C", (), kw)()
    monkeypatch.setattr(LS, "LangevinConfig", config)
    monkeypatch.setattr(LS, "prepare", lambda cfg: {"arrays": None, "t_max": cfg.t_max, "ref": None})
    monkeypatch.setattr(LS, "run_rung", lambda cfg, a, t, s: (np.array(times[s - 1]), np.zeros(len(times[s - 1]))))
    spec = type("S", (), {"n": 3, "delta": 1.0, "censor_frac": 0.0, "time_range": None})()
    return P2.coupled_halving(spec, 0.1, "clamp", 100, cohorts=2), cfgs


def test_planted_p2s_halving_maps_a_survivor_at_the_horizon_to_gen1s_last_cell(monkeypatch):
    """P2's 2026-11-03 error (31ec891, cohorts 53 and 69): a survivor at the horizon, before the fix recorded censored
    at the sampler's t_max, a cell gen-1's law does not have. Now the paths run to gen-1's horizon and the survivor
    sits at gen-1's last node: a death in the last cell, counted in n_beyond."""
    h, cfgs = _halving_on_a_toy_law(monkeypatch, ([0.4, 1.2, 2.5], [0.4, 1.7, 2.2]))
    assert h["errors"] == [] and h["cohorts"] == 2 and h["n_beyond"] == 2 and h["t_max"] == 2.5
    assert [c["t_max"] for c in cfgs] == [2.5, 2.5]                    # run to gen-1's horizon, not the default
    assert h["mean"] == [0.0, 0.0, 0.0, 0.0]                           # (1,1,1) - (1,1,1) per cohort


def test_twin_p2s_halving_on_ordinary_records_is_unchanged(monkeypatch):
    h, _ = _halving_on_a_toy_law(monkeypatch, ([0.4, 1.2, 2.3], [0.4, 0.7, 2.2]))
    assert h["errors"] == [] and h["cohorts"] == 2 and h["n_beyond"] == 0
    assert h["mean"] == [-1.0, 1.0, 0.0, 0.0]                          # (1,1,1) - (2,0,1)


# ----------------------------------------------------------------------------- the close: capped is never PASS
def test_capped_and_incomplete_are_never_counted_as_pass_by_the_close():
    t = ACC.close_tally({"hr1": "pass", "hr3": "capped", "hr2": "incomplete", "hr4": "PASS"})
    assert not t["all_pass"] and t["pass"] == 2 and t["capped"] == ["hr3"] and t["incomplete"] == ["hr2"]
    assert not ACC.counts_as_pass("capped") and not ACC.counts_as_pass("incomplete")
    twin = ACC.close_tally({"hr1": "pass", "hr4": "PASS"})
    assert twin["all_pass"] and twin["pass"] == 2 and twin["capped"] == []
    assert not ACC.close_tally({})["all_pass"]                    # nothing to count is not a pass


# --------------------------------------------------------------------------- SR 4: only a new naveh_ok lifts it
def test_a_new_naveh_ok_naming_the_phase_and_a_higher_cap_lifts_the_refusal():
    b = ACC.stop_rule("s4", 36.0, 35.0, ACC.NavehOk("s4", 40.0, OK))
    assert b.cap_cpu_h == 40.0 and b.record()["naveh_ok"] == {"phase": "s4", "cap_cpu_h": 40.0, "said": OK}
    assert b.record()["cap_frozen_cpu_h"] == 35.0


def test_a_naveh_ok_naming_another_phase_does_not_lift_it():
    with pytest.raises(ACC.Refused, match=r"cap 35 CPU-h"):
        ACC.stop_rule("s4", 36.0, 35.0, ACC.NavehOk("s5-het", 40.0, OK))
    with pytest.raises(ACC.Refused, match=r"cap 35 CPU-h"):
        ACC.stop_rule("s4", 36.0, 35.0, [ACC.NavehOk("s5-het", 90.0, OK), ACC.NavehOk("hr3", 40.0, OK)])
    assert ACC.stop_rule("s4", 36.0, 35.0, [ACC.NavehOk("s5-het", 90.0, OK),                     # twin, in a list
                                            ACC.NavehOk("s4", 40.0, OK)]).cap_cpu_h == 40.0


# ------------------------------------------------------------------------------------- --naveh-ok on the command line
def _args(phase, ok=None, cap=None):
    return ACC.parser().parse_args([phase] + (["--naveh-ok", ok] if ok else []) +
                                   (["--naveh-ok-cap", str(cap)] if cap is not None else []))


def test_a_pilots_naveh_ok_names_the_phase_and_its_cap_and_is_recorded():
    oks = ACC.naveh_oks("p1-langevin", _args("p1-langevin", OK, 8))
    assert oks == (ACC.NavehOk("p1-langevin", 8.0, OK),)
    assert ACC.stop_rule("p1-langevin", 7.0, ACC.PILOT_CAPS["p1-langevin"], oks).cap_cpu_h == 8.0
    assert ACC.naveh_oks("p1-langevin", _args("p1-langevin")) == ()                              # twin: none given


@pytest.mark.parametrize("phase,ok,cap,match", [
    ("s4", OK, 40, "langevin_004.toml"),               # a frozen phase's cap is lifted only in its freeze file (SR 4)
    ("hr3", OK, 12, "panel_004.toml"),
    ("p1-langevin", OK, None, "--naveh-ok-cap"),       # his OK without the cap it sets
    ("p1-langevin", None, 8, "--naveh-ok"),            # a cap without his OK
    ("p1-langevin", "go ahead", 8, "date"),            # not "<date>: <his words>"
])
def test_a_naveh_ok_the_command_line_cannot_carry_is_refused(phase, ok, cap, match):
    with pytest.raises(ACC.Refused, match=match):
        ACC.naveh_oks(phase, _args(phase, ok, cap))


def test_the_pilot_caps_are_the_tasks_and_every_phase_has_a_subcommand():
    assert ACC.PILOT_CAPS == {"p5-inventory": 0.1, "p0-het-cost": 1.0, "hr2-noise": 0.2, "p1-langevin": 6.0,
                              "p2-s5-power": 6.0, "p3-cost": 3.0, "p4-jax": 1.0, "p1-pooled": 45.0,
                              "p1-pooled-b": 15.0}
    assert ACC.PHASE_NAMES == ("p5-inventory", "p0-het-cost", "hr2-noise", "panel", "reference", "hr1", "hr5", "hr2",
                               "hr3-pilot", "hr3", "hr4", "d401", "p1-langevin", "p1-pooled", "p1-pooled-b",
                               "p2-s5-power",
                               "p3-cost", "p4-jax",
                               "plants", "hr8", "s4", "paths", "multi", "s5-hom", "s5-het", "docs", "unchanged")
    assert set(ACC.PHASES) == set(ACC.PHASE_NAMES) == set(ACC.PREREQUISITES)


# ------------------------------------------------------------------------------------------------- prerequisites
def test_a_phase_with_no_prerequisite_has_nothing_missing_and_a_frozen_one_waits_for_its_freeze(monkeypatch):
    """On a tree with no freeze committed (monkeypatched: since freeze-H is committed, the real tree would let main
    start a real phase)."""
    _tree(monkeypatch, {})
    assert ACC.missing_prerequisites("p5-inventory") == []
    for phase, path in (("hr2", "panel_004.toml"), ("d401", "panel_004.toml"), ("multi", "langevin_004.toml")):
        assert any(path in m for m in ACC.missing_prerequisites(phase)), phase
    with pytest.raises(ACC.Refused, match="freeze-H"):
        ACC.main(["hr4"])


def _tree(monkeypatch, files: dict, results: dict = (), modules=("sraging.engine.mixture", "sraging.engine.pool")):
    """A committed tree of freeze files and results, as the prerequisite checks read it."""
    monkeypatch.setattr(ACC, "committed", lambda rel: rel in files or rel in dict(results))
    monkeypatch.setattr(ACC, "read_toml", lambda rel: files[rel])
    monkeypatch.setattr(ACC, "read_result", lambda name: dict(results)[ACC.result_rel(name)])
    monkeypatch.setattr(ACC, "file_sha256", lambda rel: "ab" * 32)
    monkeypatch.setattr(ACC, "has_module", lambda name: name in modules)


def test_a_freeze_counts_only_committed_with_navehs_ok(monkeypatch):
    _tree(monkeypatch, {ACC.PANEL_PATH: {"naveh_ok_h": ""}, ACC.IDS_PATH: {}})
    assert any("naveh_ok_h is empty" in m for m in ACC.missing_prerequisites("hr2"))
    _tree(monkeypatch, {ACC.PANEL_PATH: {"naveh_ok_h": "2026-11-01: OK"}, ACC.IDS_PATH: {}})
    assert ACC.missing_prerequisites("hr2") == []                                               # twin


def test_a_result_prerequisite_needs_its_status_from_an_acceptance_run(monkeypatch):
    frz = {ACC.PANEL_PATH: {"naveh_ok_h": "2026-11-01: OK"}, ACC.IDS_PATH: {}}
    run = {"status_for_acceptance": "acceptance", "tree_dirty": False}
    _tree(monkeypatch, frz, {ACC.result_rel("panel"): {**run, "status": "fail"}})
    assert any("panel_004.json" in m and "fail" in m for m in ACC.missing_prerequisites("reference"))
    _tree(monkeypatch, frz, {ACC.result_rel("panel"): {**run, "status": "pass",
                                                       "status_for_acceptance": "development"}})
    assert any("development" in m for m in ACC.missing_prerequisites("reference"))
    _tree(monkeypatch, frz, {ACC.result_rel("panel"): {**run, "status": "pass"}})
    assert ACC.missing_prerequisites("reference") == []                                         # twin


def test_hr1_waits_for_the_reference_to_be_pinned_by_its_sha256(monkeypatch):
    res = {ACC.result_rel("reference"): {"status": "written", "status_for_acceptance": "acceptance"}}
    frz = {ACC.PANEL_PATH: {"naveh_ok_h": "2026-11-01: OK", "reference": {"sha256": "cd" * 32}}, ACC.IDS_PATH: {}}
    _tree(monkeypatch, frz, res)
    assert any("not pinned" in m for m in ACC.missing_prerequisites("hr1"))
    frz[ACC.PANEL_PATH]["reference"]["sha256"] = "ab" * 32
    assert ACC.missing_prerequisites("hr1") == []                                               # twin
    res[ACC.result_rel("reference")]["status"] = "fail"                  # a failed reference is never pinned
    assert any("'fail'" in m for m in ACC.missing_prerequisites("hr1"))


def test_a_phase_waits_for_the_code_it_runs(monkeypatch):
    _tree(monkeypatch, {}, modules=("sraging.engine.mixture",))
    assert any("sraging.engine.pool" in m for m in ACC.missing_prerequisites("p0-het-cost"))
    _tree(monkeypatch, {})
    assert ACC.missing_prerequisites("p0-het-cost") == []                                       # twin


def test_main_dispatches_only_once_the_prerequisites_hold(monkeypatch):
    monkeypatch.setitem(ACC.PHASES, "p5-inventory", ACC._todo("p5-inventory"))     # never runs a real phase here
    with pytest.raises(SystemExit, match="not implemented yet"):
        ACC.main(["p5-inventory"])


# ------------------------------------------------------------------------------- US1's phases (tasks T025), no run
def test_the_heterogeneity_halfs_phases_are_registered_and_the_rest_wait_for_their_stories():
    done = {"p5-inventory", "hr2-noise", "panel", "reference", "hr1", "hr2", "d401",          # US1 (T025)
            "p0-het-cost", "hr3-pilot", "hr3", "hr4", "hr5",                                  # US2 (T036)
            "s4", "p3-cost",                                                                  # US4 (T080, T081)
            "p2-s5-power", "s5-het", "s5-hom",                                                # US5 (T087-T089)
            "paths",                                                                          # US7 (T098)
            "multi",                                                                          # US8 (T109)
            "p1-langevin", "p4-jax", "plants", "hr8",                                         # US3 (T072-T074)
            "p1-pooled", "p1-pooled-b",                                                       # T111b, T111c
            "docs", "unchanged"}                                                              # the close (§5)
    assert {p for p, f in ACC.PHASES.items() if f.__name__.startswith("phase_")} == done == set(ACC.PHASE_NAMES)


def test_a_frozen_cap_is_read_from_the_freeze_and_a_shared_one_is_spent_by_its_phases(monkeypatch):
    frz = {ACC.PANEL_PATH: {"naveh_ok_h": "2026-11-01: OK", "caps": {"reference": 2.0, "hr2": 0.5}}, ACC.IDS_PATH: {}}
    ref = {"status": "written", "status_for_acceptance": "acceptance", "stop_rule": {"cpu_spent_h": 0.3}}
    _tree(monkeypatch, frz)
    assert ACC.frozen_cap("hr2")[0] == 0.5 and ACC.frozen_cap("hr1")[0] == 2.0
    _tree(monkeypatch, frz, {ACC.result_rel("reference"): ref})
    left, rec = ACC.frozen_cap("hr1")
    assert left == pytest.approx(1.7) and rec["spent_by"] == {"reference": 0.3} and rec["key"] == "reference"
    assert ACC.frozen_cap("reference")[0] == 2.0                                    # its own spend is not counted
    with pytest.raises(ACC.Refused, match="d401"):
        ACC.frozen_cap("d401")                                                      # a cap the freeze does not set


def test_a_projection_needs_p0s_committed_per_node_cost(monkeypatch):
    _tree(monkeypatch, {})
    with pytest.raises(ACC.Refused, match="p0_het_cost_004.json is not committed"):
        ACC.p0_per_node()
    res = {"status": "written", "status_for_acceptance": "acceptance",
           "per_node_cpu_s": {"value": 0.01, "value_and_grad": 0.033}, "compile_cpu_s": 4.0}
    _tree(monkeypatch, {}, {ACC.result_rel("p0-het-cost"): res})
    assert ACC.p0_per_node()["value_and_grad"] == 0.033                                         # the twin
    _tree(monkeypatch, {}, {ACC.result_rel("p0-het-cost"): dict(res, status_for_acceptance="development")})
    with pytest.raises(ACC.Refused, match="development"):
        ACC.p0_per_node()


def test_the_noise_pilot_takes_the_h_with_the_smallest_floor_and_ten_times_it():
    rows = [{"h": h, "abs_diff": d} for h, ds in ((1e-3, (2e-6, 5e-6)), (1e-4, (3e-8, 1e-7)), (1e-5, (4e-7, 2e-7)))
            for d in ds]
    s = ACC.noise_summary(rows)
    assert s["h"] == 1e-4 and s["tol"] == pytest.approx(1e-6) and s["floor"]["0.001"] == 5e-6
    assert ACC.noise_summary([]) == {"floor": {}, "h": None, "tol": None}


def test_the_plants_cross_where_the_lowest_gauss_hermite_node_meets_zero():
    assert ACC.plant_crossing(5) == pytest.approx(1 / 2.856970, rel=1e-6)            # research R10: z = -2.857
    assert ACC.plant_crossing(3) == pytest.approx(3 ** -0.5, rel=1e-12)


def test_fr009s_rule_is_gen1s_161_node_law():
    from sraging import generator as GEN
    r = ACC.gen1_rule()
    d = GEN._het_nodes(0.2, 161)
    assert (r.M, r.spread, r.definition) == (161, 0.2, "gen1_hat")
    assert r.z.tobytes() == d["z"].tobytes() and r.w.tobytes() == d["W"].tobytes()
    assert abs(r.w.sum() - 1.0) < 1e-14 and r.s_min > 0


def test_fr009_fails_off_by_more_than_01_and_is_incomplete_on_an_invalid_reference():
    row = {"cohort_id": "c", "ref_valid": True, "abs_diff": 0.08}
    assert ACC.fr009_verdict([row, dict(row, cohort_id="d")])["status"] == "pass"              # the twin
    assert ACC.fr009_verdict([row, dict(row, cohort_id="d", abs_diff=0.12)])["over_tol"] == ["d"]
    assert ACC.fr009_verdict([row, dict(row, cohort_id="d", ref_valid=False)])["status"] == "incomplete"
    assert ACC.fr009_verdict([])["status"] == "incomplete"


def test_a_combined_status_is_a_pass_only_when_every_part_passes():
    assert ACC.combined_status(["pass", "pass"]) == "pass"
    assert ACC.combined_status(["pass", "incomplete"]) == "incomplete"
    assert ACC.combined_status(["incomplete", "fail"]) == "fail"
    assert ACC.combined_status(["pass", "capped"]) == "capped"


def test_the_reference_cells_are_every_panel_cohort_at_four_sd_with_the_composite_subset_marked():
    rows = [{"cohort_id": "a", "stratum": "syn-sharp", "roles": ["panel", "subset-composite@0.2"]},
            {"cohort_id": "b", "stratum": "human-10k", "roles": ["panel"]}]
    cells = ACC.reference_cells(rows)
    assert len(cells) == 8 and [c for c in cells if c[3]] == [("a", "syn-sharp", 0.2, True)]


def test_a_ladder_the_freeze_did_not_freeze_is_refused():
    ok = {"ladder": {"rungs": [5, 8, 11, 17, 25, 38, 57, 85, 128, 192], "threshold": 0.05, "cap": 192}}
    ACC.require_frozen_ladder(ok)                                                                # the twin
    with pytest.raises(ACC.Refused, match="ladder"):
        ACC.require_frozen_ladder({"ladder": dict(ok["ladder"], threshold=0.1)})


@pytest.mark.engine
def test_hr2s_point_and_plant_machinery_on_a_small_cohort():
    """cd_vs_grad agrees on the truncated normal; the normal_frac_sd plant at M = 5 jumps and is seen across v*."""
    import numpy as np
    from sraging import Cohort, SRParams
    rng = np.random.default_rng(3)
    t = np.sort(np.floor(rng.gamma(9.0, 14.0, 300)) + 1.0)
    coh = Cohort.from_arrays(t, (rng.random(300) < 0.9).astype(int), delta=1.0)
    theta = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
    ev = ACC.evaluator_for(coh, 0.0)
    pt = ACC.cd_vs_grad(ev, theta, 0.35, 11, 1e-4)
    assert pt["abs_diff"] < 1e-4 * max(1.0, abs(pt["grad"]))
    plant = ACC.plant_row(coh, theta, 0.0, 5, 1e-4)
    assert plant["jump"] > 1e-3 and plant["abs_diff"] > 10 * plant["jump"]
    assert plant["v_cross"] == ACC.plant_crossing(5)


V6 = os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset",   # release 0.1.1: the lab's checkouts
                  "sr_calibration_tester/data/library/v6/manifest.json")


@pytest.mark.skipif(not os.path.exists(V6), reason="NOT APPLICABLE: the v6 library is absent on this host")
def test_a_realfull_cohort_loads_with_its_window_and_hazard_rate():
    import math
    import numpy as np
    man, specs = ACC.v6_specs()
    cid = "realfull-real-Sweden_M_1910_hetro-c2-s0"
    coh, theta, h = ACC.v6_case(cid, man, specs)
    lo, hi = specs[cid]["window"]
    assert coh.window == (lo, hi) and np.all(np.asarray(coh.ages) >= lo) and np.all(np.asarray(coh.ages) <= hi)
    assert h == theta.h_ext == math.exp(-specs[cid]["external_hazard"])
    plain, _, h0 = ACC.v6_case("het-k0.01-p00-c0-s0", man, specs)
    assert plain.window is None and h0 == 0.0 and plain.n == specs["het-k0.01-p00-c0-s0"]["n"]


# ------------------------------------------------------------------------------- US2's phases (tasks T036), no run
def test_hr3_pilots_cpu_counts_against_hr3s_cap(monkeypatch):
    frz = {ACC.PANEL_PATH: {"naveh_ok_h": "2026-11-01: OK", "caps": {"hr3-pilot": 1.0, "hr3": 10.0}}, ACC.IDS_PATH: {}}
    pilot = {"status": "written", "status_for_acceptance": "acceptance", "stop_rule": {"cpu_spent_h": 0.6}}
    _tree(monkeypatch, frz, {ACC.result_rel("hr3-pilot"): pilot})
    left, rec = ACC.frozen_cap("hr3")
    assert left == pytest.approx(9.4) and rec["spent_by"] == {"hr3-pilot": 0.6}
    assert ACC.frozen_cap("hr3-pilot")[0] == 1.0                                                 # the twin


def test_hr3_takes_a_capped_pilot_only_with_navehs_ok_naming_hr3(monkeypatch):
    """Naveh 2026-10-29 (T052): a capped hr3-pilot's projection serves hr3 only under a committed [[naveh_ok_cap]]
    naming hr3; a written pilot needs none; either way the pilot must carry the projection."""
    pan = {"naveh_ok_h": "2026-11-01: OK"}
    pilot = {"status": "capped", "status_for_acceptance": "acceptance", "hr3_projection_cpu_h": 24.9}

    def missing(freeze, result):
        _tree(monkeypatch, {ACC.PANEL_PATH: freeze, ACC.IDS_PATH: {}}, {ACC.result_rel("hr3-pilot"): result})
        return ACC.missing_prerequisites("hr3")
    assert any("naming hr3" in m for m in missing(pan, pilot))
    other = {**pan, "naveh_ok_cap": [{"phase": "hr4", "cap_cpu_h": 30.0, "said": OK}]}
    assert any("naming hr3" in m for m in missing(other, pilot))                              # an OK for another phase
    ok = {**pan, "naveh_ok_cap": [{"phase": "hr3", "cap_cpu_h": 30.0, "said": OK}]}
    assert missing(ok, pilot) == []                                                           # the twin
    assert missing(pan, {**pilot, "status": "written"}) == []                                 # written needs no OK
    no_projection = {k: v for k, v in pilot.items() if k != "hr3_projection_cpu_h"}
    assert any("hr3_projection_cpu_h" in m for m in missing(ok, no_projection))


def test_hr3_and_hr4_need_their_frozen_constants():
    with pytest.raises(ACC.Refused, match=r"\[rules.HR-3\] lacks \['start', 'profile', 'sharp_strata'\]"):
        ACC.hr3_rules({"rules": {}})
    r = ACC.hr3_rules({"rules": {"HR-3": {"start": {"eta": 1.5}, "profile": {"parameter": "epsilon", "factors": [0.8]},
                                          "sharp_strata": ["syn-sharp"]}}})
    assert r["sds"] == [0.2, 0.3] and r["threshold"] == 0.05                                     # the twin
    with pytest.raises(ACC.Refused, match="HR-4"):
        ACC.hr4_rules({"rules": {"HR-4": {"cohort": "x"}}})
    assert ACC.hr4_rules({"rules": {"HR-4": {"cohort": "x", "stated_W": 5}}})["stated_W"] == 5


def test_hr3s_cells_are_the_subset_at_two_sd_and_the_pilot_one_sharp_one_human():
    rows = [{"cohort_id": c, "stratum": s, "roles": ["panel", "subset-hr3"]} for c, s in
            (("a", "syn-broad"), ("b", "syn-sharp"), ("c", "human-25k"), ("d", "human-100k"))]
    rows.append({"cohort_id": "e", "stratum": "syn-mid", "roles": ["panel"]})
    rules = {"sds": [0.2, 0.3], "sharp_strata": ["syn-sharp", "human-25k", "human-100k"]}
    cells = ACC.hr3_cells(rows, rules)
    assert len(cells) == 8 and {c[0] for c in cells} == {"a", "b", "c", "d"}
    assert [c for c in cells if not c[3]] == [("a", "syn-broad", 0.2, False), ("a", "syn-broad", 0.3, False)]
    assert ACC.hr3_cells(rows, rules, pilot=True) == [("b", "syn-sharp", 0.2, True), ("c", "human-25k", 0.2, True)]


def test_hr5s_cells_sum_each_hr1_cells_statements_and_the_blind_ones():
    st = [{"phase": "guide", "M": 6, "t_hom_cpu_s": 0.01, "factor": 0.9, "stated_cpu_s": 0.054, "measured_cpu_s": 0.05},
          {"phase": "rung", "M": 38, "t_hom_cpu_s": 0.01, "factor": 0.9, "stated_cpu_s": 0.342, "measured_cpu_s": 0.4}]
    (c,) = ACC.hr5_cells([{"id": "x/sd0.2", "M_used": 38, "cost": {"statements": st}}])
    assert c["stated_cpu_s"] == pytest.approx(0.396) and c["measured_cpu_s"] == pytest.approx(0.45)
    assert c["blind_stated_cpu_s"] == pytest.approx(2 * 11 * 0.01 * 0.9)


def test_p0s_projection_is_from_the_probe_and_inside_its_cap_and_its_summary_is_what_the_projections_read():
    import json
    import p0_het_cost_004 as P0
    probe = json.load(open(os.path.join(REPO, "specs", "004-samplers", "evidence", "cost_probe.json")))
    p = P0.projection_cpu_h(probe)
    assert 0.05 < p < ACC.PILOT_CAPS["p0-het-cost"]
    assert len(P0.COHORTS) == 6 and set(P0.COHORTS) < set(__import__("panel_004").HELD_OUT)
    row = {"n": 100, "per_node_value_cpu_s": 0.009, "per_node_value_and_grad_cpu_s": 0.03, "compile_cpu_s": 5.0,
           "f_value": 0.9, "f_grad": 3.2, "log_sd_value": 0.1,
           "gradients": {"per_node_rel_vs_cd": 1e-8, "vmapped_rel_vs_cd": 1e-8}, "finite_at_192": {"finite": True}}
    s = P0.summarise([row, dict(row, per_node_value_cpu_s=0.011)], [{"value_equal": True, "grad_equal": {"eta": True}}])
    assert s["per_node_cpu_s"] == {"value": pytest.approx(0.01), "value_and_grad": 0.03} and s["compile_cpu_s"] == 5.0
    assert s["identity_smoke_all_equal"] and s["all_finite_at_192"]


@pytest.mark.engine
def test_hr3s_pair_and_hr4s_cell_on_a_small_cohort():
    """The pair: both arms agree at the common M and are checked; the plant stops at M_low. The cell: pooled bytes
    equal in process, and the remainder plant differs where it drops a node."""
    import numpy as np
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.engine.inputs import Heterogeneity
    from sraging.engine.pool import NodePool
    rng = np.random.default_rng(3)
    t = np.sort(np.floor(rng.gamma(9.0, 14.0, 300)) + 1.0)
    coh = Cohort.from_arrays(t, (rng.random(300) < 0.9).astype(int), delta=1.0)
    theta = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
    grid = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
    e = Evaluator(coh, numerics="v1", grid=grid, free=("eta", "epsilon"), fixed={"beta": 0.08, "kappa": 0.1},
                  heterogeneity=Heterogeneity(parameter="xc", sd=0.2))
    rules = {"start": {"eta": 1.2}, "profile": {"parameter": "epsilon", "factors": [0.9]}, "sharp_strata": []}
    out = ACC.hr3_pair(e, theta, 0.2, rules, "x/sd0.2", False)
    assert [p["kind"] for p in out["pairs"]] == ["max", "profile"]
    assert all(abs(p["ll_adaptive"] - p["ll_full"]) < 1e-3 and p["checked"] for p in out["pairs"])   # 1.4e-6 [M]
    assert out["plant"]["M_plant"] <= out["plant"]["M_common"]
    e7 = Evaluator(coh, numerics="v1", grid=grid, heterogeneity=Heterogeneity(parameter="xc", sd=0.2, M=7,
                                                                                check=False))
    with NodePool(e7, workers=2) as pool:
        cell = ACC.hr4_cell(e7, theta, 2, 7, pool, stated_W=5)
    assert all(r["equal"] for r in cell["rows"]) and cell["plant"]["dropped"] == 1 and cell["plant"]["differs"]


# ------------------------------------------------------------------------------- US4's phases (tasks T080-T081), no run
def test_an_unknown_pilot_configuration_is_refused():
    import presets_004 as PR                                  # every role is decided (tests/unit/test_presets_004.py)
    with pytest.raises(PR.PresetsUndecided, match="not decided"):
        PR.role("p9")


def test_the_langevin_phases_wait_for_p1s_committed_result(monkeypatch):
    _tree(monkeypatch, {})
    with pytest.raises(ACC.Refused, match="p1_langevin_004.json is not committed"):
        ACC.p1_result()


def test_p3_draws_gen1_only_inside_its_domain():
    import p3_cost_004 as P3
    from sraging import SRParams
    p = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, h_ext=1e-3)
    spec, why = P3.cohort_spec(p, "xc", 2000, 3, 1.0, 120.0)
    assert why is None and spec.heterogeneous and spec.het_var == 0.2 and spec.external_hazard == pytest.approx(
        -__import__("math").log(1e-3)) and spec.rho_eta == pytest.approx(6e-4 * 120.0 ** 2)
    spec, why = P3.cohort_spec(p, "eta", 2000, 3, 1.0, 120.0)
    assert spec is None and "xc only" in why
    rows = [{"config": "in-table", "n": 2000, "gen1": {"in_domain": True, "cpu_s": 3.0}, "langevin": {"cpu_s": 1.0}},
            {"config": "out-of-table", "n": 2000, "gen1": {"in_domain": False, "reason": why},
             "langevin": {"cpu_s": 1.5}}]
    b = P3.boundary(rows)
    assert [r["engine"] for r in b["rows"]] == ["gen-1", "langevin"] and b["rows"][1]["reason"] == why


# ------------------------------------------------------------------------------- US5's phases (tasks T087-T089), no run
def test_s5_hom_draws_002s_configuration_on_sragings_kernel():
    pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")
    import numpy as np
    import score_004 as SC
    out = SC.SraSession().call("simulate_sr_hetro", steps=601, dt=0.5, npeople=60, seed=1_000_010, eta=6e-4, beta=0.08,
                               kappa=0.1, epsilon=8e-3, xc=1.0, xc_var=0.2)
    assert out["death_times"].shape == (60,) and set(np.unique(out["events"])) <= {0, 1}
    assert np.all(out["death_times"] <= 300.0)                            # SRK's horizon (steps - 1) dt
    with pytest.raises(ValueError, match="simulate_sr_hetro only"):
        SC.SraSession().call("fpt_full")
    assert SC.summarise({"pure": {"verdict": "pass", "engine_grid_certified": True}}) == "pass"
    assert SC.summarise({"pure": {"verdict": "pass", "engine_grid_certified": False}}) == "fail"
    assert SC.summarise({}) == "fail"


def test_p2s_scaling_rules():
    import p2_s5_power_004 as P2
    assert P2.smallest_detectable(100, 200) == pytest.approx(0.96)            # KL detectable ~ 1/R
    h = {"mean": [0.5, -1.0, 0.0, 2.0], "se": [0.0, 0.0, 0.0, 0.0]}
    # S5's bias at dt from the halving (Naveh 2026-11-03): x 1/(1 - 2^-p), an upper bound with + 2 SE
    assert P2.bias_factor(0.5) == pytest.approx(1 / (1 - 2 ** -0.5)) == pytest.approx(3.4142, abs=1e-4)
    assert P2.bias_factor(1.0) == 2.0
    assert P2.step_bias_in_se(h, 400, [10.0] * 4, 1.0) == pytest.approx([2.0, 4.0, 0.0, 8.0])
    assert P2.step_bias_in_se(h, 400, [10.0] * 4, 0.5) == pytest.approx([3.4142, 6.8284, 0.0, 13.657], rel=1e-4)
    up = P2.step_bias_in_se({"mean": [0.5], "se": [0.25]}, 400, [10.0], 1.0)
    assert up == pytest.approx([(0.5 + 0.5) * 2.0 / 0.5])                    # (|mean| + 2 SE) x factor, upper bound
    assert P2.kl([0.5, 0.5], [0.5, 0.5]) == 0.0 and P2.kl([1.0, 0.0], [0.5, 0.5]) == pytest.approx(math.log(2))


def test_p4_projects_from_p1s_m1_rung_not_its_finest():
    """P4 draws at the base step (m = 1): its projection reads that rung of P1's cell, not P1's top-level
    steps_per_individual, which holds the cell's finest rung (m = 64 on a clamp cell; 2026-11-01, P4 refused at 15.2)."""
    p1 = {"steps_per_individual": {"human|hom|clamp": 1_540_777.0},
          "cells": [{"id": "human|hom|clamp", "ladder": [{"m": 1, "steps_per_individual": 23_908.0},
                                                         {"m": 2, "steps_per_individual": 47_816.0}]}]}
    assert ACC.p4_steps_per_individual(p1, "human|hom|clamp") == 23_908.0
    assert ACC.p4_steps_per_individual(p1, "mice_F|hom|clamp") is None                        # twin: no such cell


# ------------------------------------------------- the plants frozen in LANG (T119 prep): used, re-computed, compared
def test_a_frozen_plant_is_used_beside_its_value_now_and_twin_equal_passes():
    out = ACC.frozen_plants("s4", {"a": 0.25, "b": 0.5}, {"a": lambda: 0.25, "b": lambda: 0.5})
    assert out == {"a": {"frozen": 0.25, "now": 0.25}, "b": {"frozen": 0.5, "now": 0.5}}


def test_planted_code_drift_since_freeze_l_is_refused_naming_both_values():
    with pytest.raises(ACC.Refused, match=r"'b' is 0.5 in .*but the code computes 0.5000001 now: it drifted"):
        ACC.frozen_plants("s4", {"a": 0.25, "b": 0.5}, {"a": lambda: 0.25, "b": lambda: 0.5000001})


def test_a_plant_value_not_filled_in_lang_is_refused():
    for frozen in ({}, {"a": None}, {"a": {"t119": "multi_004.plant_size"}}):
        with pytest.raises(ACC.Refused, match="is not filled"):
            ACC.frozen_plants("multi", frozen, {"a": lambda: 0.1})


def test_each_phase_compares_langs_plant_values_with_its_function_now(monkeypatch):
    import multi_004 as MU
    import paths_004 as PA
    import s4_fp_langevin_004 as S4
    grid = {"clamp": [{"id": "Jack_Russell/v0/clamp", "preset": "Jack_Russell", "parameter": None, "cv": 0.0}]}
    s4 = {"plant_cells": ["Jack_Russell/v0/clamp"], "plant_delta": {"Jack_Russell/v0/clamp": 0.0117}}
    paths = {"n": 25000, "plant": {"preset": "mice_F", "het": "hom", "age": 131.9, "delta": 0.0431}}
    multi = {"preset": "mice_F", "plants": [{"cell": "xc+eta", "drop": "xc", "predicted_sup_dS": 0.0538}]}
    rows = [{"id": "xc+eta", "members": {"xc": 0.2, "eta": 0.2}}]
    for now in (None, 1e-12):                                       # the twin (equal), then the planted drift
        monkeypatch.setattr(S4, "plant_delta", lambda params, het=None: {"delta": 0.0117 + (now or 0.0)})
        monkeypatch.setattr(PA, "plant_delta", lambda params, age, n, het=None: {"delta": 0.0431 + (now or 0.0)})
        monkeypatch.setattr(MU, "plant_size", lambda params, d, drop, **kw: {"predicted_sup_dS": 0.0538 + (now or 0.0)})
        checks = [("s4", ACC.s4_plant_checks(s4, grid)), ("paths", ACC.paths_plant_check(paths)),
                  ("multi", ACC.multi_plant_checks(multi, rows))]
        for phase, (frozen, compute) in checks:
            if now is None:
                assert all(v["frozen"] == v["now"] for v in ACC.frozen_plants(phase, frozen, compute).values())
            else:
                with pytest.raises(ACC.Refused, match=f"\\[{phase}\\] REFUSED: .*drifted since freeze-L"):
                    ACC.frozen_plants(phase, frozen, compute)


def test_multis_projection_counts_the_plant_sizes_recheck():
    assert ACC.multi_plant_check_cpu_h({"plants": [{}] * 6}) == pytest.approx(6 * 200 / 3600)
    assert ACC.multi_plant_check_cpu_h({"plants": [{}] * 6, "tensor_cpu_s_per_cell": 100}) == pytest.approx(1 / 6)


# ------------------------------------------------- A6's re-check, prepared (LANG's [s5].recheck_at_step turns it on)
def test_s5_hets_projection_counts_the_recheck_only_when_lang_asks():
    import p2_s5_power_004 as P2
    s5 = {"R": 274, "cpu_per_cohort_s": 8.0}
    assert ACC.s5_het_projected_cpu_h(s5) == pytest.approx(274 * 8.0 / 3600)
    with_rc = ACC.s5_het_projected_cpu_h({**s5, "recheck_at_step": True})
    assert with_rc == pytest.approx((274 + P2.HALVING_COHORTS * ACC.RECHECK_RUNG_FACTOR) * 8.0 / 3600)


def test_the_recheck_without_p2s_sd_is_refused_by_name():
    with pytest.raises(ACC.Refused, match="sd_per_replicate"):
        ACC.s5_recheck(None, {"R": 4, "key": 0, "p_hat": {"clamp": 0.5}}, 0.1, "clamp", None, None)


def test_the_recheck_reuses_s5s_law_and_gradient_and_records_a5s_bound(monkeypatch):
    import p2_s5_power_004 as P2
    seen = {}

    def halving(spec, dt, wall, key0, cohorts=100, *, base=None, d=None):
        seen.update(base=base, d=d, dt=dt, key=key0, cohorts=cohorts)
        return {"mean": [0.5, 0.0, 0.0, 0.0], "se": [0.25, 0.0, 0.0, 0.0], "cohorts": cohorts, "errors": []}
    monkeypatch.setattr(P2, "coupled_halving", halving)
    s5 = {"R": 400, "key": 7, "p_hat": {"clamp": 1.0}, "sd_per_replicate": [10.0, 10.0, 10.0, 10.0]}
    rc = ACC.s5_recheck("spec", s5, 0.05, "clamp", "LAW", "GRAD", cohorts=3)
    assert seen == {"base": "LAW", "d": "GRAD", "dt": 0.05, "key": 7, "cohorts": 3}
    assert rc["bias_in_se"] == P2.step_bias_in_se(rc["halving"], 400, [10.0] * 4, 1.0) == pytest.approx(
        [4.0, 0.0, 0.0, 0.0])                                         # (0.5 + 2 x 0.25) x 2 / (10 / 20)
    assert rc["dt"] == 0.05 and rc["cpu_s"] >= 0.0 and "upper bound" in rc["bias_in_se_is"]
    monkeypatch.setattr(P2, "coupled_halving", lambda *a, **k: {"mean": None, "cohorts": 1, "errors": []})
    assert "error" in ACC.s5_recheck("spec", s5, 0.05, "clamp", "LAW", "GRAD", cohorts=3)   # no halving: an error


def test_the_recheck_on_a_tiny_gen1_spec_runs_p2s_halving_at_its_step():
    """A real halving on a tiny homogeneous gen-1 spec (n 20, 2 cohorts, a coarse step), its law and gradient computed
    once outside and reused."""
    import s5_gen1_004 as S5
    from sraging import generator as GEN
    spec = GEN.CohortSpec(name="tiny", rho_eta=2.125427697723628, rho_beta=0.1, rho_eps=0.003981071705534973,
                          kappa=0.01, n=20, L_phys=100.0, seed=1, delta=1.0)
    base = S5.law(spec)
    d = S5.spec_gradient(spec, base)
    s5 = {"R": 274, "key": 5, "p_hat": {"clamp": 0.5}, "sd_per_replicate": [222.7, 17.4, 5.5, 1.1]}
    rc = ACC.s5_recheck(spec, s5, 0.05, "clamp", base, d, cohorts=2)
    h = rc["halving"]
    assert "error" not in rc and h["cohorts"] == 2 and h["dt"] == 0.05 and h["t_max"] == S5.horizon(spec)
    assert len(rc["bias_in_se"]) == 4 and all(b >= 0 for b in rc["bias_in_se"])


def test_a6s_step_and_its_c2_refusal():
    import p2_s5_power_004 as P2
    a = P2.a6_step(16, 7.8125e-4, 32, "S5 runs at m 32")
    assert (a["m"], a["dt"], a["factor"], a["m_max"], a["above_m_max"]) == (32, 3.90625e-4, 2, 32, False)
    with pytest.raises(ValueError, match="above gate C2's m_max 32"):
        P2.a6_step(16, 7.8125e-4, 64, "S5 runs at m 64")
    b = P2.a6_step(16, 7.8125e-4, 64, f"S5 runs at m 64; {P2.A6_C2_CLAUSE}")
    assert (b["dt"], b["above_m_max"]) == (1.953125e-4, True)
    for m in (16, 48, 8):
        with pytest.raises(ValueError, match="power-of-two multiple"):
            P2.a6_step(16, 7.8125e-4, m, P2.A6_C2_CLAUSE)


def test_s5_homs_projection_counts_002s_feature_workers_not_its_parent_alone():
    s002 = {"cpu_s": 1.0, "features": {"pure": {"cpu_s": 500.0}, "window": {"cpu_s": 1099.0}}}
    assert ACC.s5_hom_projected_cpu_h(s002) == pytest.approx(1600.0 / 3600)
    assert ACC.s5_hom_projected_cpu_h({"cpu_s": 1.0}) == pytest.approx(1.0 / 3600)     # twin: no features, the parent


def test_s5_homs_projection_reads_each_of_002s_runs_and_each_features_latest():
    """002 re-ran its amended window and ran heterogeneity on its own: score_002.json alone gave 0.61 CPU-h, and the
    run at 252ad3a spent ~5 (Naveh 2026-11-04)."""
    import json
    import score_test as ST
    s002 ={"cpu_s": 1.0, "features": {"pure": {"cpu_s": 500.0}, "window": {"cpu_s": 1066.0}}}
    win = {"cpu_s": 1.0, "features": {"window": {"cpu_s": 132.0}}}
    het = {"cpu_s": 1.0, "features": {"heterogeneity": {"cpu_s": 15911.0}}}
    assert ACC.s5_hom_projected_cpu_h(s002, win, het) == pytest.approx((3.0 + 500 + 132 + 15911) / 3600)
    runs = [json.load(open(os.path.join(REPO, "tests", "acceptance", "results", f))) for f in ACC.S5_HOM_002_RUNS]
    assert set().union(*(r["features"] for r in runs)) == set(ST.FEATURES)            # every feature priced
    assert 4.5 < ACC.s5_hom_projected_cpu_h(*runs) < 5.0 and ACC.frozen_cap("s5-hom")[0] == 1.5   # above: refused


# ------------------------------------------------- rows across worker processes: wall time only (Naveh 2026-11-04 ask)
def _tiny_s4_rows():
    rows = [{"id": f"mice_F/{q or 'v0'}/cv{cv:g}/{w}", "preset": "mice_F", "parameter": q, "cv": cv, "wall": w, "n": 300,
             "m": 1, "dt": 0.05, "m_max": 2, "p1_step": "mice_F|hom|clamp"}
            for q, cv, w in ((None, 0.0, "clamp"), (None, 0.0, "reflect"), ("xc", 0.2, "clamp"))]
    frozen = {"p_hat": 0.5, "pooled_b": 1e-3, "D_key_mean": 0.01}
    return rows, {r["id"]: 40400 + i for i, r in enumerate(rows)}, frozen


def _budget(cap_cpu_h=100.0, clock=None):
    return ACC.Budget("s4", 0.0, cap_cpu_h, cap_cpu_h, None, "test", clock or time.process_time)


def test_s4s_rows_are_the_same_at_one_and_three_workers(tmp_path):
    rows, keys, frozen = _tiny_s4_rows()
    out = {}
    for W in (1, 3):
        b = _budget()
        got = ACC.robust_run_pool("s4", b, rows, "s4_row_task", lambda r: (r, keys[r["id"]], frozen), W,
                                  label=lambda r: r["id"], path=str(tmp_path / f"w{W}.jsonl"))
        out[W] = [{k: v for k, v in g.items() if k != "cpu_s"} for g in got]
        assert not b.capped and (W == 1 or b.worker_cpu_s > 0)          # the workers' CPU reaches the budget
    assert [r["id"] for r in out[3]] == [r["id"] for r in rows]          # in cell order, whatever finished first
    assert out[1] == out[3]                                                # the same rows: D, KS, p, M_ref, bytes-exact


def test_at_the_cap_no_new_row_starts_and_a_raising_row_is_its_error_row(tmp_path):
    rows, keys, frozen = _tiny_s4_rows()
    bad = {**rows[0], "id": "nope/v0/cv0/clamp", "wall": "nope"}           # simulate refuses the wall in the worker
    cells = [bad, rows[0], rows[1], rows[2]]
    keys = {**keys, bad["id"]: 1}
    b = _budget(cap_cpu_h=1e-6, clock=lambda: 0.0)                         # reached as soon as one row's CPU is counted
    got = ACC.robust_run_pool("s4", b, cells, "s4_row_task", lambda r: (r, keys[r["id"]], frozen), 2,
                              label=lambda r: r["id"], path=str(tmp_path / "cap.jsonl"))
    assert b.capped and [g["id"] for g in got] == ["nope/v0/cv0/clamp", rows[0]["id"]]   # two in flight, kept
    assert got[0]["error"]["type"] == "UnsupportedOptionError" and "nope" in got[0]["error"]["message"]
    assert "error" not in got[1] and got[0]["error"]["traceback"]


def test_workers_is_refused_by_name_where_no_phase_takes_it():
    import argparse
    ACC.check_workers(argparse.Namespace(phase="s4", workers=3))
    ACC.check_workers(argparse.Namespace(phase="multi", workers=None))     # twin: not given
    with pytest.raises(ACC.Refused, match="--workers is taken by s4, paths, p1-pooled only"):
        ACC.check_workers(argparse.Namespace(phase="multi", workers=2))
    with pytest.raises(ACC.Refused, match="an integer >= 1"):
        ACC.check_workers(argparse.Namespace(phase="s4", workers=0))


# ------------------------------------------------------------- the close: docs (T135) and unchanged (T139), §5
def _marker_files(tmp_path, shown):
    res = tmp_path / "r.json"
    res.write_text('{"a": {"b": [0.123456, 7]}, "s": "pass"}')
    tex = tmp_path / "x.tex"
    tex.write_text("A number \\verify{r.json}{a.b[0]}{.3g}{" + shown + "} and a status \\verify{r.json}{s}{}{pass}.\n")
    return str(tmp_path), "x.tex"


def test_docs_a_planted_marker_that_does_not_recompute_fails_and_its_twin_passes(tmp_path):
    base, tex = _marker_files(tmp_path, "0.124")                       # planted: format(0.123456, ".3g") = 0.123
    rows = ACC.docs_markers([tex], base=base)
    assert [r["match"] for r in rows] == [False, True] and rows[0]["recomputed"] == "0.123" and rows[0]["text"] == "0.124"
    assert ACC.docs_verdict({"ok": True}, rows)["status"] == "fail"
    twin = ACC.docs_markers([tex], base=_marker_files(tmp_path, "0.123")[0])
    assert all(r["match"] for r in twin) and ACC.docs_verdict({"ok": True}, twin)["status"] == "pass"


def test_docs_a_failed_build_no_marker_or_an_unparsed_marker_fails(tmp_path):
    base, tex = _marker_files(tmp_path, "0.123")
    rows = ACC.docs_markers([tex], base=base)
    assert ACC.docs_verdict({"ok": False}, rows)["status"] == "fail"               # the PDF did not build
    assert ACC.docs_verdict({"ok": True}, [])["status"] == "fail"                  # nothing to check is not a pass
    (tmp_path / "y.tex").write_text("\\verify{r.json}{s}{}\n")                     # three groups, not four
    bad = ACC.docs_markers(["y.tex"], base=base)
    assert bad and not bad[0]["match"] and ACC.docs_verdict({"ok": True}, bad)["status"] == "fail"
    (tmp_path / "z.tex").write_text("\\verify{nope.json}{s}{}{pass}\n")         # its result file is not there
    missing = ACC.docs_markers(["z.tex"], base=base)
    assert len(missing) == 1 and missing[0]["recomputed"] is None and "FileNotFoundError" in missing[0]["error"]
    assert ACC.docs_verdict({"ok": True}, missing)["status"] == "fail"


@pytest.mark.skipif(not __import__("shutil").which("pdflatex"), reason="NOT APPLICABLE: no pdflatex on this host")
def test_docs_build_twice_into_a_scratch_directory_and_a_latex_error_fails(tmp_path):
    src = tmp_path / "doc"
    src.mkdir()
    (src / "main.tex").write_text("\\documentclass{article}\\begin{document}One page.\\end{document}\n")
    out = tmp_path / "out"
    out.mkdir()
    b = ACC.docs_build(str(out), cwd=str(src))
    assert b["ok"] and b["returncodes"] == [0, 0] and b["pages"] == 1 and not (src / "main.pdf").exists()
    (src / "main.tex").write_text("\\documentclass{article}\\begin{document}\\undefinedcommand\\end{document}\n")
    bad = ACC.docs_build(str(tmp_path / "out"), cwd=str(src))                     # planted: halt-on-error
    assert not bad["ok"] and bad["returncodes"][0] != 0


def _junit(path, cases):
    """A junit file: cases [(classname, name, kind)] with kind ok, failure, error or skipped."""
    import xml.etree.ElementTree as ET
    suite = ET.Element("testsuite", tests=str(len(cases)), failures=str(sum(k == "failure" for *_, k in cases)),
                       errors=str(sum(k == "error" for *_, k in cases)),
                       skipped=str(sum(k == "skipped" for *_, k in cases)))
    for cls, name, kind in cases:
        tc = ET.SubElement(suite, "testcase", classname=cls, name=name)
        if kind != "ok":
            ET.SubElement(tc, kind, message=f"planted {kind}")
    ET.ElementTree(suite).write(path)


def test_unchanged_a_group_runs_in_its_own_pytest_and_reads_its_junit(monkeypatch, tmp_path):
    seen = []

    def fake_run(argv, **kw):
        seen.append(argv)
        xml = next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml="))
        _junit(xml, [("tests.unit.test_register", "test_a", "ok"), ("tests.unit.test_register", "test_b", "failure")])
        return type("P", (), {"returncode": 1, "stdout": "1 failed, 1 passed in 0.1s\n", "stderr": ""})()
    monkeypatch.setattr(ACC.subprocess, "run", fake_run)
    g = ACC.unchanged_group("unit", ["tests/unit"], str(tmp_path))
    assert seen[0][1:4] == ["-m", "pytest", "-q"] and "tests/unit" in seen[0]
    assert (g["tests"], g["failures"], g["passed"], g["returncode"]) == (2, 1, 1, 1)
    assert g["failed_list"][0][0] == "tests.unit.test_register::test_b" and len(g["cases"]) == 2


def _groups(**over):
    ok = lambda cases: {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0, "passed": len(cases),  # noqa: E731
                        "returncode": 0, "skipped_list": [], "failed_list": [],
                        "cases": [{"classname": c, "name": "t", "kind": "ok"} for c in cases]}
    g = {"unit": ok(["tests.unit.test_generator_checksums", "tests.unit.test_register", "tests.unit.test_x"]),
         "parity": ok(["tests.parity.test_engine_t1_het", "tests.parity.test_curves"]),
         "results_001_to_003g": ok(["tests.acceptance.test_results_003g"])}
    g.update(over)
    return g


def test_unchanged_passes_only_when_every_group_and_every_named_check_holds():
    assert ACC.unchanged_verdict_004(_groups())["status"] == "PASS"                           # the twin
    failing = _groups()
    failing["results_001_to_003g"] = {**failing["results_001_to_003g"], "failures": 1, "returncode": 1,
                                      "failed_list": [("tests.acceptance.test_results_003g::t", "planted")]}
    v = ACC.unchanged_verdict_004(failing)
    assert v["status"] == "FAIL" and not v["groups_ok"]
    no_t1 = _groups()
    no_t1["parity"] = {**no_t1["parity"], "cases": [c for c in no_t1["parity"]["cases"]
                                                     if c["classname"] != "tests.parity.test_engine_t1_het"]}
    v = ACC.unchanged_verdict_004(no_t1)                                # spec 002's T1 did not run: not a pass
    assert v["status"] == "FAIL" and not v["named"]["parity_t1_normal_frac_sd_v033"]["ok"]
    skipped = _groups()
    skipped["unit"]["cases"] = [dict(c, kind="skipped") if c["classname"] == "tests.unit.test_register" else c
                                for c in skipped["unit"]["cases"]]
    assert not ACC.unchanged_verdict_004(skipped)["named"]["register"]["ok"]


def test_the_close_phases_are_implemented_with_their_caps():
    assert ACC.PHASES["docs"] is ACC.phase_docs and ACC.PHASES["unchanged"] is ACC.phase_unchanged
    assert ACC.CLOSE_CAPS == {"docs": 0.1, "unchanged": 2.0}
    assert set(ACC.UNCHANGED_GROUPS_004) == {"unit", "parity", "results_001_to_003g"}
    assert "tests/acceptance/test_results_003g.py" in ACC.UNCHANGED_GROUPS_004["results_001_to_003g"]


def test_multis_composite_cross_checks_the_tensor_at_the_freezes_bar_not_the_cells():
    """A9 (Naveh 2026-11-05, T126): the composite rule is compared with the tensor reference built at
    [multi].composite.bar (1e-4). At the cells' crit/10 (M 8 x 5) the reference sits 6.3e-4 from the converged one,
    inside its own bar but not the contract's 1e-4, so the first run read the cross-check's bar, not the tensor code."""
    import multi_004 as MU
    seen = {}

    def fake(params, d, **kw):
        seen.update(kw, d=d)
        return {"sup_dS": 0.0}
    spec = {"n": 25000, "composite": {"cell": "xc+eta", "panels": [16, 32], "order": 8, "agree": 1e-4, "bar": 1e-4}}
    by_id = {"xc+eta": {"members": {"xc": 0.2, "eta": 0.2}}}
    orig = MU.composite
    MU.composite = fake
    try:
        ACC.multi_composite(spec, by_id, None)
        assert seen == {"panels": (16, 32), "order": 8, "agree": 1e-4, "bar": 1e-4, "d": {"xc": 0.2, "eta": 0.2}}
        no_bar = {**spec, "composite": {k: v for k, v in spec["composite"].items() if k != "bar"}}
        with pytest.raises(ACC.Refused, match="A9"):          # the twin: never crit/10 by default
            ACC.multi_composite(no_bar, by_id, None)
    finally:
        MU.composite = orig
