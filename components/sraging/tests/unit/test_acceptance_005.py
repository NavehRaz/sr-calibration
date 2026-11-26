"""Spec 005's acceptance frame (tools/acceptance_005.py; tasks T003, T023; contracts/acceptance.md; research R10, R11,
R14): the phase registry and its caps, the not_applicable path for the records, the dirty-tree refusal, `capped`, and
the exit's comparison logic. Each planted case beside its twin. No real data: the records are planted in tmp_path."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_003 as A3  # noqa: E402
import acceptance_005 as ACC  # noqa: E402

H = 3600.0


class Clock:
    """A fake process clock: CPU seconds, advanced by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def plant_records(tmp_path):
    """A manifest in the shape of cleaning_records_005.toml whose files all exist under tmp_path, pinned right."""
    files = {}
    for rec in ("mosquito", "fly"):
        d = tmp_path / rec
        (d / "cohorts_allcages").mkdir(parents=True)
        (d / "cleaning.csv").write_text("cohort,cage_id,median,steepness_iqr,distance,kept,n\n")
        (d / "cleaning.csv.stamp.json").write_text("{}")
        (d / "cohorts_allcages" / "a.csv").write_text("age,event,cage_id\n1,1,c1\n")
        files[rec] = d
    (files["fly"] / "cleaning_floors.json").write_text("{}")
    for f in ("null_drops.csv", "null_by_group.csv"):
        (files["fly"] / f).write_text("k\n")
    lines = []
    for rec, d in files.items():
        extra = ["cleaning_floors.json", "null_drops.csv", "null_by_group.csv"] if rec == "fly" else []
        lines += [f"[{rec}]", f'dir = "{d}"', 'rule = "planted"', "k = [2.5]", 'kept_column = "kept"', "",
                  f"[{rec}.sha256]"]
        lines += [f'"{f}" = "{_sha(d / f)}"' for f in ["cleaning.csv", "cleaning.csv.stamp.json", *extra]]
        lines += ["", f"[{rec}.cohorts_allcages]", f'"a.csv" = "{_sha(d / "cohorts_allcages" / "a.csv")}"', ""]
    path = tmp_path / "records.toml"
    path.write_text("\n".join(lines))
    return path, files


# --------------------------------------------------------------------------------------------- the phase registry
def test_the_phases_and_their_caps_are_the_contracts():
    assert ACC.PHASES == ("exit", "km", "noise", "workers", "presets", "units", "unchanged")
    assert ACC.CAPS == {"exit": 0.1, "km": 0.1, "noise": 0.5, "workers": 8.0, "presets": 0.1, "units": 0.1,
                        "unchanged": 1.0}
    assert sum(ACC.CAPS.values()) <= 10.0                 # inside SC-009's 9 CPU-h plus unchanged's 1 (2026-11-09)


def test_the_rules_file_carries_the_same_caps_and_is_frozen_with_navehs_ok():
    """T004: rules_005.toml repeats the caps (a drift between the tool and the freeze fails here); frozen at T047
    (Naveh, 2026-11-09), with no value left to a pilot."""
    rules = ACC.read_rules()
    assert rules["caps"] == ACC.CAPS
    assert rules["frozen"] is True and rules["naveh_ok"].startswith("2026-11-09")
    assert rules["rules"]["HR-8"]["loglik_tolerance"] == 9.55e-11
    assert "pilot" not in [v for r in rules["rules"].values() for v in r.values() if isinstance(v, str)]
    assert {f"HR-{i}" for i in range(1, 11)} <= set(rules["rules"])


# ------------------------------------------------------------------------------------------ not_applicable (R10)
def test_records_pinned_and_present_have_no_problem(tmp_path):
    path, _ = plant_records(tmp_path)
    assert ACC.records_problems(path) == []                                              # the twin


def test_a_missing_record_file_is_not_applicable_naming_it(tmp_path):
    path, files = plant_records(tmp_path)
    os.remove(files["fly"] / "null_by_group.csv")
    problems = ACC.records_problems(path)
    assert len(problems) == 1 and "missing" in problems[0] and "null_by_group.csv" in problems[0]
    out = ACC.not_applicable(problems)
    assert out["status"] == "not_applicable" and out["reasons"] == problems
    assert not ACC.counts_as_pass(out["status"])


def test_a_changed_record_file_is_not_applicable_naming_it(tmp_path):
    path, files = plant_records(tmp_path)
    with open(files["mosquito"] / "cohorts_allcages" / "a.csv", "a") as fh:
        fh.write("2,0,c1\n")
    problems = ACC.records_problems(path)
    assert len(problems) == 1 and "sha256" in problems[0] and os.path.join("cohorts_allcages", "a.csv") in problems[0]
    assert ACC.not_applicable(problems)["status"] == "not_applicable"


def test_the_exit_phase_writes_not_applicable_never_pass_when_a_record_is_missing(tmp_path, monkeypatch):
    path, files = plant_records(tmp_path)
    os.remove(files["mosquito"] / "cleaning.csv")
    monkeypatch.setattr(ACC, "RECORDS", str(path))
    monkeypatch.setattr(A3, "RESULTS", str(tmp_path / "results"))
    with pytest.raises(SystemExit) as ex:
        ACC.main(["exit", "--dev"])
    assert ex.value.code == 3
    out = json.load(open(tmp_path / "results" / "exit_005.json"))
    assert out["status"] == "not_applicable" and out["status_for_acceptance"] == "development"
    assert any("cleaning.csv" in r and "missing" in r for r in out["reasons"])


def test_the_real_manifest_names_every_record_file():
    """The committed manifest's entries, as records_files reads them: cleaning.csv, its stamp and every per-cage file
    of each record, and the fly's floors and nulls."""
    rows = ACC.record_files(ACC.RECORDS)
    names = {(r, os.path.basename(p)) for r, p, _ in rows}
    assert {("mosquito", "cleaning.csv"), ("fly", "cleaning.csv"), ("fly", "null_drops.csv")} <= names
    assert sum(1 for r, p, _ in rows if r == "mosquito" and "cohorts_allcages" in p) == 19
    assert sum(1 for r, p, _ in rows if r == "fly" and "cohorts_allcages" in p) == 82


# ---------------------------------------------------------------------------------------- the envelope (Run, XI)
def test_a_dirty_tree_is_refused_and_a_dev_run_goes_on_labelled_development(monkeypatch, tmp_path):
    monkeypatch.setattr(ACC, "read_rules", lambda: {"frozen": True, "caps": ACC.CAPS, "rules": {}})   # only the dirt
    monkeypatch.setattr(A3, "tree_dirty", lambda: True)
    with pytest.raises(SystemExit) as ex:
        ACC.start("km", dev=False)
    assert "REFUSED" in str(ex.value) and "dirty" in str(ex.value)
    monkeypatch.setattr(A3, "RESULTS", str(tmp_path))
    run = ACC.start("km", dev=True)                                                       # the twin
    out = run.write({"status": "pass"})
    assert out["status_for_acceptance"] == "development" and out["phase"] == "km"


def test_a_non_dev_run_before_the_freeze_is_refused(monkeypatch):
    """T047: no non-dev run before rules_005.toml is frozen with Naveh's OK."""
    monkeypatch.setattr(A3, "tree_dirty", lambda: False)
    monkeypatch.setattr(ACC, "read_rules", lambda: {"frozen": False, "caps": ACC.CAPS, "rules": {}})
    with pytest.raises(SystemExit) as ex:
        ACC.start("km", dev=False)
    assert "REFUSED" in str(ex.value) and "frozen" in str(ex.value) and "T047" in str(ex.value)


# ------------------------------------------------------------------------------------------------ caps (R14)
def test_a_phase_past_its_cap_reads_capped_keeping_the_finished_cells():
    clock = Clock()
    b = ACC.Budget("exit", ACC.CAPS["exit"], clock=clock, children=False)
    ran = []

    def work(cell):
        clock.t += 0.04 * H
        ran.append(cell)
        return {"cell": cell}
    rows = ACC.run_capped(b, range(5), work)
    assert ran == [0, 1, 2] and len(rows) == 3                    # 0.12 CPU-h after the third: past 0.1
    assert b.capped and ACC.phase_status("pass", b) == "capped" and not ACC.counts_as_pass("capped")
    rec = b.record()
    assert rec["cap_cpu_h"] == 0.1 and rec["capped"] and rec["cpu_spent_h"] == pytest.approx(0.12)


def test_a_phase_inside_its_cap_keeps_its_status():
    clock = Clock()
    b = ACC.Budget("exit", ACC.CAPS["exit"], clock=clock, children=False)

    def work(cell):
        clock.t += 0.01 * H
        return {"cell": cell}
    assert len(ACC.run_capped(b, range(5), work)) == 5                                   # the twin
    assert not b.capped and ACC.phase_status("pass", b) == "pass" and ACC.counts_as_pass("pass")


def test_only_pass_counts_as_pass():
    for s in ("fail", "capped", "not_applicable", "development", "incomplete", "", None):
        assert not ACC.counts_as_pass(s)
    assert ACC.counts_as_pass("pass")


def test_every_phase_is_built():
    assert set(ACC.PHASE_FUNCS) == set(ACC.PHASES)
    assert all(f.__name__ == f"phase_{name}" for name, f in ACC.PHASE_FUNCS.items())


# ------------------------------------------------------------------------------------- exit's comparison (T023)
TOL = {"distance_abs": 1e-12, "steepness_abs": 1e-12, "floors_rel": 1e-12}


def _fake():
    """A 3-row record and our rows that reproduce it exactly."""
    record = [{"cohort": "a", "cage_id": 1, "median": 30.0, "steepness_iqr": 2.5, "distance": 0.4, "kept": True,
               "n": 40},
              {"cohort": "a", "cage_id": 2, "median": 31.0, "steepness_iqr": 2.0, "distance": 1.3, "kept": False,
               "n": 41},
              {"cohort": "b", "cage_id": "x", "median": 12.0, "steepness_iqr": 4.0, "distance": 0.9, "kept": True,
               "n": 39}]
    ours = {ACC.exit_key(r["cohort"], r["cage_id"]): {"verdict": "kept" if r["kept"] else "flagged",
                                                      "distance": r["distance"], "median": r["median"],
                                                      "steepness": r["steepness_iqr"], "n": r["n"]} for r in record}
    return ours, record


def test_the_comparison_passes_an_exact_reproduction():
    ours, record = _fake()
    out = ACC.compare_record(ours, record, kept_col="kept", distance_col="distance", tol=TOL)
    assert out["mismatches"] == [] and out["n"] == 3 and (out["n_flagged_ours"], out["n_flagged_record"]) == (1, 1)


def test_one_flipped_verdict_fails_the_comparison():
    ours, record = _fake()
    ours[ACC.exit_key("a", 1)]["verdict"] = "flagged"
    out = ACC.compare_record(ours, record, kept_col="kept", distance_col="distance", tol=TOL)
    assert [m["what"] for m in out["mismatches"]] == ["verdict"] and out["flips"] == 1


def test_a_distance_error_of_1e_11_fails_and_1e_13_passes():
    ours, record = _fake()
    ours[ACC.exit_key("b", "x")]["distance"] = 0.9 + 1e-11
    out = ACC.compare_record(ours, record, kept_col="kept", distance_col="distance", tol=TOL)
    assert [m["what"] for m in out["mismatches"]] == ["distance"]
    ours[ACC.exit_key("b", "x")]["distance"] = 0.9 + 1e-13
    assert ACC.compare_record(ours, record, kept_col="kept", distance_col="distance", tol=TOL)["mismatches"] == []


def test_a_median_off_by_one_ulp_a_missing_cage_and_a_wrong_n_fail():
    ours, record = _fake()
    ours[ACC.exit_key("a", 2)]["median"] = float(np.nextafter(31.0, 32.0))           # medians are exact
    del ours[ACC.exit_key("b", "x")]
    ours[ACC.exit_key("a", 1)]["n"] = 41
    out = ACC.compare_record(ours, record, kept_col="kept", distance_col="distance", tol=TOL)
    assert sorted(m["what"] for m in out["mismatches"]) == ["median", "missing", "n"]


def test_the_floors_compare_relatively():
    assert ACC.compare_floors((0.1678, 0.3065), (0.1678, 0.3065), TOL["floors_rel"])["ok"]
    assert not ACC.compare_floors((0.1678 * (1 + 1e-11), 0.3065), (0.1678, 0.3065), TOL["floors_rel"])["ok"]


def test_cage_ids_compare_as_the_reader_types_them():
    assert ACC.exit_key("a", 3) == ACC.exit_key("a", 3.0) == ACC.exit_key("a", "3")
    assert ACC.exit_key("a", "M1") != ACC.exit_key("a", "M2")


# ------------------------------------------------------------------------------------------------- units (T026)
def test_units_round_trips_are_within_hr8s_bound_and_the_proposal_rule():
    trips = ACC.units_round_trips()
    assert trips["params_rel"] <= 1e-15 and trips["cohort_rel"] <= 1e-15
    assert ACC.proposed_tolerance(3e-14) == pytest.approx(3e-13) and ACC.proposed_tolerance(0.0) == 1e-12
    c = ACC.units_cohort()
    assert c.n == ACC.UNITS_N and c.delta == 1.0 and c.n_censored == ACC.UNITS_N // 7 and c.time_unit == "days"


# ------------------------------------------------------------------------------------- workers (T037; WEXAC)
def test_the_runtime_pins_are_the_projects_dependencies_as_uv_lock_resolves_them():
    pins = ACC.runtime_pins()
    assert pins["packages"] == {"numpy": "2.4.3", "scipy": "1.17.1", "jax": "0.11.1", "jaxlib": "0.11.1",
                                "ml-dtypes": "0.6.0", "opt-einsum": "3.4.0"}
    assert pins["requires_python"] == "==3.14.*" and pins["python"].startswith("3.14.")


def test_the_runtime_stack_must_match_exactly_and_a_difference_is_refused_by_name():
    """T037 (the coordinator, 2026-11-08): the Python version and sraging's numerical runtime stack against uv.lock,
    exactly; anything else in the freeze (a Linux venv's, say) is recorded, not compared."""
    pins = {"requires_python": "==3.14.*", "python": "3.14.6",
            "packages": {"numpy": "2.4.3", "scipy": "1.17.1", "jax": "0.11.1", "ml-dtypes": "0.6.0"}}
    freeze = "jax==0.11.1\nml_dtypes==0.6.0\nnumpy==2.4.3\nscipy==1.17.1\npandas==2.3.3\nappdirs==1.4\n"
    assert ACC.runtime_problems(freeze, "3.14.6", pins) == []                              # the twin
    probs = ACC.runtime_problems(freeze.replace("numpy==2.4.3", "numpy==2.4.2").replace("scipy==1.17.1\n", ""),
                                 "3.14.5", pins)
    assert any("numpy" in p and "2.4.2" in p and "2.4.3" in p for p in probs)
    assert any("scipy" in p and "missing" in p for p in probs) and any("python" in p and "3.14.5" in p for p in probs)
    assert not any("pandas" in p or "appdirs" in p for p in probs)
    assert any("python" in p for p in ACC.runtime_problems(freeze, "3.13.9", {**pins, "python": "3.13.9"}))


def _job(thetas):
    """A fake job record: per cohort, the identity of its in-process, W = 1 and W = 2 fits."""
    rec = {n: {"theta_hex": {"eta": float(t).hex()}, "loglik_hex": float(-t).hex(), "claims": {"eta": [t, 2 * t]}}
           for n, t in thetas.items()}
    return {"inproc": rec, "w1": {k: dict(v) for k, v in rec.items()}, "w2": {k: dict(v) for k, v in rec.items()}}


def test_identical_runs_compare_equal_and_a_result_swapped_between_two_cohorts_fails():
    job = _job({"c1": 0.1, "c2": 0.2, "c3": 0.3})
    assert ACC.compare_workers(job) == []                                                  # the twin
    job["w2"]["c1"], job["w2"]["c2"] = job["w2"]["c2"], job["w2"]["c1"]                    # planted swap
    probs = ACC.compare_workers(job)
    assert any("c1" in p and "w2" in p for p in probs) and any("c2" in p for p in probs)
    job2 = _job({"c1": 0.1, "c2": 0.2, "c3": 0.3})
    del job2["w1"]["c3"]
    assert any("c3" in p and "missing" in p for p in ACC.compare_workers(job2))


def test_the_pilot_job_compares_its_one_cohort_at_w1_only_and_still_catches_a_difference():
    """P1 (rules_005.toml HR-6's pilot): one cohort in-process and spawned at W = 1; the record names its runs."""
    job = _job({"w005_a": 0.1})
    del job["w2"]
    job["runs"] = ["w1"]
    assert ACC.compare_workers(job) == []                                                  # the twin
    job["w1"]["w005_a"] = {**job["w1"]["w005_a"], "loglik_hex": float(-0.2).hex()}         # planted difference
    assert any("w005_a" in p and "w1" in p and "loglik_hex" in p for p in ACC.compare_workers(job))
    full = _job({"c": 0.1})
    full["w2"] = {}
    assert any("missing from w2" in p for p in ACC.compare_workers(full))        # a full record must have W = 2


def test_the_workers_cap_counts_the_lsf_jobs_cpu():
    """P1 (2026-11-08): the cap block read 0.001 CPU-h after a 2.1 CPU-h job; the job's CPU now counts."""
    b = ACC.Budget("workers", ACC.CAPS["workers"])
    assert not b.reached()
    b.remote_cpu_s = ACC.CAPS["workers"] * 3600.0
    assert b.reached() and b.record()["cpu_spent_h"] >= ACC.CAPS["workers"]


def test_the_workers_dry_run_prints_the_exact_commands_and_contacts_nothing(monkeypatch, capsys):
    def no_remote(*a, **k):
        raise AssertionError("the dry run must not contact WEXAC")
    monkeypatch.setattr(ACC, "_remote", no_remote)
    with pytest.raises(SystemExit) as ex:
        ACC.main(["workers", "--dry-run"])
    assert ex.value.code in (0, None)
    out = capsys.readouterr().out
    assert "git -C" in out and "archive" in out and "rsync -az --delete" in out and "wexac:sraging_005/repo/" in out
    assert "sraging_cov003/venv/bin/python -c" in out and "importlib.metadata" in out and "bsub" in out and "-n 3" in out
    assert "span[hosts=1]" in out and "workers-job" in out and "bjobs" in out and "platform.python_version" in out
    assert "PYTHONPATH=$HOME/sraging_005/repo/src:$HOME/sraging_005/repo/tools" in out
    assert ACC.WEXAC["host"] == "wexac" and ACC.WEXAC["repo"] == "sraging_005/repo"
    assert "--pilot" not in out                                       # the acceptance run is the full job
    with pytest.raises(SystemExit):
        ACC.main(["workers", "--dry-run", "--dev"])
    assert "workers_job.json --pilot" in capsys.readouterr().out      # a dev run is P1's job


def test_the_job_cohorts_are_three_distinct_small_ones():
    cs = ACC.workers_cohorts()
    assert len(cs) == 3 and len({tuple(c.ages) for c in cs.values()}) == 3 and all(c.n <= 200 for c in cs.values())


# ------------------------------------------------------------------------------------------- km (HR-3, T048 code)
def test_km_against_lifelines_on_synthetic_ties_agrees_and_the_planted_tie_order_fails():
    pytest.importorskip("lifelines")
    worst = ACC.km_synthetic(seed=1, reps=20)
    assert worst["max_dS"] <= 1e-12 and worst["max_dband"] <= 1e-10 and worst["median_mismatches"] == 0
    assert worst["cohorts"] == 21                                     # 20 random + the (1, 2, 2, 2, 3) example
    p = ACC.km_planted_tie()
    assert p["ours"] == pytest.approx(0.6) and p["planted"] == pytest.approx(8 / 15) and p["fails"]


# ------------------------------------------------------------------------------------------ noise (HR-4, T051 code)
def test_a_rate_within_4_binomial_se_passes_and_one_beyond_fails():
    ok = ACC.rate_cell(0.0905, 0.0900, 12000, 12000)
    assert ok["ok"] and ok["se"] == pytest.approx((0.09025 * 0.90975 * (2 / 12000)) ** 0.5, rel=1e-12)
    assert not ACC.rate_cell(0.120, 0.090, 12000, 12000)["ok"]                     # a null without LOO, say


def test_group_spreads_follow_the_fly_redos_construction():
    """mult = robust sd of the pooled leave-one-out relative deviations / r(m), over the floor (clean.py:150-164),
    computed by the library's pooled_spread, which clean's own-spread null uses (sr-naveh_005 B2)."""
    from sraging import cleaning as CL
    cohorts = {"c1": ([1.0, 1.1, 0.9, 1.05], [2.0, 2.2, 1.9, 2.1]), "c2": ([3.0, 3.3, 2.8], [1.0, 1.2, 0.9])}
    floors = (0.05, 0.1)
    got = ACC.group_spread(cohorts, floors, seed=7)
    dm, ds = [], []
    for ml, st in cohorts.values():
        r = CL.loo_correction(len(ml), seed=7)
        dm += list(CL.loo_relative_deviations(ml) / r)
        ds += list(CL.loo_relative_deviations(st) / r)
    assert got == (CL.robust_sigma(dm) / floors[0], CL.robust_sigma(ds) / floors[1])
    s = CL.pooled_spread(cohorts, seed=7)
    assert got == (s[0] / floors[0], s[1] / floors[1])


def test_group_expected_sums_the_own_spread_null_over_the_groups_cohort_sizes():
    from sraging.cleaning import noise_baseline
    floors, spread = (0.05, 0.1), (0.08, 0.15)
    e = ACC.group_expected({3: 2, 4: 1}, 2.5, floors, spread, reps=500, seed=3)
    want = 2 * noise_baseline(3, 2.5, floors=floors, spread=spread, reps=500, seed=3)["own_spread"]["expected_flags"] \
        + noise_baseline(4, 2.5, floors=floors, spread=spread, reps=500, seed=3)["own_spread"]["expected_flags"]
    assert e["expected"] == pytest.approx(want, rel=1e-15) and e["var"] > 0


def test_the_no_floors_plant_is_scatter_phi_under_a_rule_without_floors_and_fails_far_above_the_reference():
    """sr-naveh_005 N1: the plant judges scatter of size phi by a rule with floors (0, 0) (the own-spread entry with
    spread = phi), not zero scatter (the floor entry with floors (0, 0), which read 0.000). At the fly's floors and
    its m = 3, k = 2.5 cell (rate 0.090 over 12,000 cages; seed 20261108, R = 4,000) the reviewer measured 0.514;
    the plant must fail far above the reference, not below it."""
    phi = (0.04839711413364872, 0.1862177855477855)                     # the fly's pooled floors (noise_005.json)
    ref = {"m": "3", "k": "2.5", "reps": "4000", "rate": "0.09", "cages": "12000"}
    planted = {p["defect"]: p for p in ACC.noise_plants(phi, ref)}
    nf = planted["no floors"]
    assert nf["fails"] and nf["ours"] == pytest.approx(0.514, abs=0.001) and nf["diff"] > 50 * nf["se"]
    assert nf["ours"] > 5 * nf["theirs"]
    assert planted["no leave-one-out"]["fails"] and planted["no leave-one-out"]["ours"] < 0.09
    from sraging.cleaning import noise_baseline                          # the degenerate reading, for the record
    zero = noise_baseline(3, 2.5, floors=(0.0, 0.0), spread=phi, reps=4000, seed=ACC.NOISE_SEED)["floor"]
    assert np.isnan(zero["rate"]) and zero["reason"].startswith("not measurable")


# ---------------------------------------------------------------------------------- unchanged (SC-008, T053 code)
def test_the_unchanged_groups_are_the_suites_the_earlier_results_tests_and_the_closure_check():
    g = ACC.unchanged_groups()
    assert g["unit"] == ["tests/unit"] and g["parity"] == ["tests/parity"]
    assert "tests/acceptance/test_results_004.py" in g["results"] and "tests/acceptance/test_results.py" in g["results"]
    assert not any("005" in p for p in g["results"])
    assert g["closure"] == ["tests/acceptance/test_results_005.py::test_claims_closure_untouched"]


CLOSED_AT_004, CLOSED_AT = "759c003", "d604a97"      # 005's base and merge (spec 006 T046: 006 changes the closure)


def _at_005s_merge(monkeypatch):
    """Spec 003's reading as it stood on 005's branch: from 004's merge to 005's (spec 006 changes the closure by
    design, so HEAD no longer reads "no closure change"; the bound is test_results_005.py's, spec 006 T001)."""
    real = A3.git
    monkeypatch.setattr(A3, "merge_base", lambda *a, **k: CLOSED_AT_004)
    monkeypatch.setattr(A3, "git", lambda *a, **k: real(*[CLOSED_AT if x == "HEAD" else x for x in a], **k))


def test_spec003s_unchanged_reads_no_closure_change_and_no_claims_rerun_here(monkeypatch):
    _at_005s_merge(monkeypatch)
    r = ACC.spec003_reading()
    assert r["changed"] == [] and r["violations"] == [] and r["spec_exempt"] == [] and not r["rerun_required"]
    assert ACC.spec003_ok(r)


def test_a_closure_change_makes_spec003s_reading_require_the_rerun(monkeypatch):
    real = A3.git

    def git(*a, **k):
        if a[:2] == ("diff", "--name-only") and "src/sraging/cohort.py" in a:
            return "src/sraging/cohort.py"
        if a[0] == "merge-base":
            return "BASE"
        if a[0] == "diff" and "--name-status" in a:
            return ""
        return real(*a, **k)
    monkeypatch.setattr(A3, "git", git)
    r = ACC.spec003_reading()
    assert r["changed"] == ["src/sraging/cohort.py"] and r["rerun_required"] and not ACC.spec003_ok(r)


def test_the_unchanged_phase_runs_its_groups_and_spec003s_reading(monkeypatch, tmp_path):
    """The phase's plumbing on one tiny group (a real pytest subprocess), results redirected; a planted failing group
    fails the phase."""
    reading, real_reading = None, ACC.spec003_reading

    def bounded():
        nonlocal reading
        if reading is None:
            with pytest.MonkeyPatch.context() as mp:
                _at_005s_merge(mp)
                reading = real_reading()
        return reading
    monkeypatch.setattr(ACC, "spec003_reading", bounded)
    monkeypatch.setattr(A3, "RESULTS", str(tmp_path))
    one = "tests/unit/test_units.py::test_the_unit_table_is_the_closed_list"
    monkeypatch.setattr(ACC, "unchanged_groups", lambda: {"unit": [one]})
    ACC.main(["unchanged", "--dev"])
    out = json.load(open(tmp_path / "unchanged_005.json"))
    assert out["status"] == "pass" and out["groups"]["unit"]["passed"] == 1 and not out["spec003"]["rerun_required"]
    bad = tmp_path / "test_planted_fail.py"
    bad.write_text("def test_x():\n    assert False\n")
    monkeypatch.setattr(ACC, "unchanged_groups", lambda: {"unit": [str(bad)]})
    ACC.main(["unchanged", "--dev"])
    assert json.load(open(tmp_path / "unchanged_005.json"))["status"] == "fail"


# ------------------------------------------------------------- workers: resuming a job whose poller died (2026-11-09)
def test_a_resume_counts_only_when_head_differs_from_the_jobs_head_in_the_driver_alone():
    assert ACC.resume_problems("a" * 40, diff=["tools/acceptance_005.py", "tests/unit/test_acceptance_005.py",
                                               "tests/acceptance/results/km_005.json", "specs/005-x/review/r.md"]) == []
    assert ACC.resume_problems("a" * 40, diff=["tests/unit/test_cleaning.py", "uv.lock"]) != []   # tests and the lock
    probs = ACC.resume_problems("a" * 40, diff=["tools/acceptance_005.py", "src/sraging/fitset.py"])
    assert len(probs) == 1 and "src/sraging/fitset.py" in probs[0]                       # the planted code change


def test_the_remote_tree_check_compares_explicit_hashes_and_names_each_difference(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "head.tar").write_text("tar")
    (tmp_path / ".cache" / "pycache").mkdir(parents=True)
    (tmp_path / ".cache" / "pycache" / "a.pyc").write_text("bytecode")
    local = ACC.tree_sha256(tmp_path)
    assert list(local) == ["src/a.py"]                                     # the tarball and bytecode are skipped
    assert ACC.tree_problems(local, dict(local)) == []                     # the twin
    probs = ACC.tree_problems(local, {"src/a.py": "0" * 64, "src/b.py": "1" * 64})
    assert any("a.py" in p and "content differs" in p for p in probs)
    assert any("b.py" in p and "only" in p for p in probs)
    assert any("missing" in p for p in ACC.tree_problems(local, {}))
    assert "sha256sum" in ACC.remote_tree_command()[-1] and "./.cache/*" in ACC.remote_tree_command()[-1]


def test_polling_waits_out_transient_failures_and_gives_up_after_its_retries():
    calls = iter([RuntimeError("ssh: connect"), RuntimeError("ssh: connect"), "RUN", "DONE"])

    def remote(cmd):
        x = next(calls)
        if isinstance(x, Exception):
            raise x
        return x
    assert ACC.poll_job(["bjobs", "<JOBID>"], "7", retries=2, wait_s=0, remote=remote) == "DONE"

    def down(cmd):
        raise RuntimeError("ssh: connect")
    with pytest.raises(RuntimeError):
        ACC.poll_job(["bjobs", "<JOBID>"], "7", retries=2, wait_s=0, remote=down)


def test_resume_and_job_head_go_together():
    with pytest.raises(SystemExit, match="together"):
        ACC.main(["workers", "--resume", "469166"])
