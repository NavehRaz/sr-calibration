"""Spec 003's acceptance tools, the parts that decide a verdict without fitting (sr-manager gate 4-7, CHEAPEN; the
coordinator's walker panel): the kmax confirm walk and cap moves, the coverage error origin and side scores, the speed
load rule, the panel writer, the import closure and untouched check of the `unchanged` phase. Fast; fits nothing.
"""
import math
import os
import tomllib

import pytest

import acceptance_003 as A
import acceptance_003_coverage as CV
import acceptance_003_kmax as KM
import acceptance_003_speed as SP
import acceptance_003_walker as WK


# ------------------------------------------------------------------------------------------------ kmax
def _batches(confirmed):
    calls = []

    def run_batch(ids):
        calls.append(list(ids))
        return {c: {"cohort_id": c, "confirmed": c in confirmed, "fit300": {"beta_upper": 1.0}} for c in ids}
    return run_batch, calls


def test_kmax_confirms_along_the_seeded_list_until_ten_and_fits_the_controls_first():
    draw = [f"c{i:02d}" for i in range(30)]
    confirmed = {c for i, c in enumerate(draw) if i % 3 != 1}          # 2 of every 3
    run_batch, calls = _batches(confirmed)
    kept, conf, tried, ended = KM.confirm_along(draw, ["n0", "n1", "n2"], run_batch, 4)
    assert kept == [c for c in draw if c in confirmed][:10] and not ended
    assert calls[0][-3:] == ["n0", "n1", "n2"] and all("n0" not in b for b in calls[1:])
    assert tried == draw[:16] and all(c in conf for c in tried)       # stops at the batch that completes ten


def test_kmax_records_a_list_that_ends_short():
    draw = [f"c{i}" for i in range(12)]
    run_batch, _ = _batches({"c1", "c5"})
    kept, _, tried, ended = KM.confirm_along(draw, [], run_batch, 5)
    assert kept == ["c1", "c5"] and ended and tried == draw


def test_kmax_cap_moves_count_a_vanished_cap_as_moved():
    assert KM._moved(1.0, 10.0) == pytest.approx(1.0)
    assert KM._moved(1.0, float("nan")) == math.inf                     # no finite cap at kappa_max 3000
    assert KM._moved(1.0, None, refit_ok=False) is None                  # the refit failed: not a measurement
    assert KM._moved(float("nan"), 2.0) is None and KM._moved(None, 2.0) is None


def test_kmax_reads_kappa_at_the_beta_cap_node_from_the_full_z():
    ll0 = -100.0
    nodes = [(1e-3, ll0, [0.0, math.log(1e-3), 0.0, math.log(1.0)]),
             (1e-2, ll0 - 1.0, [0.0, math.log(1e-2), 0.0, math.log(300.0)]),          # inside 3.84: sets the cap
             (1e-1, ll0 - 10.0, [0.0, math.log(1e-1), 0.0, math.log(5.0)])]
    assert KM.kappa_at_cap(nodes, ll0 - 0.5) == pytest.approx(300.0)
    assert KM.kappa_at_cap([], ll0) is None


# ------------------------------------------------------------------------------------------------ coverage
def _raise_from(path, exc=FloatingPointError):
    code = compile(f"def f():\n    raise {exc.__name__}('x')\n", path, "exec")
    ns = {exc.__name__: exc}
    exec(code, ns)
    try:
        ns["f"]()
    except exc as ex:
        return ex


def test_an_error_raised_in_the_frozen_code_is_told_from_sraging_s_own():
    assert CV.error_origin(_raise_from(os.path.join(CV.FROZEN, "estimator.py"))) == "frozen"
    assert CV.error_origin(_raise_from(os.path.join(CV.SRAGING, "claims", "walker.py"))) == "sraging"
    assert CV.error_origin(_raise_from("/elsewhere/lib.py")) == "harness"
    inner = _raise_from(os.path.join(CV.FROZEN, "fp_jax.py"))
    try:
        raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert CV.error_origin(outer) == "frozen"                        # a cause raised in the frozen code


def _row(cid, claim, truth=None):
    return {"cohort_id": cid, "status": "ok", "truth": truth or {"eta": 1e-4, "beta": 3e-3, "epsilon": 2e-3,
                                                                  "kappa": 0.5}, "claims": [claim]}


def test_widened_claims_are_scored_against_their_claims_profile():
    w = {"parameter": "eta", "shape": "upper_bound", "lower": None, "upper": 1e-3, "label": "",
         "widening": {"claims_profile": {"shape": "closed", "lower": 2e-4, "upper": 5e-4}, "upper_moved_dex": 0.3,
                      "lower_moved_dex": None}}
    s = CV.widened_scores([_row("a", w)], "eta")
    assert s["n"] == 1 and s["widened"]["coverage"] == 1.0 and s["claims_profile"]["coverage"] == 0.0
    assert s["shape_moves"] == {"closed->upper_bound": 1}


def test_negligible_costly_errors_and_the_rate_among_truths_above_3x():
    from sraging.claims import combinations as K
    t = {"eta": 1e-4, "beta": 3e-3, "epsilon": 2e-3, "kappa": 0.5}
    v = K.value("rbar_over_eta", t)
    r = -0.1                                                            # log10(cap / floor) on the labelled claims

    def neg(upper, labelled=True):
        return {"parameter": "rbar_over_eta", "shape": "upper_bound", "lower": None, "upper": upper,
                "label": "negligible: x" if labelled else "",
                "label_info": {"kind": "negligible", "log10_cap_over_floor": r} if labelled else None}
    rows = [dict(_row("a", neg(2 * v), t), negligible_floor={"floor": 2 * v / 10 ** r}),     # floor 2.5v: below 3x
            dict(_row("b", neg(v / 10), t), negligible_floor={"floor": v / 10 / 10 ** r}),   # floor 0.13v: COSTLY
            dict(_row("c", neg(v / 10, False), t), negligible_floor={"floor": v / 10})]      # above 3x, unlabelled
    s = CV.negligible_scores(rows)
    assert (s["labelled"], s["costly_errors"], s["truths_above_3x_floor"], s["floor_mismatch"]) == (2, 1, 2, 0)
    assert s["costly_error_rate"] == 0.5 and s["labelled_rate_among_truths_above_3x"] == 0.5
    rows[1]["negligible_floor"] = {"floor": v}                          # the tool's floor is not the label's
    assert CV.negligible_scores(rows)["floor_mismatch"] == 1


def test_the_label_kinds_include_those_label_info_drops():
    c = {"label_info": {"kind": "box_limited"}, "label": "open above: box-limited (kappa<=300) · negligible: x"}
    assert CV.label_kinds(c) == {"box_limited", "negligible"}


# ------------------------------------------------------------------------------------------------ speed
def test_speed_busy_rule_allows_its_own_process_after_the_start():
    rec = tomllib.load(open(os.path.join(A.REPO, "tests", "acceptance", "decisions_003.toml"), "rb"))["speed_load"]
    assert (SP.LOAD_MAX, SP.OWN_LOAD) == (rec["load_max"], rec["own_load"]) == (5.0, 1.0)
    L = SP.LOAD_MAX
    s = [{"at": "start", "load": [L - 0.4, 0, 0]}, {"at": "c1", "load": [L + 0.8, 0, 0]},
         {"at": "end", "load": [L + 0.9, 0, 0]}]
    assert SP.busy(s) == []
    assert SP.busy([{"at": "start", "load": [L + 0.2, 0, 0]}])
    assert SP.busy(s + [{"at": "end", "load": [L + 1.4, 0, 0]}])
    c = SP.cores()
    assert c["logical"] == os.cpu_count() and isinstance(c["levels"], list)


# ------------------------------------------------------------------------------------------------ the panel
def test_the_panel_writer_round_trips_inline_tables_and_the_cuts():
    panel = {"walker": {"cohorts": [{"id": "pure-a", "name": "profile_jax2", "why": "pure, \"q\""}], "n": 1},
             "coverage": {"profile_jax2": ["a", "b"], "gated": [["profile_jax2", "pure"]],
                          "cuts": {"het": {"cohorts": ["h1"], "reason": "0 informative"}}}}
    assert tomllib.loads(A._toml(panel)) == panel


def test_the_pilot_panel_cuts_coverage_and_lists_every_kmax_candidate():
    """sr-manager CHEAPEN: het, the hazard arm under profile_jax2 and the realfull C. elegans cohorts are not fitted
    under profile_jax2, each cut recorded with its reason; gate 4: the kmax list is every candidate, seeded."""
    if not os.path.isdir(A.J2):
        pytest.skip(f"NOT APPLICABLE: no J2 at {A.J2}")
    import oracle_jax2 as OJ
    lib = OJ.library()
    panel, extra = A.build_panel()
    cov = panel["coverage"]
    arms = {lib.spec(c).arm for c in cov["profile_jax2"]}
    assert arms == {"pure", "censor", "realfull"}
    assert not any(lib.spec(c).real_name == "celegans" for c in cov["profile_jax2"])
    assert {lib.spec(c).arm for c in cov["profile_jax2_h"]} == {"hazard"}
    assert set(cov["cuts"]) == {"het", "hazard_under_profile_jax2", "realfull_celegans"}
    assert all(v["cohorts"] and v["reason"] for v in cov["cuts"].values())
    assert len(panel["kmax"]["draw"]) == panel["kmax"]["candidates_total"] == len(set(panel["kmax"]["draw"]))
    assert tomllib.loads(A._toml(panel)) == panel


def test_stored_runtime_thetas_reads_only_jax2_1_0_0_s_runtime():
    if not os.path.isdir(A.J2):
        pytest.skip(f"NOT APPLICABLE: no J2 at {A.J2}")
    st = A.stored_runtime_thetas()
    assert st and all(p["path"].endswith(("__v100on.json", "__dev6on.json")) or p["path"].count("__") == 1
                      for p in st.values())
    assert not any("dev4" in p["path"] or "dev6off" in p["path"] for p in st.values())


def test_stored_fits_read_r3d6_l5_s_nan_strings_as_floats(tmp_path, monkeypatch):
    """Measured 2026-10-06: the fits phase compared 12 of 18 stored fits unequal, every diff a float NaN against
    R3d6/L5's string "nan" (its writer cannot put NaN in JSON). The loader reads those strings back; nothing else."""
    assert math.isnan(A.j2_floats("nan")) and A.j2_floats("-inf") == -math.inf and A.j2_floats("none") == "none"
    d = tmp_path / "reports" / "rc2" / "R3d6" / "L5"
    d.mkdir(parents=True)
    rec = {"cohort": "c1", "name": "jax2", "cpu_s": 5.0, "claims": [{"label": "nan"}],
           "result": {"theta": [1.0], "ci": {"beta@0.95": ["nan", 2.0]}, "verdict": "identified"}}
    (d / "c1__jax2.json").write_text(__import__("json").dumps(rec))
    monkeypatch.setattr(A, "J2", str(tmp_path))
    st = A.stored_fits()[("c1", "jax2")]
    lo, hi = st["result"]["ci"]["beta@0.95"]
    assert math.isnan(lo) and hi == 2.0 and st["result"]["verdict"] == "identified"
    assert st["claims"] == [{"label": "nan"}]            # the result only: the claims compared equal as stored
    assert st["meta"]["cpu_s"] == 5.0 and st["_path"].endswith("L5/c1__jax2.json")


def test_the_walker_panel_draw():
    """The coordinator's [walker] table: up to 3 kappa-hat-at-300 cohorts under their stored fit's name, 12 pure under
    profile_jax2, 5 hazard under profile_jax2_h, no repeats, the same on every call."""
    if not os.path.isdir(A.J2):
        pytest.skip(f"NOT APPLICABLE: no J2 at {A.J2}")
    import oracle_jax2 as OJ
    lib = OJ.library()
    S = A.sets_c()
    pure = sorted(c for c in S["C2"]["J"]["MEDIUM"] if c.startswith("pure-k"))
    hz = sorted(c for c in S["C2"]["H"]["MEDIUM_H"] if c.startswith("hazard-") and lib.spec(c).window is None)
    w = A.walker_draw(lib, pure, hz)
    ids = [r["id"] for r in w["cohorts"]]
    names = [r["name"] for r in w["cohorts"]]
    k = A.WALKER_K300 - w["kappa300_shortfall"]
    assert len(ids) == len(set(ids)) == k + A.WALKER_PURE + A.WALKER_HAZARD
    assert names[k:] == ["profile_jax2"] * A.WALKER_PURE + ["profile_jax2_h"] * A.WALKER_HAZARD
    assert all("kappa-hat at 300" in r["why"] for r in w["cohorts"][:k])
    assert A.walker_draw(lib, pure, hz) == w


# ------------------------------------------------------------------------------------------------ unchanged
def test_the_import_closure_follows_function_level_imports_and_package_inits():
    cl = A.import_closure(("acceptance_claims",))
    assert "src/sraging/calibrators/profile_v033.py" in cl            # imported inside claims_worker
    assert {"src/sraging/__init__.py", "src/sraging/calibrators/__init__.py", "tools/acceptance.py",
            "tools/parity_checks.py", "tools/oracle_v033.py"} <= set(cl)
    full = A.import_closure()
    assert {"src/sraging/compat/v033/estimator.py", "src/sraging/compat/v033/fp_jax.py", "src/sraging/result.py",
            "src/sraging/registry.py", "src/sraging/labels.py"} <= set(full)


def test_untouched_allows_only_the_jax2_oracle_snapshot_added(monkeypatch):
    out = "A\ttests/oracle/jax2_3bc5775/x.py\nM\tsrc/sraging/engine/grid.py\nM\ttests/oracle/v033/a.py"
    monkeypatch.setattr(A, "git", lambda *a, **k: out)
    assert A.untouched_problems() == ["M\tsrc/sraging/engine/grid.py", "M\ttests/oracle/v033/a.py"]
    monkeypatch.setattr(A, "git", lambda *a, **k: "BASE" if a[0] == "merge-base" else "")    # base resolved, no diff
    assert A.untouched_problems() == []


def test_untouched_exempts_a_src_change_of_comments_and_docstrings_only(monkeypatch):
    """Spec 003g T075 (Naveh 2026-10-29), planted: 85072be's kind of change passes; code, a non-.py file, a deletion
    and an oracle docstring do not."""
    base = '"""The ageing clock."""\ndef f(x):\n    """Doc."""\n    return x + 1  # ageing\n'
    files = {
        "src/sraging/engine/doc.py": '"""The production clock."""\ndef f(x):\n    """New doc."""\n    return x + 1  # prod\n',
        "src/sraging/engine/code.py": '"""The ageing clock."""\ndef f(x):\n    """Doc."""\n    return x + 2  # ageing\n',
        "src/sraging/engine/data.toml": "a = 2\n",
        "tests/oracle/v033/o.py": '"""Reworded."""\ndef f(x):\n    """Doc."""\n    return x + 1  # ageing\n',
    }
    diff = "\n".join(["M\tsrc/sraging/engine/doc.py", "M\tsrc/sraging/engine/code.py", "M\tsrc/sraging/engine/data.toml",
                      "D\tsrc/sraging/engine/gone.py", "M\ttests/oracle/v033/o.py"])

    def fake_git(*args, **kw):
        if args[0] == "diff":
            return diff
        if args[0] == "merge-base":
            return "BASE"
        rev, path = args[1].split(":", 1)
        return base if rev == "BASE" else files[path]
    monkeypatch.setattr(A, "git", fake_git)
    doc = []
    assert A.untouched_problems(doc) == ["M\tsrc/sraging/engine/code.py", "M\tsrc/sraging/engine/data.toml",
                                         "D\tsrc/sraging/engine/gone.py", "M\ttests/oracle/v033/o.py"]
    assert doc == ["src/sraging/engine/doc.py"]
    assert A.without_docstrings("def f(:") is None


# ---------------------------------------- found at spec 004 T145: an unresolvable base never reads as "no change"
NO_MAIN = "[unchanged] REFUSED: no merge-base with main (create or fetch the main branch)"


def _git_recording(base):
    """git with `merge-base` answering `base` ("" as git prints when main is missing) and every diff empty; the calls."""
    calls = []

    def git(*a, **k):
        calls.append(a)
        return base if a[0] == "merge-base" else ""
    return git, calls


def test_an_unresolvable_main_refuses_by_name_and_never_reads_as_no_change(monkeypatch):
    git, calls = _git_recording("")                                        # planted: a clone with no local main
    monkeypatch.setattr(A, "git", git)
    for check in (A.merge_base, A.untouched_problems):
        with pytest.raises(SystemExit) as ex:
            check()
        assert str(ex.value) == NO_MAIN
    assert not [c for c in calls if c[0] == "diff"]                          # no diff was read as "nothing changed"
    with pytest.raises(SystemExit) as ex:                                    # the phase refuses before it writes
        A.phase_unchanged(__import__("types").SimpleNamespace(only=None, dev=True, workers=1))
    assert str(ex.value) == NO_MAIN


def test_a_resolvable_main_takes_every_diff_from_its_merge_base(monkeypatch):
    git, calls = _git_recording("abc123")                                  # the twin: main resolves
    monkeypatch.setattr(A, "git", git)
    assert A.merge_base() == "abc123" and A.untouched_problems() == []
    assert ("diff", "--name-status", "abc123", "HEAD", "--", *A.UNTOUCHED) in calls


# ------------------------------------------------- spec 004 T144: a named engine exemption, held to the claims re-run
def test_t144s_list_is_the_tasks_ten_engine_files_and_not_the_stepper():
    listed = dict(A.SPEC_ENGINE_CHANGES["004"])
    assert listed == {"src/sraging/engine/inputs.py": "T017", "src/sraging/engine/quadrature.py": "T016",
                      "src/sraging/engine/mixture.py": "T018-T020, T034", "src/sraging/engine/evaluator.py": "T020, T034",
                      "src/sraging/engine/__init__.py": "T020, T034", "src/sraging/engine/pool.py": "T031",
                      "src/sraging/engine/cost.py": "T032", "src/sraging/engine/series.py": "T033",
                      "src/sraging/engine/density.py": "T094", "src/sraging/engine/tensor_reference.py": "T107"}
    assert not {"src/sraging/engine/stepper.py", "src/sraging/engine/stepper_gen.py"} & set(listed)


def _diff_git(diff):
    return lambda *a, **k: diff if a[0] == "diff" else "BASE"


def test_t144_an_unlisted_engine_file_changed_is_a_violation_and_a_listed_one_is_exempt(monkeypatch):
    monkeypatch.setattr(A, "git", _diff_git("A\tsrc/sraging/engine/new_thing.py\nA\tsrc/sraging/engine/quadrature.py"))
    exempt = []
    assert A.untouched_problems([], exempt) == ["A\tsrc/sraging/engine/new_thing.py"]           # planted: unlisted
    assert exempt == [{"path": "src/sraging/engine/quadrature.py", "status": "A", "spec": "004", "task": "T016"}]
    assert A.untouched_problems() == ["A\tsrc/sraging/engine/new_thing.py",                   # no collector given:
                                      "A\tsrc/sraging/engine/quadrature.py"]                  # strict, as before


def _checks(n):
    return {k: {"passed": n, "total": 40} for k in ("claims_identity", "theta_hat", "value_bytes", "grad_bytes")}


def test_t144_the_exemption_holds_at_40_of_40_and_a_listed_file_is_a_violation_again_at_39():
    exempt = [{"path": "src/sraging/engine/quadrature.py", "status": "A", "spec": "004", "task": "T016"}]
    assert A.claims_exempt_ok(_checks(40))                                                       # the twin
    assert A.exempt_violations(exempt, _checks(40)) == []
    assert not A.claims_exempt_ok(_checks(39))                                                   # planted: 39/40
    assert A.exempt_violations(exempt, _checks(39)) == ["A\tsrc/sraging/engine/quadrature.py"]
    bad = _checks(40)
    bad["grad_bytes"] = {"passed": 40, "total": 39}                                              # not 40 compared
    assert not A.claims_exempt_ok(bad) and not A.claims_exempt_ok({}) and not A.claims_exempt_ok(None)


def test_t144_compat_v033_the_oracles_and_their_tools_stay_strict_whatever_the_list_says(monkeypatch):
    monkeypatch.setattr(A, "SPEC_ENGINE_CHANGES", {"004": (("src/sraging/compat/v033/estimator.py", "T999"),
                                                           ("tests/oracle/v033/a.py", "T999"),
                                                           ("tools/oracle_fp_piecewise.py", "T999"))})
    monkeypatch.setattr(A, "git", _diff_git("A\tsrc/sraging/compat/v033/estimator.py\nA\ttests/oracle/v033/a.py\n"
                                            "A\ttools/oracle_fp_piecewise.py"))
    exempt = []
    assert A.untouched_problems([], exempt) == ["A\tsrc/sraging/compat/v033/estimator.py", "A\ttests/oracle/v033/a.py",
                                                "A\ttools/oracle_fp_piecewise.py"] and exempt == []


def test_t144_check_claims_001_needs_the_40_of_40_rerun_when_spec_exempt_is_not_empty():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "test_results_003_t144", os.path.join(os.path.dirname(os.path.dirname(__file__)), "acceptance",
                                              "test_results_003.py"))
    R3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R3)
    good = {"status": "pass", "untouched": {"violations": [], "spec_exempt": [{"path": "src/sraging/engine/cost.py"}]},
            "closure": {"files": ["src/sraging/calibrators/profile_v033.py"], "changed_vs_base": []},
            "rerun_required": True, "claims_status": "pass", "checks": _checks(40),
            "srbench_record": {"status": "not_applicable", "reason": "not re-run"}}
    R3.check_claims_001(good)                                                                    # the twin
    with pytest.raises(AssertionError):
        R3.check_claims_001({**good, "checks": _checks(39)})                                     # planted: 39/40
    with pytest.raises(AssertionError):
        R3.check_claims_001({**good, "rerun_required": False, "claims_status": "not re-run"})     # exempt, no re-run


# ----------------------------------- the oracle tool's additions-only exemption (Naveh 2026-11-05, beside T144's list)
ORACLE = "tools/oracle_fp_piecewise.py"
OLD_TOOL = ('"""The oracle tool."""\nimport os\n\n_SRK = {}\n\n\ndef _op_a(a):\n    return 1\n\n\n'
            'OPS = {\n    "ping": lambda a: {"value": 1},\n    "a": _op_a,\n}\n')


def _tool(extra_defs="", extra_ops="", edit=None):
    """OLD_TOOL with `extra_defs` inserted before OPS and `extra_ops` lines inside it; `edit` (old, new) changes a line."""
    t = OLD_TOOL.replace("\n\nOPS = {", "\n" + extra_defs + "\n\nOPS = {").replace('    "a": _op_a,\n',
                                                                                    '    "a": _op_a,\n' + extra_ops)
    return t.replace(*edit) if edit else t


def _tool_git(new_text):
    """git as untouched_problems calls it, on OLD_TOOL at BASE and `new_text` at HEAD, the diffs made by difflib."""
    import difflib
    old_l, new_l = OLD_TOOL.splitlines(keepends=True), new_text.splitlines(keepends=True)
    u0 = "".join(difflib.unified_diff(old_l, new_l, f"a/{ORACLE}", f"b/{ORACLE}", n=0))
    plus = sum(1 for x in u0.splitlines() if x.startswith("+") and not x.startswith("+++"))
    minus = sum(1 for x in u0.splitlines() if x.startswith("-") and not x.startswith("---"))

    def git(*a, **k):
        if a[0] == "diff" and "--name-status" in a:
            return f"M\t{ORACLE}"
        if a[0] == "diff" and "--numstat" in a:
            return f"{plus}\t{minus}\t{ORACLE}"
        if a[0] == "diff":
            return u0
        if a[0] == "merge-base":
            return "BASE"
        rev, path = a[1].split(":", 1)
        return OLD_TOOL if rev == "BASE" else new_text
    return git


NEW_OPS = ('def _helper(x):\n    return x\n\n\ndef _op_b(a):\n    return _helper(2)\n', '    "b": _op_b,\n')


def test_the_oracle_tool_additions_only_is_exempt_and_held_to_the_claims(monkeypatch):
    monkeypatch.setattr(A, "git", _tool_git(_tool(*NEW_OPS)))
    exempt = []
    assert A.untouched_problems([], exempt) == []                                                  # the twin
    assert [(e["path"], e["spec"], e["task"], e["rule"]) for e in exempt] == [
        (ORACLE, "004", "T070, T072, T074", "additions-only")]
    assert A.exempt_violations(exempt, _checks(40)) == []
    assert A.exempt_violations(exempt, _checks(39)) == [f"M\t{ORACLE}"]                            # planted: 39/40
    assert A.untouched_problems() == [f"M\t{ORACLE}"]                                             # no collector: strict


@pytest.mark.parametrize("new_text,why", [
    (_tool(*NEW_OPS, edit=("    return 1\n", "    return 3\n")), "deletes or changes"),          # a changed line
    (_tool("def _op_a(a):\n    return 5\n", ""), "redefines"),                                     # an existing _op_
    (_tool("import sys\n", ""), "not a new function"),                                            # a module import
    (_tool("X = 1\n", ""), "not a new function"),                                                 # module-level code
    (_tool("def op_public(a):\n    return 1\n", ""), "not a new function"),                       # not a _ name
])
def test_the_oracle_tool_is_a_violation_unless_its_change_is_additions_only(monkeypatch, new_text, why):
    monkeypatch.setattr(A, "git", _tool_git(new_text))
    exempt = []
    assert A.untouched_problems([], exempt) == [f"M\t{ORACLE}"] and exempt == []
    assert any(why in r for r in A.oracle_additions_problems(ORACLE, "BASE")), A.oracle_additions_problems(ORACLE, "BASE")


def test_other_oracle_tools_stay_strict_even_additions_only(monkeypatch):
    other = "tools/oracle_v033.py"
    monkeypatch.setattr(A, "git", lambda *a, **k: f"M\t{other}" if a[0] == "diff" else "BASE")
    assert A.untouched_problems([], []) == [f"M\t{other}"]
    assert other not in {p for p, *_ in A.SPEC_ORACLE_ADDITIONS["004"]}


# ------------------------------------------------------------------------------------------------ provenance
def test_the_tool_fits_with_its_own_checkouts_src():
    """A run in a worktree must not import the main tree's sraging through the venv's editable install (found
    2026-10-06: the checks phase fits in the tool's own process)."""
    assert A.own_src_problem() is None
    import sraging
    real = sraging.__file__
    try:
        sraging.__file__ = "/elsewhere/src/sraging/__init__.py"
        assert "not this checkout" in A.own_src_problem()
    finally:
        sraging.__file__ = real


def test_the_brute_force_starts_span_beta_and_kappa_over_the_whole_box():
    """SC-004 as amended 2026-10-06: the cube (+-1.5 dex around theta-hat) cannot reach a branch 9-10 dex away in
    beta; the whole-box Sobol starts do, with eta and eps still in the cube."""
    import numpy as np
    bounds = [(math.log(1e-12), math.log(1e3))] * 3 + [(math.log(1e-3), math.log(300.0))]
    z_hat = np.log([2e-4, 8e-9, 3.5e-5, 300.0])
    starts = np.array(WK.bf_starts(z_hat, bounds, 32, [], seed=20261005))
    lb = starts[:, 1] / math.log(10)
    assert lb.max() - lb.min() > 12 and (lb > 0).any()                 # beta reaches 1-1000
    lk = starts[:, 3] / math.log(10)
    assert lk.min() < -2 and lk.max() > 2
    cube = starts[:64]
    assert (np.abs(cube[:, [0, 2]] - z_hat[[0, 2]]) <= 1.5 * math.log(10) + 1e-9).all()


def test_the_archive_check_is_the_walkers_own_contradiction_test():
    """Since the walker re-walks a contradicted side, step 1's archive check is that pass's convergence check: the
    two must agree on what contradicts a walk (beyond an end by more than 0.01 dex, at deviance below 3.8415)."""
    import numpy as np
    from sraging.claims import walker as W
    w = {"ll_ref": 0.0, "ci": {"ci_lo": math.exp(1.0), "ci_hi": math.exp(2.0)}}
    x = [0.9, 2.01, 2.2]                    # log(eta/eps): 0.043 dex below the lower end; 0.004 and 0.087 dex above
    Z = np.array([[v, 0.0, 0.0, 0.0] for v in x])
    LL = np.array([-1.0, -1.0, -3.0])       # deviance 2, 2, 6
    checks = {c["side"]: c for c in WK.archive_check("eta_over_eps", w, Z, LL)}
    con, _ = W.contradictions("eta_over_eps", w, Z, LL)
    assert not checks["lower"]["ok"] and list(con) == ["lower"] and list(con["lower"]) == [0]
    assert checks["upper"]["ok"]            # inside but within 0.01 dex; the far one is outside
