"""Spec 003b's acceptance tool (T014): `verify` (FR-001) and `census` (FR-006), each on planted cases with a twin.
The comparator tests are pure; `verify` reads jax2's git objects (read-only) and the census test loads the 1.0.1
oracle's srbench, so it runs in a subprocess (it must not meet another srbench in this process). Both skip where the
jax2 checkouts or the tester's library v6 are absent."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

import acceptance_003 as A3
import acceptance_003b as ACC

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HAVE_J2 = os.path.isdir(ACC.J2_REPO) and os.path.exists(ACC.CENSUS)
CID = "censor-k0.01-p05-c1-s1"


# --------------------------------------------------------------------------------------------- bit equality
@pytest.mark.parametrize("a,b,same", [
    (0.1, 0.1, True), (0.1, float(np.nextafter(0.1, 1.0)), False), (float("nan"), float("nan"), True),
    (0.0, -0.0, False), (3600, np.int64(3600), True), (3600, 3600.0, False), (True, np.bool_(True), True),
    (True, False, False), (True, 1, False), (None, None, True), (None, 0.0, False), (np.float64(0.25), 0.25, True),
])
def test_census_values_compare_bit_for_bit_and_by_kind(a, b, same):
    assert ACC.bit_equal(a, b) is same


# --------------------------------------------------------------------------------------------- the census comparator
def _row(cid, view="J", dt5=0.0004709215631307541, in5=True):
    return {"cid": cid, "view": view, "lib": "v6", "cv": 0.0377, "cv_c": 0.0376, "in5": in5, "N5": 3600,
            "dt5": dt5, "L": 99.2455, "dl": 0.0101, "n": 1000}


def test_a_planted_dt_is_caught_and_its_twin_is_equal():
    truth = {f: v for f, v in _row("x").items() if f in ACC.CENSUS_FIELDS}
    rows = [_row("equal"), _row("planted", dt5=float(np.nextafter(truth["dt5"], 1.0)))]
    res = ACC.census_compare(rows, lambda cid, view: dict(truth))
    assert res["rows"] == 2 and res["equal"] == 1
    assert [d["cid"] for d in res["differences"]] == ["planted"]
    assert list(res["differences"][0]["fields"]) == ["dt5"]
    assert res["table"][0][:3] == ["equal", "J", True] and res["table"][1][:3] == ["planted", "J", False]
    assert res["in_scope"] == {"jax2": {"J": 2, "H": 0}, "sraging": {"J": 2, "H": 0}}


def test_the_scope_counts_are_kept_per_view_and_per_side():
    truth = {f: v for f, v in _row("x").items() if f in ACC.CENSUS_FIELDS}
    rows = [_row("a", "J"), _row("b", "H", in5=False)]
    res = ACC.census_compare(rows, lambda cid, view: dict(truth, in5=(view == "J")))
    assert res["equal"] == 2 and res["views"] == {"J": 1, "H": 1}
    assert res["in_scope"]["sraging"] == res["in_scope"]["jax2"] == {"J": 1, "H": 0}


# --------------------------------------------------------------------------------------------- verify (real, read-only)
@pytest.mark.skipif(not HAVE_J2, reason="the jax2 freeze checkout or its reports are absent on this host")
def test_verify_passes_on_the_handoff_and_a_planted_pin_fails_only_its_check():
    pins = ACC.panel()["pins"]
    good = ACC.verify_checks(pins)
    assert all(r["ok"] for r in good), [r for r in good if not r["ok"]]
    names = [r["check"] for r in good]
    assert {"tag", "version_lock", "listing_digest", "srbench", "srk", "file:handoff"} <= set(names)
    assert sum(n.startswith("runtime:") for n in names) == 4
    planted = dict(pins, listing_digest="0" * 64, runtime=dict(pins["runtime"], **{"fp_jax.py": "0" * 64}),
                   handoff="0" * 64)
    bad = {r["check"] for r in ACC.verify_checks(planted) if not r["ok"]}
    assert bad == {"listing_digest", "runtime:fp_jax.py", "file:handoff", "version_lock"}   # the lock pins fp_jax too


@pytest.mark.skipif(not HAVE_J2, reason="the jax2 freeze checkout or its reports are absent on this host")
def test_the_verify_phase_writes_a_development_result_with_dev(tmp_path, monkeypatch):
    monkeypatch.setattr(A3, "RESULTS", str(tmp_path))
    ACC.main(["verify", "--dev"])
    out = json.load(open(tmp_path / "verify_003b.json"))
    assert out["status"] == "pass" and out["status_for_acceptance"] == "development" and not out["failed"]
    assert out["provenance"]["version"] == "1.0.1" and out["provenance"]["lock_sha256"].startswith("e3e6e2fe")


# --------------------------------------------------------------------------------------------- census (real, subprocess)
CENSUS_SCRIPT = r"""
import json, sys
import numpy as np
sys.path.insert(0, "tools")
import acceptance_003 as A3
import acceptance_003b as A
A3.RESULTS = sys.argv[1]
cid = sys.argv[2]
real = A.census_rows

def planted(path, want):
    rows = [r for r in real(path, want) if r["cid"] == cid]
    j = [r for r in rows if r["view"] == "J"][0]
    return rows + [dict(j, dt5=float(np.nextafter(j["dt5"], 1.0)))]

A.census_rows = planted
try:
    A.main(["census", "--dev", "--only", cid])
except SystemExit as ex:
    print("EXIT", ex.code)
"""


@pytest.mark.skipif(not (HAVE_J2 and os.path.isdir(os.environ.get("SRAGING_TESTER_V6", ACC.DROPBOX))),
                    reason="the census or the tester's library v6 is absent on this host")
def test_the_census_phase_on_a_real_row_and_its_planted_dt_twin(tmp_path):
    r = subprocess.run([sys.executable, "-c", CENSUS_SCRIPT, str(tmp_path), CID], cwd=REPO, capture_output=True,
                       text=True, timeout=600)
    assert "EXIT 1" in r.stdout, r.stdout + r.stderr
    out = json.load(open(tmp_path / "census_003b.json"))
    assert out["status"] == "fail" and out["status_for_acceptance"] == "development" and out["full"] is False
    assert out["rows"] == 2 and out["equal"] == 1                          # the real J row equal, its twin caught
    assert [(d["cid"], d["view"], list(d["fields"])) for d in out["differences"]] == [(CID, "J", ["dt5"])]
    assert out["table"][0][:3] == [CID, "J", True]


# --------------------------------------------------------------------------------------------- parity (T015)
CITED = {"fp_likelihood.py": "a" * 64, "fp_gof.py": "b" * 64}


def _run_fit_side(srk=None, **diag):
    res = {"theta": [1e-3, 0.5, 0.02, 0.3], "loglik": -100.25, "wall_s": 3.0,
           "diagnostics": {"compile_s": 1.0, "seed_s": 0.5, "mle_s": 2.0, "solver": {"N": 900, "dt": 0.002},
                           "exc_counts": {}, **diag}}
    if srk is not None:
        res["diagnostics"]["srk_sha"] = srk
    claims = [{"parameter": "eta", "shape": "closed", "lower": 1e-4, "upper": 1e-2, "point": 1e-3, "level": 0.95}]
    return res, claims, {"gap_nats": 0.0, "extents": {"eta": {"cpu_s": 0.1}}}


ORACLE_SRK = {"root": "/somewhere/oracle", "files": dict(CITED)}


def test_run_fit_level_ignores_timing_and_the_srk_location_and_compares_the_rest():
    port = _run_fit_side()
    oracle = _run_fit_side(srk=ORACLE_SRK)
    oracle[0]["wall_s"], oracle[0]["diagnostics"]["mle_s"], oracle[2]["extents"]["eta"]["cpu_s"] = 9.0, 7.0, 5.0
    assert ACC.run_fit_diff(port, oracle, CITED) == {"equal": True, "diff_paths": [], "srk": []}


def test_run_fit_level_catches_an_unlisted_key_and_a_changed_srk_file():
    port = _run_fit_side(polish=True)                                  # planted: a key only the port has
    oracle = _run_fit_side(srk=dict(ORACLE_SRK, files=dict(CITED, **{"fp_gof.py": "c" * 64})))
    d = ACC.run_fit_diff(port, oracle, CITED)
    assert not d["equal"] and d["diff_paths"] == ["/result/diagnostics/polish: present only in port"]
    assert d["srk"] and "fp_gof.py" in d["srk"][0]
    twin = ACC.run_fit_diff(_run_fit_side(), _run_fit_side(srk=ORACLE_SRK), CITED)
    assert twin["equal"]
    missing = ACC.run_fit_diff(_run_fit_side(), _run_fit_side(), CITED)   # the oracle must carry its fingerprint
    assert not missing["equal"] and missing["srk"] == ["oracle: no diagnostics/srk_sha"]


class _FR:
    def __init__(self, diag):
        from sraging.result import Claim
        self.cohort, self.status = {"name": "c1"}, "ok"
        self.claims = (Claim("eta", "closed", 1e-4, 1e-2, 1e-3, 0.95, 1, "a label"),)
        self.diagnostics = diag


class _CS:
    def __init__(self, diag, method_name="mle_profile_jax2"):
        import types
        self.cohort_id, self.method_name, self.status = "c1", method_name, "ok"
        self.claims = (types.SimpleNamespace(parameter="eta", shape="closed", lower=1e-4, upper=1e-2, point=1e-3,
                                             level=0.95, n_components=1),)
        self.diagnostics = diag
        self.wall_clock_seconds, self.cpu_seconds = 1.0, 2.0


def _fr_cs(est_port=1e-3, est_oracle=1e-3, extra_port=None):
    method = {"loglik": -100.25, "switches": {"dt_rule": True}, "estimate": {"schema": 1, "theta_hat": {"eta": None}}}
    fd = {**method, "code": "sraging 0.x", "leverage": {"label": "ok"}, "floor_report": {}, "L_SR": {}, "box_lsr": {},
          "assumptions": "no external mortality", "mle_s": 3.0, **(extra_port or {})}
    fd["estimate"] = {"schema": 1, "theta_hat": {"eta": est_port}}
    cd = {**method, "code": "oracle line", "srk_sha": ORACLE_SRK, "mle_s": 9.0}
    cd["estimate"] = {"schema": 1, "theta_hat": {"eta": est_oracle}}
    return ACC.fitresult_view(_FR(fd), "profile_jax2"), ACC.claimset_view(_CS(cd))


def test_fitresult_level_leaves_out_sraging_keys_location_and_timing():
    f, c = _fr_cs()
    assert ACC.fitresult_diff(f, c, CITED) == {"equal": True, "diff_paths": [], "srk": []}


def test_fitresult_level_compares_the_estimate_block():
    f, c = _fr_cs(est_port=float(np.nextafter(1e-3, 1.0)))
    d = ACC.fitresult_diff(f, c, CITED)
    assert not d["equal"] and d["diff_paths"][0].startswith("/diagnostics/estimate/theta_hat/eta:")


def test_fitresult_level_catches_an_unlisted_key_and_the_wrong_method():
    f, c = _fr_cs(extra_port={"box_points": []})
    assert ACC.fitresult_diff(f, c, CITED)["diff_paths"] == ["/diagnostics/box_points: present only in port"]
    f, _ = _fr_cs()
    c = ACC.claimset_view(_CS({"loglik": -100.25, "switches": {"dt_rule": True}, "srk_sha": ORACLE_SRK,
                               "estimate": {"schema": 1, "theta_hat": {"eta": 1e-3}}}, method_name="mle_profile_jax"))
    assert ACC.fitresult_diff(f, c, CITED)["diff_paths"] == ["/method_name: 'mle_profile_jax2' != 'mle_profile_jax'"]


def _fake_worker(cid, name, mode, cited, library="v6"):
    equal = cid != "differs"
    return {"cohort_id": cid, "rows": [{"cohort_id": cid, "level": "run_fit", "equal": equal,
                                        "diff_paths": [] if equal else ["/result/x: 1 != 2"]}]}


def test_the_first_difference_stops_the_run_and_its_twin_runs_through():
    pairs = [{"cid": c, "name": "profile_jax2", "mode": "full"} for c in ("a", "differs", "b")]
    out, stopped = ACC.run_pairs(pairs, 1, CITED, worker=_fake_worker, pool=False)
    assert stopped and [r["cohort_id"] for r in out] == ["a", "differs"]
    twin = [dict(p, cid=p["cid"].replace("differs", "c")) for p in pairs]
    out, stopped = ACC.run_pairs(twin, 1, CITED, worker=_fake_worker, pool=False)
    assert not stopped and len(out) == 3
    assert ACC.is_difference({"cohort_id": "x", "error": "boom"})


def test_the_mle_only_keywords_are_the_ports_own_run_fit_call(monkeypatch):
    """The port's MLE stage passes estimator.fit exactly the keywords the port's run_fit passes (stages added), on
    both registered names; a planted dropped keyword is caught."""
    from sraging import Cohort
    from sraging.calibrators import profile_jax2 as PJ
    import sraging.compat.jax2_101.estimator as E1
    P1 = PJ.PORTS["1.0.1"]
    c = Cohort.from_csv(os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv"), delta=1.0)
    obs = PJ.observation(c)
    seen = []

    class Stop(Exception):
        pass

    def capture(*a, **kw):
        seen.append(kw)
        raise Stop
    monkeypatch.setattr(E1, "fit", capture)
    for name in ("profile_jax2", "profile_jax2_h"):
        seen.clear()
        with pytest.raises(Stop):
            P1.claims.run_fit(obs, **PJ.run_kwargs(P1.classes[name]))
        with pytest.raises(Stop):
            ACC.port_mle_only(obs, name, P1)
        run_fit_kw, mle_kw = seen
        assert mle_kw.pop("stages") == ("mle",)
        assert mle_kw == run_fit_kw
        dropped = {k: v for k, v in mle_kw.items() if k != "extent_z"}
        assert dropped != run_fit_kw


# --------------------------------------------------------------------------------------------- identity-off, v100 (T017, T018)
@pytest.mark.skipif(not HAVE_J2, reason="jax2's G1/cohorts.txt is absent on this host")
def test_the_identity_panel_is_g1s_eleven_lines():
    g1 = ACC.g1_lines()
    lines = ACC.panel()["identity"]["lines"]
    assert len(g1) == 11 and [(x["name"], x["cid"]) for x in g1] == [(x["name"], x["cid"]) for x in lines]
    assert [x["cpu_s"] for x in g1] == [x["cpu_s"] for x in lines]


def test_g1_lines_parse_and_skip_comments(tmp_path):
    p = tmp_path / "cohorts.txt"
    p.write_text("# NAME COHORT_ID PATH CPU_S\n\njax2    pure-a   identity   750.2\njax2_h  pure-a  identity 894.6\n")
    assert ACC.g1_lines(str(p)) == [{"name": "jax2", "cid": "pure-a", "path": "identity", "cpu_s": 750.2},
                                    {"name": "jax2_h", "cid": "pure-a", "path": "identity", "cpu_s": 894.6}]


def _lev(**kw):
    block = {"version": "003f/2", "label": "flag", "R": {"eta/eps": 1.592}, "cost": {"cpu_s": 74.1, "share_of_fit": 0.05},
             "refit": {"theta": [1e-3, 0.5], "cpu_s": 57.1}, "provenance": {"version": "1.0.0"}}
    block.update(kw)
    return block


def test_the_leverage_block_compares_003fs_keys_and_lists_added_ones():
    stored = _lev()
    new = _lev(cost={"cpu_s": 80.0, "share_of_fit": 0.06}, refit={"theta": [1e-3, 0.5], "cpu_s": 60.0}, g0=0.0)
    d = ACC.leverage_diff(new, stored)
    assert d == {"equal": True, "diff_paths": [], "added_keys": ["g0"], "missing_keys": []}
    planted = ACC.leverage_diff(_lev(R={"eta/eps": float(np.nextafter(1.592, 2.0))}), stored)
    assert not planted["equal"] and planted["diff_paths"][0].startswith("/leverage/R/eta/eps:")
    gone = ACC.leverage_diff({k: v for k, v in stored.items() if k != "label"}, stored)
    assert not gone["equal"] and gone["missing_keys"] == ["label"]


# --------------------------------------------------------------------------------------------- lev-measure, itp, ac11 (T024-T025)
def test_the_pilot_is_003fs_seven_human_hazard_cohorts_and_itp():
    pc = ACC.pilot_cohorts()
    assert [x["cid"] for x in pc][-1] == ACC.ITP_TAG and len(pc) == 8
    assert all(x["cid"].startswith("hazard-real-") and len(x["sha256"]) == 64 for x in pc[:7])


def test_the_bound_tolerance_read_changes_the_free_set_and_restores_the_constant(monkeypatch):
    from sraging import leverage as LEV
    from sraging.calibrators import profile_jax2 as PJ
    n = 6
    G = np.arange(n * 4, dtype=float).reshape(n, 4) + 1.0
    monkeypatch.setattr(LEV, "terms_and_derivatives", lambda c, name, z, **k: {"G": G, "H": -np.eye(4)})
    box = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
    monkeypatch.setattr(PJ, "fit_inputs", lambda c, name, port: {"box": box})
    z = np.log([1e-3, 1e3, 0.02, 0.3]) - np.array([0.0, 5e-5, 0.0, 0.0])          # beta 5e-5 (log) below its bound
    out = ACC.bound_tol_sensitivity(None, "profile_jax2", z, PJ.PORTS["1.0.1"], "all4")
    free = {x["bound_tol"]: x["free"] for x in out}
    assert free[1e-6] == [True, True, True, True] and free[1e-4] == [True, False, True, True]
    assert LEV.BOUND_TOL == 1e-6
    monkeypatch.setattr(LEV, "influence", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("planted")))
    with pytest.raises(RuntimeError):
        ACC.bound_tol_sensitivity(None, "profile_jax2", z, PJ.PORTS["1.0.1"], "all4")
    assert LEV.BOUND_TOL == 1e-6                                                 # restored after an error too


def _itp_rows(shift=0.0, label="flag", row=1):
    th = {"eta": 3.58e-5, "beta": 0.0317, "epsilon": 0.00237, "kappa": 0.001}
    refit = [th[p] * (math.exp(shift) if p == "eta" else 1.0) for p in ("eta", "beta", "epsilon", "kappa")]
    with_ = {"leverage": {"label": label, "dropped": {"row": row}, "refits": {"mle_stage": {"theta": refit}},
                          "refit": {"start": "mle_stage"}}, "warnings": ["LEVERAGE"] if label == "flag" else []}
    return with_, {"theta_hat": th}


def test_the_itp_verdict_and_its_twins():
    v = ACC.itp_verdict(*_itp_rows(), age0_row=1)
    assert v["AC1"] and v["AC2"] and v["refit_vs_fit_without_max_log"] == 0.0
    assert not ACC.itp_verdict(*_itp_rows(shift=1e-6), age0_row=1)["AC2"]          # not equal to round-off
    assert not ACC.itp_verdict(*_itp_rows(row=7), age0_row=1)["AC1"]              # another death dropped
    assert not ACC.itp_verdict(*_itp_rows(label="no_flag"), age0_row=1)["AC2"]


REG = {"generator_version": "sraging-gen-1", "late_rule": {"factor_of_max_age": 1.5},
       "planted": [{"name": "pA", "seeds": {"100": 11, "300": 12},
                    "spec": {"rho_eta": 1.9606713963716613, "rho_beta": 0.1, "rho_eps": 0.03162277660168379,
                             "kappa": 0.01, "L_phys": 100.0, "delta": 1.0}}],          # v6 pure-k0.01-p02's point
       "rule": {"quantities": "all4"}, "ceilings": {"held_out": 0.3, "small_n": 0.3, "twins": 0.3}}


def test_the_planted_panel_has_a_twin_and_two_plants_per_spec_and_n():
    items = ACC.planted_cohorts(REG)
    assert [x["cid"] for x in items] == [f"planted-pA-n{n}-{v}" for n in (100, 300) for v in ("twin", "age0", "late")]


def test_a_planted_cohort_is_its_twin_plus_one_death():
    twin, r0 = ACC.planted_cohort(REG, "planted-pA-n100-twin")
    age0, r1 = ACC.planted_cohort(REG, "planted-pA-n100-age0")
    late, r2 = ACC.planted_cohort(REG, "planted-pA-n100-late")
    assert r0 is None and twin.n == 100 and r1 == r2 == 101 and age0.n == late.n == 101
    assert np.array_equal(np.asarray(age0.ages)[:100], np.asarray(twin.ages))     # the twin's arrays, unchanged
    assert age0.ages[-1] == 0.0 and age0.events[-1] == 1
    assert late.ages[-1] == round(max(twin.ages) * 1.5) and late.events[-1] == 1
    with pytest.raises(RuntimeError, match="generator"):
        ACC.planted_cohort(dict(REG, generator_version="sraging-gen-0"), "planted-pA-n100-twin")


def test_ac11_scores_flag_rates_per_null_and_detection_by_n():
    def row(mode, cid, label, planted_row=None, dropped=None):
        return {"cohort_id": cid, "mode": mode, "label": label, "labels": {"all4": label, "lead_pair": "no_flag",
                                                                           "raw": label},
                "planted_row": planted_row, "planted_dropped": planted_row is not None and dropped == planted_row}
    rows = ([row("held_out", f"h{i}", "flag" if i == 0 else "no_flag") for i in range(12)]
            + [row("small_n", f"s{i}", "no_flag") for i in range(25)]
            + [row("planted", "planted-pA-n100-twin", "no_flag"),
               row("planted", "planted-pA-n100-age0", "flag", 101, 101),
               row("planted", "planted-pA-n100-late", "flag", 101, 7)])
    s = ACC.ac11_score(rows, REG)
    h = s["nulls"]["held_out"]["per_set"]["all4"]
    assert (h["flags"], h["n"]) == (1, 12) and h["wilson"] == pytest.approx(ACC.wilson(1, 12))
    assert s["nulls"]["held_out"]["pass"] == (ACC.wilson(1, 12)[1] <= 0.3)
    assert s["nulls"]["small_n"]["per_set"]["all4"]["flags"] == 0 and s["nulls"]["twins"]["per_set"]["all4"]["n"] == 1
    assert s["detection"]["age0"]["100"]["detected"] == 1 and s["detection"]["late"]["100"]["detected"] == 0


def test_ac11_refuses_an_empty_or_stale_registration(monkeypatch):
    from sraging.leverage_rule import RULE
    monkeypatch.setattr(ACC, "panel", lambda: {"ac11": {}})
    with pytest.raises(SystemExit, match="lacks"):
        ACC.ac11_registration()
    full = dict(REG, held_out={}, small_n={}, rule=dict(RULE, ratio=9.0))
    monkeypatch.setattr(ACC, "panel", lambda: {"ac11": full})
    with pytest.raises(SystemExit, match="not sraging.leverage_rule.RULE"):
        ACC.ac11_registration()
    monkeypatch.setattr(ACC, "panel", lambda: {"ac11": dict(full, rule=dict(RULE))})
    assert ACC.ac11_registration()["rule"] == dict(RULE)                           # the twin


# --------------------------------------------------------------------------------------------- the registration (T029)
def test_the_committed_ac11_registration_is_the_design_naveh_chose():
    """12 held-out, all 47 small-n (Naveh: "All 47"), 6 points x 3 n x (twin, age-0, late) = 54 planted, ceilings
    0.25 / 0.15 ("Held-out 0.25, small-n 0.15"), the rule in force, the seeds by their stated formula."""
    from sraging.leverage_rule import RULE
    reg = ACC.ac11_registration()
    assert len(reg["held_out"]["sha256"]) == 12 and len(reg["small_n"]["sha256"]) == 47
    assert len(reg["planted"]) == 6 and len(ACC.planted_cohorts(reg)) == 54
    assert reg["ceilings"] == {"held_out": 0.25, "small_n": 0.15} and dict(reg["rule"]) == dict(RULE)
    assert all(p["spec"]["delta"] == 1.0 for p in reg["planted"])
    assert [p["seeds"] for p in reg["planted"]] == [{str(n): 2026102000 + 10 * i + j for j, n in
                                                     enumerate((100, 300, 1000))} for i in range(6)]
    assert not set(reg["held_out"]["sha256"]) & set(reg["small_n"]["sha256"])
    assert all(c.startswith("nlad-") and "-n150-" in c for c in reg["small_n"]["sha256"])


@pytest.mark.skipif(not HAVE_J2 or not os.path.isdir(os.environ.get("SRAGING_TESTER_V6", ACC.DROPBOX)),
                    reason="the tester's library v6 is absent on this host")
def test_the_registered_arrays_are_the_libraries():
    import hashlib
    import oracle_jax2_101 as OJ1
    reg = ACC.ac11_registration()
    for null in ("held_out", "small_n"):
        for cid, sha in reg[null]["sha256"].items():
            p = os.path.join(OJ1.LIBRARY_V6, "cohorts", f"{cid}.npz")
            assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha, cid
