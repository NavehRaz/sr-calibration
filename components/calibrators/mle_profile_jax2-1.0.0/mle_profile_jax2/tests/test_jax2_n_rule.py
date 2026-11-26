"""rc2 I2: the N rule for large n (`estimator._n_rule`, `_grid(n_rule=True)`) and its optional companion, the
CV-aware dt cap (`_grid(dt_rule=True)`).

  * `_n_rule` as a pure function: the ladder 900 * 2^k, the cap, monotone in n and in 1/CV, the measured
    anchors of reports/rc2/I2, degenerate CV -> the cap, the record;
  * `_km_cv` against an independent computation (uncensored: the trapezoid-on-KM-steps CV equals the sample CV
    up to the trapezoid's half-step; censored: KM restricted moments);
  * the grid on EVERY library cohort (subprocess, SRK): off is dev3's grid exactly; on changes N only
    (L, dt, tau_max, Gtol, sampling_dt, the ceiling and the floor's binds/dt unchanged), N = max(N_off, rule),
    so the floor composes (floor-bound cohorts keep dt = 3e-4 and N >= 1800); dt_rule changes dt only,
    dt = min(dt_off, CV / K); the census of N values;
  * the adapter: N_RULE on and DT_RULE off for both names (dev4 defaults), recorded in diagnostics.switches,
    and run_fit's defaults call the estimator exactly as dev3 did (so the rc1 identity harness is unaffected).
"""
from __future__ import annotations

import json
import math
import os

import pytest

import _jax2_testlib as T


@pytest.fixture(scope="module")
def est():
    return T.pkg_module("estimator")


LADDER = (900, 1800, 3600)


# ------------------------------------------------------------------------------------------------ _n_rule

def test_rule_is_on_the_ladder_capped_and_monotone(est):
    ns = [100, 300, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 1000000]
    cvs = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
    for n in ns:
        prev = None
        for cv in sorted(cvs, reverse=True):              # sharper and sharper: N never decreases
            N, rec = est._n_rule(n, cv)
            assert N in LADDER and N <= est._NR_N_MAX
            assert rec["N_rule"] == N and rec["n"] == n and rec["cv"] == cv
            assert prev is None or N >= prev
            prev = N
    for cv in cvs:
        prev = None
        for n in ns:                                      # larger and larger: N never decreases
            N, _ = est._n_rule(n, cv)
            assert prev is None or N >= prev
            prev = N


def test_rule_meets_its_target_unless_capped(est):
    for n in (500, 1000, 3000, 10000, 30000, 100000):
        for cv in (0.02, 0.05, 0.1, 0.2, 0.4):
            N, rec = est._n_rule(n, cv)
            k = int(round(math.log2(N / 900)))
            e = rec["e900_pred"] * est._NR_Q ** k
            assert abs(e - rec["e_pred"]) <= 1e-12 * max(1.0, e)
            if N < est._NR_N_MAX:
                assert e <= est._NR_TARGET                       # met, and the next lower rung would not be:
                assert k == 0 or rec["e900_pred"] * est._NR_Q ** (k - 1) > est._NR_TARGET
            assert rec["capped"] == (rec["e900_pred"] * est._NR_Q ** k > est._NR_TARGET)


def test_rule_anchors_from_the_measurement(est):
    """Cohorts of reports/rc2/I2 (n, the rule's KM CV) -> the N the fitted model asks for."""
    assert est._n_rule(1000, 0.40)[0] == 900          # broad, small: production grid
    assert est._n_rule(2000, 0.20)[0] == 900
    assert est._n_rule(10000, 0.15)[0] == 1800        # measured E900 0.24-0.33 on p03 / censor p03
    assert est._n_rule(10000, 0.044)[0] == 3600       # pure-k0.01-p00-c1: measured E900 1.75 (capped)
    assert est._n_rule(25000, 0.25)[0] == 1800        # predicted E900 0.245 -> 0.147 at 1800
    assert est._n_rule(25000, 0.17)[0] == 3600        # predicted E900 0.348 -> 0.209 at 1800 (> 0.2)
    assert est._n_rule(100000, 0.135)[0] == 3600      # human n = 1e5: wants 7200, capped at 3600


def test_rule_degenerate_cv_takes_the_cap_and_bad_n_raises(est):
    for cv in (0.0, -1.0, float("nan"), float("inf")):
        N, rec = est._n_rule(1000, cv)
        assert N == est._NR_N_MAX and rec["capped"] and rec["degenerate_cv"]
    with pytest.raises(ValueError):
        est._n_rule(0, 0.1)


def test_rule_constants_are_the_reported_ones(est):
    assert (est._NR_A, est._NR_B, est._NR_C) == (-3.034, 0.428, -0.900)
    assert est._NR_TARGET == 0.2 and est._NR_Q == 0.6 and est._NR_N_MAX == 3600
    assert est._DTR_K == 80.0
    assert est._NR_N_MAX in LADDER                       # the cap must be a rung, or N could overshoot it


# ------------------------------------------------------------------------------------------------ _km_cv

def _km_steps(t, e):
    """SRK.fp_likelihood.kaplan_meier, re-implemented (the test process must not import SRK)."""
    import numpy as np
    t = np.asarray(t, float)
    e = np.asarray(e, int)
    o = np.argsort(t)
    t, e = t[o], e[o]
    S, at_risk, tk, Sk = 1.0, t.size, [0.0], [1.0]
    for i in range(t.size):
        if e[i] == 1 and at_risk > 0:
            S *= (1.0 - 1.0 / at_risk)
        tk.append(t[i])
        Sk.append(S)
        at_risk -= 1
    tk, Sk = np.asarray(tk), np.asarray(Sk)
    return tk, Sk, float(np.sum(np.diff(tk) * (Sk[1:] + Sk[:-1]) / 2.0))


def test_km_cv_uncensored_is_the_sample_cv(est):
    import numpy as np
    rng = np.random.default_rng(3)
    for cv_true in (0.03, 0.1, 0.3):
        t = rng.normal(100.0, 100.0 * cv_true, 20000)
        t = t[t > 0]
        tk, Sk, L = _km_steps(t, np.ones(t.size, int))
        cv = est._km_cv(tk, Sk, L)
        assert abs(cv / (t.std() / t.mean()) - 1) < 0.02, (cv, t.std() / t.mean())


def test_km_cv_censored_uses_the_km_restricted_moments(est):
    import numpy as np
    rng = np.random.default_rng(4)
    t = rng.gamma(20.0, 5.0, 5000)
    e = np.ones(t.size, int)
    e[t > 120] = 0
    t = np.minimum(t, 120.0)                              # administrative censoring at 120
    tk, Sk, L = _km_steps(t, e)
    m2 = float(np.sum(np.diff(tk) * ((2 * tk * Sk)[1:] + (2 * tk * Sk)[:-1]) / 2.0))
    assert est._km_cv(tk, Sk, L) == pytest.approx(math.sqrt(m2 - L * L) / L, rel=1e-12)


# ----------------------------------------------------------------------------------- the grid, every cohort

@pytest.mark.sut
def test_grid_rule_on_every_library_cohort():
    if not os.path.isdir(os.path.join(T.ROOT, "data", "library", "v2", "cohorts")):
        pytest.skip("library v2 cohorts are not built in this checkout")
    out = T.run_py("""
import sys, json, os, math, importlib.util, collections
import numpy as np
from srbench import npcompat
from SRK import fp_likelihood as FPL
s = importlib.util.spec_from_file_location("g.new", os.path.join("methods", "mle_profile_jax2", "estimator.py"))
est = importlib.util.module_from_spec(s); sys.modules["g.new"] = est; s.loader.exec_module(est)
from srbench.library.load import Library
lib = Library(version="v2")
n = bad = 0; census = collections.Counter(); floor_N = collections.Counter(); dt_capped = 0; problems = []
for sp in lib.specs:
    t, e = lib.data(sp.cohort_id); lib._cache.clear()
    t = np.asarray(t, float); e = np.asarray(e, int); d = sp.delta if sp.delta else None
    off = est._grid(t, e, d)
    off2 = est._grid(t, e, d, n_rule=False, dt_rule=False)
    on = est._grid(t, e, d, n_rule=True)
    dtr = est._grid(t, e, d, dt_rule=True)
    n += 1
    tk, Sk, L = FPL.kaplan_meier(t, e); cv = est._km_cv(tk, Sk, float(L))
    Nr, rec = est._n_rule(t.size, cv)
    ok = repr(off) == repr(off2)                                       # off: dev3's grid exactly
    ok &= "grid_rule" not in off[3] and "grid_rule" in on[3] and "grid_rule" in dtr[3]
    ok &= on[0] == off[0] and on[2] == off[2]                          # L, ceiling
    for k in ("dt", "tau_max", "Gtol", "scheme", "sampling_dt"):
        ok &= on[1].get(k) == off[1].get(k)
    ok &= on[1]["N"] == max(off[1]["N"], Nr) and on[3]["N"] == on[1]["N"]
    ok &= on[3]["binds"] == off[3]["binds"] and on[3]["dt"] == off[3]["dt"]
    ok &= on[3]["grid_rule"]["N_rule"] == Nr and on[3]["grid_rule"]["N_base"] == off[1]["N"]
    ok &= on[3]["grid_rule"]["cv"] == cv
    ok &= dtr[1]["N"] == off[1]["N"] and dtr[1]["dt"] == min(off[1]["dt"], cv / est._DTR_K)
    ok &= dtr[3]["nsteps"] == int(dtr[1]["tau_max"] / dtr[1]["dt"])
    if off[3]["binds"]:
        ok &= on[1]["dt"] == 3e-4 and on[1]["N"] >= 1800
        floor_N[on[1]["N"]] += 1
    dt_capped += dtr[1]["dt"] < off[1]["dt"]
    census[on[1]["N"]] += 1
    if not ok:
        bad += 1; problems.append(sp.cohort_id)
print(json.dumps(dict(n=n, bad=bad, problems=problems[:10], census={str(k): v for k, v in sorted(census.items())},
                      floor_N={str(k): v for k, v in sorted(floor_N.items())}, dt_capped=dt_capped)))
""", timeout=3600)
    assert out["_rc"] == 0, out["_stderr"]
    assert out["n"] == 2290 and out["bad"] == 0, out["problems"]
    assert sum(out["floor_N"].values()) == 48                          # the 48 floor-bound v1 cohorts
    assert set(out["census"]) <= {"900", "1800", "3600"}
    print(f"\nN rule census (2,290 v2 cohorts): {out['census']}; floor-bound: {out['floor_N']}; "
          f"dt_rule would cap dt on {out['dt_capped']}")


# ------------------------------------------------------------------------------------------------ adapter

def test_adapter_defaults_and_switch_record():
    """Both names: N_RULE on, DT_RULE off (dev4); run_fit's defaults pass neither to the estimator (dev3's call,
    so tests/identity_rc1.py still compares like with like); calibrate passes the class constants."""
    out = T.run_py("""
import json, os, importlib.util, inspect, sys
s = importlib.util.spec_from_file_location("jax2_tests.nrule.adapter", os.path.join("methods", "mle_profile_jax2", "__init__.py"))
ad = importlib.util.module_from_spec(s); sys.modules[s.name] = ad; s.loader.exec_module(ad)
sig = inspect.signature(ad.run_fit)
src_run = inspect.getsource(ad.run_fit); src_cal = inspect.getsource(ad._MleProfileJax2Base.calibrate)
src_cs = inspect.getsource(ad._MleProfileJax2Base.claimset)
print(json.dumps(dict(
    jax2=[ad.MleProfileJax2.N_RULE, ad.MleProfileJax2.DT_RULE],
    jax2_h=[ad.MleProfileJax2H.N_RULE, ad.MleProfileJax2H.DT_RULE],
    defaults=[sig.parameters["n_rule"].default, sig.parameters["dt_rule"].default],
    run_passes_only_when_on=('**({"n_rule": True} if n_rule else {})' in src_run
                             and '**({"dt_rule": True} if dt_rule else {})' in src_run),
    calibrate_passes=("n_rule=self.N_RULE" in src_cal and "dt_rule=self.DT_RULE" in src_cal),
    switches_record=("n_rule=bool(self.N_RULE)" in src_cs and "dt_rule=bool(self.DT_RULE)" in src_cs))))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["jax2"] == [True, False] and out["jax2_h"] == [True, False]
    assert out["defaults"] == [False, False]
    assert out["run_passes_only_when_on"] and out["calibrate_passes"] and out["switches_record"]


@pytest.mark.sut
def test_fit_threads_the_rule_to_the_grid():
    """estimator.fit(n_rule=True) solves on the rule's N and records it; off records nothing new (MLE stage only,
    small synthetic cohort, rule forced to 1800 by a patched `_n_rule`)."""
    out = T.run_py("""
import sys, json, os, importlib.util
import numpy as np
from srbench import npcompat
def load(k, p):
    s = importlib.util.spec_from_file_location(k, p); m = importlib.util.module_from_spec(s)
    sys.modules[k] = m; s.loader.exec_module(m); return m
P = os.path.join("methods", "mle_profile_jax2")
est = load("nr.est", os.path.join(P, "estimator.py")); fj = load("nr.fj", os.path.join(P, "fp_jax.py"))
rng = np.random.default_rng(11)
t = np.round(rng.gamma(30.0, 1.0 / 30.0, 300) * 100.0); t = np.maximum(t, 1.0); e = np.ones(t.size, int)
est._n_rule = lambda n, cv: (1800, dict(n=int(n), cv=float(cv), N_rule=1800, forced=True))
on = est.fit(fj, t, e, delta=1.0, stages=("mle",), n_rule=True)
off = est.fit(fj, t, e, delta=1.0, stages=("mle",))
print(json.dumps(dict(N_on=on["diagnostics"]["solver"]["N"], N_off=off["diagnostics"]["solver"]["N"],
                      rec_on=on["diagnostics"]["dt_floor"].get("grid_rule"),
                      rec_off="grid_rule" in off["diagnostics"]["dt_floor"],
                      dt_same=on["diagnostics"]["solver"]["dt"] == off["diagnostics"]["solver"]["dt"])))
""", timeout=1200)
    assert out["_rc"] == 0, out["_stderr"]
    assert out["N_on"] == 1800 and out["N_off"] == 900 and out["dt_same"]
    assert out["rec_on"]["N_rule"] == 1800 and out["rec_on"]["forced"] and out["rec_off"] is False
