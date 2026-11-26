"""Regression tests pinning the AS-IS behaviour of known v0.3.3 defects (T062; register D-001..D-021).

These defects are deliberately ported unchanged (spec 001 ports v0.3.3 verbatim). A test here failing
means the behaviour changed -- which must happen only through a divergence-register entry (VII).
"""
import inspect

import numpy as np

from sraging.compat.v033 import claims, estimator, fp_curvefit, fp_gof, fp_jax, fp_likelihood


def test_d001_seed_rho_cap():
    assert fp_curvefit.RHO_CEIL == 300.0
    assert inspect.signature(fp_curvefit._fit_one_arm).parameters["rho_ceil"].default == 300.0


def test_d004_interval_mass_is_a_difference():
    """The interval mass is computed as a clipped difference of survivals (cancels when S ~ 1)."""
    src = inspect.getsource(fp_jax.make_loglik)
    assert "p_phys = jnp.clip(S_i - S_hi, 0.0, None)" in src
    assert "jnp.maximum(jnp.log(jnp.clip(p_phys, _P_FLOOR, None)), _LOG_FLOOR)" in src


def test_d007_kaplan_meier_depends_on_row_order():
    """Appendix A B13: the same tied data in different row orders give different KM means."""
    t = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
    means = {round(float(fp_likelihood.kaplan_meier(t, np.array(e))[2]), 4)
             for e in ([1, 1, 0, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1])}
    assert means == {1.75, 1.8333, 1.8}


def test_d008_unknown_scheme_runs_backward_euler():
    kw = dict(N=120, dt=0.02, tau_max=4.0, Gtol=1e-9, return_survival=True)
    be = fp_likelihood.fpt_full(2.0, 1.0, 0.1, 1.0, scheme="be", **kw)
    other = fp_likelihood.fpt_full(2.0, 1.0, 0.1, 1.0, scheme="crank-nicolson", **kw)
    bdf2 = fp_likelihood.fpt_full(2.0, 1.0, 0.1, 1.0, scheme="bdf2", **kw)
    assert np.asarray(other[-1]).tobytes() == np.asarray(be[-1]).tobytes()
    assert np.asarray(bdf2[-1]).tobytes() != np.asarray(be[-1]).tobytes()


def test_d012_n_profile_is_inert():
    src = inspect.getsource(estimator.fit)
    assert "per_side = max(24, (int(n_profile) - 1) // 2 + 3)" in src
    assert max(24, (int(11) - 1) // 2 + 3) == max(24, (int(43) - 1) // 2 + 3) == 24


def test_d013_one_sided_mapping_follows_the_code():
    assert claims.VERDICT_TO_SHAPE["one-sided:<"] == "upper_bound"
    assert claims.VERDICT_TO_SHAPE["one-sided:>"] == "lower_bound"


def _res(ci, verdicts):
    return {"theta": [1e-3, 0.02, 2e-3, 1.0], "ci": ci, "loglik": -10.0, "dt_ceiling": None,
            "diagnostics": {"verdicts": verdicts, "profiles": {"eta": []}}}


def test_d014_n_components_is_always_one():
    cl, _ = claims.build(_res({f"{p}@0.95": (1e-4, 1e-2) for p in ("eta", "beta", "epsilon", "kappa")},
                              {p: "identified" for p in ("eta", "beta", "epsilon", "kappa")}))
    assert [c["n_components"] for c in cl] == [1, 1, 1, 1]


def test_d015_all_starts_failed_maps_to_four_none_claims():
    res = {"theta": [1e-3, 0.02, 2e-3, 1.0], "ci": {}, "loglik": float("nan"), "dt_ceiling": None,
           "diagnostics": {"error": "all starts failed"}}
    cl, diag = claims.build(res)
    assert [c["shape"] for c in cl] == ["none"] * 4 and diag["error"] == "all starts failed"


def test_d016_two_chi2_thresholds():
    from scipy.stats import chi2
    assert fp_gof._CHI2_1[0.95] == 3.8415
    assert float(chi2.ppf(0.95, 1)) != 3.8415
    assert "thr = float(chi2.ppf(max(levels), 1))" in inspect.getsource(estimator.fit)


def test_d017_sampling_dt_scalar_branch_only():
    src = inspect.getsource(fp_jax.make_loglik)
    assert "np.ndim(sampling_dt) == 0 and not (float(sampling_dt) > 0)" in src



def test_d018_closed_interval_ends_at_the_beta_box_edge():
    """The Drosophila cohort (first README example, now review evidence): theta-hat on the beta box edge, eta/epsilon 'closed' with no box note, and the
    likelihood still higher outside the box (stored runs; no new fit)."""
    import json
    import os
    res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "specs", "001-reference-fit", "review", "evidence", "drosophila")
    r = json.load(open(os.path.join(res_dir, "drosophila_simons2024_25C_2pct_batch1.json")))
    ridge = json.load(open(os.path.join(res_dir, "drosophila_simons2024_25C_2pct_batch1.ridge.json")))
    assert np.isclose(r["theta_hat"]["beta"], estimator.box_for(300.0)["beta"][1])
    by = {c["parameter"]: c for c in r["claims"]}
    for p in ("eta", "epsilon"):
        assert by[p]["shape"] == "closed"
        assert by[p]["label"] == "two-sided 95% profile-likelihood interval"
    assert r["warnings"] == []                    # stored before BOX_EDGE existed: four checks, all quiet
    assert ridge["max_gain_beyond_box"] > 0
    # the code: BOX_EDGE (FR-017b) now flags exactly this fit
    from sraging.calibrators import profile_v033
    from sraging.cohort import Cohort
    from sraging.result import Claim
    from sraging.warnings import evaluate
    c = Cohort.from_csv(os.path.join(res_dir, "drosophila_simons2024_25C_2pct_batch1.csv"), delta=2.0)
    claims = tuple(Claim(**{k: v for k, v in x.items() if k != "label"}, label=x["label"]) for x in r["claims"])
    ws, checks = evaluate(c, claims, r["diagnostics"], theta_hat=r["theta_hat"], box=profile_v033.SETTINGS["box"])
    edge = [w for w in ws if w.code == "BOX_EDGE"]
    assert len(edge) == 1 and edge[0].data["parameters"] == ["beta"]


def test_d019_named_cohorts_differ_between_hosts():
    """Stored measurement (the oracle cannot be run on WEXAC from here): the named cohorts' stored reference
    records differ from this host's oracle run; the quick tier matches (claims.json, adapter phase)."""
    import json
    import os
    m = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                    "specs", "001-reference-fit", "review", "measurements.json")))["host"]
    assert m["n"] == 10 and m["n_differ"] == 10 and m["n_shape_flips"] == 2
    assert m["max_abs_dex"] > 1.0


def test_d020_ridge_keeps_eta_over_epsilon():
    """Stored measurement of a MODEL property (the Drosophila ridge): over beta = 30..30000 the log-likelihood
    changes by < 0.5 nats while eta/epsilon stays within 15 %."""
    import json
    import os
    ev = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                      "specs", "001-reference-fit", "review", "evidence", "drosophila")
    ridge = json.load(open(os.path.join(ev, "drosophila_simons2024_25C_2pct_batch1.ridge.json")))
    pts = [p for p in ridge["points"] if 30.0 <= p["beta"] <= 30000.0]
    ratio = [p["eta"] / p["epsilon"] for p in pts]
    assert abs(ridge["span_30_to_30000"]) < 0.5 and max(ratio) / min(ratio) < 1.15


def test_d021_loglik_moves_at_the_ulp_scale():
    """Runs the code: on the small test cohort, a one-ulp move of one coordinate of log(theta) changes the value
    function by much more than its slope predicts (noise, not the smooth change), and far less than 1.92 nats."""
    import os
    from sraging.cohort import Cohort
    c = Cohort.from_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "small_cohort.csv"),
                        delta=1.0)
    t, e = np.asarray(c.ages, float), np.asarray(c.events, int)
    L, kw, _ = estimator.solver_grid(t, e, 1.0)
    f, _ = fp_jax.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                              sampling_dt=kw.get("sampling_dt"))
    _, vg = fp_jax.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                               sampling_dt=kw.get("sampling_dt"))
    z = np.log([1e-3, 0.05, 0.01, 1.0])
    f0, g = float(f(z)), np.asarray(vg(z)[1])
    ratios, moves = [], []
    for i in range(3):                            # eta, beta, epsilon (kappa's slope is ~0 here)
        z1 = z.copy()
        z1[i] = np.nextafter(z1[i], np.inf)
        moves.append(abs(float(f(z1)) - f0))
        ratios.append(moves[-1] / abs(g[i] * (z1[i] - z[i])))
    assert max(ratios) > 10 and max(moves) < 1e-3


# ---- spec 003 (T010): register entries D-201..D-215, jax2 1.0.0 against v0.3.3 and against the engine -------
import math  # noqa: E402
import types  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from sraging.compat.jax2 import claims as j2_claims  # noqa: E402
from sraging.compat.jax2 import estimator as j2_estimator  # noqa: E402
from sraging.compat.jax2 import fp_jax as j2_fp_jax  # noqa: E402
from sraging.compat.jax2 import observation_model as j2_observation_model  # noqa: E402
from sraging.compat.v033 import estimator as v033_estimator  # noqa: E402
from sraging.engine import grid as engine_grid  # noqa: E402
from sraging.engine import observation as engine_observation  # noqa: E402

def test_d201_death_mass_from_absorbed_mass():
    """D-201: the interval death's mass is F(t+delta) - F(t) where F(t+delta) < S(t); the S difference would be 0."""
    tau = jnp.array([0.0, 1.0, 2.0])
    S = jnp.ones(3)                                    # S ~ 1: the survivals agree to the last digit
    F = jnp.array([0.0, 1e-20, 3e-20])
    m, _S_i, m_S, m_F, use_F = j2_fp_jax._death_mass(S, F, jnp.array(0.0), jnp.array(1.0), tau)
    assert bool(use_F) and float(m_S) == 0.0
    assert float(m) == float(m_F) == 1e-20
    src = inspect.getsource(j2_fp_jax._F_next)
    assert "h * jnp.sum(jnp.minimum(q, 0.0))" in src and "bflux_coef" in src      # the scheme's own mass balance
    assert "_F_next_g" in inspect.getsource(j2_fp_jax._march_forward_g)           # the graded mesh carries it too


def test_d202_exact_time_density_from_F():
    """D-202: exact-time density from F where F's larger operand is below S's; the S stencil gives round-off (here 0)."""
    nsteps, dt = 4, 0.1
    S = jnp.ones(nsteps + 1)
    F_raw = jnp.array([0.0, 1e-20, 2e-20, 3e-20, 4e-20])
    p = jnp.zeros(nsteps + 1)                          # -gradient(S) on a flat S
    p_new, use_F = j2_fp_jax._density_F(F_raw, S, p, jnp.array(nsteps), dt, nsteps)
    assert bool(use_F[1]) and abs(float(p_new[1]) - 1e-19) < 1e-30
    assert float(p[1]) == 0.0
    assert "exact_from_F" in inspect.signature(j2_estimator.fit).parameters
    assert "exact_from_F" in inspect.getsource(j2_estimator.fit)                  # fit hands it to make_loglik
    assert "assumes the uniform grid" in inspect.getsource(j2_fp_jax.make_loglik)  # refused with a graded mesh


def test_d203_basin_refit_loop_constants_and_library_starts():
    """D-203: a re-fit loop (<= 3 rounds, thr/2 trigger, eta ladder, library starts) that v0.3.3's estimator does not have."""
    assert (j2_estimator._REFIT_MAX, j2_estimator._LIB_K) == (3, 3)
    assert (j2_estimator._LADDER_RUNGS, j2_estimator._LADDER_DEX, j2_estimator._REFIT_RESEED) == (3, 1.0, True)
    fit_src = inspect.getsource(j2_estimator.fit)
    assert "tol = 0.5 * thr" in fit_src and "_basin_refit(obj, pts_all, ll, bounds4, nm_opts, tol)" in fit_src
    assert "len(rounds) >= _REFIT_MAX" in fit_src
    assert not hasattr(v033_estimator, "_basin_refit") and not hasattr(v033_estimator, "_library_starts")
    # _library_starts: the K points nearest in the profiled parameter, none more than _DEV_STOP below the MLE's level
    lib = [(-100.0 - 0.1 * j, np.array([0.0, 0.01 * j, float(j), 0.0])) for j in range(6)]
    lib.append((-100.0 - 3 * j2_estimator._DEV_STOP, np.array([0.0, 9.0, 0.5, 0.0])))   # far below: excluded
    out = j2_estimator._library_starts(lib, 2, [0, 1, 3], math.exp(0.4), -100.0)
    assert len(out) == j2_estimator._LIB_K and out[0].tolist() == [0.0, 0.0, 0.0]
    assert all(z[1] != 9.0 for z in out)
    assert j2_estimator._library_starts([], 2, [0, 1, 3], 1.0, -100.0) == []


def test_d204_nelder_mead_fallback_identical_to_v033():
    """D-204: the zero-gradient plateau test and the z_alt / Nelder-Mead fallback are textually identical in v0.3.3."""
    a = inspect.getsource(j2_estimator._lbfgs)
    b = inspect.getsource(v033_estimator._lbfgs)
    assert 'method="Nelder-Mead"' in a and 'method="Nelder-Mead"' in b
    assert "plateau = (not np.isfinite(v0)) or (not np.any(g0))" in a
    assert "plateau = (not np.isfinite(v0)) or (not np.any(g0))" in b
    assert a[a.index("    if no_nm:"):] == b[b.index("    if no_nm:"):]


def test_d205_censor_tie_relabel_trigger():
    """D-205: relabels only when every censored record shares one age that is the maximum age; otherwise no-op."""
    relabel = j2_observation_model.censor_tie_relabel
    ages, ev = [1.0, 2.0, 3.0, 3.0, 3.0], [1, 1, 1, 0, 0]
    a, e, rec = relabel(ages, ev, 1.0)
    assert rec["applied"] and rec["n_relabelled"] == 1 and e == [1, 1, 0, 0, 0] and a == ages   # ages never move
    _, e, rec = relabel(ages, ev, 1.0, enabled=False)                                          # detector only
    assert rec["detected"] and not rec["applied"] and e == ev
    _, e, rec = relabel([1.0, 2.0, 3.0, 3.0, 4.0], [1, 1, 1, 0, 0], 1.0)                      # censored at two ages
    assert not rec["detected"] and e == [1, 1, 1, 0, 0] and "more than one age" in rec["reason"]
    _, e, rec = relabel([1.0, 2.0, 3.0, 3.0, 5.0], [1, 1, 1, 0, 1], 1.0)                      # a record beyond tc
    assert not rec["detected"] and e == [1, 1, 1, 0, 1] and "beyond the censoring age" in rec["reason"]
    _, e, rec = relabel(ages, ev, None)                                                        # no delta
    assert not rec["detected"] and e == ev and rec["reason"] == "no delta"


def test_d206_self_consistency_widening():
    """D-206: SC_EDGE_DEX = 0.01; a widened edge within 0.01 dex of the box edge opens the side; the point never moves."""
    assert j2_claims.SC_EDGE_DEX == 0.01
    kw = {"parameter": "beta", "shape": "closed", "lower": 1.0, "upper": 10.0, "point": 3.0}
    box = (1e-12, 1e3)
    new, rec = j2_claims.sc_widen_claim(kw, {"n": 1, "qmin": 0.5, "qmax": 999.0}, box)
    assert new["shape"] == "lower_bound" and "upper" not in new and new["lower"] == 0.5
    assert rec["opened"] == ["upper"] and new["point"] == 3.0
    new, _ = j2_claims.sc_widen_claim(kw, {"n": 1, "qmin": 2.0, "qmax": 900.0}, box)           # 900 is 0.046 dex below 1e3
    assert new["shape"] == "closed" and new["upper"] == 900.0 and new["lower"] == 1.0
    new, _ = j2_claims.sc_widen_claim(kw, {"n": 1, "qmin": 2.0, "qmax": 5.0}, box)             # inside: unchanged
    assert new == kw
    # the evidence is the memo of every evaluated point within thr/2 of the best (no likelihood call)
    src = inspect.getsource(j2_estimator.sc_extents)
    assert "LL >= ll_best - half" in src and "half = 0.5 * float(thr)" in src
    assert "_f_cache" in inspect.getsource(j2_estimator._sc_archive)


def test_d207_n_rule():
    """D-207: E900 = 10^(-3.034 + 0.428 log n - 0.900 log CV), N = 900 * 2^k, cap 3,600 (target 0.2 nats, q 0.6)."""
    assert (j2_estimator._NR_A, j2_estimator._NR_B, j2_estimator._NR_C) == (-3.034, 0.428, -0.900)
    assert (j2_estimator._NR_TARGET, j2_estimator._NR_Q) == (0.2, 0.6)
    assert (j2_estimator._NR_N0, j2_estimator._NR_N_MAX) == (900, 3600)
    assert j2_estimator._DTR_K == 80.0
    assert inspect.signature(j2_estimator._grid).parameters["dt_rule"].default is False       # DT_RULE stays off by default
    N, rec = j2_estimator._n_rule(100000, 0.3)
    e900 = 10.0 ** (-3.034 + 0.428 * math.log10(1e5) - 0.900 * math.log10(0.3))
    assert math.isclose(rec["e900_pred"], e900, rel_tol=1e-12)
    assert N == 3600 and not rec["capped"]                                                     # 0.377 -> 0.226 -> 0.136
    assert j2_estimator._n_rule(100, 0.5)[0] == 900                                            # already under the target
    N, rec = j2_estimator._n_rule(1000, 0.0)                                                   # degenerate CV takes the cap
    assert N == 3600 and rec["capped"] and rec["degenerate_cv"]
    assert "N = max(N, N_r)" in inspect.getsource(j2_estimator._grid)


def test_d208_graded_early_mesh():
    """D-208: 20 BE steps of 1e-8, then x1.05 steps up to dt, then the uniform tail with its count rounded DOWN."""
    assert j2_estimator._EARLY_ALL_MESH == {"dt0": 1e-8, "ratio": 1.05}
    assert j2_estimator._EARLY_NSTART == 20 and j2_estimator._FLOOR_EARLY == {"dt0": 1e-8, "ratio": 1.02}
    dt, tau_max = 3e-4, 20.0
    DT, TAU, AL, A, B = j2_fp_jax.early_schedule(dt, tau_max, 20, 1e-8, 1.05)
    assert np.all(DT[:20] == 1e-8) and np.all(AL[:20] == 1.0) and np.all(B[:20] == 0.0)        # the BE prefix
    assert math.isclose(DT[20], 1.05e-8, rel_tol=1e-12)                                        # then geometric growth
    n_u = int(np.sum(DT == dt))
    Tg = TAU[len(DT) - n_u]
    assert n_u == int((tau_max - Tg) / dt) and TAU[-1] <= tau_max                              # tail rounded down
    assert "early_all" in inspect.getsource(j2_estimator._grid)


def test_d209_floored_dt_ceiling():
    """D-209: where (delta/L)/3 < 3e-4 the ceiling is 3e-4 and N = 1800 (v0.3.3: dt = delta/L/3, N = 900)."""
    assert (j2_estimator._DT_FLOOR, j2_estimator._N_FLOOR) == (3e-4, 1800)
    src = inspect.getsource(j2_estimator._grid)
    assert "if ceiling < _DT_FLOOR" in src and "N = _N_FLOOR" in src
    t = np.arange(1.0, 1001.0)
    e = np.ones_like(t, dtype=int)
    delta = 0.01                                                                               # delta/L ~ 2e-5: the floor binds
    _L, kw, ceiling, floor = j2_estimator._grid(t, e, delta)
    assert floor["binds"] and ceiling == 3e-4 and kw["N"] == 1800 and kw["dt"] <= 3e-4
    _L, kw033, ceiling033 = v033_estimator.solver_grid(t, e, delta)
    assert kw033["N"] == 900 and ceiling033 < 3e-4 and kw033["dt"] == ceiling033


def test_d210_exception_policy():
    """D-210: swallowed solver exceptions are counted and re-raised (the port fixes strict True, reads no environment)."""
    assert j2_estimator._strict() is True
    assert "os.environ" not in inspect.getsource(j2_estimator._strict)
    assert j2_estimator._ALWAYS_RAISE == (IndexError, ValueError, TypeError)
    obj = types.SimpleNamespace(exc_counts={})
    ex = RuntimeError("solver")
    assert j2_estimator._note_exc(obj, ex) is True and obj.exc_counts == {"RuntimeError": 1}
    assert j2_estimator._note_exc(obj, ex) is True and obj.exc_counts == {"RuntimeError": 1}   # counted once per object
    assert "_note_exc" not in inspect.getsource(v033_estimator)                                # v0.3.3 swallows silently
    assert "pass" in inspect.getsource(v033_estimator._lbfgs)


def test_d211_external_hazard_box_and_triggers():
    """D-211: the h box (1e-6/L, 10/L) with absolute guards, h@lo is not a degenerate corner, z has 5 entries."""
    lo, hi = j2_estimator._h_box(2.0)
    assert (lo, hi) == (max(1e-6 / 2.0, 1e-12), min(10.0 / 2.0, 1e6)) == (5e-7, 5.0)
    assert j2_fp_jax.z_size(True) == 5 and j2_fp_jax.z_size(False) == 4
    assert j2_estimator.PARAMS_SR == ("eta", "beta", "epsilon", "kappa")
    obj = types.SimpleNamespace(params=("eta", "beta", "epsilon", "kappa", "h"), L=1.0)
    z = np.zeros(5)
    z[4] = -5.0
    assert j2_estimator._triggers(obj, z, [4], [(-5.0, 1.0)]) == ()                            # h at its floor: not a corner
    z[0] = -5.0
    assert j2_estimator._triggers(obj, z, [0], [(-5.0, 1.0)]) == ("eta@lo",)                   # an SR parameter is
    assert "!= \"h\"" in inspect.getsource(j2_estimator._triggers)
    assert "def _triggers" in inspect.getsource(v033_estimator) and "!= \"h\"" not in inspect.getsource(v033_estimator._triggers)


def test_d212_hazard_bracket_on_both_routes():
    """D-212: jax2 applies m + S_hi (-expm1(-h dl)) on whichever route _death_mass chose and e^{-ht} in log space."""
    core = j2_fp_jax.hazard_interval_core(jnp.array(0.0), jnp.array(1.0), 1e-20, 1.0)
    assert float(core) == 1e-20                                                                # expm1: no cancellation
    src = inspect.getsource(j2_fp_jax.make_loglik)
    assert "m, S_i, _mS, _mF, _uF = _death_mass(S, F, teval, thi, tau)" in src                 # one mass, either route
    assert "hazard_interval_core(m, S_hi, hz, dl)" in src
    assert "jnp.log(jnp.clip(core, _P_FLOOR, None)) - ht" in src                               # e^{-ht} in log space
    eng = inspect.getsource(engine_observation)
    assert "register D-004: the F route, only where use_F holds" in eng                        # the engine's bracket: F route only
    assert "p_phys = jnp.where(use_F, p_F, p_phys)" in eng


def test_d213_hazard_death_past_the_stop_is_floored():
    """D-213: with the hazard a death at or beyond the march's early stop scores the -700 floor (B5); the engine keeps D-115."""
    src = inspect.getsource(j2_fp_jax.make_loglik)
    assert "past = died & (teval >= tau[k])" in src and "past = died & (teval > tau[k])" in src
    assert "jnp.where(past, 0.0, hazard_interval_core(m, S_hi, hz, dl))" in src
    assert "jnp.where(past, 0.0, hazard_exact_core(p_phys, S_i, hz))" in src
    assert j2_fp_jax._LOG_FLOOR == -700.0
    assert "beyond_march" in inspect.getsource(engine_observation)                             # the engine's own bookkeeping


def test_d214_mesh_and_window_floor_vs_engine():
    """D-214: jax2's graded mesh has a 20-step BE prefix and a tail rounded down; the engine's has neither; window floor 1e-12."""
    dt, tau_max = 3e-4, 20.0
    DT, TAU, *_ = j2_fp_jax.early_schedule(dt, tau_max, 20, 1e-8, 1.05)
    nodes = np.asarray(engine_grid.graded_start(1e-8, 1.05, dt, tau_max))
    assert int(np.sum(DT == 1e-8)) == 20 and int(np.sum(np.diff(nodes) == 1e-8)) == 1          # BE prefix of 20 vs one step
    assert TAU[-1] <= tau_max <= nodes[-1]                                                     # rounded down vs rounded up
    assert "jnp.log(jnp.clip(S_lo, 1e-12, 1.0))" in inspect.getsource(j2_fp_jax.make_loglik)
    # spec 006 (Naveh's Db): the engine clips S(t_lo) at jax2's 1e-12 in both numerics (D-108 retired); the mesh
    # difference stands
    assert engine_observation.S_LO_CLIP == 1e-12
    assert "jnp.maximum(S_lo_w, S_LO_CLIP)" in inspect.getsource(engine_observation)
    assert "register D-108: the common log+ floor only" not in inspect.getsource(engine_observation)


def test_d215_raw_box_depends_on_time_unit():
    """D-215: the eta, beta, epsilon box is [1e-12, 1e3] in raw time units, identical in v0.3.3; rho_beta's box scales with L."""
    box = j2_estimator.box_for(300.0)
    assert box == v033_estimator.box_for(300.0)
    for p in ("eta", "beta", "epsilon"):
        assert box[p] == (1e-12, 1e3)
    assert list(inspect.signature(j2_estimator.box_for).parameters) == ["kappa_max"]          # no time-unit argument
    L_years, L_days = 80.0, 80.0 * 365.0
    assert math.isclose((box["beta"][1] * L_days) / (box["beta"][1] * L_years), 365.0)         # rho_beta's upper edge moves 365x
