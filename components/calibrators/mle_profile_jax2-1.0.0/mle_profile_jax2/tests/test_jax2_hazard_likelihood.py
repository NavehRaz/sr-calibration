"""L0 (B2): the hazard / window likelihood of `fp_jax.make_loglik`.

  * the hazard cores vs mpmath at 50 digits (PHASE_B_DESIGN.txt L0: S in [1e-300, 1-1e-14], h*delta in
    [1e-12, 10], relative error <= 1e-14);
  * with both switches OFF the lowered program is rc1's, text for text, and values / gradients are rc1's
    bit for bit (exact times, delta, censoring);
  * the per-observation terms (`with_terms`) sum to the value, with the hazard, the window and both;
  * the window with lo_eff = 0 conditions on S(0) = 1 (adds nothing);
  * the hazard at a tiny h nests the 4-parameter likelihood EXACTLY as the algebra says (the L1b identity);
  * against the SRK oracle `fp_loglik_physical(external_hazard=, time_range=, sampling_dt=)` on a small
    synthetic cohort (subprocess: SRK must stay out of the pytest process, see tests/conftest.py).
The full-size rungs (L1 24 cohorts x 6 points, L1b, L2) are scripts under reports/phaseB/scripts/.
"""
from __future__ import annotations

import math

import pytest

import _jax2_testlib as T

pytestmark = pytest.mark.sut


@pytest.fixture(scope="module")
def mods():
    T.jax_env()
    return T.pkg_module("fp_jax"), T.rc1_module("fp_jax")


def _cohort(censor=True, seed=0, n=300):
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.maximum(np.floor(rng.gamma(9.0, 11.0, n)), 0.5)
    e = np.ones(t.size, int)
    if censor:
        e[t >= 140] = 0
        t[t >= 140] = 140.0
    return t, e, float(np.mean(t))


KW = dict(N=200, dt=4e-3, tau_max=6.0, Gtol=1e-11)


# ---------------------------------------------------------------- the hazard cores vs mpmath

TINY = 2.2250738585072014e-308       # the smallest NORMAL double; XLA CPU flushes anything below to 0


def test_hazard_cores_vs_mpmath():
    """The two hazard brackets, and the floored log term, against 50-digit mpmath on the SAME double inputs,
    over the design's domain S in [1e-300, 1 - 1e-14], h*delta in [1e-12, 10] (x interval masses m from 0 to
    ~S).

    MEASURED (B2): XLA's CPU backend flushes SUBNORMAL doubles (< 2.2e-308) to zero, inputs and results
    (NumPy does not). So the 1e-14 relative bound is asserted where every operand and intermediate is 0 or
    normal; where one is subnormal the flush loses at most two subnormals, |error| <= 2 * TINY, which is
    asserted exactly (B2 deviation D-B2-5). Its largest effect on a log term is log(1 + 2 TINY / P_FLOOR)
    = 4.5e-4 nats, and only for terms within ~25 nats of the -700 floor; the floored term is asserted to
    1e-14 (relative to max(1, |term|)) wherever the exact core is >= 1e14 * TINY."""
    import mpmath
    import numpy as np
    fj = T.pkg_module("fp_jax")
    T.jax_env()
    import jax
    import jax.numpy as jnp
    mpmath.mp.dps = 50
    S_vals = np.concatenate([np.logspace(-300, -1, 60), 1.0 - np.logspace(-1, -14, 40)])
    hd_vals = np.logspace(-12, 1, 40)                 # h * delta in [1e-12, 10]
    ratios = np.array([0.0, 1e-300, 1e-200, 1e-30, 1e-16, 1e-8, 1e-3, 0.1, 1.0, 1e3, 1e10])
    S_hi, hd, r = np.meshgrid(S_vals, hd_vals, ratios, indexing="ij")
    S_hi, hd, r = S_hi.ravel(), hd.ravel(), r.ravel()
    m = r * S_hi * np.where(r > 1, 1e-12, 1.0)        # interval masses from 0 up to ~S_hi
    h = np.full(S_hi.shape, 0.37)
    dl = hd / h
    tt = np.full(S_hi.shape, 3.0)
    core = np.asarray(jax.jit(fj.hazard_interval_core)(jnp.asarray(m), jnp.asarray(S_hi), jnp.asarray(h),
                                                       jnp.asarray(dl)))
    term = np.asarray(jax.jit(lambda m_, s_, h_, d_, t_: jnp.maximum(
        jnp.log(jnp.clip(fj.hazard_interval_core(m_, s_, h_, d_), fj._P_FLOOR, None)) - h_ * t_, fj._LOG_FLOOR))(
        jnp.asarray(m), jnp.asarray(S_hi), jnp.asarray(h), jnp.asarray(dl), jnp.asarray(tt)))
    worst_core = worst_term = worst_flush_abs = 0.0
    n_rel = n_flush = n_term = 0
    for j in range(core.size):
        ex_q = -mpmath.expm1(-mpmath.mpf(h[j]) * mpmath.mpf(dl[j]))
        prod = mpmath.mpf(S_hi[j]) * ex_q
        ex = mpmath.mpf(m[j]) + prod
        normal = (m[j] == 0 or m[j] >= TINY) and (prod == 0 or prod >= TINY)
        err = abs(mpmath.mpf(core[j]) - ex)
        if normal and ex > 0:
            n_rel += 1
            worst_core = max(worst_core, float(err / ex))
        elif not normal:
            n_flush += 1
            worst_flush_abs = max(worst_flush_abs, float(err - 1e-14 * ex))
        if ex >= 1e14 * TINY:
            n_term += 1
            ext = max(mpmath.log(ex) - mpmath.mpf(h[j]) * mpmath.mpf(tt[j]), mpmath.mpf(fj._LOG_FLOOR))
            worst_term = max(worst_term, float(abs(mpmath.mpf(term[j]) - ext) / max(1, abs(ext))))
        elif ex < fj._P_FLOOR:
            assert term[j] == fj._LOG_FLOOR             # clipped: exactly the floor
    # the exact-time bracket p/L + h S (normal operands)
    p = np.logspace(-300, 2, 1200)
    Si = np.resize(S_vals, p.size)
    hh = np.resize(hd_vals, p.size)
    ce = np.asarray(jax.jit(fj.hazard_exact_core)(jnp.asarray(p), jnp.asarray(Si), jnp.asarray(hh)))
    worst_exact = 0.0
    for j in range(ce.size):
        ex = mpmath.mpf(p[j]) + mpmath.mpf(hh[j]) * mpmath.mpf(Si[j])
        worst_exact = max(worst_exact, float(abs(mpmath.mpf(ce[j]) - ex) / ex))
    print(f"\nMPMATH interval core: {n_rel} all-normal points, worst rel {worst_core:.3g}; {n_flush} with a "
          f"subnormal operand, worst |err| - 1e-14 ex = {worst_flush_abs:.3g} (bound 2 TINY = {2 * TINY:.3g}); "
          f"floored term on {n_term} points, worst rel {worst_term:.3g}; exact core worst rel {worst_exact:.3g}")
    assert n_rel > 0.9 * core.size
    assert worst_core <= 1e-14 and worst_term <= 1e-14 and worst_exact <= 1e-14
    assert worst_flush_abs <= 2 * TINY


# ---------------------------------------------------------------- the off path is rc1's

@pytest.mark.parametrize("sd", [None, 1.0])
@pytest.mark.parametrize("censor", [False, True])
def test_off_path_is_rc1_bit_for_bit(mods, sd, censor):
    import numpy as np
    fj, rc = mods
    t, e, L = _cohort(censor=censor)
    a = rc.make_loglik(t, e, L, sampling_dt=sd, **KW)
    b = fj.make_loglik(t, e, L, sampling_dt=sd, **KW)
    import jax
    z0 = np.log([2e-4, 0.03, 0.002, 0.5])
    for f_a, f_b in ((a[0], b[0]), (a[1], b[1])):
        ta = f_a.lower(z0).as_text(debug_info=False)
        tb = f_b.lower(z0).as_text(debug_info=False)
        assert ta == tb, "the hazard-off program is not rc1's"
    rng = np.random.default_rng(1)
    for _ in range(4):
        z = z0 + rng.uniform(-0.7, 0.7, 4)
        assert float(a[0](z)) == float(b[0](z))
        va, ga = a[1](z)
        vb, gb = b[1](z)
        assert float(va) == float(vb) and np.array_equal(np.asarray(ga), np.asarray(gb))
    del jax


# ---------------------------------------------------------------- terms, window, nesting

@pytest.mark.parametrize("sd", [None, 1.0, "array"])
@pytest.mark.parametrize("hazard,window", [(True, None), (False, (50.0, 140.0)), (True, (50.0, 140.0))])
def test_terms_sum_to_the_value(mods, sd, hazard, window):
    import numpy as np
    fj, _ = mods
    t, e, L = _cohort()
    if window is not None:
        keep = t >= window[0]
        t, e = t[keep], e[keep]
    sdv = np.where(np.arange(t.size) % 3 == 0, 2.0, 1.0) if sd == "array" else sd
    f, vg, terms = fj.make_loglik(t, e, L, sampling_dt=sdv, hazard=hazard, window=window, with_terms=True, **KW)
    z = np.log([2e-4, 0.03, 0.002, 0.5] + ([0.4 / L] if hazard else []))
    tr = terms(z)
    v = float(f(z))
    assert math.isfinite(v)
    assert abs(float(np.sum(np.asarray(tr["term"])) + float(tr["cond"])) - v) <= 1e-9 * abs(v)
    if window is None:
        assert float(tr["cond"]) == 0.0
    else:
        S_lo = float(tr["S_lo"])
        assert 0 < S_lo <= 1 and float(tr["cond"]) == pytest.approx(-t.size * math.log(S_lo), rel=1e-13)
    vv, g = vg(z)
    assert abs(float(vv) - v) <= 1e-9 * abs(v) and np.all(np.isfinite(np.asarray(g)))


def test_window_at_zero_adds_nothing(mods):
    import numpy as np
    fj, _ = mods
    t, e, L = _cohort()
    z = np.log([2e-4, 0.03, 0.002, 0.5, 0.4 / L])
    a = float(fj.make_loglik(t, e, L, sampling_dt=1.0, hazard=True, **KW)[0](z))
    b = float(fj.make_loglik(t, e, L, sampling_dt=1.0, hazard=True, window=(0.0, 140.0), **KW)[0](z))
    assert a == b


@pytest.mark.parametrize("sd", [None, 1.0, "array"])
@pytest.mark.parametrize("window", [None, (50.0, 140.0)])
def test_nesting_identity_at_the_h_floor(mods, sd, window):
    """l5(z, h) - l4(z) equals, term by term, -h sum(t) + n h lo_eff (window) + sum_deaths log1p(x_i) with
    x_i = S_hi (1 - e^{-h dl_i}) / m_i (interval) or h S_i / (p_i/L) (exact) -- computed here in NumPy from
    the 4-D likelihood's own terms. At h = h_lo = 1e-6 / L the death part is O(h L S/p) per death, so the
    design's "l5 - l4 + h sum(t) <= 1e-9 n" cannot hold literally (B2 deviation D-B2-4); the identity can."""
    import numpy as np
    fj, _ = mods
    t, e, L = _cohort()
    if window is not None:
        keep = t >= window[0]
        t, e = t[keep], e[keep]
    sdv = np.where(np.arange(t.size) % 3 == 0, 2.0, 1.0) if sd == "array" else sd
    z4 = np.log([2e-4, 0.03, 0.002, 0.5])
    _f4, _vg4, terms4 = fj.make_loglik(t, e, L, sampling_dt=sdv, window=window, with_terms=True, **KW)
    _f5, _vg5, terms5 = fj.make_loglik(t, e, L, sampling_dt=sdv, hazard=True, window=window, with_terms=True,
                                       **KW)
    for hL in (1e-6, 1e-3):
        h = hL / L
        z5 = np.append(z4, math.log(h))
        a, b = terms4(z4), terms5(z5)
        l4 = float(np.sum(np.asarray(a["term"])) + float(a["cond"]))
        l5 = float(np.sum(np.asarray(b["term"])) + float(b["cond"]))
        core4 = np.asarray(a["core"])
        died = e == 1
        if sdv is None:
            x = h * np.asarray(a["S_i"]) / core4
        else:
            dl = np.broadcast_to(np.asarray(sdv, float), t.shape)
            S_hi = np.asarray(b["S_hi"])
            x = S_hi * (-np.expm1(-h * dl)) / core4
        pred = -h * t.sum() + float(np.sum(np.log1p(x[died])))
        if window is not None:
            pred += t.size * h * window[0]
        assert abs((l5 - l4) - pred) <= 1e-9 * t.size, (hL, l5 - l4, pred)
        design_quantity = l5 - l4 + h * t.sum()
        if hL == 1e-6:
            assert design_quantity > 1e-9 * t.size       # the literal design bound fails, as predicted ...
            assert design_quantity >= 0                  # ... in the direction L5's nesting check allows


def test_oracle_parity_small(mods):
    """The hazard / window / per-observation-delta likelihood vs SRK's fp_loglik_physical on a small cohort,
    on the method's grid shape. Subprocess (SRK)."""
    out = T.run_py("""
import json, math, importlib.util, sys, os
import numpy as np
from srbench import npcompat
from SRK import fp_likelihood as FPL
s = importlib.util.spec_from_file_location("o.fj", os.path.join("methods", "mle_profile_jax2", "fp_jax.py"))
fj = importlib.util.module_from_spec(s); sys.modules["o.fj"] = fj; s.loader.exec_module(fj)
rng = np.random.default_rng(0)
t = np.maximum(np.floor(rng.gamma(9.0, 11.0, 300)), 0.5); e = np.ones(t.size, int)
e[t >= 140] = 0; t[t >= 140] = 140.0
L = float(FPL.kaplan_meier(t, e)[2])
kw = dict(N=300, dt=2e-3, tau_max=6.0, Gtol=1e-11)
res = []
for sd in (None, 1.0, "array"):
    for win in (None, (50.0, 140.0)):
        tt, ee = (t, e) if win is None else (t[t >= win[0]], e[t >= win[0]])
        sdv = np.where(np.arange(tt.size) % 3 == 0, 2.0, 1.0) if sd == "array" else sd
        for hL in (1e-6, 0.4, 5.0):
            z = np.log([2e-4, 0.03, 0.002, 0.5, hL / L]); th = np.exp(z)
            o = FPL.fp_loglik_physical(tt, ee, th[0], th[1], th[2], 1.0, th[3], L=L, external_hazard=-math.log(th[4]),
                                       sampling_dt=sdv, time_range=win, scheme="bdf2", **kw)
            v = float(fj.make_loglik(tt, ee, L, sampling_dt=sdv, hazard=True, window=win, **kw)[0](z))
            res.append(dict(sd=str(sd), win=win, hL=hL, o=o, v=v, d=v - o, n=int(tt.size)))
print(json.dumps(dict(res=res)))
""")
    assert out["_rc"] == 0, out["_stderr"]
    worst = max(abs(r["d"]) / r["n"] for r in out["res"])
    print(f"\nORACLE small: {len(out['res'])} evaluations, worst |diff|/n = {worst:.3g}")
    for r in out["res"]:
        assert abs(r["d"]) <= max(1e-4, 1e-8 * r["n"]), r


# ---------------------------------------------------------------- B5: deaths beyond the march's early stop

@pytest.mark.parametrize("sd", [None, 1.0])
def test_hazard_past_stop_deaths_score_the_floor(mods, sd):
    """B5 (1.0.0.dev3): with the hazard on, a death beyond the march's early stop scores exactly -700, as it does
    without the hazard (rc1's SR mass there is exactly 0); censored records there keep log S[k] - h t. So such
    deaths add a constant: removing them changes the value by exactly -700 each and leaves the gradient alone,
    at every h -- the discrete stop index k no longer reaches the value through them (the B2 sawtooth)."""
    import numpy as np
    fj, _ = mods
    t, e, L = _cohort(censor=True)
    far = np.array([520.0, 560.0, 590.0])          # tau ~ 5.2-5.9 (tau_max 6): beyond S < 1e-11 at this z
    t2 = np.concatenate([t, far, [580.0]])
    e2 = np.concatenate([e, [1, 1, 1], [0]])
    z4 = np.log([2e-4, 0.03, 0.002, 0.5])
    _f4, _v4, terms4 = fj.make_loglik(t2, e2, L, sampling_dt=sd, with_terms=True, **KW)
    _f5, vg5, terms5 = fj.make_loglik(t2, e2, L, sampling_dt=sd, hazard=True, with_terms=True, **KW)
    _f5b, vg5b, _t5b = fj.make_loglik(t2[:-4], e2[:-4], L, sampling_dt=sd, hazard=True, with_terms=True, **KW)
    a = terms4(z4)
    assert np.all(np.asarray(a["term"])[-4:-1] == fj._LOG_FLOOR)      # rc1: SR mass 0 -> floor
    for hL in (1e-6, 0.4, 5.0):
        z5 = np.append(z4, math.log(hL / L))
        b = terms5(z5)
        past = np.asarray(b["past"])
        term = np.asarray(b["term"])
        assert past[-4:-1].all() and not past[-1]                   # the 3 far deaths; never a censored record
        assert not past[:-4].any()
        assert np.all(term[past] == fj._LOG_FLOOR)
        S_k = float(np.asarray(b["S_i"])[-1])
        assert 0.0 <= S_k < KW["Gtol"]                                   # held S[k], below the stop threshold
        assert term[-1] == pytest.approx(max(math.log(max(S_k, fj._P_FLOOR)) - hL / L * 580.0, fj._LOG_FLOOR),
                                         rel=1e-12)
        v, g = vg5(z5)
        vb, gb = vg5b(z5)
        cens_far = float(term[-1])
        assert float(v) == pytest.approx(float(vb) + 3 * fj._LOG_FLOOR + cens_far, rel=1e-13, abs=1e-9)
        # the far deaths add nothing to the gradient; the far censored record adds d(log S[k] - h t)
        gb_c = np.asarray(gb) + np.asarray(jax_grad_of_term(fj, t2, e2, L, sd, z5, -1))
        assert np.allclose(np.asarray(g), gb_c, rtol=1e-10, atol=1e-8), (hL, np.asarray(g), gb_c)


def jax_grad_of_term(fj, t, e, L, sd, z, j):
    """d term_j / dz by forward-mode AD through the value march (independent of the custom VJP)."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    terms = fj.make_loglik(t, e, L, sampling_dt=sd, hazard=True, with_terms=True, **KW)[2]
    return np.array([float(jax.jvp(lambda zz: terms(zz)["term"][j], (jnp.asarray(z),),
                                   (jnp.asarray(np.eye(z.size)[k]),))[1]) for k in range(z.size)])
