"""1.0.1.dev2 (estimator note 14): kappa PINNED by a mask (KFIX), the removal-on starts, the sqrt(eta)/eps walk
coordinate, the beta*rbar read-out, `sc_extents(derived=)`, and the kfix names' adapter.

  * the mask: the pinned objective's value and gradient ARE the full objective's at z[3] = log k0 (bit for bit, the
    kappa entry dropped); `_check_z` refuses an unpinned z; `inside` skips the pin; nothing fixed -> `free_fns` hands
    back the objective's own callables and `zfull` / `pin` / `zfree` their input;
  * the coordinate: `full` / `pull` against finite differences (kappa free and pinned, with and without h); `bounds`
    keep eta in its box;
  * the starts: no kappa corners under KFIX; the removal-on starts' positions for J (fidx [1, 2]) and H ([1, 2, 4]);
    every _h MLE start, ladder rung and jax2-like point has z[3] == log k0;
  * `sc_extents`: derived=None is the 1.0.1 dict; derived extents == brute force;
  * the adapter: beta_rbar == the rescaled beta claim; claim shapes (kappa none, beta present, never
    beta_over_kp1); validate_claimset; the kfix classes' constants; run_fit's keywords only when set;
  * tiny real fits (J-kfix, H-kfix, the kappa-free twin path): theta[3] == k0 and every archived z[3] == log k0, the
    combination walked and widened, the combination claim containing every qualifying archived point;
  * 1.0.1.dev3 (the dev2 implementation review, reports/rel101/IMPL_REVIEW_LOG.md): calibrate passes kappa_fixed /
    combos to run_fit only when set; kfix_claim_kwargs refuses a foreign result; the "all starts failed" path keeps
    theta[3] == k0 exactly and diagnostics.kfix; the removal-on starts are reached at all three call sites in real
    fits (and generate starts there).
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

import _jax2_testlib as T

THR = 3.841458820694124            # chi2.ppf(0.95, 1)
HALF = THR / 2
K0S = (1e-3, 0.5, 1.0)
KFIX_NAMES = ("mle_profile_jax2_kfix_k0p001", "mle_profile_jax2_kfix_k0p5", "mle_profile_jax2_kfix_k1",
              "mle_profile_jax2_h_kfix_k0p001", "mle_profile_jax2_h_kfix_k0p5", "mle_profile_jax2_h_kfix_k1")


@pytest.fixture(scope="module")
def est():
    T.jax_env()
    return T.pkg_module("estimator")


@pytest.fixture(scope="module")
def fj():
    T.jax_env()
    return T.pkg_module("fp_jax")


@pytest.fixture(scope="module")
def ad():
    """The adapter under a TEST key: it registers nothing there."""
    m = T.load("jax2_tests.kfix.adapter", os.path.join(T.PKG, "__init__.py"))
    assert m.__name__ != m._KEY
    return m


_T = (0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7)
_E = (1, 1, 1, 1, 1, 1, 1, 0)
_KW = dict(N=40, dt=0.02, tau_max=3.0, Gtol=1e-11, sampling_dt=0.1)


def _obj(est, fj, hazard=False, kappa_fixed=None):
    """test_jax2_l0_zshape's tiny cohort on a coarse grid (compiles in seconds), optionally with h and a pin."""
    import numpy as np
    box = est.box_for(300.0)
    if hazard:
        box = dict(box, h=est._h_box(1.1))
    return est._Objective(fj, np.array(_T), np.array(_E), 1.1, dict(_KW), box, hazard=hazard,
                          window=(0.5, 1.7) if hazard else None,
                          **({"kappa_fixed": kappa_fixed} if kappa_fixed is not None else {}))


@pytest.fixture(scope="module")
def objs(est, fj):
    """{(hazard, k0 or None): objective}: J and H, free and pinned at every registered kappa0."""
    return {(h, k): _obj(est, fj, hazard=h, kappa_fixed=k) for h in (False, True) for k in (None,) + K0S}


def _bounds(o):
    return [(math.log(o.box[p][0]), math.log(o.box[p][1])) for p in o.params]


# ----------------------------------------------------------------------------------------------- the mask

@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
@pytest.mark.parametrize("k0", K0S)
def test_pinned_objective_is_the_full_objective_at_log_k0(objs, hazard, k0):
    import numpy as np
    full, pin = objs[(hazard, None)], objs[(hazard, k0)]
    NP = 5 if hazard else 4
    assert pin.NP == full.NP == NP and pin.params == full.params
    assert pin.fixed == {3: math.log(k0)} and pin.free == [k for k in range(NP) if k != 3]
    assert pin.kappa_fixed == k0
    fg, fv, to_full, bfree = pin.free_fns(_bounds(pin))
    assert bfree == [_bounds(pin)[k] for k in pin.free]
    rng = np.random.default_rng(11)
    z_ref = np.log([1.0, 2.0, 0.3, 0.5] + ([0.05] if hazard else []))
    for _ in range(4):
        zf = z_ref[pin.free] + rng.normal(0, 0.2, len(pin.free))
        z = to_full(zf)
        assert z[3] == math.log(k0) and np.array_equal(z[pin.free], zf)
        v1, g1 = fg(zf)
        v2, g2 = full.nll_grad(z)
        assert v1 == v2 and np.asarray(g1).tobytes() == np.asarray(g2)[pin.free].tobytes()
        assert fv(zf) == full.nll(z)
    # the assembly helpers write the pinned float
    zz = pin.zfull(0, -1.0, [k for k in pin.free if k != 0], z_ref[[k for k in pin.free if k != 0]])
    assert zz[3] == math.log(k0) and zz[0] == -1.0
    zp = pin.pin(z_ref)
    assert zp is not z_ref and zp[3] == math.log(k0) and np.array_equal(np.delete(zp, 3), np.delete(z_ref, 3))
    assert np.array_equal(pin.zfree(z_ref), z_ref[pin.free])


@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
def test_check_z_refuses_an_unpinned_z(objs, hazard):
    import numpy as np
    pin = objs[(hazard, 1e-3)]
    z = pin.pin(np.log([1.0, 2.0, 0.3, 0.5] + ([0.05] if hazard else [])))
    pin.nll_grad(z)
    pin.nll(z)
    for bad in (np.nextafter(z[3], 0.0), np.nextafter(z[3], -1.0), float(np.log(np.exp(z[3]))) + 1e-12, 0.0):
        z2 = z.copy()
        z2[3] = bad
        with pytest.raises(ValueError, match="pinned"):
            pin.nll_grad(z2)
        with pytest.raises(ValueError, match="pinned"):
            pin.nll(z2)
    free = objs[(hazard, None)]
    z3 = z.copy()
    z3[3] = 0.0
    free.nll_grad(z3)                                   # nothing pinned: any kappa evaluates


@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
def test_inside_skips_the_pinned_coordinate_only(objs, hazard):
    pin, free = objs[(hazard, 1e-3)], objs[(hazard, None)]
    th = [1.0, 2.0, 0.3, 1e-3 * (1 - 1e-15)] + ([0.05] if hazard else [])
    assert free.inside(th) is False and pin.inside(th) is True       # exp(log(1e-3)) may sit 1 ulp off the floor
    th_out = [1e4, 2.0, 0.3, 1e-3] + ([0.05] if hazard else [])
    assert pin.inside(th_out) is False                               # the free coordinates are still checked
    if hazard:
        h_lo = pin.box["h"][0]
        assert pin.inside([1.0, 2.0, 0.3, 1e-3, h_lo * (1 - 1e-13)]) is True     # h's floor tolerance kept
        assert pin.inside([1.0, 2.0, 0.3, 1e-3, h_lo * (1 - 1e-9)]) is False


@pytest.mark.sut
def test_kappa_fixed_must_be_inside_the_box(est, fj):
    for bad in (1e-4, 301.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="kappa_fixed"):
            _obj(est, fj, kappa_fixed=bad)


@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
def test_nothing_fixed_hands_back_the_originals(objs, hazard):
    import numpy as np
    o = objs[(hazard, None)]
    assert o.fixed == {} and o.free == list(range(o.NP)) and o.kappa_fixed is None
    assert not hasattr(o, "n_removal_on")                           # no kfix state when off
    b = _bounds(o)
    fg, fv, to_full, bb = o.free_fns(b)
    assert fg == o.nll_grad and fg.__func__ is type(o).nll_grad and fg.__self__ is o
    assert fv == o.nll and fv.__func__ is type(o).nll and fv.__self__ is o
    assert bb is b
    z = np.log([1.0, 2.0, 0.3, 0.5] + ([0.05] if hazard else []))
    assert to_full(z) is z and o.zfree(z) is z and o.pin(z) is z
    # zfull is the old assembly, byte for byte
    fidx = [1, 2, 3] + ([4] if hazard else [])
    old = np.empty(o.NP)
    old[0] = -2.5
    old[fidx] = z[fidx]
    assert o.zfull(0, -2.5, fidx, z[fidx]).tobytes() == old.tobytes()


# ----------------------------------------------------------------------------------------------- the coordinate

@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
@pytest.mark.parametrize("k0", [None, 1e-3, 1.0])
def test_coordinate_full_and_pull_against_finite_differences(est, objs, hazard, k0):
    import numpy as np
    o = objs[(hazard, k0)]
    c = est._SqrtEtaOverEps(o)
    assert c.free == [k for k in o.free if k != 0]
    z_ref = np.log([1.0, 2.0, 0.3, 0.5] + ([0.05] if hazard else []))
    zi = 0.5 * z_ref[0] - z_ref[2] + 0.1                             # log c, a little off the reference point
    zf = z_ref[c.free]
    z = c.full(zi, zf)
    assert z[0] == 2.0 * (zi + z[2]) and np.array_equal(z[c.free], zf)
    assert abs(c.zval(z) - zi) < 1e-14
    if k0 is not None:
        assert z[3] == math.log(k0)
    v, g = o.nll_grad(z)
    gp = c.pull(g)
    assert gp.shape == (len(c.free),)
    h = 1e-5
    for j in range(len(c.free)):
        e = np.zeros(len(c.free))
        e[j] = h
        fd = (o.nll_grad(c.full(zi, zf + e))[0] - o.nll_grad(c.full(zi, zf - e))[0]) / (2 * h)
        assert gp[j] == pytest.approx(fd, rel=1e-5, abs=1e-6), (j, c.free[j], gp[j], fd)


@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
def test_coordinate_bounds_keep_eta_in_its_box(est, objs, hazard):
    import numpy as np
    o = objs[(hazard, 0.5)]
    c = est._SqrtEtaOverEps(o)
    (elo, ehi), (plo, phi) = o.box["eta"], o.box["epsilon"]
    assert c.box == (math.sqrt(elo) / phi, math.sqrt(ehi) / plo)
    rng = np.random.default_rng(5)
    lo_c, hi_c = math.log(c.box[0]), math.log(c.box[1])
    n_ok = 0
    for zi in list(rng.uniform(lo_c, hi_c, 300)) + [lo_c + 1e-9, hi_c - 1e-9]:
        b = c.bounds(zi)
        assert b is not None, zi
        n_ok += 1
        assert len(b) == len(c.free)
        for (lo, hi), k in zip(b, c.free):
            assert math.log(o.box[o.params[k]][0]) <= lo < hi <= math.log(o.box[o.params[k]][1])
        lo_e, hi_e = b[c.free.index(2)]
        for ze in (lo_e, hi_e, 0.5 * (lo_e + hi_e)):
            eta = math.exp(2.0 * (zi + ze))
            assert elo * (1 - 1e-12) <= eta <= ehi * (1 + 1e-12), (zi, ze, eta)
    for zi in (lo_c - 0.01, hi_c + 0.01, lo_c - 10, hi_c + 10):          # beyond c's feasible range: empty box
        assert c.bounds(zi) is None
    assert n_ok == 302


@pytest.mark.sut
@pytest.mark.parametrize("hazard", [False, True])
def test_box_status_skips_the_pinned_coordinate(est, objs, hazard):
    """BOX_FLAG (off in every name, kept dormant): kappa0 = 1e-3 IS the kappa floor by choice, never a box limit."""
    import numpy as np
    pin, free = objs[(hazard, 1e-3)], objs[(hazard, None)]
    z = pin.pin(np.log([1e-3, 1e3, 1e-2, 0.5] + ([0.05] if hazard else [])))
    s_free, s_pin = est._box_status(free, z), est._box_status(pin, z)
    assert s_free["on"] == ["beta@hi", "kappa@lo"] and "kappa" in s_free["near"]
    assert s_pin["on"] == ["beta@hi"] and "kappa" not in s_pin["near"]
    assert {k: v for k, v in s_free["near"].items() if k != "kappa"} == s_pin["near"]


@pytest.mark.sut
def test_library_starts_for_a_combination_are_nearest_in_the_combination(est, objs):
    """After a re-fit the polish also starts from library points; for a combination node they are the points nearest
    in log c = 0.5 log eta - log eps (coord.zval), not in log eta; each is the point's coord.free nuisance."""
    import numpy as np
    o = objs[(False, None)]
    c = est._SqrtEtaOverEps(o)
    lib = []
    for j, (le, lp) in enumerate([(-7.0, -4.0), (-5.0, -4.0), (-7.0, -3.0), (-9.0, -5.0), (-3.0, -1.0)]):
        lib.append((-1.0 - 0.01 * j, np.array([le, 0.1 * j, lp, 0.2])))
    zi = 0.5 * -7.0 - (-4.0)                                                     # log c of the first point: 0.5
    st = est._library_starts(lib, None, c.free, math.exp(zi), -1.0, k=2, coord=c)
    zc = [c.zval(q[1]) for q in lib]                                             # 0.5, 1.5, -0.5, 0.5, -0.5
    assert [np.asarray(s).tolist() for s in st] == [lib[0][1][c.free].tolist(), lib[3][1][c.free].tolist()]
    assert zc[0] == zc[3] == 0.5                                                 # tie: the higher loglik first
    raw = est._library_starts(lib, 0, [1, 2, 3], math.exp(-7.0), -1.0, k=2)     # the raw path: nearest in log eta
    assert [np.asarray(s).tolist() for s in raw] == [lib[0][1][[1, 2, 3]].tolist(), lib[2][1][[1, 2, 3]].tolist()]


# ----------------------------------------------------------------------------------------------- the starts

@pytest.mark.sut
def test_no_kappa_corners_under_kfix_and_the_removal_on_positions(est, objs):
    import numpy as np
    for hazard, fidx_eta, fidx_eps in ((False, [1, 2], [0, 1]), (True, [1, 2, 4], [0, 1, 4])):
        o = objs[(hazard, 1e-3)]
        free = objs[(hazard, None)]
        L = o.L
        for fidx in (fidx_eta, fidx_eps):
            b_pos = fidx.index(1)
            zf = np.log([0.3] * len(fidx))
            zf[b_pos] = math.log(1e-12)                              # rho_beta = 1.1e-12 < 1e-3: removal off
            assert est._kappa_corner_starts(o, fidx, zf) == []
            assert est._kappa_corner_starts(o, fidx + [3], np.append(zf, 0.0)) == []   # explicit: kappa is pinned
            n0 = o.n_removal_on
            st = est._removal_on_starts(o, fidx, zf)
            assert len(st) == 3 and o.n_removal_on == n0 + 3
            for rho, z in zip((1e-2, 1e-1, 1.0), st):
                assert z[b_pos] == math.log(min(max(rho / L, o.box["beta"][0]), o.box["beta"][1]))
                assert np.array_equal(np.delete(z, b_pos), np.delete(zf, b_pos))        # only beta moves
            zon = zf.copy()
            zon[b_pos] = math.log(1e-3 / L * 1.0000001)               # removal on: no starts
            assert est._removal_on_starts(o, fidx, zon) == []
            assert est._removal_on_starts(o, [k for k in fidx if k != 1], np.delete(zf, b_pos)) == []  # beta pinned
            assert est._removal_on_starts(o, fidx, None) == []
            assert est._removal_on_starts(free, fidx, zf) == []        # kappa free: kappa corners do this job
        # kappa free: the corners are still there
        assert len(est._kappa_corner_starts(free, [1, 2, 3], np.log([1e-12, 0.3, 0.5]))) == 3
        assert est._shadow_for(o, 0, fidx_eta, -9.0, [1.0, 2.0, 0.3, 1e-3] + ([0.05] if hazard else [])) is None


@pytest.mark.sut
@pytest.mark.parametrize("k0", [1e-3, 1.0])
def test_every_hazard_mle_point_is_pinned(est, fj, k0):
    """_h + KFIX's MLE stage (`_hazard_mle` with the jax2-like MLE on): every evaluated z (starts, L-BFGS-B iterates,
    ladder rungs, jax2-like searches) has z[3] == log k0 exactly -- read from the archive and from a recorder --
    the Makeham seed is mapped onto k0 keeping beta/(1+kappa), and every row's theta has kappa = k0."""
    import numpy as np
    o = _obj(est, fj, hazard=True, kappa_fixed=k0)
    o.shadow_on = True
    seen = []
    for name in ("nll", "nll_grad"):
        f = getattr(o, name)

        def rec(z, _f=f):
            seen.append(np.asarray(z, float)[3])
            return _f(z)
        setattr(o, name, rec)
    seed4 = est._kfix_map_seed([1.0, 2.0, 0.3, 0.5], k0)
    assert seed4[3] == k0 and seed4[1] == pytest.approx(2.0 * (1 + k0) / 1.5, rel=1e-15)
    bounds = _bounds(o)
    best, best_v, rec_, seed_theta = est._hazard_mle(o, np.array(_T), np.array(_E), seed4, bounds,
                                                     dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
    assert best is not None and best[3] == math.log(k0)
    Z, LL = est._sc_archive(o)
    assert Z.shape[0] > 20 and np.all(Z[:, 3] == math.log(k0))
    assert seen and all(x == math.log(k0) for x in seen)
    labels = [r["label"] for r in rec_["starts"]]
    assert labels[:2] == ["seed4+h_lo", "seed4_eps4+h_lo"] and "jax2like+h_lo" in labels
    for r in rec_["starts"]:
        if "theta" in r:
            assert r["theta"][3] == pytest.approx(k0, rel=1e-15) and r["theta0"][3] == pytest.approx(k0, rel=1e-15)
    for r in rec_["jax2like"]:
        assert r["theta"][3] == pytest.approx(k0, rel=1e-15) and len(r["theta"]) == 4
    assert o.z_shadow0[3] == math.log(k0)
    mk = rec_["makeham"]
    if mk["theta"] is not None:
        th, tk = mk["theta"], mk["theta_kfix"]
        assert "makeham" in labels and tk[3] == k0 and tk[1] == pytest.approx(th[1] * (1 + k0) / (1 + th[3]))
        assert [tk[0], tk[2], tk[4]] == [th[0], th[2], th[4]]
    assert any("ll" in r for r in rec_["ladder"]) or rec_["ladder"] == []
    assert seed_theta[3] == k0                                      # 1.0.1.dev3: exactly (`_theta_of`)


# ----------------------------------------------------------------------------------------------- sc_extents

def _archive(seed=3, m=300):
    import numpy as np
    rng = np.random.default_rng(seed)
    Z = np.log(np.array([[1e-3, 0.5, 1e-2, 1.0]])) + rng.normal(0, 0.4, (m, 4))
    LL = -rng.exponential(1.5, m)
    Z[:5, 0] = np.nan                                                 # unknown coordinates: skipped where needed
    LL[5:8] = np.nan
    Z[8, 1] = math.log(1e4)                                           # out of the box: ignored
    return Z, LL


def test_sc_extents_derived_none_is_the_1_0_1_dict(est):
    Z, LL = _archive()
    box = est.box_for(300.0)
    args = (Z, LL, ("eta", "beta", "epsilon", "kappa"), box, -0.1, THR)
    for sz in (False, True):
        a = est.sc_extents(*args, store_z=sz)
        b = est.sc_extents(*args, store_z=sz, derived=None)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        assert set(a) == {"thr", "half_thr", "ll_hat", "ll_best", "gap_nats", "stale", "stale_nats", "rtol_box",
                          "n_points", "n_in_box", "n_out_of_box", "n_qualifying", "best", "extents"}
        c = est.sc_extents(*args, store_z=sz, derived=dict(est.DERIVED))
        d = dict(c)
        d.pop("derived_extents")
        assert json.dumps(d, sort_keys=True) == json.dumps(a, sort_keys=True)          # derived only ADDS


def test_sc_extents_derived_extents_are_the_brute_force_ones(est):
    import numpy as np
    Z, LL = _archive(seed=4)
    box = est.box_for(300.0)
    P = ("eta", "beta", "epsilon", "kappa")
    sc = est.sc_extents(Z, LL, P, box, -0.1, THR, store_z=True, derived=dict(est.DERIVED))
    TH = np.exp(Z)
    ok = np.isfinite(LL)
    for k, p in enumerate(P):
        known = ~np.isnan(Z[:, k])
        ok &= ~known | ((TH[:, k] >= box[p][0] * (1 - 1e-9)) & (TH[:, k] <= box[p][1] * (1 + 1e-9)))
    llb = max(-0.1, LL[ok].max())
    q = ok & (LL >= llb - HALF)
    rb = lambda k: 1.0 - k * math.log1p(1.0 / k)                      # noqa: E731 - typed again, independently
    brute = {"sqrt_eta_over_eps": [math.sqrt(TH[j, 0]) / TH[j, 2] if np.isfinite(Z[j, 0]) else None
                                   for j in range(len(LL))],
             "beta_rbar": [TH[j, 1] * rb(TH[j, 3]) for j in range(len(LL))]}
    for name, vals in brute.items():
        idx = [j for j in range(len(LL)) if q[j] and vals[j] is not None]
        ext = sc["derived_extents"][name]
        assert ext["n"] == len(idx) > 50
        jmin = min(idx, key=lambda j: vals[j])
        jmax = max(idx, key=lambda j: vals[j])
        assert ext["qmin"] == pytest.approx(vals[jmin], rel=1e-13) and ext["qmax"] == pytest.approx(vals[jmax], rel=1e-13)
        assert ext["ll_qmin"] == LL[jmin] and ext["ll_qmax"] == LL[jmax]
        assert np.asarray(ext["z_qmin"], float).tobytes() == Z[jmin].tobytes()
        assert np.asarray(ext["z_qmax"], float).tobytes() == Z[jmax].tobytes()
    empty = est.sc_extents(Z[:0], LL[:0], P, box, -0.1, THR, derived=dict(est.DERIVED))
    assert empty["derived_extents"]["sqrt_eta_over_eps"] == dict(n=0, qmin=None, qmax=None)


def test_rbar_is_the_tester_definition(est, ad):
    from srbench.score.combos import removal_mean_factor
    for k in (1e-3, 0.02, 0.5, 1.0, 2.0, 300.0):
        assert est.rbar(k) == pytest.approx(removal_mean_factor(k), rel=1e-12)
        assert ad.rbar(k) == est.rbar(k)
    assert ad.rbar(1.0) == pytest.approx(1 - math.log(2.0), rel=1e-15)


# ----------------------------------------------------------------------------------------------- the adapter

def _fake_res(theta, hazard=False):
    """A fit result shaped like a KFIX fit: no kappa ci / verdict, combo_grid, derived extents, kfix record."""
    ci = {"eta@0.95": (5e-4, 2e-3), "beta@0.95": (0.05, 5.0), "epsilon@0.95": (5e-3, 2e-2)}
    verdicts = {"eta": "identified", "beta": "identified", "epsilon": "identified"}
    ext = lambda lo, hi: {"n": 9, "qmin": lo, "qmax": hi}            # noqa: E731
    sc = {"ll_hat": 0.0, "ll_best": 0.0, "gap_nats": 0.0, "stale": "none",
          "extents": {"eta": ext(6e-4, 1.5e-3), "beta": ext(0.04, 4.0), "epsilon": ext(5e-3, 2e-2),
                      "kappa": ext(theta[3], theta[3])},
          "derived_extents": {"sqrt_eta_over_eps": ext(1.8, 4.5)}}
    dg = {"verdicts": verdicts, "self_consistency": sc, "profiles": {"eta": [], "sqrt_eta_over_eps": []},
          "kfix": {"kappa_fixed": theta[3], "n_components": {"eta": 1, "beta": 2, "epsilon": 1}},
          "combo_grid": {"sqrt_eta_over_eps": {"kind": "walk", "centre": math.sqrt(theta[0]) / theta[2],
                                               "ci": {"sqrt_eta_over_eps@0.95": [2.0, 5.0]},
                                               "verdict": "identified", "n_components": 1}}}
    return {"theta": list(theta), "ci": ci, "loglik": 0.0, "wall_s": 1.0, "diagnostics": dg}


@pytest.mark.parametrize("k0", K0S)
def test_kfix_claims_beta_rbar_is_the_rescaled_beta_claim(ad, k0):
    res = _fake_res([1e-3, 0.5, 1e-2, k0])
    claims, block = ad.final_claim_kwargs(res, True)
    out, combo, rec = ad.kfix_claim_kwargs(res, claims, k0, True)
    beta = next(c for c in out if c["parameter"] == "beta")
    assert beta == next(c for c in claims if c["parameter"] == "beta")         # beta AT kappa0, unchanged
    assert beta["lower"] == 0.04 and beta["upper"] == 5.0                    # (the self-consistency widening)
    br = combo["beta_rbar"]
    r = 1.0 - k0 * math.log1p(1.0 / k0)
    assert br["shape"] == beta["shape"] and br["level"] == beta["level"]
    for key in ("lower", "upper", "point"):
        assert br[key] == beta[key] * r
    assert br["n_components"] == 2                                          # the beta profile's
    assert rec["rbar_kappa0"] == r and rec["kappa0"] == k0
    # a one-sided / declined beta claim keeps its shape
    for shape_ci, shape in (((0.05, float("nan")), "lower_bound"), ((float("nan"), float("nan")), "none")):
        res2 = _fake_res([1e-3, 0.5, 1e-2, k0])
        res2["ci"]["beta@0.95"] = shape_ci
        res2["diagnostics"]["verdicts"]["beta"] = {"lower_bound": "one-sided:>", "none": "flat"}[shape]
        c2, _b = ad.final_claim_kwargs(res2, True)
        _o, cc, _r = ad.kfix_claim_kwargs(res2, c2, k0, True)
        assert cc["beta_rbar"]["shape"] == shape
        assert ("upper" in cc["beta_rbar"]) is False
        assert ("n_components" in cc["beta_rbar"]) == (shape != "none")


def test_kfix_claim_shapes(ad):
    res = _fake_res([1e-3, 0.5, 1e-2, 1e-3])
    claims, block = ad.final_claim_kwargs(res, True)
    res["ci"]["kappa@0.95"] = (0.01, 10.0)                               # even a stray kappa interval ...
    res["diagnostics"]["verdicts"]["kappa"] = "identified"
    claims2, _ = ad.final_claim_kwargs(res, True)
    for cl in (claims, claims2):
        out, combo, rec = ad.kfix_claim_kwargs(res, cl, 1e-3, True)
        P = [c["parameter"] for c in out]
        assert P == ["eta", "beta", "epsilon", "kappa"]
        assert out[3] == {"parameter": "kappa", "shape": "none", "level": ad.LEVEL}   # ... is declined
        assert out[1]["shape"] == "closed" and "point" in out[1]
        assert set(combo) == {"sqrt_eta_over_eps", "beta_rbar"}
        assert "beta_over_kp1" not in json.dumps([out, combo, rec])
    sq = combo["sqrt_eta_over_eps"]
    # the walked combination's claim: profile_ci's [2.0, 5.0] widened down to the derived extent 1.8
    assert sq["shape"] == "closed" and sq["lower"] == 1.8 and sq["upper"] == 5.0 and sq["n_components"] == 1
    assert sq["point"] == pytest.approx(math.sqrt(1e-3) / 1e-2)
    assert rec["sqrt_eta_over_eps_widening"]["changed"] is True
    # without self-consistency: the profile's own interval
    sq0, rec0 = ad.combo_claim_kwargs(res, "sqrt_eta_over_eps", False)
    assert (sq0["lower"], sq0["upper"], rec0) == (2.0, 5.0, None)
    assert ad.combo_claim_kwargs(res, "beta_rbar", True) == (None, None)     # no such record in a kfix fit


def test_claimset_of_every_kfix_name_validates(ad):
    from srbench.contract.types import Observation
    from srbench.contract.validate import validate_claimset
    from srbench.provenance import CodeProvenance
    obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
    prov = CodeProvenance(package="SRK", resolved_path="/nowhere")
    by_name = {c.NAME: c for c in ad.CLASSES}
    for name in KFIX_NAMES:
        cls = by_name[name]
        res = _fake_res([1e-3, 0.5, 1e-2, cls.KAPPA0] + ([0.01] if cls.HAZARD else []))
        cs = cls().claimset(obs, res, prov, 0.0, 0.0)
        validate_claimset(cs, cls().descriptor, obs)
        assert cs.claim_for("kappa").shape == "none" and cs.claim_for("beta").shape == "closed"
        cc = cs.diagnostics["combo_claims"]
        assert set(cc) == {"sqrt_eta_over_eps", "beta_rbar"}
        assert "beta_over_kp1" not in json.dumps(cs.diagnostics, default=str)
        assert "profiles" in cs.diagnostics                                    # KEEP_PROFILES
        assert cs.diagnostics["switches"]["kappa_fixed"] == cls.KAPPA0
        assert cs.diagnostics["switches"]["combos"] == ["sqrt_eta_over_eps"]
        assert cs.diagnostics["switches"]["box_flag"] is False
        assert cs.diagnostics["kfix"]["claims"]["kappa0"] == cls.KAPPA0
    # J / H: no combo claims, profiles dropped, no kfix switch keys (1.0.1.dev1's record)
    for cls in (ad.MleProfileJax2, ad.MleProfileJax2H):
        res = _fake_res([1e-3, 0.5, 1e-2, 1.0] + ([0.01] if cls.HAZARD else []))
        cs = cls().claimset(obs, res, prov, 0.0, 0.0)
        assert "combo_claims" not in cs.diagnostics and "profiles" not in cs.diagnostics
        assert "kappa_fixed" not in cs.diagnostics["switches"] and "combos" not in cs.diagnostics["switches"]


def test_kfix_classes_and_versions(ad):
    from version_lock import versions
    v = versions()
    assert set(v) == {"mle_profile_jax2", "mle_profile_jax2_h"} | set(KFIX_NAMES)
    assert set(v.values()) == {"1.0.1"}                                     # 1.0.1: the freeze
    assert [c.NAME for c in ad.CLASSES] == ["mle_profile_jax2", "mle_profile_jax2_h"] + list(KFIX_NAMES)
    assert ad.KFIX_KAPPAS == K0S
    for c in ad.CLASSES:
        assert issubclass(c, ad._MleProfileJax2Base) and c.__init__ is object.__init__
        assert (c.INNER_POLISH_J, c.BRACKET_BISECT, c.EXTENT_Z, c.BOX_FLAG) == (True, True, True, False)
        assert (c.SELF_CONSISTENT, c.N_RULE, c.DT_RULE, c.FLOOR_EARLY, c.EARLY_ALL, c.SHADOW, c.EXACT_F,
                c.RELABEL) == (True, True, True, True, True, True, True, True)      # DT_RULE on: 1.0.1.dev4
        if c.KAPPA0 is None:
            assert c.NAME in ("mle_profile_jax2", "mle_profile_jax2_h")
            assert c.COMBOS == () and c.KEEP_PROFILES is False
            continue
        base = "mle_profile_jax2_h" if c.HAZARD else "mle_profile_jax2"
        assert c.NAME == ad.kfix_name(base, c.KAPPA0) and c.KAPPA0 in K0S
        assert (c.HAZARD, c.WINDOW) == ((True, True) if c.HAZARD else (False, False))
        assert c.COMBOS == ("sqrt_eta_over_eps",) and c.KEEP_PROFILES is True
        assert ad.capabilities_for(c.HAZARD, c.WINDOW, c.RELABEL) == (
            ("external_hazard", "fit_window", "censoring") if c.HAZARD else ("censoring",))
    assert [ad.kappa_tag(k) for k in K0S] == ["0p001", "0p5", "1"]


@pytest.mark.sut
def test_run_fit_passes_the_kfix_keywords_only_when_set(ad, monkeypatch):
    from types import SimpleNamespace
    from srbench.contract.types import Observation
    calls = []

    def fake_fit(fj, ages, events, **kw):
        calls.append(dict(kw))
        return {"theta": [1e-3, 1.0, 1e-2, 1.0], "ci": {}, "loglik": 0.0, "diagnostics": {"exc_counts": {}}}
    prep = SimpleNamespace(ages=(1.0, 2.0), events=(1, 1), window_eff=None, censor_tie={}, record={})
    stubs = {"observation_model": SimpleNamespace(prepare=lambda *a, **k: prep),
             "fp_jax": SimpleNamespace(), "estimator": SimpleNamespace(fit=fake_fit)}
    monkeypatch.setattr(ad, "_sibling", lambda name: stubs[name])
    monkeypatch.setattr(ad, "version_lock_guard", lambda: None)
    obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
    # 1.0.1.dev1's run_fit keyword SET (the plumbing under test), not 1.0.1's constants: dt_rule=False is kept on
    # purpose (every name passes DT_RULE = True since 1.0.1.dev4; test_calibrate_passes_... checks that)
    base = dict(relabel=True, hazard=False, window=False, shadow=True, exact_from_F=True, self_consistent=True,
                floor_early=True, n_rule=True, dt_rule=False, early_all=True, inner_polish_j=True,
                bracket_bisect=True, extent_z=True, box_flag=False)
    ad.run_fit(obs, **base)
    k101 = sorted(calls[-1])
    assert "kappa_fixed" not in k101 and "combos" not in k101 and "box_flag" not in k101
    ad.run_fit(obs, **base, kappa_fixed=None, combos=())
    assert sorted(calls[-1]) == k101
    ad.run_fit(obs, **base, kappa_fixed=1e-3)
    assert sorted(calls[-1]) == sorted(k101 + ["kappa_fixed"]) and calls[-1]["kappa_fixed"] == 1e-3
    ad.run_fit(obs, **base, combos=("sqrt_eta_over_eps",))
    assert sorted(calls[-1]) == sorted(k101 + ["combos"]) and calls[-1]["combos"] == ("sqrt_eta_over_eps",)


#: 1.0.1.dev1's `calibrate` -> `run_fit` keywords (5f48381), the J / H call of every later version
_RUN_FIT_KW_DEV1 = {"relabel", "hazard", "window", "shadow", "exact_from_F", "self_consistent", "floor_early",
                    "n_rule", "dt_rule", "early_all", "inner_polish_j", "bracket_bisect", "extent_z", "box_flag"}


def test_calibrate_passes_the_kfix_keywords_only_when_set(ad, monkeypatch):
    """1.0.1.dev3: J / H's calibrate calls run_fit with 1.0.1.dev1's keyword set exactly (no kappa_fixed: None /
    combos: () keys, which a run_fit capture would record); a kfix name adds exactly its KAPPA0 and COMBOS."""
    import srbench.provenance as P
    from srbench.contract.types import Observation
    from srbench.provenance import CodeProvenance
    calls = []

    class _Recorded(Exception):
        pass

    def fake_run_fit(observation, **kw):
        calls.append(dict(kw))
        raise _Recorded("recorded")                      # calibrate stores it as an error ClaimSet: no fit
    monkeypatch.setattr(ad, "run_fit", fake_run_fit)
    monkeypatch.setattr(ad, "env_guard", lambda: None)
    monkeypatch.setattr(ad, "version_lock_guard", lambda: None)
    monkeypatch.setattr(P, "resolve_sut", lambda: CodeProvenance(package="SRK", resolved_path="/nowhere"))
    monkeypatch.setenv("SRK_CACHE_READONLY", os.environ.get("SRK_CACHE_READONLY", "1"))
    obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
    for cls in ad.CLASSES:
        cs = cls().calibrate(obs)
        assert cs.status == "error:_Recorded"
        kw = calls[-1]
        if cls.KAPPA0 is None:
            assert set(kw) == _RUN_FIT_KW_DEV1, cls.NAME
        else:
            assert set(kw) == _RUN_FIT_KW_DEV1 | {"kappa_fixed", "combos"}, cls.NAME
            assert kw["kappa_fixed"] == cls.KAPPA0 and kw["combos"] == ("sqrt_eta_over_eps",)
        assert kw["box_flag"] is False and kw["hazard"] is cls.HAZARD and kw["window"] is cls.WINDOW
        assert kw["dt_rule"] is True                     # 1.0.1.dev4: every name, scoped in the estimator
    assert len(calls) == len(ad.CLASSES)


def test_kfix_claim_kwargs_refuses_a_foreign_result(ad):
    """1.0.1.dev3: beta_rbar is scaled by rbar(kappa0); a result pinned at another kappa, or not pinned at all, is
    refused instead of silently rescaled."""
    res = _fake_res([1e-3, 0.5, 1e-2, 0.5])
    claims, _b = ad.final_claim_kwargs(res, True)
    ad.kfix_claim_kwargs(res, claims, 0.5, True)                                 # its own kappa0: accepted
    for k0 in (1e-3, 1.0, 0.5 * (1 + 1e-15)):
        with pytest.raises(ValueError, match="pinned kappa at"):
            ad.kfix_claim_kwargs(res, claims, k0, True)
    res["diagnostics"].pop("kfix")
    with pytest.raises(ValueError, match="not a kappa-fixed fit"):
        ad.kfix_claim_kwargs(res, claims, 0.5, True)


# ----------------------------------------------------------------------------------------------- tiny real fits

def _tiny():
    import numpy as np
    rng = np.random.default_rng(1)
    u = rng.random(80)
    t = np.floor(np.log(1 - np.log(u) / 0.05) / 0.3 / 0.5) * 0.5 + 0.5
    e = np.ones_like(t, dtype=int)
    e[t > 12] = 0
    return np.minimum(t, 12.0), e


KW_TINY = dict(N=40, dt=0.02)
SW = dict(self_consistent=True, inner_polish=True, bracket_bisect=True, extent_z=True)    # 1.0.1's estimator switches


@pytest.fixture(scope="module")
def tiny_kfits(est, fj):
    """Three tiny real fits (80 records on a 0.5 grid, solver N=40), each with the archive captured:
    J-kfix (k0 = 1e-3), H-kfix (k0 = 0.5, SHADOW on) and the twin's kappa-free path (both combinations).
    1.0.1.dev3: every `_removal_on_starts` call is recorded by its CALLER (seen["ro_sites"]: caller -> [calls,
    calls that returned starts, starts])."""
    t, e = _tiny()
    real = est._sc_archive
    real_ro = est._removal_on_starts
    out = {}

    def run(key, **kw):
        seen = {}
        sites = {}

        def spy(obj):
            Z, LL = real(obj)
            seen.update(Z=Z.copy(), LL=LL.copy(), params=obj.params, fixed=dict(obj.fixed))
            return Z, LL

        def ro_spy(obj, fidx, zf):
            st = real_ro(obj, fidx, zf)
            rec = sites.setdefault(sys._getframe(1).f_code.co_name, [0, 0, 0])
            rec[0] += 1
            rec[1] += bool(st)
            rec[2] += len(st)
            return st
        est._sc_archive = spy
        est._removal_on_starts = ro_spy                  # a module global: every call site looks it up per call
        try:
            r = est.fit(fj, t, e, delta=0.5, solver_kw=KW_TINY, **SW, **kw)
        finally:
            est._sc_archive = real
            est._removal_on_starts = real_ro
        seen["ro_sites"] = sites
        out[key] = (r, seen)
    run("J", kappa_fixed=1e-3, combos=("sqrt_eta_over_eps",))
    run("H", kappa_fixed=0.5, combos=("sqrt_eta_over_eps",), hazard=True, shadow=True)
    run("twin", combos=("sqrt_eta_over_eps", "beta_rbar"))
    return out


@pytest.mark.sut
@pytest.mark.parametrize("key,k0", [("J", 1e-3), ("H", 0.5)])
def test_real_kfix_fit_is_pinned_everywhere(tiny_kfits, ad, key, k0):
    import numpy as np
    r, seen = tiny_kfits[key]
    d = r["diagnostics"]
    assert r["theta"][3] == k0 and np.isfinite(r["loglik"])
    assert seen["fixed"] == {3: math.log(k0)} and seen["Z"].shape[0] > 100
    assert np.all(seen["Z"][:, 3] == math.log(k0))                    # every evaluated point
    assert "kappa@0.95" not in r["ci"] and "kappa" not in d["verdicts"] and "kappa" not in d["profiles"]
    assert "kappa" not in d["grid"] and "kappa" not in d["bracket_bisect"]["params"]
    kf = d["kfix"]
    assert kf["kappa_fixed"] == k0 and kf["z_fixed"] == math.log(k0) and kf["free"] == [k for k in range(
        len(seen["params"])) if k != 3]
    assert kf["seed4_mapped"][3] == k0 and kf["seed4_raw"][3] != k0
    assert kf["shadow_chains"] == ("n/a: kappa is fixed (no other kappa basin)" if key == "H" else None)
    assert set(kf["n_components"]) == set(d["verdicts"])
    for rnd in d["basin_fix"]["rounds"]:
        for st in rnd["starts"]:
            assert st["theta"][3] == pytest.approx(k0, rel=1e-15)
    if key == "H":
        assert d["hazard"]["shadow"]["n_solves"] == 0                 # no shadow chains under KFIX
        assert all(row["theta"][3] == pytest.approx(k0, rel=1e-15) for row in d["hazard"]["mle"]["jax2like"])
    # the combination: walked, profiled, widened
    cg = d["combo_grid"]["sqrt_eta_over_eps"]
    assert cg["kind"] == "walk" and cg["n_nodes"] >= 5 and cg["centre"] == pytest.approx(
        math.sqrt(r["theta"][0]) / r["theta"][2], rel=1e-12)
    assert set(cg) >= {"ci", "verdict", "n_components", "recentred", "down", "up", "bracket_bisect"}
    rows = d["profiles"]["sqrt_eta_over_eps"]
    assert len(rows) == cg["n_nodes"] and rows == sorted(rows, key=lambda q: q["value"])
    # the claims: the adapter's, contract-valid, kappa declined, beta at kappa0, combination claims present
    claims, block = ad.final_claim_kwargs(r, True)
    out, combo, rec = ad.kfix_claim_kwargs(r, claims, k0, True)
    assert out[3]["shape"] == "none" and set(combo) == {"sqrt_eta_over_eps", "beta_rbar"}
    from srbench.contract.types import Claim
    for kw in out:
        Claim(**kw)
    # the combination claim contains every qualifying archived point (the strict rule, from the archive)
    Z, LL = seen["Z"], seen["LL"]
    TH = np.exp(Z)
    box = {p: ad.BOUNDS[p] for p in ("eta", "beta", "epsilon", "kappa")}
    if key == "H":
        box["h"] = tuple(d["hazard"]["box"])
    inb = np.all([(TH[:, k] >= box[p][0] * (1 - 1e-9)) & (TH[:, k] <= box[p][1] * (1 + 1e-9))
                  for k, p in enumerate(seen["params"])], axis=0) & np.isfinite(LL)
    q = inb & (LL >= max(r["loglik"], LL[inb].max()) - HALF)
    vals = np.sqrt(TH[q, 0]) / TH[q, 2]
    sq = combo["sqrt_eta_over_eps"]
    if "lower" in sq:
        assert vals.min() >= sq["lower"] * (1 - 1e-12)
    if "upper" in sq:
        assert vals.max() <= sq["upper"] * (1 + 1e-12)
    ext = d["self_consistency"]["derived_extents"]["sqrt_eta_over_eps"]
    assert ext["n"] == int(q.sum()) and ext["qmin"] == pytest.approx(vals.min(), rel=1e-12)
    print(f"\n{key}-kfix k0={k0}: sqrt(eta)/eps {sq}; beta_rbar {combo['beta_rbar']}; removal-on starts "
          f"{kf['removal_on']['n_starts']}; archive {Z.shape[0]} points")


_RO_SITES = {"_polish_edges", "_polish_inner", "_bisect_brackets"}     # the three kappa-corner call sites


@pytest.mark.sut
def test_real_fits_reach_the_removal_on_starts_at_all_three_sites(tiny_kfits):
    """1.0.1.dev3 (review): the removal-on starts are CALLED at all three kappa-corner sites in real J-kfix and
    H-kfix fits (gated on kappa_fixed, not on NP -- kfix 0.1.2's `np == 3` never fires with h), they GENERATE starts
    in both fits (MEASURED 2026-10-08: J 6 and H 9, all at `_polish_edges`), and diagnostics.kfix.removal_on counts
    exactly what they generated. Kappa free (the twin's path): called at the same sites, never a start."""
    made = {}
    for key in ("J", "H", "twin"):
        r, seen = tiny_kfits[key]
        sites = seen["ro_sites"]
        print(f"\n{key}: removal-on calls by site (calls, non-empty, starts): {sites}")
        assert set(sites) == _RO_SITES, (key, sites)
        assert all(n > 0 for n, _ne, _ns in sites.values())
        made[key] = sum(ns for _n, _ne, ns in sites.values())
        if key == "twin":
            assert made[key] == 0 and "kfix" not in r["diagnostics"]
        else:
            assert made[key] == r["diagnostics"]["kfix"]["removal_on"]["n_starts"]
    assert made["J"] > 0 and made["H"] > 0, made


@pytest.mark.sut
@pytest.mark.parametrize("hazard,k0", [(False, 0.5), (True, 1e-3)])
def test_failed_kfix_fit_keeps_k0_exact_and_its_kfix_record(est, fj, ad, monkeypatch, hazard, k0):
    """1.0.1.dev3 (review): "all starts failed" (every MLE search returns a non-finite value) -> theta[3] == k0
    EXACTLY (the _h seed was exp(log k0) = 1e-3 + 1 ulp before dev3), diagnostics.kfix present (n_components None),
    and the kfix claims still build (every one declined). The two SRK curve-fit seeds are stubbed (speed: the path
    under test starts after them; the Makeham seed declines, so _h's seed is start 1, exp(log k0) before dev3)."""
    import numpy as np
    import SRK.fp_curvefit as CF
    t, e = _tiny()
    monkeypatch.setattr(CF, "fp_curvefit_seed", lambda *a, **k: {"eta": 2e-3, "beta": 0.3, "epsilon": 0.05, "k": 0.7})
    monkeypatch.setattr(est, "_makeham_seed", lambda obj, t_, e_: dict(status=["stubbed"], theta=None, raw=None,
                                                                        nfev=None, wall_s=None, cpu_s=None))
    monkeypatch.setattr(est, "_lbfgs", lambda obj, fg, fv, z0, *a, **k: (np.inf, np.asarray(z0, float), False))
    r = est.fit(fj, t, e, delta=0.5, solver_kw=KW_TINY, kappa_fixed=k0, hazard=hazard, **SW)
    d = r["diagnostics"]
    assert d["error"] == "all starts failed" and not np.isfinite(r["loglik"]) and r["ci"] == {}
    assert r["theta"][3] == k0 and len(r["theta"]) == (5 if hazard else 4)
    assert d["kfix"]["kappa_fixed"] == k0 and d["kfix"]["n_components"] is None
    claims, _b = ad.final_claim_kwargs(r, True)
    out, combo, _rec = ad.kfix_claim_kwargs(r, claims, k0, True)
    assert [c["shape"] for c in out] == ["none"] * 4 and combo["beta_rbar"]["shape"] == "none"


@pytest.mark.sut
def test_real_twin_path_walks_and_reads_out(tiny_kfits, ad):
    import numpy as np
    r, seen = tiny_kfits["twin"]
    d = r["diagnostics"]
    assert seen["fixed"] == {} and "kfix" not in d and "kappa@0.95" in r["ci"]       # kappa free
    assert len(set(np.round(seen["Z"][:, 3], 6))) > 10
    cg = d["combo_grid"]
    assert set(cg) == {"sqrt_eta_over_eps", "beta_rbar"}
    br = cg["beta_rbar"]
    assert br["kind"] == "readout" and br["centre"] == pytest.approx(r["theta"][1] * ad.rbar(r["theta"][3]), rel=1e-12)
    rows = d["profiles"]["beta_rbar"]
    assert len(rows) == br["n_nodes"] >= 3 and rows == sorted(rows, key=lambda q: q["value"])
    assert set(d["self_consistency"]["derived_extents"]) == {"sqrt_eta_over_eps", "beta_rbar"}
    for name in ("sqrt_eta_over_eps", "beta_rbar"):
        kw, rec = ad.combo_claim_kwargs(r, name, True)
        assert kw["parameter"] == name and kw["shape"] in ("closed", "lower_bound", "upper_bound", "none")
        lo, hi = ad.COMBO_BOX[name]
        for key in ("lower", "upper", "point"):
            if key in kw:
                assert lo <= kw[key] <= hi


@pytest.mark.sut
def test_fit_refuses_bad_combos(est, fj):
    t, e = _tiny()
    for bad, kw in ((("beta_over_kp1",), {}), (("sqrt_eta_over_eps", "sqrt_eta_over_eps"), {}),
                    (("beta_rbar",), {"kappa_fixed": 1.0})):
        with pytest.raises(ValueError, match="combos|beta\\*rbar"):
            est.fit(fj, t, e, delta=0.5, solver_kw=KW_TINY, combos=bad, **kw)
