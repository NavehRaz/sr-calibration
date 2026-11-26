"""L0: a z of the wrong length raises in f / vg / nll / nll_grad / inside, and the exception policy
(count; re-raise under SRB_STRICT; ALWAYS re-raise IndexError / ValueError / TypeError)."""
from __future__ import annotations

import math

import pytest

import _jax2_testlib as T

pytestmark = pytest.mark.sut


@pytest.fixture(scope="module")
def mods():
    T.jax_env()
    return T.pkg_module("fp_jax"), T.pkg_module("estimator")


@pytest.fixture(scope="module")
def small(mods):
    """A tiny cohort on a coarse grid: compiles in seconds."""
    import numpy as np
    fj, est = mods
    t = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7])
    e = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    kw = dict(N=40, dt=0.02, tau_max=3.0, Gtol=1e-11, sampling_dt=0.1)
    obj = est._Objective(fj, t, e, 1.1, kw, est.box_for(300.0))
    return obj


def test_z_size(mods):
    fj, _ = mods
    assert fj.z_size(False) == 4 and fj.z_size(True) == 5


@pytest.mark.parametrize("n", [3, 5])
def test_wrong_length_z_raises_in_f_and_vg(small, n):
    import numpy as np
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        small.f(np.zeros(n))
    with pytest.raises(ValueError, match=r"shape \(4,\)"):
        small.vg(np.zeros(n))


def test_right_length_z_evaluates(small):
    import numpy as np
    z = np.log([1.0, 2.0, 0.3, 0.5])
    v = float(small.f(z))
    vv, g = small.vg(z)
    assert math.isfinite(v) and math.isfinite(float(vv)) and np.all(np.isfinite(np.asarray(g)))
    assert np.asarray(g).shape == (4,)


@pytest.mark.parametrize("n", [3, 5])
def test_wrong_length_raises_in_objective(small, n):
    import numpy as np
    with pytest.raises(ValueError):
        small.nll(np.zeros(n))
    with pytest.raises(ValueError):
        small.nll_grad(np.zeros(n))
    with pytest.raises(ValueError):
        small.inside([1.0] * n)
    assert small.inside([1.0] * 4) is True


def test_objective_params_are_the_sr_four(small):
    assert small.params == ("eta", "beta", "epsilon", "kappa") and small.NP == 4
    assert small.exc_counts == {}
    assert not hasattr(small, "h_mk") and not hasattr(small, "n_h_starts")   # no hazard state when off


@pytest.fixture(scope="module")
def small_h(mods):
    """The same cohort with the hazard on (5-D z) and a window."""
    import numpy as np
    fj, est = mods
    t = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7])
    e = np.array([1, 1, 1, 1, 1, 1, 1, 0])
    kw = dict(N=40, dt=0.02, tau_max=3.0, Gtol=1e-11, sampling_dt=0.1)
    box = dict(est.box_for(300.0), h=est._h_box(1.1))
    return est._Objective(fj, t, e, 1.1, kw, box, hazard=True, window=(0.5, 1.7))


@pytest.mark.parametrize("n", [3, 4, 6])
def test_wrong_length_z_raises_with_hazard(small_h, n):
    """With the hazard on, a length-4 z must NOT evaluate (JAX would read h = exp(z[3]) = kappa)."""
    import numpy as np
    with pytest.raises(ValueError, match=r"shape \(5,\)"):
        small_h.f(np.zeros(n))
    with pytest.raises(ValueError, match=r"shape \(5,\)"):
        small_h.vg(np.zeros(n))
    with pytest.raises(ValueError):
        small_h.nll(np.zeros(n))
    with pytest.raises(ValueError):
        small_h.nll_grad(np.zeros(n))
    with pytest.raises(ValueError):
        small_h.inside([1.0] * n)


def test_hazard_objective_evaluates_and_boxes_h(small_h, mods):
    import numpy as np
    _, est = mods
    assert small_h.params == ("eta", "beta", "epsilon", "kappa", "h") and small_h.NP == 5
    h_lo, h_hi = small_h.box["h"]
    assert h_lo == pytest.approx(1e-6 / 1.1) and h_hi == pytest.approx(10.0 / 1.1)
    z = np.log([1.0, 2.0, 0.3, 0.5, 0.2])
    v, g = small_h.nll_grad(z)
    assert math.isfinite(v) and np.asarray(g).shape == (5,) and np.all(np.isfinite(g))
    assert math.isfinite(small_h.nll(z))
    # h exactly at its floor through exp(log(h_lo)) is inside; clearly below / above is not
    assert small_h.inside([1.0, 2.0, 0.3, 0.5, float(np.exp(np.log(h_lo)))])
    assert not small_h.inside([1.0, 2.0, 0.3, 0.5, h_lo * 0.999])
    assert not small_h.inside([1.0, 2.0, 0.3, 0.5, h_hi * 1.001])
    assert small_h.nll(np.log([1.0, 2.0, 0.3, 0.5, h_lo * 0.5])) == np.inf
    with pytest.raises(ValueError):
        est._h_box(float("inf"))


def test_triggers_skip_h_at_floor_but_keep_h_at_ceiling(small_h, mods):
    import numpy as np
    _, est = mods
    obj = small_h
    h_lo, h_hi = obj.box["h"]
    fidx = [1, 2, 3, 4]
    bounds = [(math.log(obj.box[obj.params[k]][0]), math.log(obj.box[obj.params[k]][1])) for k in fidx]
    z = np.log([1.0, 2.0, 0.3, 0.5, h_lo])
    assert est._triggers(obj, z, fidx, bounds) == ()
    z = np.log([1.0, 2.0, 0.3, 0.5, h_hi])
    assert est._triggers(obj, z, fidx, bounds) == ("h@hi",)
    z = np.log([1.0, 2.0, 0.3, 1e-3, h_lo])                 # kappa at its floor still triggers
    assert est._triggers(obj, z, fidx, bounds) == ("kappa@lo",)


def test_step0_and_hazard_starts(small_h, small, mods):
    import numpy as np
    _, est = mods
    obj = small_h
    h_lo = obj.box["h"][0]
    th = [1.0, 2.0, 0.3, 0.5, h_lo]
    assert est._step0(obj, "h", th) == 0.5                          # at the floor
    assert est._step0(obj, "h", th[:4] + [h_lo * 9.9]) == 0.5       # within 1 dex of it
    assert est._step0(obj, "h", th[:4] + [h_lo * 10.1]) == est._STEP0_DEX
    for p in ("eta", "beta", "epsilon", "kappa"):
        assert est._step0(obj, p, th) == est._STEP0_DEX
        assert est._step0(small, p, th[:4]) == est._STEP0_DEX       # hazard off: rc1's first step
    obj.h_mk, obj.mk_eps, obj.h_mle = 0.05, 0.4, 0.2
    zf = np.log([2.0, 0.3, 0.5, 0.2])                               # nuisance z of an eta node: (beta,eps,kap,h)
    st = est._hazard_starts(obj, [1, 2, 3, 4], zf)
    hs = sorted(round(float(np.exp(s[3])), 12) for s in st)
    # floor, Makeham h (the MLE h equals zf's own h -> deduplicated), plus the (eps, h) pair
    assert len(st) == 3
    assert round(h_lo, 12) in hs and hs.count(0.05) == 2
    pair = [s for s in st if abs(np.exp(s[1]) - 0.4) < 1e-12]
    assert len(pair) == 1 and abs(np.exp(pair[0][3]) - 0.05) < 1e-12
    assert est._hazard_starts(obj, [0, 1, 2, 3], np.log([1.0, 2.0, 0.3, 0.5])) == []   # the h profile
    assert est._hazard_starts(small, [1, 2, 3], np.log([2.0, 0.3, 0.5])) == []          # hazard off
    obj.h_mk = obj.mk_eps = obj.h_mle = None


@pytest.mark.parametrize("exc", [IndexError, ValueError, TypeError])
def test_programming_errors_always_reraised_and_counted_once(small, mods, monkeypatch, exc):
    """rc1's `_lbfgs` and `_profile_ll_at` swallowed ANY exception ((nan, None)); IndexError / ValueError /
    TypeError must surface even off-strict. Raised inside `_lbfgs`'s try, it passes `_lbfgs`'s site and
    then `_profile_ll_at`'s: counted ONCE."""
    _, est = mods

    def boom(*a, **k):
        raise exc("synthetic")

    monkeypatch.setattr(est, "minimize", boom)
    monkeypatch.delenv("SRB_STRICT", raising=False)
    small.exc_counts.clear()
    with pytest.raises(exc, match="synthetic"):
        est._profile_ll_at(small, [1.0, 2.0, 0.3, 0.5], 0, 1.0, [1, 2, 3])
    assert small.exc_counts == {exc.__name__: 1}
    small.exc_counts.clear()


def test_other_exceptions_counted_and_swallowed_unless_strict(small, mods, monkeypatch):
    _, est = mods

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(est, "_lbfgs", boom)
    monkeypatch.delenv("SRB_STRICT", raising=False)
    small.exc_counts.clear()
    ll, z = est._profile_ll_at(small, [1.0, 2.0, 0.3, 0.5], 0, 1.0, [1, 2, 3])
    assert math.isnan(ll) and z is None                    # rc1's behaviour off-strict ...
    assert small.exc_counts == {"RuntimeError": 1}         # ... but now counted
    monkeypatch.setenv("SRB_STRICT", "1")
    with pytest.raises(RuntimeError, match="synthetic"):
        est._profile_ll_at(small, [1.0, 2.0, 0.3, 0.5], 0, 1.0, [1, 2, 3])
    assert small.exc_counts == {"RuntimeError": 2}
    small.exc_counts.clear()


def test_strict_flag_values(mods, monkeypatch):
    _, est = mods
    for v, want in (("1", True), ("yes", True), ("0", False), ("", False)):
        monkeypatch.setenv("SRB_STRICT", v)
        assert est._strict() is want
    monkeypatch.delenv("SRB_STRICT", raising=False)
    assert est._strict() is False
