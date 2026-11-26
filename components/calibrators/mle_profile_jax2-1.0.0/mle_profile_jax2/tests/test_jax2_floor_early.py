"""rc2 I3: the floor early-time fix -- the graded early-time mesh (`fp_jax.early_schedule`, `_make_march_g`,
`make_loglik(early=)`) and its wiring (`estimator._grid(floor_early=)`).

  * the schedule: BE start, geometric growth bounded by `ratio`, BDF2's exact 1.5 / 2 / -0.5 on the uniform
    part, tau_max respected, bad arguments refused;
  * OFF: `make_loglik(early=None)` lowers to rc1's program, text for text;
  * the uniform limit (dt0 = dt): the graded code reproduces the uniform march to round-off;
  * the hand-written reverse sweep on the graded grid equals forward-mode AD through the value march, with and
    without the hazard, and the checkpointed sweep equals the stored one bit for bit;
  * the defect it fixes, on a synthetic E. coli-like cohort with a death in the first sampling bin: the
    uniform floor grid is > 1 nat off a converged reference, the graded start within 0.01 nats;
  * `_grid(floor_early=True)` adds the mesh only where the floor binds (subprocess: SRK).
The full-size evidence (the 22 floor records, the 3-grid check) is in reports/rc2/I3-floor/.
"""
from __future__ import annotations

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
EARLY = dict(dt0=1e-8, ratio=1.02)


def test_schedule(mods):
    import numpy as np
    fj, _ = mods
    for dt, tm, ns, dt0, r in ((3e-4, 2.0, 20, 1e-8, 1.02), (4e-3, 6.0, 20, 1e-9, 1.05), (1e-3, 1.0, 3, 1e-3, 1.5)):
        DT, TAU, AL, A, B = fj.early_schedule(dt, tm, ns, dt0, r)
        assert DT.size + 1 == TAU.size and np.all(DT > 0) and np.all(np.diff(TAU) > 0)
        assert TAU[0] == 0.0 and TAU[-1] <= tm + 1e-12 and tm - TAU[-1] < dt
        assert np.all(DT[:ns] == dt0) and np.all(AL[:ns] == 1.0) and np.all(A[:ns] == 1.0) and np.all(B[:ns] == 0.0)
        w = DT[ns:] / DT[ns - 1:-1]
        assert np.all(w <= r * (1 + 1e-12)) and np.all(w > 0)
        uni = DT == dt
        assert uni[-1] and np.all(uni[np.argmax(uni):])               # graded first, then uniform to the end
        k = np.flatnonzero(uni)[1:]
        k = k[k >= ns]
        assert np.all(AL[k] == 1.5) and np.all(A[k] == 2.0) and np.all(B[k] == -0.5)   # BDF2, exactly
        np.testing.assert_allclose(AL[ns:], A[ns:] + B[ns:], rtol=1e-15)              # consistency (sum = alpha)
    with pytest.raises(ValueError):
        fj.early_schedule(3e-4, 2.0, 20, 1e-3, 1.02)                 # dt0 > dt
    with pytest.raises(ValueError):
        fj.early_schedule(3e-4, 2.0, 20, 1e-8, 2.5)                  # ratio beyond VSBDF2's zero-stability
    with pytest.raises(ValueError):
        fj.early_schedule(3e-4, 2.0, 0, 1e-8, 1.02)                  # no BE start
    with pytest.raises(ValueError):
        fj.early_schedule(3e-4, 1e-3, 20, 1e-8, 1.02)                # nothing left for the uniform part
    t, e, L = _cohort()
    with pytest.raises(ValueError):
        fj.make_loglik(t, e, L, sampling_dt=1.0, early=dict(dt0=1e-8, ratio=1.02, typo=1), **KW)


@pytest.mark.parametrize("sd", [None, 1.0])
def test_off_is_rc1_program(mods, sd):
    import numpy as np
    fj, rc = mods
    t, e, L = _cohort()
    a = rc.make_loglik(t, e, L, sampling_dt=sd, **KW)
    b = fj.make_loglik(t, e, L, sampling_dt=sd, early=None, **KW)
    z0 = np.log([2e-4, 0.03, 0.002, 0.5])
    for f_a, f_b in ((a[0], b[0]), (a[1], b[1])):
        assert f_a.lower(z0).as_text(debug_info=False) == f_b.lower(z0).as_text(debug_info=False)


def test_uniform_limit(mods):
    """dt0 = dt: no geometric steps, so the schedule IS the uniform grid (TAU to an ulp)."""
    import numpy as np
    fj, _ = mods
    t, e, L = _cohort()
    f_u, vg_u = fj.make_loglik(t, e, L, sampling_dt=1.0, **KW)
    f_g, vg_g = fj.make_loglik(t, e, L, sampling_dt=1.0, early=dict(dt0=KW["dt"], ratio=1.1), **KW)
    rng = np.random.default_rng(3)
    z0 = np.log([2e-4, 0.03, 0.002, 0.5])
    for _ in range(3):
        z = z0 + rng.uniform(-0.5, 0.5, 4)
        vu, gu = vg_u(z)
        vg, gg = vg_g(z)
        assert abs(float(f_g(z)) - float(f_u(z))) <= 1e-10 * abs(float(f_u(z)))
        np.testing.assert_allclose(np.asarray(gg), np.asarray(gu), rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("hazard", [False, True])
def test_graded_vjp_equals_forward_mode(mods, hazard):
    import jax
    import numpy as np
    fj, _ = mods
    t, e, L = _cohort()
    kw = dict(KW, sampling_dt=1.0, hazard=hazard, early=dict(dt0=1e-7, ratio=1.05))
    f, vg = fj.make_loglik(t, e, L, **kw)
    _fc, vgc = fj.make_loglik(t, e, L, q_budget_bytes=1, **kw)        # force the checkpointed sweep
    jf = jax.jit(jax.jacfwd(f))
    rng = np.random.default_rng(5)
    z0 = np.log([2e-4, 0.03, 0.002, 0.5] + ([0.2 / L] if hazard else []))
    for _ in range(3):
        z = z0 + rng.uniform(-0.4, 0.4, z0.size)
        v, g = vg(z)
        vc, gc = vgc(z)
        gf = np.asarray(jf(z))
        assert float(v) == float(f(z)) or abs(float(v) - float(f(z))) <= 1e-12 * abs(float(v))
        np.testing.assert_allclose(np.asarray(g), gf, rtol=1e-6, atol=1e-8)
        assert float(vc) == float(v) and np.array_equal(np.asarray(gc), np.asarray(g))


def test_first_bin_death_is_resolved(mods):
    """The defect: a death recorded in the first sampling bin of an E. coli-like cohort (delta/L ~ 1e-4) at
    het-real-ecoli-c0-s1's MLE corner and one of its Phase A edge nodes. Only the early part of the march
    matters for these deaths (the march is causal), so tau_max is short. Reference: uniform dt = 5e-8
    (measured: 3e-5 nats from dt = 2e-8 at this node)."""
    import numpy as np
    fj, _ = mods
    L, delta = 5.281252405028264, 0.000524838
    t = np.array([0.5 * delta, 0.0105])                               # the first-bin clamp; tau 5e-5 and 2e-3
    e = np.ones(t.size, int)
    z_hat = np.log([59.86684052069404, 999.9999999999998, 70.38804292048007, 0.001])
    z_node = np.array([3.6316057526174657, 6.39398801350146, 3.7792954674146597, -6.907755278982137])
    common = dict(N=900, tau_max=0.02, Gtol=1e-11, sampling_dt=delta)

    def dll(**kw):
        f, _vg = fj.make_loglik(t, e, L, **dict(common, **kw))
        return float(f(z_node)) - float(f(z_hat))

    ref = dll(dt=5e-8)
    floor = dll(dt=3e-4)
    graded = dll(dt=3e-4, early=EARLY)
    assert abs(floor - ref) > 1.0, (floor, ref)                       # the uniform floor grid: ~1.9 nats off
    assert abs(graded - ref) < 0.01, (graded, ref)                    # the graded start: resolved


def test_grid_floor_early_only_where_the_floor_binds():
    import os
    if not os.path.isdir(os.path.join(T.ROOT, "data", "library", "v1", "cohorts")):
        pytest.skip("library v1 cohorts are not built in this checkout")
    out = T.run_py("""
import sys, json, os, importlib.util
import numpy as np
from srbench import npcompat
def load(k, p):
    s = importlib.util.spec_from_file_location(k, p); m = importlib.util.module_from_spec(s)
    sys.modules[k] = m; s.loader.exec_module(m); return m
est = load("fe.new", os.path.join("methods", "mle_profile_jax2", "estimator.py"))
from srbench.library.load import Library
lib = Library(version="v1")
res = {}
for cid in ("het-real-ecoli-c0-s1", "pure-real-celegans-c1-s0", "pure-k0.01-p00-c0-s0", "realfull-real-Sweden_F_1910_homo-c0-s0"):
    obs = lib.observation(cid)
    t, e, d = np.asarray(obs.ages, float), np.asarray(obs.events, int), obs.delta
    off = est._grid(t, e, d); dflt = est._grid(t, e, d, floor_early=False); on = est._grid(t, e, d, floor_early=True)
    res[cid] = dict(binds=off[3]["binds"], off_eq_default=repr(off) == repr(dflt), on_eq_off=repr(on) == repr(off),
                    on_kw_early=on[1].get("early"), on_floor_early=on[3].get("early"),
                    on_rest_eq=repr({k: v for k, v in on[1].items() if k != "early"}) == repr(off[1]),
                    sg=repr(est.solver_grid(t, e, d, floor_early=True)) == repr(on[:3]),
                    fi=repr(est.floor_info(t, e, d, floor_early=True)) == repr(on[3]))
print(json.dumps(res))
""", timeout=900)
    assert out["_rc"] == 0, out["_stderr"]
    for cid in ("het-real-ecoli-c0-s1", "pure-real-celegans-c1-s0"):
        r = out[cid]
        assert r["binds"] and r["off_eq_default"] and not r["on_eq_off"] and r["on_rest_eq"], (cid, r)
        assert r["on_kw_early"] == {"dt0": 1e-8, "ratio": 1.02} == r["on_floor_early"], (cid, r)
        assert r["sg"] and r["fi"], (cid, r)
    for cid in ("pure-k0.01-p00-c0-s0", "realfull-real-Sweden_F_1910_homo-c0-s0"):
        r = out[cid]
        assert not r["binds"] and r["off_eq_default"] and r["on_eq_off"] and r["on_kw_early"] is None, (cid, r)
