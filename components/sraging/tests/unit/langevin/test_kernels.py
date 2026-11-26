"""The numba kernels (tasks T058; research R13, R30): one step dt with the age k dt computed from k, the Brownian-bridge
crossing probability after the wall, the death at the end of its step, a competing external death time, survivors
censored at t_max, and the wall at 0 (D-410: SRK's clamp by default, reflect selectable)."""
import inspect
import math

import numpy as np
import pytest

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

from sraging import SRParams  # noqa: E402
from sraging.langevin import kernels as K  # noqa: E402
from sraging.langevin import rng as RNG  # noqa: E402

KEY = 11
NO_REC = np.zeros(0, dtype=np.int64)
P = dict(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, xc=1.0)


def arrays(n, t_ext=math.inf, **over):
    p = {**P, **over}
    return {k: np.full(n, float(p[k])) for k in ("eta", "beta", "epsilon", "kappa", "xc")} | {"t_ext": np.full(n, t_ext)}


def run(n=200, dt=0.5, t_max=400.0, wall="clamp", kernel="bridge", split=1, **kw):
    a = arrays(n, **kw)
    idx = np.arange(n)
    return K.run_chunk(0, n, RNG.hash64_array(KEY, idx), RNG.hash64_array(KEY, idx, "bridge"), a["eta"], a["beta"],
                       a["epsilon"], a["kappa"], a["xc"], a["t_ext"], dt, t_max, K.WALLS[wall], kernel == "bridge",
                       split, NO_REC)[:3]


def test_the_age_at_step_k_is_k_dt_from_k_with_no_substep():
    dt, k = 0.37, 1_000_003
    x, xi = 0.4, -0.3
    want = x + dt * (P["eta"] * (k * dt) - P["beta"] * x / (x + P["kappa"])) + math.sqrt(2 * P["epsilon"] * dt) * xi
    assert K.step(x, k, dt, P["eta"], P["beta"], P["epsilon"], P["kappa"], xi, K.WALLS["clamp"]) == want
    names = set(inspect.signature(K.run_chunk.py_func).parameters)
    assert not names & {"time_step_multiplier", "nsteps", "step_size", "m", "multiplier"}


def test_the_bridge_probability_its_cutoff_and_when_it_is_evaluated():
    eps, dt, xc = 8e-3, 0.5, 1.0
    var = 2 * eps * dt
    x, x2 = 0.95, 0.97
    assert K.bridge_p(x, x2, xc, eps, dt) == math.exp(-2 * (xc - x) * (xc - x2) / var)
    assert K.bridge_p(0.2, 0.25, xc, eps, dt) == -1.0               # 2 dx1 dx2 >= 30 var: the e^-30 cutoff
    assert 2 * 0.8 * 0.75 >= 30 * var
    assert K.bridge_p(0.0, 0.97, xc, eps, dt) == -1.0               # x = 0: not evaluated (SRK's x > 0 guard)
    assert K.bridge_p(x, 1.01, xc, eps, dt) == -1.0                 # an end at or above xc: a direct crossing
    assert K.bridge_p(1.0, 0.9, xc, eps, dt) == -1.0


def test_the_death_time_is_the_end_of_the_kill_step():
    """With epsilon = 0 the path is the Euler recursion itself: the death is at (k + 1) dt of the first k with x >= xc."""
    dt = 0.5
    t, kind, _ = run(n=3, dt=dt, epsilon=0.0)
    x, k = 0.0, 0
    while True:
        x = max(0.0, x + dt * (P["eta"] * k * dt - P["beta"] * x / (x + P["kappa"])))
        if x >= P["xc"]:
            break
        k += 1
    assert np.all(t == (k + 1) * dt) and np.all(kind == K.SR_DEATH)


def test_the_hazard_is_a_competing_time_that_stops_the_path():
    t, kind, _ = run(n=50, t_ext=12.25)
    assert np.all(t == 12.25) and np.all(kind == K.EXTERNAL)
    t2, kind2, _ = run(n=50, t_ext=math.inf)                                                     # the twin
    assert np.all(kind2 == K.SR_DEATH) and np.all(t2 > 12.25)
    assert np.all(t2 / 0.5 == np.round(t2 / 0.5))                                                 # on the dt grid


def test_survivors_at_t_max_are_censored_there():
    t, kind, x_end = run(n=50, t_max=20.0)
    assert np.all(t == 20.0) and np.all(kind == K.CENSORED) and np.all((x_end >= 0) & (x_end < 1.0))


def test_a_draw_is_a_function_of_the_key_and_the_individual_only():
    a = run(n=40)
    b = run(n=40)
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a, b))
    n = 40
    arr = arrays(n)
    idx = np.arange(n)
    part = K.run_chunk(10, 25, RNG.hash64_array(KEY, idx), RNG.hash64_array(KEY, idx, "bridge"), arr["eta"],
                       arr["beta"], arr["epsilon"], arr["kappa"], arr["xc"], arr["t_ext"], 0.5, 400.0, 0, True, 1,
                       NO_REC)
    assert part[0].tobytes() == a[0][10:25].tobytes()                                             # any chunk


def test_the_euler_kernel_draws_no_bridge_and_dies_later_on_average():
    tb, _, _ = run(n=400, kernel="bridge")
    te, _, _ = run(n=400, kernel="euler")
    assert np.all(te >= tb) and te.mean() > tb.mean()                # the same normals: the bridge only adds kills


def test_the_coupled_half_step_rung_uses_each_polar_pair_twice():
    """split = 2 runs dt/2 on the normals the dt path sums in pairs, (xi_a + xi_b)/sqrt 2: the two rungs follow one
    Brownian path, so their death times are close, and the dt rung's own draw is unchanged by the rung's existence."""
    t1, _, _ = run(n=400, dt=0.5, split=1)
    t2, _, _ = run(n=400, dt=0.5, split=2)                            # the kernel steps h = dt / 2
    assert np.median(np.abs(t1 - t2)) <= 1.0 and not np.array_equal(t1, t2)
    t3, _, _ = run(n=400, dt=0.25, split=1)                           # an uncoupled dt/2 sample is another draw
    assert np.median(np.abs(t1 - t2)) < np.median(np.abs(t1 - t3)) / 5


def test_d410_default_wall_is_clamp():
    """D-410 (ported-as-is): the default wall is SRK's clamp max(0, .), which leaves exact zeros in young-age damage;
    reflect (|.|) leaves none."""
    from sraging.langevin import LangevinConfig
    assert LangevinConfig(params=SRParams(**{k: P[k] for k in ("eta", "beta", "epsilon", "kappa")}), n=1, key=1,
                          dt=0.5).wall == "clamp"
    _, kind, x_clamp = run(n=2000, t_max=2.0, wall="clamp")
    _, _, x_refl = run(n=2000, t_max=2.0, wall="reflect")
    assert np.all(kind == K.CENSORED)
    assert np.sum(x_clamp == 0.0) > 100 and np.sum(x_refl == 0.0) == 0
    assert K.step(0.0, 0, 0.5, 1e-9, 0.0, 8e-3, 0.1, -1.0, K.WALLS["clamp"]) == 0.0
    assert K.step(0.0, 0, 0.5, 1e-9, 0.0, 8e-3, 0.1, -1.0, K.WALLS["reflect"]) > 0.0
