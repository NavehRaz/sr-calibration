"""Spec 004's pilot P4, a JAX Langevin prototype (tasks T073; research R19; OD-7 (a), decided). A prototype: never
imported by `src`. The phase `p4-jax` in tools/acceptance_004.py runs it in a subprocess (cap 1 CPU-h).

**The kernel** (`make_kernel`): the numba kernel's scheme (sraging.langevin.kernels): an Euler step of length dt
from age k dt, the wall after it (clamp max(0, .) or reflect |.|), the kill at X >= xc and, for the bridge kernel, the
Brownian-bridge test exp(-2 (xc - x)(xc - x~) / (2 epsilon dt)) under SRK's e^-30 cutoff; the competing external time
t_ext (drawn outside, as the numba path does). `lax.scan` over the steps, `vmap` over the individuals, jax.random for
the noise (so its draws are not the numba kernel's, only its law). A scan runs every step for everyone: the dead carry
their state to t_max, the price of the design measured here.
**Single-threaded XLA:** `run` sets XLA_FLAGS (--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1)
before it imports jax, so the CPU compared is one core's, as the numba kernel's (workers = 1).
**The comparison** (`compare`): one preset, 20 paired repeats (each repeat a numba draw and a JAX draw of the same
configuration with their own keys, alternating which goes first), compilation apart; the ratio CPU_jax / CPU_numba
per pair, the mean log ratio with its one-sided 95 % upper bound (Student t); the decision: a JAX spec opens iff the
upper bound of the ratio is <= 1.2. The two samples' two-sample KS is recorded beside it (the same law, or the
ratio compares different work).

    python tools/p4_jax_langevin_004.py run --preset mice_F --n 25000 --repeats 20 --out result.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

REPEATS = 20
THRESHOLD = 1.2                     # OD-7 (a): the upper 95 % bound of the ratio
CUTOFF = 30.0                       # sraging.langevin.kernels.CUTOFF
XLA_ONE_CORE = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def make_kernel(nsteps: int, dt: float, t_max: float, wall: str = "clamp", bridge: bool = True):
    """jit(vmap(one individual's lax.scan over nsteps)) -> (times, kinds) for arrays of parameters and keys."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    w_clamp = wall == "clamp"

    def one(eta, beta, eps, kappa, xc, t_ext, key):
        sd = jnp.sqrt(2.0 * eps * dt)
        var = 2.0 * eps * dt

        def body(carry, k):
            x, alive, T, kind, key = carry
            key, k1, k2 = jax.random.split(key, 3)
            t1 = (k + 1) * dt
            ext = alive & (t_ext <= t1)                        # the competing death comes first (numba's order)
            xn = x + dt * (eta * (k * dt) - beta * x / (x + kappa)) + sd * jax.random.normal(k1)
            xn = jnp.maximum(xn, 0.0) if w_clamp else jnp.abs(xn)
            dx1, dx2 = xc - x, xc - xn
            evalp = (x < xc) & (xn < xc) & (x > 0.0) & (2.0 * dx1 * dx2 < CUTOFF * var)
            p = jnp.where(evalp, jnp.exp(-2.0 * dx1 * dx2 / var), -1.0)
            cross = (xn >= xc) | (bridge & (p >= 0.0) & (jax.random.uniform(k2) < p))
            sr = alive & ~ext & cross
            T = jnp.where(ext, t_ext, jnp.where(sr, t1, T))
            kind = jnp.where(ext, 2, jnp.where(sr, 1, kind))
            alive = alive & ~ext & ~sr
            return (jnp.where(alive, xn, x), alive, T, kind, key), None

        init = (jnp.asarray(0.0), jnp.asarray(True), jnp.asarray(float(t_max)), jnp.asarray(0), key)
        (x, alive, T, kind, _), _ = lax.scan(body, init, jnp.arange(nsteps))
        T = jnp.where(alive & (t_ext <= t_max), t_ext, T)        # an external death after the last full step
        kind = jnp.where(alive & (t_ext <= t_max), 2, kind)
        return T, kind

    return jax.jit(jax.vmap(one))


def configuration(preset: str, n: int, *, dt=None, wall="clamp", key=1):
    """The numba side's LangevinConfig and prepared state (its t_max and parameter draws) for the preset."""
    import presets_004 as PR
    from sraging.langevin import sample as LS
    cfg = LS.LangevinConfig(params=PR.preset(preset), n=int(n), key=int(key), dt=float(dt or PR.base_dt(preset)),
                            wall=wall, error="ks")
    return cfg, LS.prepare(cfg)


def numba_draw(cfg, st) -> tuple:
    from sraging.langevin import sample as LS
    c0 = time.process_time()
    t, kd = LS.run_rung(cfg, st["arrays"], st["t_max"], 1)[:2]
    return t, kd, time.process_time() - c0


def jax_draw(kernel, st, seed) -> tuple:
    import jax
    import jax.numpy as jnp
    a = st["arrays"]
    n = a["eta"].size
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), n)
    args = [jnp.asarray(a[k]) for k in ("eta", "beta", "epsilon", "kappa", "xc", "t_ext")]
    c0 = time.process_time()
    T, kind = kernel(*args, keys)
    T.block_until_ready()
    return np.asarray(T), np.asarray(kind), time.process_time() - c0


def ratio_ci(cpu_jax, cpu_numba) -> dict:
    """The paired log ratios' mean and one-sided 95 % upper bound (Student t), as ratios; the OD-7 decision."""
    from scipy.stats import t as tdist
    lr = np.log(np.asarray(cpu_jax, float) / np.asarray(cpu_numba, float))
    k = lr.size
    m = float(lr.mean())
    up = m + float(tdist.ppf(0.95, k - 1)) * float(lr.std(ddof=1)) / math.sqrt(k) if k > 1 else math.inf
    return {"n_pairs": int(k), "mean_log_ratio": m, "ratio_geomean": math.exp(m), "upper_95_ratio": math.exp(up),
            "threshold": THRESHOLD, "jax_spec_opens": math.exp(up) <= THRESHOLD}


def compare(preset: str, n: int, repeats: int = REPEATS, *, dt=None, wall="clamp", key=1) -> dict:
    """P4: compilation apart, then `repeats` paired draws (the order alternating), the ratio's CI and the decision."""
    from scipy.stats import ks_2samp
    cfg, st = configuration(preset, n, dt=dt, wall=wall, key=key)
    nsteps = int(math.floor(st["t_max"] / cfg.dt + 1e-9))
    c0 = time.process_time()
    kernel = make_kernel(nsteps, cfg.dt, st["t_max"], wall=wall)
    jax_draw(kernel, st, 0)                                       # compiles (and runs once): apart
    compile_cpu = time.process_time() - c0
    numba_draw(cfg, st)                                           # numba's compilation, apart
    cj, cn, ks = [], [], []
    for r in range(int(repeats)):
        if r % 2 == 0:
            tn, kn, a = numba_draw(cfg, st)
            tj, kj, b = jax_draw(kernel, st, 1000 + r)
        else:
            tj, kj, b = jax_draw(kernel, st, 1000 + r)
            tn, kn, a = numba_draw(cfg, st)
        cn.append(a)
        cj.append(b)
        if r == 0:
            res = ks_2samp(tn, tj)
            ks = {"D": float(res.statistic), "p_value": float(res.pvalue), "n": int(n)}
    return {"preset": preset, "n": int(n), "dt": cfg.dt, "t_max": st["t_max"], "nsteps": nsteps, "wall": wall,
            "repeats": int(repeats), "cpu_numba_s": cn, "cpu_jax_s": cj, "compile_cpu_s": compile_cpu,
            "two_sample_ks": ks, "xla_flags": os.environ.get("XLA_FLAGS"), **ratio_ci(cj, cn)}


def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n", type=int, default=25_000)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ["XLA_FLAGS"] = XLA_ONE_CORE                        # before jax is imported (it is not, above)
    out = compare(a.preset, a.n, a.repeats)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(run())
