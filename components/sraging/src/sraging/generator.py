"""Synthetic cohorts drawn from the SR model: sraging's replacement of `SRK.benchmark.cohorts.materialize` (spec 003d).

    from sraging.generator import CohortSpec, materialize
    co = materialize(CohortSpec(name=..., rho_eta=..., rho_beta=..., rho_eps=..., kappa=..., n=..., L_phys=...,
                                seed=..., delta=..., censor_frac=0.0))
    co.death_times, co.events          # recorded ages (multiples of delta, ascending) and 1 death / 0 censored

What a cohort is (specs/003d-cohort-generator/research.md):
- R2. The survival curve is the engine's uniform march (BDF2 after `nstart` backward-Euler steps, numerics "v1"), in
  the dimensionless tau-mean = 1 convention (SRK's): theta = (rho_eta, rho_eps, rho_beta y/(y + kappa)) at the faces.
  The CDF sampled is the march's absorbed mass F, piecewise linear between grid nodes (R4).
- R5. One PCG64 generator per cohort, `default_rng(seed)`, drawing only `.random(n)`: the death uniforms first
  (SRK's homogeneous stream, so a homogeneous cohort is an individual-level twin of its library-v5 cohort), then the
  hazard uniforms, then the heterogeneity uniforms, each only when the feature is on.
- R6. Heterogeneity in xc: s = 1 + het_var z with z standard normal truncated to s > 0; K node curves uniformly
  spaced in z, at (rho_eta/s, rho_beta/s, rho_eps/s^2, kappa/s). An individual's s is continuous (its own uniform);
  its death time inverts the CDF interpolated linearly in z between the two bracketing nodes, so the cohort's law is
  exactly the hat-weighted mixture of node curves (`_het_nodes`), which the convergence and G-tests compute.
- External hazard: a competing exponential at rate exp(-external_hazard) per age unit, by inverse CDF.
- R7. Censoring at one administrative time C: the delta-grid point nearest the model's observed survival quantile
  S_obs(t*) = censor_frac. Everyone with t >= C is recorded at C, censored; no death is recorded at C.
- Recording: floor(t / delta) * delta, no clamp, so first-bin deaths are at 0.
- R8. time_range = (lo, hi) is the tester's `apply_window`, closed at both ends: recorded ages below lo are dropped,
  ages above hi are censored at hi, a death recorded exactly at hi stays a death.

The grid error is stated with every cohort (constitution IX): lambda = 2 n KL(P_gen || P_ref) and dlnCV of the
binned outcome distribution against a grid with N and K doubled and dt halved, from the version's point table when
the configuration is in it, else computed. Every unsupported field or value is refused by name (constitution VIII).
"""
from __future__ import annotations

import datetime as _dt
import functools
import json
import math
import os
import platform
import sys
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from sraging.compat.v033 import runtime as _runtime
from sraging.errors import UnsupportedOptionError

# x64 and one XLA thread before any JAX array is built here: the parameters of the first curve in a process were built
# as float32 before the engine's import enabled x64 (found 2026-10-14; test_the_first_curve_in_a_fresh_process_...).
_runtime.configure()

GENERATOR_VERSION = "sraging-gen-1"   # spec 003d T035: the version tagged and handed to the tester
# Frozen by the refinement sweep (spec 003d T021: tests/acceptance/results/sweep_003d.json, run 2 at 4dbbc12, PASS on
# all 569 library configurations); a different grid is a different version.
GRID = {"N": 6000, "dt": 2e-4, "tau_max": 30.0, "Gtol": 1e-12, "nstart": 20, "K": 161}
HET_U_TAIL = 1e-9            # R6: the node range is [max(ndtri(u), -(1 - s_min)/het_var), ndtri(1 - u)]
HET_S_MIN = 0.05
CENSOR_SUPPORT_MIN = 1e-12   # R7: S_obs(C) below this means C lies beyond the support
# Heterogeneous configurations are accepted at n <= HET_N_CAP (Naveh, 2026-10-13, after the sweep pilot): above it the
# generator's node error may be detectable, and every such cohort says so in its grid error (spec 003d SC-001).
HET_N_CAP = 20000
TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generator_point_table.json")

_REQUIRED = ("name", "rho_eta", "rho_beta", "rho_eps", "kappa", "n", "L_phys", "seed", "delta")
_DEFAULTS = {"censor_frac": 0.0, "external_hazard": None, "het_param": None, "het_var": 0.0, "time_range": None}
FIELDS = _REQUIRED + tuple(_DEFAULTS)


# ----------------------------------------------------------------------------------------------------- the spec
@dataclass(frozen=True, init=False)
class CohortSpec:
    """The tester's fields, by keyword only (contracts/python-api.md). Every other field is refused by name."""
    name: str
    rho_eta: float
    rho_beta: float
    rho_eps: float
    kappa: float
    n: int
    L_phys: float
    seed: int
    delta: float
    censor_frac: float
    external_hazard: float | None
    het_param: str | None
    het_var: float
    time_range: tuple | None

    def __init__(self, *args, **kw):
        if args:
            raise UnsupportedOptionError("positional", "CohortSpec takes keywords only (constitution VIII: no "
                                         "positional parameter vectors); pass each field by name")
        for k in kw:
            if k not in FIELDS:
                raise UnsupportedOptionError(k, f"CohortSpec has no field {k!r} (SRK's defect, defect_kw, group and "
                                             f"notes are not generator inputs); fields: {', '.join(FIELDS)}")
        for k in _REQUIRED:
            if k not in kw:
                raise UnsupportedOptionError(k, f"CohortSpec needs {k!r}")
        v = {**_DEFAULTS, **kw}
        _validate(v)
        for k in FIELDS:
            object.__setattr__(self, k, v[k])

    @property
    def point(self) -> dict:
        return {"rho_eta": self.rho_eta, "rho_beta": self.rho_beta, "rho_eps": self.rho_eps, "kappa": self.kappa}

    @property
    def heterogeneous(self) -> bool:
        return self.het_param == "xc" and self.het_var > 0


def _num(v, k, cond, why):
    x = v[k]
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or not cond(float(x)):
        raise UnsupportedOptionError(k, f"{k}={x!r}: {why}")
    v[k] = float(x)


def _int(v, k, lo):
    x = v[k]
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or int(x) < lo:
        raise UnsupportedOptionError(k, f"{k}={x!r}: must be an integer >= {lo}")
    v[k] = int(x)


def _on_grid(x, delta):
    r = x / delta
    return abs(r - round(r)) <= 1e-9 * max(1.0, abs(r))


def _validate(v):
    if not isinstance(v["name"], str) or not v["name"]:
        raise UnsupportedOptionError("name", "name must be a non-empty string")
    for k in ("rho_eta", "rho_eps", "kappa"):
        _num(v, k, lambda x: x > 0, "must be finite and > 0")
    _num(v, "rho_beta", lambda x: x >= 0, "must be finite and >= 0")
    _int(v, "n", 1)
    _num(v, "L_phys", lambda x: x > 0, "must be finite and > 0")
    _int(v, "seed", 0)
    if v["delta"] is None:
        raise UnsupportedOptionError("delta", "delta=None (continuous ages) is not supported: every cohort is "
                                     "recorded on a delta grid (spec 003d FR-002)")
    _num(v, "delta", lambda x: x > 0, "must be finite and > 0")
    _num(v, "censor_frac", lambda x: 0.0 <= x < 1.0, "must be in [0, 1)")
    if v["external_hazard"] is not None:
        _num(v, "external_hazard", lambda x: True, "must be finite (the rate is exp(-external_hazard))")
    _num(v, "het_var", lambda x: x >= 0, "must be finite and >= 0 (the SD of s)")
    if v["het_param"] is not None and v["het_param"] != "xc":
        raise UnsupportedOptionError("het_param", f"het_param={v['het_param']!r}: only 'xc' is supported")
    if v["het_param"] is None and v["het_var"] > 0:
        raise UnsupportedOptionError("het_param", "het_var > 0 needs het_param='xc'")
    tr = v["time_range"]
    if tr is not None:
        try:
            lo, hi = (float(tr[0]), float(tr[1]))
            ok = len(tr) == 2 and math.isfinite(lo) and math.isfinite(hi) and 0 <= lo < hi
        except (TypeError, ValueError, IndexError):
            ok = False
        if not ok:
            raise UnsupportedOptionError("time_range", f"time_range={tr!r}: must be (lo, hi) with 0 <= lo < hi")
        if not (_on_grid(lo, v["delta"]) and _on_grid(hi, v["delta"])):
            raise UnsupportedOptionError("time_range", f"time_range={tr!r} is off the delta={v['delta']} grid")
        v["time_range"] = (lo, hi)


# ----------------------------------------------------------------------------------------------------- the curve
@functools.lru_cache(maxsize=None)
def _march(N, nsteps, dt, nstart, Gtol):
    """The engine's compiled value march for one static grid (compilation only; it changes no numerics)."""
    import jax
    from sraging.engine import stepper as ST
    return jax.jit(ST.make_march(ST.MarchSpec(N=N, nsteps=nsteps, dt=dt, nstart=nstart, Gtol=Gtol, carry_F=True),
                                 "value"))


def _curve(point: dict, grid: dict) -> dict:
    """The curve for (point, grid), memoised per process (a pure cache: the same arrays a recomputation gives; callers
    never mutate them). A library build then solves each point's curves (161 nodes for a heterogeneous point) once,
    not once per cohort."""
    return _curve_cached(tuple(sorted(point.items())), tuple(sorted(grid.items())))


@functools.lru_cache(maxsize=200)
def _curve_cached(point_items, grid_items):
    return _solve_curve(dict(point_items), dict(grid_items))


def _solve_curve(point: dict, grid: dict) -> dict:
    """tau, F (absorbed mass) and S on the grid up to the march's stop, refusing a horizon that does not absorb."""
    import jax.numpy as jnp
    N, dt, tau_max, Gtol = int(grid["N"]), float(grid["dt"]), float(grid["tau_max"]), float(grid["Gtol"])
    nsteps = int(round(tau_max / dt))
    yf = (np.arange(N) + 0.5) / N
    theta = (jnp.asarray(point["rho_eta"]), jnp.asarray(point["rho_eps"]),
             jnp.asarray(point["rho_beta"] * yf / (yf + point["kappa"])))
    S, F = _march(N, nsteps, dt, int(grid["nstart"]), Gtol)(theta)
    S, F = np.asarray(S, float), np.asarray(F, float)
    i = np.arange(nsteps + 1)
    hit = np.flatnonzero((S < Gtol) & (i * dt > 0.3))
    if not hit.size:
        raise UnsupportedOptionError("tau_max", f"the survival at tau_max={tau_max} is {S[-1]:.3g} >= Gtol={Gtol:g}: "
                                     f"the horizon does not absorb, so draws would be truncated")
    k = int(hit[0])
    F, S = F[: k + 1], S[: k + 1]
    return {"tau": np.arange(k + 1) * dt, "F": F, "S": S,
            "max_abs_S_plus_F_minus_1": float(np.max(np.abs(S + F - 1.0)))}


def curve_moments(c: dict) -> dict:
    """The continuous (unbinned) mean and CV of a curve's first-passage time, by the trapezoid rule on S."""
    tau, S = c["tau"], c["S"]
    mean = float(np.trapezoid(S, tau))
    m2 = float(np.trapezoid(2.0 * tau * S, tau))
    return {"mean": mean, "cv": math.sqrt(max(m2 - mean * mean, 0.0)) / mean}


# ------------------------------------------------------------------------------------------------- heterogeneity
def _het_z(u, sigma):
    """z from u by the quantile of the standard normal truncated to s = 1 + sigma z > 0."""
    a = ndtr(-1.0 / sigma)
    return ndtri(a + np.asarray(u, float) * (1.0 - a))


def _het_nodes(sigma, K) -> dict:
    """K nodes uniform in z, and the exact hat-function mixture weights of the truncated normal over them; the mass
    beyond either end goes to the end node."""
    za = max(float(ndtri(HET_U_TAIL)), -(1.0 - HET_S_MIN) / sigma)
    zb = float(ndtri(1.0 - HET_U_TAIL))
    z = np.linspace(za, zb, int(K))
    Z = 1.0 - ndtr(-1.0 / sigma)
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    P = np.diff(ndtr(z))                      # mass of each interval
    M1 = phi[:-1] - phi[1:]                   # integral of z phi(z) over each interval
    dz = np.diff(z)
    W = np.zeros(z.size)
    W[:-1] += (z[1:] * P - M1) / dz           # the falling hat of the left node
    W[1:] += (M1 - z[:-1] * P) / dz           # the rising hat of the right node
    W[0] += ndtr(z[0]) - ndtr(-1.0 / sigma)
    W[-1] += 1.0 - ndtr(z[-1])
    return {"z": z, "s": 1.0 + sigma * z, "W": W / Z}


def _node_point(point: dict, s: float) -> dict:
    """xc -> xc s in dimensionless terms (SRK cohorts.py 499-501)."""
    return {"rho_eta": point["rho_eta"] / s, "rho_beta": point["rho_beta"] / s, "rho_eps": point["rho_eps"] / s ** 2,
            "kappa": point["kappa"] / s}


def _bracket(z, nodes_z):
    """(j, w, n_outside): the interval of each z and its weight on the right node; outside the range, the end node."""
    z = np.asarray(z, float)
    K = nodes_z.size
    dz = nodes_z[1] - nodes_z[0]
    pos = (z - nodes_z[0]) / dz
    j = np.clip(np.floor(pos), 0, K - 2).astype(int)
    w = np.clip(pos - j, 0.0, 1.0)
    below, above = z < nodes_z[0], z > nodes_z[-1]
    j = np.where(below, 0, np.where(above, K - 2, j))
    w = np.where(below, 0.0, np.where(above, 1.0, w))
    return j, w, int(below.sum() + above.sum())


# ----------------------------------------------------------------------------------------------------- the model
def _model(spec: CohortSpec, grid: dict) -> dict:
    """Node curves on one common tau grid (each padded with its final value), their mixture weights, and nodes."""
    if spec.heterogeneous:
        nodes = _het_nodes(spec.het_var, grid["K"])
        curves = [_curve(_node_point(spec.point, s), grid) for s in nodes["s"]]
        W = nodes["W"]
    else:
        nodes, curves, W = None, [_curve(spec.point, grid)], np.ones(1)
    L = max(c["F"].size for c in curves)
    Fm = np.vstack([np.concatenate([c["F"], np.full(L - c["F"].size, c["F"][-1])]) for c in curves])
    tau = np.arange(L) * float(grid["dt"])
    return {"tau": tau, "Fm": Fm, "W": W, "nodes": nodes,
            "max_abs_S_plus_F_minus_1": max(c["max_abs_S_plus_F_minus_1"] for c in curves)}


def _S_obs(m: dict, spec: CohortSpec, t):
    """The observed survival at physical times t: the node mixture's 1 - F at tau = t / L, times the hazard's."""
    tau = np.asarray(t, float) / spec.L_phys
    Fmix = sum(Wj * np.interp(tau, m["tau"], Fj) for Wj, Fj in zip(m["W"], m["Fm"]))
    S = np.clip(1.0 - Fmix, 0.0, 1.0)
    if spec.external_hazard is not None:
        S = S * np.exp(-math.exp(-spec.external_hazard) * np.asarray(t, float))
    return S


# ----------------------------------------------------------------------------------------------------- the draws
def _rng(seed):
    return np.random.default_rng(seed)


def _uniforms(spec: CohortSpec) -> dict:
    rng = _rng(spec.seed)
    u = {"death": rng.random(spec.n), "hazard": None, "het": None}
    if spec.external_hazard is not None:
        u["hazard"] = rng.random(spec.n)
    if spec.heterogeneous:
        u["het"] = rng.random(spec.n)
    return u


def _inverse(u, tau, Fm, j, w):
    """tau with (1 - w) F_j(tau) + w F_{j+1}(tau) = u, the CDFs piecewise linear between grid nodes; a u above the
    combined CDF's end is placed at the last node and counted."""
    u = np.asarray(u, float)
    K, L = Fm.shape
    jn = np.minimum(j + 1, K - 1)

    def cdf(idx):
        return (1.0 - w) * Fm[j, idx] + w * Fm[jn, idx]
    lo, hi = np.zeros(u.size, int), np.full(u.size, L, int)        # smallest idx with cdf(idx) >= u
    for _ in range(int(math.ceil(math.log2(L + 1))) + 1):
        act = lo < hi
        mid = (lo + hi) // 2
        below = cdf(np.minimum(mid, L - 1)) < u
        lo = np.where(act & below, mid + 1, lo)
        hi = np.where(act & ~below, mid, hi)
    beyond = lo >= L
    i = np.clip(lo, 1, L - 1)
    c0, c1 = cdf(i - 1), cdf(i)
    frac = np.where(c1 > c0, (u - c0) / np.where(c1 > c0, c1 - c0, 1.0), 1.0)
    t = tau[i - 1] + np.clip(frac, 0.0, 1.0) * (tau[i] - tau[i - 1])
    t = np.where(lo == 0, tau[0], t)
    t = np.where(beyond, tau[-1], t)
    return t, int(beyond.sum())


def _hazard_times(u, external_hazard):
    return -np.log1p(-np.asarray(u, float)) / math.exp(-external_hazard)


# --------------------------------------------------------------------------------------- censoring and recording
def _censor_time(S_obs, delta, f, t_end):
    """C: of the two delta-grid points around t* (S_obs(t*) = f, by bisection; S_obs non-increasing), the one whose
    S_obs is closer to f, never 0 (Naveh 2026-10-14: nearest in time missed the target by up to 0.33 where a sharp
    cohort dies within one or two coarse bins). Refuses a C beyond the support."""
    lo, hi = 0.0, float(t_end)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(S_obs(np.array([mid]))[0]) > f:
            lo = mid
        else:
            hi = mid
    k = math.floor(hi / delta)
    cands = [c for c in (k * delta, (k + 1) * delta) if c > 0.0]
    surv = [float(S_obs(np.array([c]))[0]) for c in cands]
    C = cands[int(np.argmin([abs(s - f) for s in surv]))]           # ties: the earlier
    if float(S_obs(np.array([C]))[0]) < CENSOR_SUPPORT_MIN:
        raise UnsupportedOptionError("censor_frac", f"censor_frac={f}: the censoring time C={C} lies beyond the "
                                     f"model's support (S_obs(C) < {CENSOR_SUPPORT_MIN:g})")
    return C


def _record(t, C, delta):
    """(ages, events): deaths at floor(t/delta) delta, below C; t >= C censored at C."""
    t = np.asarray(t, float)
    k = np.floor(t / delta)
    if C is None:
        return k * delta, np.ones(t.size, int)
    m = int(round(C / delta))
    cens = t >= C
    ages = np.where(cens, C, np.minimum(k, m - 1) * delta)
    return ages, np.where(cens, 0, 1).astype(int)


def _apply_time_range(ages, events, window):
    """The tester's apply_window (srbench/library/observation.py), closed at both ends."""
    lo, hi = float(window[0]), float(window[1])
    keep = ages >= lo
    ages, events = ages[keep], events[keep]
    if not ages.size:
        raise UnsupportedOptionError("time_range", f"time_range={window} keeps no one")
    past = ages > hi
    return np.where(past, hi, ages), np.where(past, 0, events), int(ages.size)


# ------------------------------------------------------------------------------------- binned law and grid error
def binned_distribution(spec: CohortSpec, grid: dict, model: dict | None = None, C: float | None = None) -> dict:
    """The exact law of a cohort's recorded outcome (before time_range): death cells at k delta and, with
    censoring, one censored cell at C. n-independent. `C` imposes a censoring time (a reference grid is compared
    under the generator's own C: C is the cohort's design, fixed per version, and a refinement can flip t* across a
    rounding edge)."""
    m = model or _model(spec, grid)
    d = spec.delta
    t_end = float(m["tau"][-1]) * spec.L_phys
    if spec.censor_frac > 0:
        if C is None:
            C = _censor_time(lambda t: _S_obs(m, spec, t), d, spec.censor_frac, t_end)
        kmax = int(round(C / d))
    else:
        C = None
        kmax = int(math.floor(t_end / d)) + 1
    edges = np.arange(kmax + 1) * d
    S = _S_obs(m, spec, edges)
    P = S[:-1] - S[1:]
    ages = edges[:-1]
    if C is not None:
        ages, P = np.append(ages, C), np.append(P, S[-1])
        events = np.append(np.ones(kmax, int), 0)
    else:
        P[-1] += S[-1]
        events = np.ones(kmax, int)
    return {"ages": ages, "events": events, "P": np.clip(P, 0.0, None), "C": C}


def binned_cv(b: dict) -> float:
    a, P = b["ages"], b["P"] / b["P"].sum()
    mean = float(np.dot(P, a))
    return math.sqrt(max(float(np.dot(P, a * a)) - mean * mean, 0.0)) / mean


def kl(P, Q) -> float:
    """KL(P || Q) over matching cells. Q is floored at 1e-300: an early cell where one grid's absorbed mass is
    exactly 0 and the other's is ~1e-20 then contributes ~1e-17, while a cell with real mass under P and none under
    Q still dominates (P = 1e-3 gives 0.69)."""
    P, Q = np.asarray(P, float), np.maximum(np.asarray(Q, float), 1e-300)
    m = P > 0
    return float(np.sum(P[m] * np.log(P[m] / Q[m])))


def kl_binned(P, Q, n, min_expected: float = 5.0) -> float:
    """KL(P || Q) after merging cells left to right until n Q >= min_expected (the remainder joins the last merged
    cell): the G-test's binning (spec 003d SC-003). A cell no test of n individuals could resolve then cannot dominate
    lambda: in the full sweep's run 1, far-tail cells with Q exactly 0 and P ~1e-8 (0.001 expected individuals at
    n = 100,000) gave lambda up to 25 through kl()'s floor (Naveh, 2026-10-14)."""
    Pm, Qm, p, q = [], [], 0.0, 0.0
    for a, b in zip(np.asarray(P, float), np.asarray(Q, float)):
        p, q = p + a, q + b
        if n * q >= min_expected:
            Pm.append(p), Qm.append(q)
            p = q = 0.0
    if Pm:
        Pm[-1] += p
        Qm[-1] += q
    else:
        return 0.0
    return kl(Pm, Qm)


def reference_grid(grid: dict) -> dict:
    return {**grid, "N": 2 * int(grid["N"]), "dt": float(grid["dt"]) / 2.0, "K": 2 * int(grid["K"])}


def configuration_key(spec: CohortSpec) -> list:
    """What fixes a cohort's law: the point, delta/L, censoring, h L and heterogeneity."""
    hL = None if spec.external_hazard is None else math.exp(-spec.external_hazard) * spec.L_phys
    return [spec.rho_eta, spec.rho_beta, spec.rho_eps, spec.kappa, spec.delta / spec.L_phys, spec.censor_frac, hL,
            spec.het_var if spec.heterogeneous else 0.0]


def grid_error(spec: CohortSpec, grid: dict, model: dict | None = None) -> dict:
    """KL and dlnCV of the binned law on `grid` against the reference grid (N, K doubled; dt halved), the reference
    under the generator's own censoring time."""
    bg = binned_distribution(spec, grid, model)
    br = binned_distribution(spec, reference_grid(grid), None, bg["C"])
    n = min(bg["P"].size, br["P"].size)
    if bg["P"].size != br["P"].size:              # the support's last cell may differ by the march's stop
        Pg = np.append(bg["P"][: n - 1], bg["P"][n - 1:].sum())
        Pr = np.append(br["P"][: n - 1], br["P"][n - 1:].sum())
    else:
        Pg, Pr = bg["P"], br["P"]
    return {"kl": kl_binned(Pg, Pr, spec.n), "dlncv": math.log(binned_cv(bg) / binned_cv(br))}


@functools.lru_cache(maxsize=4)
def _table(path, version, grid_items):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    if d.get("generator_version") != version or tuple(sorted(d.get("grid", {}).items())) != grid_items:
        return None
    return {json.dumps(r["key"]): r for r in d["rows"]}


def _grid_error_for(spec, grid, model):
    rows = _table(TABLE_PATH, GENERATOR_VERSION, tuple(sorted(grid.items())))
    row = rows.get(json.dumps(configuration_key(spec))) if rows else None
    if row is not None:
        kl_, d, src = row["kl"], row["dlncv"], "the version's point table"
    else:
        ge = grid_error(spec, grid, model)
        kl_, d, src = ge["kl"], ge["dlncv"], "computed against the reference grid"
    # the bars of spec 003d SC-001: homogeneous lambda <= 0.05 and 4/3 |dlnCV| <= 1/2 sqrt(0.1/(2n)); heterogeneous
    # (Naveh 2026-10-14) lambda <= 0.2 at min(n, HET_N_CAP) and the tester's rule |dlnCV| <= sqrt(0.1/(2n))
    het = spec.heterogeneous
    out = {"lambda": 2.0 * spec.n * kl_, "dlncv": d,
           "lambda_tolerance": 0.2 if het else 0.05,
           "dlncv_bound": math.sqrt(0.1 / (2.0 * spec.n)) if het else 0.5 * math.sqrt(0.1 / (2.0 * spec.n)) * 0.75,
           "reference_grid": reference_grid(grid), "source": src,
           "accepted_n": min(spec.n, HET_N_CAP) if spec.heterogeneous else spec.n}
    cav = []
    if spec.heterogeneous and spec.n > HET_N_CAP:
        cav.append(f"heterogeneous cohorts are accepted at n <= {HET_N_CAP} (spec 003d SC-001, Naveh 2026-10-13); this "
                   f"cohort has n = {spec.n}")
    # Naveh 2026-10-14 (D1): any cohort whose own lambda or dlnCV at its real n exceeds its bar says so, with numbers
    if out["lambda"] > out["lambda_tolerance"]:
        cav.append(f"lambda {out['lambda']:.3g} at n = {spec.n} exceeds the bar {out['lambda_tolerance']}: the grid "
                   f"error may be detectable at this size")
    if abs(d) > out["dlncv_bound"]:
        cav.append(f"|dlnCV| {abs(d):.3g} at n = {spec.n} exceeds the bound {out['dlncv_bound']:.3g}")
    out["caveats"] = cav
    out["caveat"] = "; ".join(cav) or None
    return out


# ----------------------------------------------------------------------------------------------------- materialize
@dataclass(frozen=True)
class GeneratedCohort:
    spec: CohortSpec
    death_times: np.ndarray
    events: np.ndarray
    n_observed: int
    censor_time: float | None
    realised_censored_fraction: float
    grid_error: dict
    provenance: dict
    caveats: tuple = ()


def _provenance(spec: CohortSpec, grid: dict) -> dict:
    from sraging.provenance import _git, _version
    return {"generator_version": GENERATOR_VERSION, "grid": dict(grid), "prng": "PCG64", "seed": spec.seed,
            "draw_order": ["death", "hazard" if spec.external_hazard is not None else None,
                           "het" if spec.heterogeneous else None],
            "package": {"version": _version("sraging"), **_git()},
            "libraries": {"python": sys.version.split()[0],
                          **{d: _version(d) for d in ("numpy", "scipy", "jax", "jaxlib")}},
            "host": {"platform": platform.platform(), "machine": platform.machine()},
            "created_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}


def materialize(spec: CohortSpec) -> GeneratedCohort:
    """Draw the cohort `spec` describes (module docstring). Deterministic given (spec, GENERATOR_VERSION)."""
    if not isinstance(spec, CohortSpec):
        raise UnsupportedOptionError("spec", "materialize takes a sraging.generator.CohortSpec")
    grid = dict(GRID)
    m = _model(spec, grid)
    u = _uniforms(spec)
    if spec.heterogeneous:
        j, w, n_out = _bracket(_het_z(u["het"], spec.het_var), m["nodes"]["z"])
    else:
        j, w, n_out = np.zeros(spec.n, int), np.zeros(spec.n), 0
    tau, n_beyond = _inverse(u["death"], m["tau"], m["Fm"], j, w)
    t = tau * spec.L_phys
    if spec.external_hazard is not None:
        t = np.minimum(t, _hazard_times(u["hazard"], spec.external_hazard))
    C = None
    if spec.censor_frac > 0:
        C = _censor_time(lambda x: _S_obs(m, spec, x), spec.delta, spec.censor_frac,
                         float(m["tau"][-1]) * spec.L_phys)
    ages, events = _record(t, C, spec.delta)
    order = np.lexsort((events, ages))
    ages, events = ages[order], events[order]
    realised = float((events == 0).mean())
    n_obs = ages.size
    if spec.time_range is not None:
        ages, events, n_obs = _apply_time_range(ages, events, spec.time_range)
    ge = _grid_error_for(spec, grid, m)
    ge.update({"n_beyond_cdf": n_beyond, "n_outside_het_nodes": n_out,
               "max_abs_S_plus_F_minus_1": m["max_abs_S_plus_F_minus_1"]})
    cav = list(ge["caveats"])
    if C is not None and abs(realised - spec.censor_frac) > 0.1:   # Naveh 2026-10-14 (D2)
        cav.append(f"realised censored fraction {realised:.3f} against censor_frac {spec.censor_frac}: on this delta "
                   f"grid no administrative time C comes closer (spec 003d R7)")
    return GeneratedCohort(spec=spec, death_times=np.ascontiguousarray(ages, dtype=np.float64),
                           events=np.ascontiguousarray(events, dtype=int), n_observed=int(n_obs), censor_time=C,
                           realised_censored_fraction=realised, grid_error=ge, provenance=_provenance(spec, grid),
                           caveats=tuple(cav))
