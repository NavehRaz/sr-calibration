"""The FP damage density among the living at requested ages: the truth-free reference for Langevin paths (spec 004
FR-022; research R25; data-model "Damage density"; contracts/python-api.md §7).

    dd = damage_density(params, ages, heterogeneity=None, grid=None)
    dd.cdf_alive(x, age)          # F(x | alive, age), piecewise linear in physical damage x

- **The march** is the engine's own, with every step stored: `stepper._march_forward(..., store="all")`, a read-only
  use (stepper.py is not edited, so gen-1's compiled march is untouched). It runs only to the oldest age.
- **The density at age a** is q on the damage grid (cell averages, clipped at 0 as the march clips), interpolated
  linearly in tau = a / L between stored steps; S(a) = h sum q, the march's own survival interpolated alike.
- **The conditional CDF:** F(x | alive, a) = sum_k w_k G_k(x / xc_k, a) / sum_k w_k S_k(a), with G_k node k's living
  mass below y on its grid (piecewise linear), xc_k = xc s_k for xc heterogeneity and xc otherwise. The external
  hazard does not enter: it kills independently of X.
- **Its node count** comes from R26's curve check: the ladder's first rung M with sup |F_M - F_next| <= curve_tol (1e-4)
  over the ages and a damage grid; unconverged at the cap reads so.
- **The grid error** sup_x |F_(N, dt) - F_(2N, dt/2)| per age is recorded beside it.
Refused: a heterogeneity tuple (R28's tensor reference is internal), normal_frac_sd, segments, a negative age, and an
age beyond the march's horizon or its early stop.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np

import jax
import jax.numpy as jnp

from sraging.engine import stepper as ST
from sraging.engine.inputs import Heterogeneity, refuse_zero_spread
from sraging.errors import UnsupportedOptionError

CURVE_TOL = 1e-4
N_DEFAULT, DT_DEFAULT = 900, 1e-3           # the model grid's (sraging.engine.curves.STANDARD)
X_CHECK = 401                               # the damage grid the curve check and the grid error read
SCALED = ("eta", "beta", "epsilon", "kappa", "xc")


@dataclass(frozen=True, eq=False)
class DamageDensity:
    ages: np.ndarray
    S: np.ndarray                           # S(a) at each age (the mixture's, the SR survival without the hazard)
    grid: dict
    M: int | None
    curve_check: dict
    grid_error: tuple                       # sup_x |F - F_(2N, dt/2)| per age
    grid_error_grid: dict
    nodes: dict                             # {"xc": the node thresholds, "w": weights, "s": scales}
    S_grid: tuple = field(repr=False, default=())      # (tau, S) of the homogeneous or first node's march
    _G: tuple = field(repr=False, default=())          # per age: (per node: cumulative living mass on y edges)

    def cdf_alive(self, x, age):
        """F(x | alive, age) at physical damage x (scalar or array); age is one of the requested ages."""
        j = int(np.flatnonzero(np.isclose(self.ages, float(age), rtol=0, atol=1e-12))[0])
        return _cdf(self._G[j], self.nodes, np.asarray(x, float))


def _cdf(Gs, nodes, x):
    num = np.zeros_like(x, dtype=float)
    den = 0.0
    for G, xc, w in zip(Gs, nodes["xc"], nodes["w"]):
        y = np.linspace(0.0, 1.0, G.size)
        num = num + w * np.interp(x / xc, y, G, left=0.0, right=G[-1])
        den += w * G[-1]
    return num / den


@functools.lru_cache(maxsize=8)
def _march(N, nsteps, dt, nstart, Gtol):
    """The engine's march with every step stored (read-only use of stepper._march_forward)."""
    return jax.jit(lambda theta: ST._march_forward(theta, N, nsteps, dt, nstart, Gtol, store="all"))


def _theta(L, N, vals):
    """(rho_eta, rho_eps, rem) in units of each node's own threshold (the engine's Assembly.theta, xc divided out)."""
    xc = vals["xc"]
    rb, rh, re, kap = L * vals["beta"] / xc, L * L * vals["eta"] / xc, L * vals["epsilon"] / xc ** 2, vals["kappa"] / xc
    yf = (np.arange(N) + 0.5) / N
    return (jnp.asarray(rh), jnp.asarray(re), jnp.asarray(rb * yf / (yf + kap)))


def _node(params, scale, L, N, dt, nsteps, taus, nstart=20, Gtol=1e-11):
    """One node's (S on its steps, per age: the cumulative living mass on the y edges)."""
    vals = {k: float(getattr(params, k)) for k in SCALED}
    vals = {k: v * scale.get(k, 1.0) for k, v in vals.items()}
    S_raw, Q, k_stop = _march(N, nsteps, dt, nstart, Gtol)(_theta(L, N, vals))
    S_raw, Q, k_stop = np.asarray(S_raw), np.asarray(Q), int(k_stop)
    h = 1.0 / N
    p0 = np.zeros(N)
    p0[0] = 1.0 / h
    out = []
    for tau in taus:
        n = int(math.floor(tau / dt))
        f = tau / dt - n
        if n + 1 > k_stop and f > 0 or n > k_stop:            # this node is dead by then (S < Gtol): no living mass
            out.append(np.zeros(N + 1))
            continue
        a = p0 if n == 0 else np.maximum(Q[n], 0.0)
        b = np.maximum(Q[min(n + 1, nsteps)], 0.0)
        q = (1.0 - f) * a + f * b
        out.append(np.concatenate([[0.0], np.cumsum(q) * h]))
    return (np.arange(nsteps + 1) * dt, S_raw), out


def _nodes(params, het, M):
    from sraging.engine import quadrature as Q
    if het is None:
        return {"s": np.ones(1), "w": np.ones(1), "parameter": None}
    r = Q.truncated_normal_rule(het.sd, M)
    return {"s": np.asarray(r.s), "w": np.asarray(r.w), "parameter": het.parameter}


def _evaluate(params, ages, het, M, L, N, dt):
    taus = np.asarray(ages, float) / L
    nsteps = int(math.ceil(float(np.max(taus)) / dt)) + 1
    nd = _nodes(params, het, M)
    per_node, S_grid = [], None
    for s in nd["s"]:
        sg, Gs = _node(params, {nd["parameter"]: float(s)} if nd["parameter"] else {}, L, N, dt, nsteps, taus)
        S_grid = S_grid or sg
        per_node.append(Gs)
    xc = float(params.xc) * (nd["s"] if nd["parameter"] == "xc" else np.ones(nd["s"].size))
    nodes = {"xc": xc, "w": nd["w"], "s": nd["s"]}
    G_by_age = tuple(tuple(per_node[k][j] for k in range(len(per_node))) for j in range(taus.size))
    S = np.array([sum(w * G[-1] for G, w in zip(G_by_age[j], nd["w"])) for j in range(taus.size)])
    return G_by_age, nodes, S, S_grid


def _x_grid(params, nodes):
    return np.linspace(0.0, float(np.max(nodes["xc"])), X_CHECK)


def damage_density(params, ages, *, heterogeneity=None, grid=None, curve_tol=CURVE_TOL, **kw) -> DamageDensity:
    """contracts/python-api.md §7. `grid`: a GridPolicy("fixed", ...) (its L, N, dt, tau_max), or None for the model
    grid (L from the model's mean, N 900, dt 1e-3)."""
    from sraging.engine import mixture as MIX
    if "segments" in kw:
        raise UnsupportedOptionError("segments", "damage_density takes no age segments (spec 007)")
    if kw:
        raise UnsupportedOptionError(sorted(kw)[0], f"unknown option {sorted(kw)[0]!r}")
    if isinstance(heterogeneity, (tuple, list)):
        raise UnsupportedOptionError("heterogeneity", "a heterogeneity tuple goes only through the internal tensor "
                                     "reference (OD-11 (a)); the public density takes one parameter")
    if heterogeneity is not None:
        if not isinstance(heterogeneity, Heterogeneity) or heterogeneity.definition != "truncated_normal":
            raise UnsupportedOptionError("heterogeneity", "the damage density takes one truncated_normal "
                                         "Heterogeneity")
        refuse_zero_spread(heterogeneity, "damage_density")
    a = np.asarray(ages, float).ravel()
    if a.size < 1 or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise UnsupportedOptionError("ages", f"ages are finite and >= 0, got {ages!r}")
    if grid is None:
        from sraging.engine.curves import model_L
        L, N, dt, tau_max = model_L(params), N_DEFAULT, DT_DEFAULT, math.inf
    else:
        L, N, dt, tau_max = float(grid.L), int(grid.N), float(grid.dt), float(grid.tau_max)
    if float(np.max(a)) / L > tau_max:
        raise UnsupportedOptionError("ages", f"age {float(np.max(a)):g} is beyond the grid's horizon tau_max = "
                                     f"{tau_max:g} L (the march's stop)")
    het = heterogeneity
    check = {"verdict": "homogeneous", "M": None, "tol": curve_tol, "sup_dF": 0.0, "ladder": []}
    if het is None:
        M = None
    elif het.M != "auto":
        M = het.M
        check = {"verdict": "fixed", "M": M, "tol": curve_tol, "sup_dF": None, "ladder": []}
    else:
        M, check = None, None
        prev = None
        ladder = []
        for j, Mr in enumerate(MIX.RUNGS):
            G, nodes, _, _ = _evaluate(params, a, het, Mr, L, N, dt)
            x = _x_grid(params, nodes)
            Fs = np.array([_cdf(G[i], nodes, x) for i in range(a.size)])
            if prev is not None:
                d = float(np.max(np.abs(Fs - prev[1])))
                ladder.append({"M": prev[0], "M_next": Mr, "sup_dF": d})
                if d <= curve_tol:
                    M, check = prev[0], {"verdict": "converged", "M": prev[0], "tol": curve_tol, "sup_dF": d,
                                         "ladder": ladder}
                    break
            prev = (Mr, Fs)
        if M is None:
            M, check = MIX.CAP, {"verdict": "unconverged_at_cap", "M": MIX.CAP, "tol": curve_tol,
                                 "sup_dF": ladder[-1]["sup_dF"], "ladder": ladder}
    G, nodes, S, S_grid = _evaluate(params, a, het, M, L, N, dt)
    if np.any(S <= 0):
        raise UnsupportedOptionError("ages", f"ages {a[S <= 0].tolist()} are beyond the march's early stop (S < Gtol "
                                     f"for every node): no one is alive there")
    G2, nodes2, _, _ = _evaluate(params, a, het, M, L, 2 * N, dt / 2)
    x = _x_grid(params, nodes)
    gerr = tuple(float(np.max(np.abs(_cdf(G[i], nodes, x) - _cdf(G2[i], nodes2, x)))) for i in range(a.size))
    return DamageDensity(ages=a, S=S, grid={"L": L, "N": N, "dt": dt, "tau_max": tau_max}, M=M, curve_check=check,
                         grid_error=gerr, grid_error_grid={"N": 2 * N, "dt": dt / 2}, nodes=nodes, S_grid=S_grid,
                         _G=G)
