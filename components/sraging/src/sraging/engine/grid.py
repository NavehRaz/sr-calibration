"""Grid policies (research R9; spec FR-020; data-model.md).

A grid policy turns a cohort (or, for "model", a parameter set) into a `Grid`: the time unit L, the
damage cells N, the time mesh, the horizon, the early-stop tolerance and the scheme. Every result
records the policy's name and version and the grid it produced. No environment variable is read.

- "v033" (version 1): v0.3.3's production grid -- the Kaplan-Meier mean L (the unstable-sort KM of
  register D-007), auto_grid's horizon and step, and the dt ceiling (delta/L)/3 when 0 < delta/L < the
  auto step, N = 900, Gtol = 1e-11. It is `compat.v033.estimator.solver_grid` itself, called on the
  cohort's arrays, so the equality is by construction (and tested on spec 001's panel).
- "fixed" (version 1): explicit N, a uniform step `dt` or a node list `nodes`, tau_max, Gtol and L.
- "model" (version 1): for curves without a cohort; L comes from the model's own mean by a coarse probe
  (as SRK's fp_curves.auto_L). Its probe needs the engine's march, so it is resolved in
  `sraging.engine.curves` (User Story 6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sraging.errors import UnsupportedOptionError

POLICIES = ("v033", "fixed", "model")
SCHEMES = ("bdf2", "be")
NSTART = 20


@dataclass(frozen=True)
class Grid:
    """The grid actually used. `nodes` is None for a uniform mesh (then `dt` and `nsteps` define it, as
    v0.3.3: tau_n = n*dt, nsteps = int(tau_max/dt)); otherwise the graded node list (tau_0 = 0)."""

    L: float
    N: int
    dt: float | None
    nsteps: int
    tau_max: float
    Gtol: float
    scheme: str
    nstart: int
    dt_ceiling: float | None
    policy: str
    policy_version: int
    nodes: tuple | None = None

    @property
    def uniform(self) -> bool:
        return self.nodes is None

    def tau(self) -> np.ndarray:
        if self.nodes is None:
            return np.arange(self.nsteps + 1) * self.dt
        return np.asarray(self.nodes, float)

    def summary(self) -> dict:
        d = {"policy": self.policy, "policy_version": self.policy_version, "L": self.L, "N": self.N,
             "tau_max": self.tau_max, "Gtol": self.Gtol, "scheme": self.scheme, "nstart": self.nstart,
             "dt_ceiling": self.dt_ceiling, "nsteps": self.nsteps}
        if self.nodes is None:
            d["mesh"] = {"kind": "uniform", "dt": self.dt}
        else:
            t = np.asarray(self.nodes, float)
            d["mesh"] = {"kind": "graded", "n_nodes": int(t.size), "dt_min": float(np.min(np.diff(t))),
                         "dt_max": float(np.max(np.diff(t))), "tau_end": float(t[-1])}
        return d


@dataclass(frozen=True)
class GridPolicy:
    name: str
    N: int | None = None
    dt: float | None = None
    nodes: tuple | None = None
    tau_max: float | None = None
    Gtol: float = 1e-11
    L: float | None = None
    version: int = 1

    def __init__(self, name: str, *, N=None, dt=None, nodes=None, tau_max=None, Gtol=1e-11, L=None):
        if name not in POLICIES:
            raise UnsupportedOptionError(name, f"unknown grid policy {name!r}; policies are {POLICIES}")
        if name == "fixed":
            if (dt is None) == (nodes is None):
                raise UnsupportedOptionError("dt/nodes", "grid policy 'fixed' needs exactly one of dt or nodes")
            for k, v in (("N", N), ("tau_max", tau_max), ("L", L)):
                if v is None:
                    raise UnsupportedOptionError(k, f"grid policy 'fixed' needs {k}")
            if not (isinstance(N, int) and N >= 3):
                raise ValueError(f"N must be an integer >= 3, got {N!r}")
            if nodes is not None:
                t = np.asarray(nodes, float)
                if t.ndim != 1 or t.size < 3 or t[0] != 0.0 or not np.all(np.diff(t) > 0):
                    raise ValueError("nodes must be a 1-D increasing list starting at 0 with >= 3 entries")
                nodes = tuple(float(x) for x in t)
            if not (float(L) > 0 and float(tau_max) > 0):
                raise ValueError("L and tau_max must be > 0")
        elif any(v is not None for v in (N, dt, nodes, tau_max, L)):
            raise UnsupportedOptionError(name, f"grid policy {name!r} takes no N, dt, nodes, tau_max or L")
        for k, v in (("name", name), ("N", N), ("dt", dt), ("nodes", nodes), ("tau_max", tau_max),
                     ("Gtol", float(Gtol)), ("L", None if L is None else float(L)), ("version", 1)):
            object.__setattr__(self, k, v)

    def grid(self, cohort, scheme: str = "bdf2") -> Grid:
        """The grid for a cohort (policies "v033" and "fixed")."""
        if scheme not in SCHEMES:
            raise UnsupportedOptionError(scheme, f"unknown scheme {scheme!r}; schemes are {SCHEMES}")
        if self.name == "model":
            raise UnsupportedOptionError("model", "grid policy 'model' is for curves without a cohort "
                                         "(sraging.engine.curves)")
        if self.name == "fixed":
            if self.nodes is None:
                return Grid(L=self.L, N=self.N, dt=float(self.dt), nsteps=int(self.tau_max / self.dt),
                            tau_max=float(self.tau_max), Gtol=self.Gtol, scheme=scheme, nstart=NSTART,
                            dt_ceiling=None, policy="fixed", policy_version=self.version)
            t = np.asarray(self.nodes, float)
            return Grid(L=self.L, N=self.N, dt=None, nsteps=int(t.size - 1), tau_max=float(t[-1]),
                        Gtol=self.Gtol, scheme=scheme, nstart=NSTART, dt_ceiling=None, policy="fixed",
                        policy_version=self.version, nodes=self.nodes)
        return v033_grid(cohort, scheme)


def v033_grid(cohort, scheme: str = "bdf2") -> Grid:
    """v0.3.3's production grid, by calling `compat.v033.estimator.solver_grid` on the cohort's arrays.

    A per-row delta has no v0.3.3 rule; the ceiling then uses the smallest delta (the most conservative
    choice: every row gets at least three steps per recording interval)."""
    from sraging.compat.v033.estimator import solver_grid
    t = np.asarray(cohort.ages, float)
    e = np.asarray(cohort.events, int)
    d = cohort.delta
    d_rule = None if d is None else (float(np.min(d)) if np.ndim(d) else float(d))
    L, kw, ceiling = solver_grid(t, e, d_rule)
    return Grid(L=float(L), N=int(kw["N"]), dt=float(kw["dt"]), nsteps=int(kw["tau_max"] / kw["dt"]),
                tau_max=float(kw["tau_max"]), Gtol=float(kw["Gtol"]), scheme=scheme, nstart=NSTART,
                dt_ceiling=None if ceiling is None else float(ceiling), policy="v033", policy_version=1)


def insert_edges(nodes, edges, *, rtol=1e-12):
    """Make every edge (dimensionless, sorted) a node of the mesh (research R4; FR-016).

    An edge already on a node (to rtol) changes nothing -- a grid-aligned configuration keeps the uniform
    mesh, like for like with SRK's fp_piecewise. Otherwise the NEAREST node is moved onto the edge (the
    step count is unchanged and every step stays within [dt/2, 3dt/2]) and the step that starts there is
    a BE restart (register D-113: fp_piecewise instead switched parameters at the node below the edge
    while its clock ran from the exact edge). Two edges that would claim one node -- a segment shorter
    than about one step -- are refused. Returns (nodes, restart_steps)."""
    t = np.array(nodes, float)
    restart, claimed = [], {}
    for e in np.asarray(edges, float):
        if not (t[0] < e < t[-1]):
            continue                                   # an edge beyond the horizon changes nothing marched
        j = int(np.argmin(np.abs(t - e)))
        if abs(t[j] - e) <= rtol * max(1.0, abs(e)):
            claimed.setdefault(j, e)
            continue
        if j in claimed or j == 0 or j == t.size - 1:
            raise UnsupportedOptionError("segments", f"two segment edges ({claimed.get(j)!r}, {e!r} in units of "
                                         f"L) fall within one grid step; a segment shorter than the step "
                                         f"({t[1] - t[0]!r}) cannot be marched (refine the grid)")
        claimed[j] = e
        t[j] = e
        restart.append(j)
    if not np.all(np.diff(t) > 0):
        raise UnsupportedOptionError("segments", "segment edges produced a non-increasing mesh")
    return t, sorted(restart)


def graded_start(dt0: float, ratio: float, dt: float, tau_max: float) -> tuple:
    """Nodes of a graded start (research R12): steps dt0, dt0*ratio, ... (geometric) until a step would exceed
    dt, then uniform steps of dt up to tau_max. For GridPolicy("fixed", nodes=...)."""
    if not (0 < dt0 <= dt and ratio > 1 and tau_max > dt):
        raise ValueError("graded_start needs 0 < dt0 <= dt, ratio > 1 and tau_max > dt")
    t, h = [0.0], dt0
    while h < dt and t[-1] < tau_max:
        t.append(t[-1] + h)
        h *= ratio
    n = int(np.ceil((tau_max - t[-1]) / dt))
    t.extend(t[-1] + dt * np.arange(1, n + 1))
    return tuple(float(x) for x in t)
