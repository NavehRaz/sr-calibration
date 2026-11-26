"""The constitution VII guard: the default numerics may differ from "v033" only where a register entry's
machine-checkable condition says so (research R14, revised at review T064; spec FR-023, FR-024).

For one cohort and one parameter point it evaluates both numerics term by term and classifies every
term: covered by an entry's predicate (and then held to that entry's bound), or uncovered (and then
required to be equal to UNCOVERED_ULPS ulps of the term, plus UNCOVERED_ULPS ulps of a death mass's operands
amplified by the difference that forms it: equal in bits but for the compiler, which was measured to round an
elementwise log, and the S products, differently in the two programs on entry-plus-hazard cohorts).

Values. `mass_stiffness` (D-004, D-106): |m_v1 - m_v033| <= c * u * K + floor_ulps * u, in mass space, where
K = rho_eps * dtau * N^2 is the solve's diffusion number (Evaluator.stiffness) and m_v033 is taken as 0 when
the v033 term is on the -700 floor. The S route's error is the tridiagonal solve's mass residual, which grows
with K (sr-naveh T064 N2: (0.24-0.50) u K across N, dtau and epsilon); c is measured by the `guard-pilot`
acceptance phase and frozen in the register. A floored v033 term therefore gets no free pass: v1 may rescue
it only to a mass the S route could not resolve. `any` (D-107, D-108): the entry defines the new value.
For an exact-time death the mass is the density times the step.

Predicates are a fixed vocabulary, joined with '&' in the register's `terms` field:
  death, censored, interval, exact, use_F, hazard, window_close (a death with t <= t_hi < t + delta),
  lo_floor (S_obs(t_lo) < 1e-12; applies to every at-risk row's conditioning).
use_F is the engine's choice; for interval deaths it is checked against the same rule recomputed from v033's
own S (1 - S(t + delta) < S(t)): the two may differ only within BAND of the switch (T064 N8).

Gradients (T064 N3):
- the weighted gradient over the uncovered terms agrees to GRAD_RTOL relative (not bits: XLA fuses the two
  programs differently; measured 4e-15 on whole sums, 1.3e-13 on a single censored row);
- over the covered terms that both numerics resolve (|dl| <= 1e-8), v1's weighted gradient agrees with
  v033's to GRAD_COVERED_RTOL;
- over the rescued terms (covered, |dl| > 1e-8 or floored under v033, not floored under v1), v1's weighted
  gradient agrees with four-point central differences of v1 (step 3e-3 to 1e-4 in log parameters, each with its
  own round-off bound; an early-stop jump straddled at one step is not a defect).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sraging.engine.stepper import _LOG_FLOOR

U = float(np.finfo(float).eps)
GRAD_RTOL = 1e-10
GRAD_COVERED_RTOL = 1e-6
RESOLVED_DL = 1e-8
BAND = 1e-9
CD_STEPS = (3e-3, 1e-3, 3e-4, 1e-4)             # four-point stencil (truncation O(h^4)): larger steps, less noise
UNCOVERED_ULPS = 4                               # XLA may round an elementwise log differently in the two programs
PREDICATES = ("death", "censored", "interval", "exact", "use_F", "hazard", "window_close", "lo_floor")
VALUE_BOUNDS = ("any", "mass_stiffness")


@dataclass
class GuardReport:
    cohort: str
    n_terms: int
    stiffness: float = 0.0
    covered: dict = field(default_factory=dict)        # entry id -> number of terms it covers
    violations: list = field(default_factory=list)     # (row, entry id or "uncovered" or check, detail)
    grad_uncovered_equal: bool = True             # within GRAD_RTOL (see the module docstring)
    grad_uncovered_rel: float = 0.0
    grad_covered_rel: float | None = None          # None: no covered resolved terms
    rescued: int = 0
    rescued_cd_ratio: float | None = None          # best ratio to its bound over CD_STEPS; None: none rescued
    use_F_disagree: int = 0                        # outside BAND

    @property
    def ok(self) -> bool:
        return not self.violations and self.grad_uncovered_equal


def _mask(pred: str, ctx: dict) -> np.ndarray:
    m = np.ones(ctx["n"], bool)
    for tok in (x.strip() for x in pred.split("&")):
        if tok not in PREDICATES:
            raise ValueError(f"unknown guard predicate {tok!r}; predicates are {PREDICATES}")
        m &= ctx[tok]
    return m


def conditions(entries) -> list[tuple[str, dict]]:
    """(id, condition table) of every register entry that changes "v1"."""
    out = []
    for e in entries:
        cond = getattr(e, "condition", None) or (e.get("condition") if isinstance(e, dict) else None)
        if cond and "v1" in cond.get("applies_to", []):
            out.append((e["id"] if isinstance(e, dict) else e.id, cond))
    return out


def masses(ev, terms, aux):
    """Each row's mass in the guard's sense: the death mass (exact times: density x step), the survival for a
    censored row; 0 for a term on the -700 floor."""
    c = ev.cohort
    died = np.asarray(c.events) == 1
    m = np.where(died, np.asarray(aux["mass"], float), np.asarray(aux["S_i"], float))
    if c.delta is None:
        step = float(np.max(np.diff(np.asarray(ev._asm.tau, float)))) * ev.grid.L
        m = np.where(died, m * step, m)
    return np.where(np.asarray(terms) <= _LOG_FLOOR + 1e-9, 0.0, m)


def _logp(x):
    x = np.asarray(x, float)
    return np.maximum(np.log(np.clip(x, np.exp(_LOG_FLOOR), None)), _LOG_FLOOR)


def _weighted_cd(ev, params, w, g, extras):
    """Best ratio |CD - g| / bound over CD_STEPS for the weighted sum of v1's terms."""
    z = np.asarray(ev._z(params, **extras), float)

    def f(x):
        t, _ = ev._terms(x)
        return float(np.sum(w * np.asarray(t)))
    v = f(z)
    gv = np.array([g[n] for n in ev.layout.names])
    best = np.inf
    for h in CD_STEPS:
        bound = 50 * U * max(abs(v), 1.0) / h + 1e-6 * float(np.max(np.abs(gv)))
        worst = 0.0
        for i in range(z.size):
            dz = np.zeros(z.size)
            dz[i] = h
            cd = (8 * (f(z + dz) - f(z - dz)) - (f(z + 2 * dz) - f(z - 2 * dz))) / (12 * h)
            worst = max(worst, abs(cd - gv[i]) / bound)
        best = min(best, worst)
        if best <= 1.0:
            break
    return best


def compare(ev_v033, ev_v1, params, conds, name="", *, segment_values=None, het_v=None) -> GuardReport:
    extras = {k: v for k, v in (("segment_values", segment_values), ("het_v", het_v)) if v is not None}
    t33, a33 = ev_v033.terms(params, **extras)
    t1, a1 = ev_v1.terms(params, **extras)
    c = ev_v1.cohort
    n = c.n
    died = np.asarray(c.events) == 1
    interval = c.delta is not None
    use_F = np.asarray(a1.get("use_F", np.zeros(n, bool)), bool)
    ctx = {"n": n, "death": died, "censored": ~died,
           "interval": np.full(n, interval), "exact": np.full(n, not interval), "use_F": use_F,
           "hazard": np.full(n, bool(ev_v1.hazard)),
           "window_close": np.zeros(n, bool), "lo_floor": np.zeros(n, bool)}
    if c.window is not None:
        lo, hi = c.window
        reach = np.asarray(c.ages) + (np.broadcast_to(np.asarray(c.delta, float), (n,)) if interval else 0.0)
        ctx["window_close"] = died & (np.asarray(c.ages) <= hi) & (reach > hi)
        if "log_S_lo" in a33 and float(a33["log_S_lo"]) <= np.log(1e-12):
            ctx["lo_floor"] = np.asarray(c.ages) >= lo
    K = ev_v1.stiffness(params, **extras)
    rep = GuardReport(cohort=name or (c.name or ""), n_terms=n, stiffness=K)
    if interval and "S_hi" in a33:                  # T064 N8: use_F against v033's own S
        S_st = np.asarray(a33.get("S_st", a33["S_i"]), float)
        S_hi = np.asarray(a33["S_hi"], float)
        mine = (1.0 - S_hi) < S_st
        off = died & (mine != use_F) & (np.abs((1.0 - S_hi) - S_st) > BAND)
        rep.use_F_disagree = int(off.sum())
        for i in np.flatnonzero(off)[:20]:
            rep.violations.append((int(i), "use_F", f"engine {bool(use_F[i])}, from v033's S {bool(mine[i])}"))
    m33, m1 = masses(ev_v033, t33, a33), masses(ev_v1, t1, a1)
    dl = np.abs(t1 - t33)
    any_rows = np.zeros(n, bool)                   # rows an "any" entry defines outright (D-107, D-108)
    for eid, cond in conds:
        if cond["value_bound"] == "any":
            any_rows |= _mask(cond["terms"], ctx)
    scored = np.ones(n, bool)                      # rows whose term is their own mass minus the conditioning
    if c.window is not None:
        scored &= (np.asarray(c.ages) >= c.window[0]) & (np.asarray(c.ages) < c.window[1])
    covered_any = np.zeros(n, bool)
    rescued = np.zeros(n, bool)
    resolved_cov = np.zeros(n, bool)
    for eid, cond in conds:
        m = _mask(cond["terms"], ctx)
        covered_any |= m
        rep.covered[eid] = int(m.sum())
        bound = cond["value_bound"]
        if bound not in VALUE_BOUNDS:
            raise ValueError(f"{eid}: unknown value_bound {bound!r}; bounds are {VALUE_BOUNDS}")
        if bound == "any":
            continue
        lim = float(cond["c_stiff"]) * U * K + float(cond["floor_ulps"]) * U
        dm = np.abs(m1 - m33)
        for i in np.flatnonzero(m & (dm > lim))[:50]:
            rep.violations.append((int(i), eid, f"|dm| = {dm[i]:.3e} > c u K + floor = {lim:.3e} (K = {K:.3g})"))
        # the term must be its own mass minus the same conditioning in both numerics: a change to the term that
        # its mass does not show is not this entry's
        cons = (t1 - _logp(a1["mass"])) - (t33 - _logp(a33["mass"]))
        for i in np.flatnonzero(m & died & scored & ~any_rows & (np.abs(cons) > 1e-9))[:50]:
            rep.violations.append((int(i), eid, f"term and mass disagree by {cons[i]:.3e} nats"))
        floored1 = t1 <= _LOG_FLOOR + 1e-9
        floored33 = t33 <= _LOG_FLOOR + 1e-9
        rescued |= m & died & ~floored1 & ((dl > RESOLVED_DL) | floored33)
        resolved_cov |= m & ~floored1 & ~floored33 & (dl <= RESOLVED_DL)
    unc = ~covered_any
    # a few ulps of the term, plus a few ulps of the death mass's operands amplified by the difference that forms
    # it: (S(t) + S(t + delta)) / m (a compiler rounding the two programs' S products differently, measured on
    # entry-plus-hazard deaths near -31 nats whose mass is a difference of numbers far larger than itself)
    amp = np.ones(n)
    if interval and "S_hi" in a33:
        s_lo = np.asarray(a33.get("S_st", a33["S_i"]), float)
        amp = np.where(died, (s_lo + np.asarray(a33["S_hi"], float)) / np.maximum(np.asarray(a33["mass"], float),
                                                                                  np.finfo(float).tiny), 1.0)
    tol = UNCOVERED_ULPS * (np.spacing(np.maximum(np.abs(t33), np.abs(t1))) + U * amp)
    for i in np.flatnonzero(unc & (np.abs(t1 - t33) > tol))[:50]:
        rep.violations.append((int(i), "uncovered", f"v033 {t33[i]!r} vs v1 {t1[i]!r}"))
    w = unc.astype(float)
    g33, g1 = ev_v033.weighted_grad(params, w, **extras), ev_v1.weighted_grad(params, w, **extras)
    scale = max(max(abs(v) for v in g33.values()), 1e-300)
    rep.grad_uncovered_rel = max(abs(g33[k] - g1[k]) for k in g33) / scale
    rep.grad_uncovered_equal = rep.grad_uncovered_rel <= GRAD_RTOL
    if resolved_cov.any():
        w = resolved_cov.astype(float)
        g33, g1 = ev_v033.weighted_grad(params, w, **extras), ev_v1.weighted_grad(params, w, **extras)
        scale = max(max(abs(v) for v in g33.values()), 1e-300)
        rep.grad_covered_rel = max(abs(g33[k] - g1[k]) for k in g33) / scale
        if rep.grad_covered_rel > GRAD_COVERED_RTOL:
            rep.violations.append((-1, "grad_covered", f"{rep.grad_covered_rel:.3e} > {GRAD_COVERED_RTOL:g} relative"))
    rep.rescued = int(rescued.sum())
    if rescued.any():
        w = rescued.astype(float)
        g1 = ev_v1.weighted_grad(params, w, **extras)
        rep.rescued_cd_ratio = _weighted_cd(ev_v1, params, w, g1, extras)
        if rep.rescued_cd_ratio > 1.0:
            rep.violations.append((-1, "grad_rescued", f"central differences {rep.rescued_cd_ratio:.2f} x bound"))
    return rep
