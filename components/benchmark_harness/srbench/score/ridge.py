"""How tightly the data pin a COMBINATION of the fitted parameters -- read from the stored fit.

A method claims intervals on the four parameters, not on functions of them. The explorer used to
build a combination from claim POINTS, so wherever a constituent was declined -- kappa on 1,314 of
1,776 J v6 records -- the combination had no estimate and drew grey. Filling it from the fitted
theta_hat alone looks like the obvious repair, and is wrong where it matters most: measured on the
same records, over each fit's own near-optimal points the spread of beta/eta exceeds 1 dex on 1,280
of 1,290 kappa-declined cohorts. An error at g(theta_hat) there is where the optimiser stopped along
a ridge the data do not resolve. theta_hat is also not the best point the fit evaluated on 489 of
them, and kappa_hat sits on a box edge on 383 (sr-manager gate and tester re-measurement,
2026-10-29).

So two things are read here, and the explorer shows them under a toggle (Naveh, 2026-10-29):

  spread       log10(max g / min g) over the record's near-optimal points -- theta_hat, the best
               evaluated point, and the eight `self_consistency.extents` points -- keeping only
               those within the method's own `half_thr` of the best log-likelihood. Each such point
               lies inside g's likelihood-ratio set, so the spread is a LOWER bound on how much the
               data leave g undetermined. Never a claim, never scored for coverage.
  error at θ̂   the combination at the fitted point, with each missing claim point replaced by
               theta_hat clipped to the bounds (exactly how the method forms a claim point), flagged
               with which inputs were filled, the fit's own staleness class, and any parameter on a
               box edge.

Everything here is read-side and costs no calibration. Records without the near-optimal layout
(other methods) are "not measurable", never a spread of zero.
"""
from __future__ import annotations

import math

from .floors import clamp
from .combos import COMBOS, combos

PARAMS = ("eta", "beta", "epsilon", "kappa")

#: A combination counts as RESOLVED by the data when its near-optimal spread is at most this many dex
#: (a factor of two). Below it, the error at the fitted point is a statement about the estimator;
#: above it, the error mostly records where along the ridge the fit stopped. Carried in the payload so
#: the views read it rather than restating it.
RESOLVED_DEX = 0.3

#: Above this many dex (a factor of ten) the data leave the combination UNRESOLVED; between the two it
#: is partly resolved. Carried in the payload beside RESOLVED_DEX.
UNRESOLVED_DEX = 1.0


def _pinned(diagnostics: dict) -> bool:
    return bool(((diagnostics or {}).get("estimate") or {}).get("pinned"))


def near_optimal_thetas(diagnostics: dict) -> list[list[float]] | None:
    """The record's own near-optimal parameter vectors, or None when it does not carry them.

    None also for a record with a PINNED parameter (kfix pins kappa): every stored point shares the
    pinned value, so a spread there is measured in a slice the data were never asked about.
    """
    d = diagnostics or {}
    est, sc = d.get("estimate") or {}, d.get("self_consistency") or {}
    ll_best, half = sc.get("ll_best"), sc.get("half_thr")
    if ll_best is None or half is None or not sc.get("extents") or _pinned(d):
        return None
    floor = ll_best - half
    pts = []

    def keep(theta, ll):
        if theta is None or ll is None or ll < floor:
            return
        t = [float(theta[p]) for p in PARAMS] if isinstance(theta, dict) else [float(v) for v in theta]
        if t not in pts:
            pts.append(t)

    keep(est.get("theta_hat"), est.get("theta_hat_loglik"))
    bp = est.get("best_point") or {}
    keep(bp.get("theta"), bp.get("loglik"))
    for e in sc["extents"].values():
        keep(e.get("theta_qmin"), e.get("ll_qmin"))
        keep(e.get("theta_qmax"), e.get("ll_qmax"))
    return pts or None


def combo_spread(thetas, name: str, floor: float | None) -> float | None:
    """log10(max/min) of combination `name` over `thetas`, each value clamped at `floor` exactly as
    `claims.log_error` clamps an estimate. None with fewer than two usable values."""
    vals = []
    for t in thetas or []:
        g = combos(t).get(name)
        if g is not None and math.isfinite(g) and g > 0:
            vals.append(clamp(g, floor))
    if len(vals) < 2:
        return None
    return math.log10(max(vals) / min(vals))


def fitted_theta(diagnostics: dict, bounds: dict | None = None) -> list[float] | None:
    """theta_hat as [eta, beta, epsilon, kappa], clipped to `bounds` where given; None when the record
    has no fitted point or pins a parameter."""
    d = diagnostics or {}
    th = (d.get("estimate") or {}).get("theta_hat")
    if not th or _pinned(d) or any(th.get(p) is None for p in PARAMS):
        return None
    out = []
    for p in PARAMS:
        v = float(th[p])
        lo_hi = (bounds or {}).get(p)
        if lo_hi:
            v = min(max(v, float(lo_hi[0])), float(lo_hi[1]))
        out.append(v)
    return out


def at_box_edge(theta, bounds: dict | None, edge_dex: float | None) -> list[str]:
    """The parameters of `theta` within `edge_dex` of a bound -- the method's own definition of an
    edge (`self_consistency.edge_dex`). Empty when either is unknown."""
    if theta is None or not bounds or edge_dex is None:
        return []
    out = []
    for p, v in zip(PARAMS, theta):
        lo_hi = bounds.get(p)
        if not lo_hi or v is None or v <= 0:
            continue
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        if (lo > 0 and abs(math.log10(v / lo)) <= edge_dex) or \
           (hi > 0 and abs(math.log10(v / hi)) <= edge_dex):
            out.append(p)
    return out


def fit_staleness(diagnostics: dict) -> tuple[str | None, float | None]:
    """(class, gap in nats) of theta_hat against the best point the fit evaluated, as the method
    recorded them: none <= 0.01 < minor <= thr/2 < major."""
    bp = (((diagnostics or {}).get("estimate") or {}).get("best_point") or {})
    if not bp:
        return None, None
    return bp.get("stale"), bp.get("gap_nats")


def combination_inputs(claim_points: dict, fitted) -> tuple[list[float], list[str]]:
    """[eta, beta, epsilon, kappa] for a combination: the claim point where there is one, the fitted
    point where there is not (NaN without one). Returns the vector and the names that were filled."""
    theta, filled = [], []
    for i, p in enumerate(PARAMS):
        v = claim_points.get(p)
        if v is None and fitted is not None:
            v = fitted[i]
            filled.append(p)
        theta.append(float("nan") if v is None else float(v))
    return theta, filled


__all__ = ["RESOLVED_DEX", "UNRESOLVED_DEX", "near_optimal_thetas", "combo_spread", "fitted_theta", "at_box_edge",
           "fit_staleness", "combination_inputs", "COMBOS"]
