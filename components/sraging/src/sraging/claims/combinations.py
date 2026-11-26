"""The combinations of constitution X (v1.1.0) and their algebra over z = log(eta, beta, epsilon, kappa[, h])
(spec 003, T028; FR-014, FR-016; data-model.md Combination and ParameterMap; research R8, R9).

    r-bar = beta * (1 - kappa ln(1 + 1/kappa))      (it already contains beta; kfix calls it beta*rbar)

Each combination is log g(z), with its analytic gradient, its range over the calibrator's box (the box image, from
the 16 corners: every g is monotone in each coordinate, since 1 - kappa ln(1 + 1/kappa) decreases in kappa), and a
projection that moves a start onto log g = c along the coordinate g depends on most simply. The coordinate order is
the port's: 0 eta, 1 beta, 2 epsilon, 3 kappa, 4 h (compat/jax2/estimator.py PARAMS_SR; fp_jax z).
"""
from __future__ import annotations

import itertools
import json
import math
import os

import numpy as np

KEYS = ("eta_over_eps", "rbar_over_eta", "sqrt_eta_over_eps", "beta_over_eta_kp1", "rbar", "beta_over_kp1")
LABEL = {"eta_over_eps": "eta/eps", "rbar_over_eta": "rbar/eta", "sqrt_eta_over_eps": "sqrt(eta)/eps",
         "beta_over_eta_kp1": "beta/(eta(kappa+1))", "rbar": "rbar", "beta_over_kp1": "beta/(1+kappa)"}
NEEDS_FITTED_KAPPA = frozenset({"beta_over_eta_kp1", "beta_over_kp1"})
HYPOTHESIS_ORDER = ("eta_over_eps", "rbar_over_eta", "sqrt_eta_over_eps", "beta_over_eta_kp1", "rbar",
                    "beta_over_kp1")          # sr-naveh's order, the pre-registered hypothesis (FR-016)
RIDGE_INVARIANT = frozenset({"eta_over_eps", "rbar_over_eta", "beta_over_eta_kp1"})   # (eta, beta, eps) -> l (eta, beta, eps)
#: g * L_SR^power is dimensionless (eta per time^2; beta, eps, h per time; kappa none): FR-021's box in L_SR units
LSR_POWER = {"eta": 2, "beta": 1, "epsilon": 1, "kappa": 0, "h": 1, "eta_over_eps": 1, "rbar_over_eta": -1,
             "sqrt_eta_over_eps": 0, "beta_over_eta_kp1": -1, "rbar": 1, "beta_over_kp1": 1}
ORDER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "order.json")
ETA, BETA, EPS, KAP = 0, 1, 2, 3


def rbar_factor(kappa):
    """f(kappa) = 1 - kappa ln(1 + 1/kappa), the mean removal fraction over X in [0, xc = 1]."""
    k = np.asarray(kappa, float)
    return 1.0 - k * np.log1p(1.0 / k)


def _dlogf_dlogk(kappa):
    """d log f / d log kappa = kappa f'(kappa) / f, f'(kappa) = -ln(1 + 1/kappa) + 1/(1 + kappa)."""
    k = float(kappa)
    return k * (-math.log1p(1.0 / k) + 1.0 / (1.0 + k)) / float(rbar_factor(k))


def log_g(key: str, z) -> float:
    z = np.asarray(z, float)
    le, lb, lp, lk = z[ETA], z[BETA], z[EPS], z[KAP]
    k = math.exp(lk)
    if key == "eta_over_eps":
        return le - lp
    if key == "sqrt_eta_over_eps":
        return 0.5 * le - lp
    if key == "rbar":
        return lb + math.log(float(rbar_factor(k)))
    if key == "rbar_over_eta":
        return lb + math.log(float(rbar_factor(k))) - le
    if key == "beta_over_kp1":
        return lb - math.log1p(k)
    if key == "beta_over_eta_kp1":
        return lb - le - math.log1p(k)
    raise KeyError(key)


def grad_log_g(key: str, z) -> np.ndarray:
    z = np.asarray(z, float)
    g = np.zeros(z.size)
    k = math.exp(z[KAP])
    if key == "eta_over_eps":
        g[ETA], g[EPS] = 1.0, -1.0
    elif key == "sqrt_eta_over_eps":
        g[ETA], g[EPS] = 0.5, -1.0
    elif key in ("rbar", "rbar_over_eta"):
        g[BETA], g[KAP] = 1.0, _dlogf_dlogk(k)
        if key == "rbar_over_eta":
            g[ETA] = -1.0
    elif key in ("beta_over_kp1", "beta_over_eta_kp1"):
        g[BETA], g[KAP] = 1.0, -k / (1.0 + k)
        if key == "beta_over_eta_kp1":
            g[ETA] = -1.0
    else:
        raise KeyError(key)
    return g


def value(key: str, theta: dict) -> float:
    z = [math.log(theta[p]) for p in ("eta", "beta", "epsilon", "kappa")]
    return math.exp(log_g(key, z))


def box_image(key: str, box: dict) -> tuple[float, float]:
    """(min, max) of g over the calibrator's box {eta, beta, epsilon, kappa: (lo, hi)} (h does not enter)."""
    vals = []
    for corner in itertools.product(*[box[p] for p in ("eta", "beta", "epsilon", "kappa")]):
        vals.append(log_g(key, np.log(np.asarray(corner, float))))
    return math.exp(min(vals)), math.exp(max(vals))


def project(key: str, z, c: float) -> np.ndarray:
    """z moved onto log g = c by changing the coordinate g depends on most simply (eta for eta/eps and
    sqrt(eta)/eps, beta otherwise). The result may leave the box; the constrained solver brings it back."""
    z = np.asarray(z, float).copy()
    lk = z[KAP]
    k = math.exp(lk)
    if key == "eta_over_eps":
        z[ETA] = c + z[EPS]
    elif key == "sqrt_eta_over_eps":
        z[ETA] = 2.0 * (c + z[EPS])
    elif key == "rbar":
        z[BETA] = c - math.log(float(rbar_factor(k)))
    elif key == "rbar_over_eta":
        z[BETA] = c + z[ETA] - math.log(float(rbar_factor(k)))
    elif key == "beta_over_kp1":
        z[BETA] = c + math.log1p(k)
    elif key == "beta_over_eta_kp1":
        z[BETA] = c + z[ETA] + math.log1p(k)
    else:
        raise KeyError(key)
    return z


def lead_group() -> tuple[tuple[str, ...], str]:
    """(the frozen lead group, its source): what a jax2 fit computes by default (Naveh, 2026-10-06, from walker step
    1's measured cost: the lead pair 1.8x the fit's evaluations, all six 6.4x). Until order.json exists, the
    pre-registered hypothesis's first two, and it says so."""
    if os.path.exists(ORDER_FILE):
        d = json.load(open(ORDER_FILE))
        return tuple(d["lead_group"]), f"measured ({d.get('source', 'order.json')})"
    return HYPOTHESIS_ORDER[:2], "hypothesis (sr-naveh's lead pair; FR-016's measurement not frozen yet)"


def order() -> tuple[tuple[str, ...], str]:
    """(the reporting order, its source). The frozen order is written by the `order` acceptance phase into
    claims/order.json (FR-016); until it exists, the pre-registered hypothesis is used and says so."""
    if os.path.exists(ORDER_FILE):
        d = json.load(open(ORDER_FILE))
        return tuple(d["order"]), f"measured ({d.get('source', 'order.json')})"
    return HYPOTHESIS_ORDER, "hypothesis (sr-naveh's order; FR-016's measurement not frozen yet)"
