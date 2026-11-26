"""Combination claims from a walk, widened by jax2's own self-consistency rule (spec 003, T030; FR-015; research R9).

The claim is built by the port's claim mapping for one quantity (`compat.jax2.claims.h_claim_kwargs`, which is
generic: the interval, the verdict and the quantity's box), with g's box image as the box. The widening is the
port's `sc_extents` over the fit's archive with a log g column appended (its box: g's box image), then the port's
`sc_widen_claim`. Nothing here changes theta-hat or a per-parameter claim.
"""
from __future__ import annotations

import numpy as np

from sraging.claims import combinations as K
from sraging.compat.jax2 import claims as JC
from sraging.compat.jax2 import estimator as JE

THR = 3.841458820694          # jax2's chi2_1(0.95) for self-consistency (estimator.py fit: chi2.ppf)
PARAMS = ("eta", "beta", "epsilon", "kappa", "h")


def claim_from_walk(key, walk, point, *, jc=None):
    """The port's one-quantity claim mapping applied to a combination's profile interval. `jc`: the port's claims
    module (spec 003b; default 1.0.0's)."""
    jc = JC if jc is None else jc
    ci, level = walk["ci"], jc.LEVEL
    kw = jc.h_claim_kwargs(point, {f"h@{level}": (ci["ci_lo"], ci["ci_hi"])}, {"h": ci["verdict"]},
                           walk["box_image"])
    kw = dict(kw, parameter=key)
    kw.pop("bounds", None)
    kw.pop("verdict", None)
    return kw


def widen(key, kw, Z, LL, ll_hat, box, box_image, *, jc=None, je=None):
    """(widened claim kw, the widening record, the extents) for one combination. `Z`, `LL`: the archive (log params,
    log-likelihoods); `box`: the calibrator's per-parameter box (with h for jax2_h). `jc`, `je`: the port's claims and
    estimator modules (spec 003b; default 1.0.0's)."""
    jc, je = (JC if jc is None else jc), (JE if je is None else je)
    Z = np.asarray(Z, float)
    params = list(PARAMS[:Z.shape[1]])
    lg = np.array([K.log_g(key, z) for z in Z]) if Z.size else np.empty(0)
    Zg = np.column_stack([Z, lg]) if Z.size else np.empty((0, len(params) + 1))
    sc = je.sc_extents(Zg, LL, params + [key], dict(box, **{key: tuple(box_image)}), ll_hat, THR)
    new, rec = jc.sc_widen_claim(kw, sc["extents"][key], box_image)
    return new, rec, sc
