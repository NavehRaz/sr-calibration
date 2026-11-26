"""Shapes and labels for weakly determined quantities (spec 003, T038; FR-020 to FR-024; constitution IV; research R11).

    box_limited   an UPPER endpoint set at a node or archive point where some coordinate (nuisances included) lies
                  within EDGE_DEX of its UPPER box edge (the widening's own edge distance, compat.jax2
                  claims.SC_EDGE_DEX): "open above" (FR-020). Lower edges never count: beta, kappa and h at their
                  floors are the model's own limits (no removal, saturated removal, no external hazard) that a wider
                  box cannot move (sr-naveh implementation review 2). Scored apart from data-determined bounds: a
                  cap set by the box scores coverage for free (constitution X incident).
    negligible    an upper bound on r-bar/eta, in units of L_SR, below the regime's negligibility floor c/sqrt(n_eff)
                  (n_eff capped at 2,000; the floors are measured: results/neg_floor_003.json). A label on an
                  upper-bound claim, scored as an upper bound; it prints log10(cap/floor) and the regime band.
    saturated     an upper bound on kappa below 0.364/sqrt(n_eff): saturated removal, never "negligible".
    widened       an upper side the self-consistency rule moved: "at least X; edge not found".
"""
from __future__ import annotations

import math

import numpy as np

EDGE_DEX = 0.01
KAPPA_FLOOR_C = 0.364          # constitution IV (sec. 8b): kappa/xc < 0.364 / sqrt(n_eff)
N_EFF_CAP = 2000
PARAMS = ("eta", "beta", "epsilon", "kappa", "h")
SYMBOL = {"eta": "eta", "beta": "beta", "epsilon": "eps", "kappa": "kappa", "h": "h"}


def n_eff(n_deaths: int) -> int:
    return int(min(max(int(n_deaths), 1), N_EFF_CAP))


def box_edge_of(z, box, params=PARAMS, edge_dex=EDGE_DEX):
    """The first coordinate of `z` (the FULL log-parameter vector, in `params` order) within `edge_dex` of its UPPER
    box edge, as 'kappa<=300'; None when every coordinate is clear of its upper edge."""
    for i, p in enumerate(params[:len(z)]):
        hi = box[p][1]
        if math.exp(z[i]) >= hi / 10 ** edge_dex:
            return f"{SYMBOL[p]}<={hi:g}"
    return None


def setting_node(nodes, ll_ref, side, thr=3.8415):
    """The node that sets an endpoint: on `side` ('lower' / 'upper'), the outermost node still inside the threshold
    2 (ll_ref - ll) <= thr. `nodes`: [(value, ll, z)]. None if there is none."""
    vals = sorted(nodes, key=lambda n: n[0])
    imax = int(np.argmax([n[1] for n in vals]))
    rng = range(imax, -1, -1) if side == "lower" else range(imax, len(vals))
    best = None
    for j in rng:
        if 2.0 * (ll_ref - vals[j][1]) <= thr:
            best = vals[j]
        else:
            break
    return best


def _label(edge, how):
    return {"kind": "box_limited", "edge": edge, "side": "upper", "log10_cap_over_floor": None, "regime": None,
            "text": f"open above: box-limited ({edge}): the upper end is set by the box{how}, not by the data"}


def box_limited_label(claim, nodes, ll_ref, box, params=PARAMS, archive_theta=None):
    """label_info for a claim whose stated UPPER endpoint is set at the box (or None). `nodes`: the profile's
    [(value, ll, full z)]; `ll_ref`: the profile's own maximum (the reference `profile_ci` measures from).
    `archive_theta`: when the widening moved the upper end, the archive point that set it (the extents'
    `theta_qmax`, in values) -- FR-020's "or archive point"."""
    if claim.upper is None:
        return None
    if archive_theta is not None:
        th = [x for x in archive_theta[:len(params)]]
        if all(x is not None and x > 0 for x in th):
            edge = box_edge_of(np.log(np.asarray(th, float)), box, params)
            if edge is not None:
                return _label(edge, " (at the archive point that widened it)")
    node = setting_node(nodes, ll_ref, "upper") if nodes else None
    if node is None or node[2] is None:
        return None
    edge = box_edge_of(node[2], box, params)
    return None if edge is None else _label(edge, "")


def negligible_label(cap_per_lsr, n_deaths, regime, floors):
    """label_info when an upper bound on r-bar/eta (in L_SR units) lies below its regime's floor (or None).
    `floors`: {regime: coefficient c}, floor = c / sqrt(n_eff)."""
    if cap_per_lsr is None or regime not in floors:
        return None
    floor = floors[regime] / math.sqrt(n_eff(n_deaths))
    r = math.log10(cap_per_lsr / floor)
    band = sorted(floors.values())
    if r >= 0:
        return None
    return {"kind": "negligible", "edge": None, "log10_cap_over_floor": r, "regime": regime, "floor": floor,
            "coefficient": floors[regime], "n_eff": n_eff(n_deaths),
            "text": (f"negligible: the cap on r-bar/eta is 10^{r:.2f} times the {regime}-regime floor "
                     f"({floors[regime]:.3g}/sqrt(n_eff); across regimes the coefficient spans {band[0]:.3g}-"
                     f"{band[-1]:.3g}): removal this small is not detectable at this n")}


def saturated_label(kappa_cap, n_deaths):
    if kappa_cap is None:
        return None
    floor = KAPPA_FLOOR_C / math.sqrt(n_eff(n_deaths))
    if kappa_cap >= floor:
        return None
    return {"kind": "saturated", "edge": None, "log10_cap_over_floor": math.log10(kappa_cap / floor), "regime": None,
            "text": f"saturated removal: kappa < {kappa_cap:.3g} (below 0.364/sqrt(n_eff) = {floor:.3g}); a small kappa "
                    f"means saturated removal, not no removal"}


def widened_text(claim):
    w = claim.widening or {}
    if not w or not w.get("upper_moved_dex") or claim.upper is None:
        return None
    return (f"at least {claim.upper:.4g}; edge not found (widened {w['upper_moved_dex']:.3g} dex from the profile's "
            f"{(w.get('claims_profile') or {}).get('upper')!s} to contain the fit's own evaluated points)")
