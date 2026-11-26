"""The combination walker's defences on synthetic likelihoods whose profiles are known in closed form (spec 003,
T026; SC-004; sr-naveh implementation review 3, 5, 6). Fast: no SR likelihood.

    * two kappa basins: the wider one is reached only from the kappa-corner starts; the planted walker without them
      is narrower than the true edge by far more than SC-004's 0.01 dex (the contract's plant);
    * a beta floor: r-bar/eta with removal negligible stops at beta's floor and reads [0, cap], not a closed claim
      with a lower end decades below the point;
    * a cliff: the jumped node is kept, so the profile crosses there instead of reading open.
Added with the walker fixes (SC-004; Naveh, 2026-10-06), each with its plant:
    * a branch 9 dex away in beta is reached only from the beta-spread starts;
    * a cliff above beta's floor, with no removal inside the threshold: the beta-floor probe opens the lower side;
    * a wide bracket across a sharp rise: the bisection puts the end within 0.01 dex (0.40 dex short without it);
    * an archived point beyond an end: `reconcile` walks from it to the true edge, and nothing contradicts the result.
Added after the ITP example (SC-004; Naveh, 2026-10-08), with its plant:
    * solves past the edge that fail unless started within 0.0005 dex: the side stops "stalled" at the true edge,
      where without the stop it crawls to "max_nodes" at the same edge, with 14 times the nodes past it.
"""
import math

import numpy as np
import pytest

from sraging.claims import walker as W

DEX = math.log(10.0)
BOX = {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3), "kappa": (1e-3, 300.0)}
BOUNDS = [(math.log(BOX[p][0]), math.log(BOX[p][1])) for p in ("eta", "beta", "epsilon", "kappa")]
THR = 3.8415


class Quadratics:
    """nll(z) = the smaller of a few quadratics in z (each a dict of {direction: (centre, sigma)} plus an offset),
    with the active one's gradient. Counts calls like the port's objective (n_g, n_f)."""

    def __init__(self, basins, extra=None):
        self.basins, self.extra, self.n_g, self.n_f = basins, extra, 0, 0

    @staticmethod
    def _q(z, basin):
        v, g = basin.get("offset", 0.0), np.zeros(4)
        for d, (m, s) in basin["terms"].items():
            a = np.asarray(d, float)
            u = (a @ z - m) / s
            v += 0.5 * u * u
            g += u / s * a
        return v, g

    def nll_grad(self, z):
        self.n_g += 1
        z = np.asarray(z, float)
        v, g = min((self._q(z, b) for b in self.basins), key=lambda t: t[0])
        if self.extra is not None:
            ev, eg = self.extra(z)
            v, g = v + ev, g + eg
        return v, g


X = (1, 0, -1, 0)                     # log(eta/eps)
RIDGE = (1, 0, 1, 0)
BETA = (0, 1, 0, 0)
KAP = (0, 0, 0, 1)
Z_HAT = np.log([1e-4, 0.1, 1e-3, 0.01])
XA = float(np.asarray(X) @ Z_HAT)


def two_basins():
    common = {RIDGE: (float(np.asarray(RIDGE) @ Z_HAT), 1.0), BETA: (math.log(0.1), 1.0)}
    a = {"terms": {X: (XA, 1.0), KAP: (math.log(0.01), 0.3), **common}}
    b = {"terms": {X: (XA, 3.0), KAP: (math.log(30.0), 1.0), **common}, "offset": 1.0}
    return Quadratics([a, b])


def edge_dex_true():
    """Basin B sets the 95% edge: 0.5 ((x - xA)/3)^2 + 1 = THR/2."""
    return 3.0 * math.sqrt(2.0 * (THR / 2.0 - 1.0)) / DEX


def test_kappa_corner_starts_reach_the_wider_basin():
    obj = two_basins()
    w = W.walk(obj, "eta_over_eps", Z_HAT, -0.0, bounds=BOUNDS, box4=BOX, seed_z=Z_HAT)
    ci = w["ci"]
    hi_dex = (math.log(ci["ci_hi"]) - XA) / DEX
    lo_dex = (XA - math.log(ci["ci_lo"])) / DEX
    assert abs(hi_dex - edge_dex_true()) <= 0.01 and abs(lo_dex - edge_dex_true()) <= 0.01, (lo_dex, hi_dex)


def test_a_walker_without_kappa_corners_is_too_narrow(monkeypatch):
    """The contract's plant: skip the kappa-corner starts; SC-004's 0.01 dex rule must catch it."""
    monkeypatch.setattr(W, "kappa_corner_starts", lambda z: [])
    obj = two_basins()
    w = W.walk(obj, "eta_over_eps", Z_HAT, -0.0, bounds=BOUNDS, box4=BOX, seed_z=Z_HAT)
    hi_dex = (math.log(w["ci"]["ci_hi"]) - XA) / DEX
    assert edge_dex_true() - hi_dex > 0.01


def test_rbar_over_eta_with_negligible_removal_reads_an_upper_bound():
    """Removal r = beta f(kappa) is penalised above 1e-3 per unit time and free below: the profile of r-bar/eta is
    flat down to beta's floor, so the lower side stops there ("beta_floor") and the claim is [0, cap]."""
    from sraging.claims import combinations as K
    z_hat = np.log([1e-4, 1e-5, 1e-3, 0.5])

    def removal(z):
        lr = K.log_g("rbar", z)                     # log(beta f(kappa))
        u = math.exp(lr) / 1e-3
        g = np.zeros(4)
        g += u * u * K.grad_log_g("rbar", z)
        return 0.5 * u * u, g
    obj = Quadratics([{"terms": {(1, 0, 0, 0): (z_hat[0], 0.3), (0, 0, 1, 0): (z_hat[2], 0.3)}}], extra=removal)
    w = W.walk(obj, "rbar_over_eta", z_hat, -obj.nll_grad(z_hat)[0], bounds=BOUNDS, box4=BOX, seed_z=z_hat)
    assert w["stops"]["lower"] == "beta_floor", w["stops"]
    assert not math.isfinite(w["ci"]["ci_lo"]) and math.isfinite(w["ci"]["ci_hi"]), w["ci"]


def test_a_cliff_node_is_kept_so_the_profile_crosses():
    """log(eta/eps) above xA + 1 costs 50 nats at once: the walk keeps the jumped node and stops as "cliff"; the
    upper end is closed at the cliff, not open."""
    def cliff(z):
        return (50.0 if float(np.asarray(X) @ z) > XA + 1.0 else 0.0), np.zeros(4)
    obj = Quadratics([{"terms": {X: (XA, 10.0), RIDGE: (float(np.asarray(RIDGE) @ Z_HAT), 1.0),
                                 BETA: (math.log(0.1), 1.0), KAP: (math.log(0.01), 1.0)}}], extra=cliff)
    w = W.walk(obj, "eta_over_eps", Z_HAT, -obj.nll_grad(Z_HAT)[0], bounds=BOUNDS, box4=BOX, seed_z=Z_HAT)
    assert w["stops"]["upper"] == "cliff", w["stops"]
    assert math.isfinite(w["ci"]["ci_hi"]) and abs(math.log(w["ci"]["ci_hi"]) - (XA + 1.0)) / DEX < 0.01


@pytest.mark.parametrize("key", ["sqrt_eta_over_eps", "rbar", "beta_over_kp1"])
def test_ridge_starts_move_eta_beta_eps_together(key):
    z = Z_HAT.copy()
    for s, d in zip(W.ridge_starts(key, z), W.RIDGE_DEX):
        assert np.allclose(s[:3] - z[:3], d * DEX) and s[3] == z[3]
    assert W.ridge_starts("eta_over_eps", z) == []


def far_beta():
    """Basin A at theta-hat's beta (1e-8); basin B at beta = 10, 9 dex away, 1 nat lower, three times wider in x."""
    common = {RIDGE: (float(np.asarray(RIDGE) @ Z_HAT), 1.0), KAP: (math.log(0.01), 1.0)}
    a = {"terms": {X: (XA, 1.0), BETA: (math.log(1e-8), 1.0), **common}}
    b = {"terms": {X: (XA, 3.0), BETA: (math.log(10.0), 1.0), **common}, "offset": 1.0}
    return Quadratics([a, b])


Z_FAR = Z_HAT.copy()
Z_FAR[1] = math.log(1e-8)


def _edges(w):
    return (XA - math.log(w["ci"]["ci_lo"])) / DEX, (math.log(w["ci"]["ci_hi"]) - XA) / DEX


def test_beta_spread_starts_reach_a_branch_nine_dex_away_in_beta(monkeypatch):
    """Measured in walker step 1: three sqrt(eta)/eps lower ends 1.1-1.2 dex too narrow, the inside set on a branch
    at beta 10-17 while theta-hat's beta was 1e-9 to 1e-8; the kappa corners move beta by at most 300x."""
    obj = far_beta()
    w = W.walk(obj, "eta_over_eps", Z_FAR, -obj.nll_grad(Z_FAR)[0], bounds=BOUNDS, box4=BOX, seed_z=Z_FAR)
    assert all(abs(e - edge_dex_true()) <= 0.01 for e in _edges(w)), _edges(w)
    monkeypatch.setattr(W, "beta_span_starts", lambda key, z, bounds, n=W.BETA_SPAN_N: [])     # the plant
    obj = far_beta()
    w = W.walk(obj, "eta_over_eps", Z_FAR, -obj.nll_grad(Z_FAR)[0], bounds=BOUNDS, box4=BOX, seed_z=Z_FAR)
    assert all(edge_dex_true() - e > 0.5 for e in _edges(w)), _edges(w)
    assert W.beta_span_starts("rbar", Z_FAR, BOUNDS) == []    # the projection moves beta itself for these keys


def _banded():
    """r-bar/eta: removal 1-5 dex below theta-hat's costs 50 nats (a cliff on the way down); none at all (beta at its
    floor) costs nothing, so no removal is inside the threshold."""
    from sraging.claims import combinations as K
    z_hat = np.log([1e-4, 0.1, 1e-3, 0.5])
    lr_hat = K.log_g("rbar", z_hat)

    def band(z):
        lr = K.log_g("rbar", z)
        return (50.0 if lr_hat - 5 * DEX < lr < lr_hat - DEX else 0.0), np.zeros(4)
    terms = {(1, 0, 0, 0): (z_hat[0], 0.3), (0, 0, 1, 0): (z_hat[2], 0.3), (0, 0, 0, 1): (z_hat[3], 1.0)}
    return Quadratics([{"terms": terms}], extra=band), z_hat


def test_the_beta_floor_probe_opens_a_lower_end_closed_by_a_cliff(monkeypatch):
    """Measured in walker step 1: an r-bar walk stopped at a cliff at 0.041 while beta's floor sat at deviance 1.7."""
    obj, z_hat = _banded()
    w = W.walk(obj, "rbar_over_eta", z_hat, -obj.nll_grad(z_hat)[0], bounds=BOUNDS, box4=BOX, seed_z=z_hat)
    assert w["stops"]["lower"] == "beta_floor" and not math.isfinite(w["ci"]["ci_lo"]), (w["stops"], w["ci"])
    assert w["probe"]["opened"] and w["probe"]["stop_before"] == "cliff" and w["probe"]["dev"] <= THR
    monkeypatch.setattr(W, "_beta_floor_probe", lambda obj, bases, bounds: None)               # the plant
    obj, z_hat = _banded()
    w = W.walk(obj, "rbar_over_eta", z_hat, -obj.nll_grad(z_hat)[0], bounds=BOUNDS, box4=BOX, seed_z=z_hat)
    assert w["stops"]["lower"] == "cliff" and math.isfinite(w["ci"]["ci_lo"])


def test_the_bisection_puts_an_end_within_0_01_dex_of_a_sharp_rise():
    """Measured in walker step 1: two ends interpolated 0.033 and 0.16 dex short across wide brackets. Planted: flat
    to 1 dex, then a quadratic rise of 0.05 dex scale; nodes at 0, +-0.5 and +-1.5 dex."""
    from sraging.compat.v033 import fp_gof as GOF
    a, s = 1.0 * DEX, 0.05 * DEX

    def plateau(z):
        d = float(np.asarray(X) @ z) - XA
        u = max(abs(d) - a, 0.0) / s
        return 0.5 * u * u, (u / s * math.copysign(1.0, d)) * np.asarray(X, float)
    obj = Quadratics([{"terms": {RIDGE: (float(np.asarray(RIDGE) @ Z_HAT), 1.0), BETA: (math.log(0.1), 1.0),
                                 KAP: (math.log(0.01), 1.0)}}], extra=plateau)
    nodes = []
    for dx in (-1.5, -0.5, 0.0, 0.5, 1.5):
        r = W._best(obj, "eta_over_eps", XA + dx * DEX, [Z_HAT], BOUNDS)
        nodes.append((XA + dx * DEX, r[1], r[0]))
    true = (a + s * math.sqrt(THR)) / DEX

    def ends(ns):
        ci = GOF.profile_ci([{"value": math.exp(c), "loglik": ll} for c, ll, _ in ns])
        return (XA - math.log(ci["ci_lo"])) / DEX, (math.log(ci["ci_hi"]) - XA) / DEX
    assert all(true - e > 0.3 for e in ends(nodes))                               # interpolated: far too narrow
    nodes, solves = W._bisect(obj, "eta_over_eps", list(nodes), Z_HAT, BOUNDS)
    assert all(abs(e - true) <= 0.01 for e in ends(nodes)) and 0 < solves <= 2 * W.BISECT_MAX


def test_reconcile_walks_from_an_archived_point_to_the_true_edge(monkeypatch):
    """A walk that missed basin B (the beta-spread plant) and one archived point in B, 0.15 dex beyond its lower end:
    the lower side reaches B's edge; the upper side, which nothing contradicts, is left as it was; and the archive no
    longer contradicts the walk. A point above the reference becomes a node."""
    monkeypatch.setattr(W, "beta_span_starts", lambda key, z, bounds, n=W.BETA_SPAN_N: [])
    obj = far_beta()
    ll_hat = -obj.nll_grad(Z_FAR)[0]
    w = W.walk(obj, "eta_over_eps", Z_FAR, ll_hat, bounds=BOUNDS, box4=BOX, seed_z=Z_FAR)
    z_b = Z_FAR.copy()
    z_b[1], z_b[0], z_b[2] = math.log(10.0), z_b[0] - 0.5 * DEX, z_b[2] + 0.5 * DEX    # x = xA - 1 dex, in B
    Z, LL = np.array([z_b]), np.array([-obj.nll_grad(z_b)[0]])
    con, _ = W.contradictions("eta_over_eps", w, Z, LL)
    assert list(con) == ["lower"]
    w2, rec = W.reconcile(obj, "eta_over_eps", w, Z, LL, z_hat=Z_FAR, ll_hat=ll_hat, bounds=BOUNDS, box4=BOX,
                          seed_z=Z_FAR)
    lo, hi = _edges(w2)
    assert abs(lo - edge_dex_true()) <= 0.01 and hi == _edges(w)[1], (lo, hi)
    assert rec["lower"]["walked_on"] and w2["reconciled"] == [rec] and w2["evals"] > w["evals"]
    assert W.contradictions("eta_over_eps", w2, Z, LL)[0] == {}
    assert W.reconcile(obj, "eta_over_eps", w2, Z, LL, z_hat=Z_FAR, ll_hat=ll_hat, bounds=BOUNDS, box4=BOX)[1] is None
    z_up = np.asarray(w2["best_z"], float).copy()                        # a point 1 nat above the reference
    con, _ = W.contradictions("eta_over_eps", w2, np.array([z_up]), np.array([w2["ll_ref"] + 1.0]))
    assert list(con) == ["above"]


def test_a_side_that_crawls_past_the_edge_stops_stalled(monkeypatch):
    """Measured on the ITP example: past r-bar/eta's lower end every free-kappa solve 0.005 dex out failed, so the
    step hovered above its floor and the side crawled 24 nodes over 0.0034 dex. Planted: a quadratic in log(eta/eps)
    of 0.1 dex scale; past its 95% edge a solve fails unless its start lies within 0.0005 dex of the target."""
    from sraging.claims import combinations as K
    sigma, near = 0.1 * DEX, 0.0005 * DEX
    edge = sigma * math.sqrt(THR)
    solve, tries = W._solve, []

    def crawl(obj, key, c, z0, bounds):
        tries.append(c)
        if abs(c - XA) > edge and abs(c - K.log_g(key, z0)) > near:
            return None
        return solve(obj, key, c, z0, bounds)

    def run():
        tries.clear()
        obj = Quadratics([{"terms": {X: (XA, sigma), RIDGE: (float(np.asarray(RIDGE) @ Z_HAT), 1.0),
                                     BETA: (math.log(0.1), 1.0), KAP: (math.log(0.01), 1.0)}}])
        w = W.walk(obj, "eta_over_eps", Z_HAT, -obj.nll_grad(Z_HAT)[0], bounds=BOUNDS, box4=BOX, seed_z=Z_HAT)
        return w, [abs(e - edge / DEX) for e in _edges(w)], len(tries)
    monkeypatch.setattr(W, "_solve", crawl)
    w, err, n = run()
    assert w["stops"] == {"lower": "stalled", "upper": "stalled"} and max(err) <= 0.01, (w["stops"], err)
    monkeypatch.setattr(W, "STALL_NODES", 10 ** 6)                                  # the plant: no stall stop
    w0, err0, n0 = run()
    assert w0["stops"] == {"lower": "max_nodes", "upper": "max_nodes"} and max(err0) <= 0.01, (w0["stops"], err0)
    past = [sum(abs(c - XA) > edge for c, _, _ in x["nodes"]) for x in (w, w0)]     # measured: 2 and 28 nodes,
    assert past[0] <= 2 * (W.STALL_NODES + 1) and past[1] > 5 * past[0] and n0 > n, (past, n, n0)   # 128, 188 solves
