"""The combination walker: the profile likelihood of a combination g(theta) over the calibrator's original box
(spec 003, T029; FR-015, FR-017; research R8).

Each node maximises the port's log-likelihood subject to log g(z) = c and z inside the box (SLSQP: the box as bounds,
one equality constraint, the port's `nll_grad` for values and gradients -- every call lands in the objective's memo,
so the walk's points join the fit's archive). The walk copies jax2's own defences against false caps
(compat/jax2/estimator.py `_walk_side`, `_polish_edges`, `_repair`, `_kappa_corner_starts`):
  * the centre node is re-solved at g(theta-hat) from every start, so a stale theta-hat does not anchor the walk;
  * steps in dex: 0.05 first, x3 when the deviance is below 1, x2 when it rose by less than 0.3, at most 2 dex;
  * a node whose deviance jumps by more than 4 nats, or is not finite, is rejected and the step divided by 3 (at
    most 24 rejections); at the minimum step (1e-4 dex) the jumped node is KEPT and the side stops as "cliff"
    (jax2 `_walk_side` keeps it, so the profile crosses there instead of reading open);
  * a side stops at deviance 10, at the box image of g (clamped to 1.01 / 0.99 of its edges), after 24 nodes, or --
    on the lower side of a combination containing beta -- once a node's beta sits at its floor ("beta_floor"): from
    there the combination is consistent with no removal, and going lower only forces eta up (r-bar/eta) or kappa to
    300, so the claim is open below, [0, cap] (sr-naveh implementation review 3); or -- past the threshold -- once
    its last 3 nodes span less than 0.01 dex ("stalled", added below);
  * mid-walk nodes start from the previous node and -- for the combinations that are not invariant on the ridge
    (eta, beta, eps) -> lambda (eta, beta, eps) -- also from the previous node moved along the ridge by the step and
    from theta-hat; any other node that fails or jumps is re-solved from theta-hat before the step is rejected;
  * the edge polish (jax2 `_polish_edges`): on each side, the first node outward at or past the threshold is
    re-solved from every start (theta-hat, the seed, the kappa corners of theta-hat and of the node itself, the h
    starts of jax2_h, the neighbouring nodes, ridge starts at +-1 dex for the non-invariant combinations, the best
    archive points near the node); if it falls inside, the edge has moved and the polish repeats at the next node
    outward (at most 12 moves), and walks on from the last node, once, if that pushes the edge past it; the node just
    inside each edge is re-solved too (`profile_ci` interpolates between the two); a monotonicity repair re-solves
    inner nodes from their outer neighbour (> 0.2 nats, 3 passes).
Added after walker step 1's archive check (SC-004; Naveh, 2026-10-06: 6 of 183 ends were contradicted by points the
fit and the walks had already evaluated):
  * for the combinations whose constraint leaves beta free (eta/eps, sqrt(eta)/eps), the centre and every polished
    node also start from beta spread across its box (6 values) -- measured: the inside set of three sqrt(eta)/eps
    lower ends lay on a branch at beta 10-17, kappa 160-300, 9-10 dex from theta-hat's beta, 1.1-1.2 dex past the end;
  * before the lower end of a combination containing beta is closed, the best fit with beta at its floor is found
    (L-BFGS-B from theta-hat and the lowest node, kappa as they are and at 1 and 300); inside the threshold, it joins
    the profile as a node and the side stops "beta_floor", open below -- measured: an r-bar walk stopped at a cliff
    at 0.041 while beta's floor sat at deviance 1.7;
  * the crossing that sets each end is bisected until its bracket is at most 0.01 dex (SC-004's resolution) --
    measured: two ends interpolated 0.033 and 0.16 dex short across wide brackets;
  * `reconcile`: a walk is final only when no archived point contradicts it -- none beyond a stated end by more than
    0.01 dex inside the threshold, none above its reference. A contradicted side gets nodes solved from the
    contradicting points (spread from the end to the farthest, and the best) and walks on from the farthest; then
    the polish, the probe, the repair and the bisection run again. The caller repeats it while the archive grows.
Added after the ITP example (SC-004; Naveh, 2026-10-08):
  * a side past the threshold whose last 3 accepted nodes span less than 0.01 dex (SC-004's resolution) stops
    "stalled": its crossing is already bracketed that finely -- measured: the ITP cohort's r-bar/eta lower side, where
    every free-kappa solve 0.005 dex further out failed, so the step (/3 on each failure, x2 on each node) hovered
    above its 1e-4-dex floor and the side took 24 nodes over 0.0034 dex (deviance 3.83-4.05): 29998 of the walk's
    42133 evaluations. With the stop, 8 nodes and 6898 evaluations, the same claims, the lead pair 3.4x the fit (6.3x).
One deviance reference throughout the polish, the repair and the interval: the profile's own maximum, which is at
least the log-likelihood at theta-hat (sr-naveh implementation review 5). The interval is the port's `profile_ci`
over the nodes (the rule jax2 uses for a parameter).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from sraging.claims import combinations as K

DEX = math.log(10.0)
STEP0, STEP_MAX, STEP_MIN = 0.05, 2.0, 1e-4
DEV_STOP, JUMP_NATS = 10.0, 4.0
MAX_NODES, MAX_REJECT, REPAIR_NATS, REPAIR_PASSES = 24, 24, 0.2, 3
POLISH_MAX_MOVES, POLISH_GAIN_TOL = 12, 1e-3          # jax2 estimator.py:173-174
FEAS_TOL = 1e-7
#: SLSQP's ftol is ABSOLUTE on the objective (the negative log-likelihood, ~1e3 nats): 1e-12 sits below its float
#: round-off, so the solver ran to maxiter (measured: 354 evaluations a solve on the small cohort's
#: beta/(eta(kappa+1))); 1e-9 nats is 1e-6 of SC-004's resolution and of the 1e-3-nat polish tolerance
SLSQP_FTOL, SLSQP_MAXITER = 1e-9, 200
KAPPA_CORNERS = (1e-3, 1.0, 300.0)
EDGE_DEX = 0.01                                       # the widening's edge distance (compat.jax2 claims.SC_EDGE_DEX)
THR = 3.8415
BETA_KEYS = frozenset({"rbar_over_eta", "beta_over_eta_kp1", "rbar", "beta_over_kp1"})
#: d log g / d lambda along the ridge (eta, beta, eps) -> lambda (eta, beta, eps), for the keys that are not invariant
RIDGE_SLOPE = {"sqrt_eta_over_eps": -0.5, "rbar": 1.0, "beta_over_kp1": 1.0}
RIDGE_DEX = (-1.0, 1.0)
BETA_SPAN_N = 6                                       # beta values spread over its box, for the keys that leave it free
PROBE_KAPPAS = (1.0, 300.0)                           # the beta-floor probe's kappa starts, beside each base's own
BISECT_DEX, BISECT_MAX = 0.01, 12                     # SC-004's resolution; 2 dex halves to it in 8 steps
STALL_NODES, STALL_DEX = 3, BISECT_DEX                # past the threshold, nodes spanning less than this: "stalled"
RECONCILE_ROUNDS, RECONCILE_SEEDS = 3, 6              # the caller's rounds; seeds spread from the end to the farthest


class _Counter:
    def __init__(self, obj):
        self.obj, self.n0 = obj, (obj.n_g, obj.n_f)

    def evals(self):
        return (self.obj.n_g - self.n0[0]) + (self.obj.n_f - self.n0[1])


def _solve(obj, key, c, z0, bounds):
    """One constrained maximisation from z0; (z, ll) or None if it did not land on the constraint inside the box."""
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    z0 = np.clip(K.project(key, z0, c), lo + 1e-12, hi - 1e-12)
    try:
        res = minimize(lambda z: obj.nll_grad(z), z0, jac=True, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "eq", "fun": lambda z: K.log_g(key, z) - c,
                                     "jac": lambda z: K.grad_log_g(key, z)}],
                       options={"maxiter": SLSQP_MAXITER, "ftol": SLSQP_FTOL})
    except (ValueError, FloatingPointError):
        return None
    z = np.clip(res.x, lo, hi)
    if not np.isfinite(res.fun) or abs(K.log_g(key, z) - c) > FEAS_TOL:
        return None
    nll, _ = obj.nll_grad(z)
    return (z, -float(nll)) if np.isfinite(nll) else None


def _best(obj, key, c, starts, bounds):
    best = None
    for z0 in starts:
        r = _solve(obj, key, c, z0, bounds)
        if r is not None and (best is None or r[1] > best[1]):
            best = r
    return best


def kappa_corner_starts(z):
    """jax2 `_kappa_corner_starts` on a full z: kappa at 1e-3, 1 and 300 with beta rescaled to keep beta/(1+kappa)."""
    out = []
    k = math.exp(z[K.KAP])
    for kc in KAPPA_CORNERS:
        if abs(math.log(kc / k)) < 1e-6:
            continue                                     # that corner IS the current solution
        s = np.asarray(z, float).copy()
        s[K.BETA] += math.log((1.0 + kc) / (1.0 + k))
        s[K.KAP] = math.log(kc)
        out.append(s)
    return out


def ridge_starts(key, z, dex=RIDGE_DEX):
    """z moved along the scaling ridge by `dex` (eta, beta, eps all x 10^dex), for the keys the ridge moves; the
    solver's projection then restores log g."""
    if key not in RIDGE_SLOPE:
        return []
    out = []
    for d in dex:
        s = np.asarray(z, float).copy()
        s[[K.ETA, K.BETA, K.EPS]] += d * DEX
        out.append(s)
    return out


def beta_span_starts(key, z, bounds, n=BETA_SPAN_N):
    """z with beta at n values spread evenly (in log) over its box (`bounds`: the walk's log box), for the keys whose
    constraint leaves beta free (the projection moves eta for those; for the others it moves beta itself, so the
    spread would be undone)."""
    if key in BETA_KEYS or bounds is None:
        return []
    lo, hi = bounds[K.BETA]
    out = []
    for i in range(n):
        s = np.asarray(z, float).copy()
        s[K.BETA] = lo + (i + 0.5) / n * (hi - lo)
        out.append(s)
    return out


def all_starts(obj, key, c, z_hat, seed_z, archive, hazard_h, node_z=None, bounds=None):
    starts = [z_hat, seed_z] + kappa_corner_starts(z_hat)
    if node_z is not None:
        starts += [node_z] + kappa_corner_starts(node_z) + ridge_starts(key, node_z)
    starts += ridge_starts(key, z_hat)
    starts += beta_span_starts(key, z_hat if node_z is None else node_z, bounds)
    if hazard_h:
        for base in ([z_hat] if node_z is None else [z_hat, node_z]):
            for hz in hazard_h:
                s = np.asarray(base, float).copy()
                s[4] = math.log(hz)
                starts.append(s)
    if archive is not None and archive[0].size:
        Z, LL = archive
        lg = np.array([K.log_g(key, z) for z in Z])
        near = np.flatnonzero(np.abs(lg - c) < DEX)              # within one dex of the node
        for j in near[np.argsort(-LL[near])][:3]:
            starts.append(Z[j])
    out = []
    for s in starts:                                  # an identical start gives the identical solve: drop repeats
        if s is None or len(s) != len(z_hat):
            continue
        s = np.asarray(s, float)
        if not any(np.array_equal(s, u) for u in out):
            out.append(s)
    return out


def _at_beta_floor(key, z, bounds):
    return key in BETA_KEYS and z[K.BETA] <= bounds[K.BETA][0] + EDGE_DEX * DEX


def _beta_floor_probe(obj, bases, bounds):
    """The best fit with beta fixed at its floor (no removal), by L-BFGS-B from each base with its own kappa and at
    PROBE_KAPPAS: (z, ll) or None. Every combination containing beta is proportional to it, so this point lies
    below every node of the walk's lower side."""
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    fixed = list(bounds)
    fixed[K.BETA] = (bounds[K.BETA][0], bounds[K.BETA][0])
    starts = []
    for base in bases:
        if base is None or len(base) != len(bounds):
            continue
        for lk in [float(base[K.KAP])] + [math.log(k) for k in PROBE_KAPPAS]:
            s = np.clip(np.asarray(base, float).copy(), lo, hi)
            s[K.BETA], s[K.KAP] = bounds[K.BETA][0], min(max(lk, lo[K.KAP]), hi[K.KAP])
            if not any(np.array_equal(s, u) for u in starts):
                starts.append(s)
    best = None
    for s in starts:
        try:
            res = minimize(lambda z: obj.nll_grad(z), s, jac=True, method="L-BFGS-B", bounds=fixed,
                           options={"maxiter": SLSQP_MAXITER})
        except (ValueError, FloatingPointError):
            continue
        z = np.clip(res.x, lo, hi)
        nll, _ = obj.nll_grad(z)
        if np.isfinite(nll) and (best is None or -float(nll) > best[1]):
            best = (z, -float(nll))
    return best


def _bisect(obj, key, nodes, z_hat, bounds, thr=THR):
    """Each end's crossing -- between the outermost node inside the threshold and its outer neighbour, the bracket
    `profile_ci` interpolates in -- halved until it is at most BISECT_DEX wide, each half solved from both bracket
    nodes and theta-hat. Returns (nodes, solves)."""
    solves = 0
    for sgn in (-1, +1):
        for _ in range(BISECT_MAX):
            nodes.sort(key=lambda n: n[0])
            ll_ref = max(n[1] for n in nodes)
            acc = [i for i, n in enumerate(nodes) if 2.0 * (ll_ref - n[1]) <= thr]
            if not acc:
                break
            i_in = acc[0] if sgn < 0 else acc[-1]
            i_out = i_in + sgn
            if not 0 <= i_out < len(nodes):
                break                                    # no node beyond: the side reads open
            (c_in, _, z_in), (c_out, _, z_out) = nodes[i_in], nodes[i_out]
            if abs(c_out - c_in) <= BISECT_DEX * DEX * (1 + 1e-9):
                break
            c_mid = 0.5 * (c_in + c_out)
            r = _best(obj, key, c_mid, [z_in, z_out, z_hat], bounds)
            solves += 1
            if r is None:
                break                                    # the bracket stays; profile_ci interpolates in it
            nodes.append((c_mid, r[1], r[0]))
    nodes.sort(key=lambda n: n[0])
    return nodes, solves


def _walk_side(obj, key, side, c0, z0, dev0, ll_ref, z_hat, bounds, c_lo, c_hi, max_nodes=MAX_NODES):
    """One side of the walk from the node (c0, z0) at deviance dev0: ([(c, ll, z)], reason). jax2 `_walk_side`'s
    step, rejection and stop rules on log g."""
    step, prev_z, prev_dev, c = STEP0, np.asarray(z0, float), float(dev0), float(c0)
    out, reason = [], "max_nodes"
    accepted = rejects = 0
    while accepted < max_nodes and rejects < MAX_REJECT:
        c_new = c + side * step * DEX
        at_edge = (side < 0 and c_new <= c_lo) or (side > 0 and c_new >= c_hi)
        if at_edge:
            c_new = c_lo if side < 0 else c_hi
            if abs(c_new - c) < 1e-9 * DEX:
                reason = "box_edge"
                break
        starts = [prev_z]                                        # the warm start, as jax2's walk
        if key in RIDGE_SLOPE:                                   # on the ridge: its continuation, and theta-hat
            s = prev_z.copy()                                    # (measured on the small cohort: without it the
            s[[K.ETA, K.BETA, K.EPS]] += (c_new - c) / RIDGE_SLOPE[key]   # sqrt(eta)/eps lower end came out
            starts += [s, z_hat]                                 # 0.026 dex narrower)
        r = _best(obj, key, c_new, starts, bounds)
        dev = float("inf") if r is None else 2.0 * (ll_ref - r[1])
        if not np.isfinite(dev) or dev - prev_dev > JUMP_NATS:   # before rejecting the step: theta-hat's start
            r2 = _best(obj, key, c_new, [z_hat], bounds)
            if r2 is not None and (r is None or r2[1] > r[1]):
                r, dev = r2, 2.0 * (ll_ref - r2[1])
        if not np.isfinite(dev) or dev - prev_dev > JUMP_NATS:
            if step > STEP_MIN:
                step = max(step / 3.0, STEP_MIN)
                rejects += 1
                continue
            if r is not None:                                    # jax2 keeps the jumped node: the profile crosses
                out.append((c_new, r[1], r[0]))
            reason = "cliff"
            break
        out.append((c_new, r[1], r[0]))
        accepted += 1
        prev_z, c = r[0], c_new
        if at_edge:
            reason = "box_edge"
            break
        if side < 0 and _at_beta_floor(key, r[0], bounds):
            reason = "beta_floor"
            break
        if dev >= DEV_STOP:
            reason = "deviance"
            break
        if dev > THR and len(out) > STALL_NODES and abs(out[-1][0] - out[-1 - STALL_NODES][0]) < STALL_DEX * DEX:
            reason = "stalled"                                   # the crossing is bracketed at SC-004's resolution
            break
        step = min(step * (3.0 if dev < 1.0 else 2.0 if dev - prev_dev < 0.3 else 1.0), STEP_MAX)
        prev_dev = dev
    return out, reason


def _ctx(key, z_hat, bounds, box4):
    """The walk's context (theta-hat, the box, g's walkable range: its box image clamped to 1.01 / 0.99 of its
    edges) and g's box image."""
    g_lo, g_hi = K.box_image(key, box4)
    return (dict(z_hat=np.asarray(z_hat, float), bounds=bounds, c_lo=math.log(g_lo * 1.01),
                 c_hi=math.log(g_hi * 0.99)), (g_lo, g_hi))


def walk(obj, key, z_hat, ll_hat, *, bounds, box4, seed_z=None, archive=None, hazard_h=()):
    """The combination's profile: {nodes: [(c, ll, z)], ci, stops, evals, gain, ll_ref, best_z, box_image, probe,
    bisections, reconciled}."""
    cnt = _Counter(obj)
    z_hat = np.asarray(z_hat, float)
    c_hat = K.log_g(key, z_hat)
    ctx, image = _ctx(key, z_hat, bounds, box4)
    centre = (c_hat, float(ll_hat), z_hat)
    r = _best(obj, key, c_hat, all_starts(obj, key, c_hat, z_hat, seed_z, archive, hazard_h, bounds=bounds), bounds)
    if r is not None and r[1] > centre[1] + POLISH_GAIN_TOL:
        centre = (c_hat, r[1], r[0])
    nodes = [centre]
    stops = {}
    for side in (-1, +1):                       # the reference: the centre, at least the log-likelihood at theta-hat
        ns, stops[side] = _walk_side(obj, key, side, c_hat, centre[2], 0.0, centre[1], **ctx)
        nodes += ns
    return _finish(obj, key, nodes, stops, ll_hat, seed_z, archive, hazard_h, ctx, image, cnt)


def _finish(obj, key, nodes, stops, ll_hat, seed_z, archive, hazard_h, ctx, image, cnt, evals0=0, history=()):
    """The edge polish, the beta-floor probe, the repair, the bisection and the interval, on a walk's nodes."""
    z_hat, bounds = ctx["z_hat"], ctx["bounds"]
    nodes.sort(key=lambda n: n[0])
    nodes, ll_ref = _polish(obj, key, nodes, stops, z_hat, seed_z, archive, hazard_h, ctx)
    probe = None
    if key in BETA_KEYS and stops[-1] not in ("box_edge", "beta_floor"):
        r = _beta_floor_probe(obj, [z_hat, nodes[0][2]], bounds)
        if r is not None:
            probe = {"dev": float(2.0 * (max(ll_ref, r[1]) - r[1])), "log10_g": float(K.log_g(key, r[0]) / DEX),
                     "stop_before": stops[-1], "opened": False}
            if probe["dev"] <= THR:             # no removal fits: the lower side is open, as at a beta-floor node
                nodes.append((K.log_g(key, r[0]), r[1], r[0]))
                stops[-1], probe["opened"] = "beta_floor", True
    nodes = _repair(obj, key, nodes, bounds)
    nodes, n_bisect = _bisect(obj, key, nodes, z_hat, bounds)
    ll_ref = max(ll_ref, max(n[1] for n in nodes))
    from sraging.compat.v033 import fp_gof as GOF
    rows = [{"value": math.exp(c), "loglik": ll} for c, ll, _ in nodes]
    ci = GOF.profile_ci(rows, level=0.95)
    best = max(nodes, key=lambda n: n[1])
    return {"key": key, "nodes": [(float(c), float(ll), [float(x) for x in z]) for c, ll, z in nodes], "ci": ci,
            "stops": {("lower" if s < 0 else "upper"): v for s, v in stops.items()},
            "evals": evals0 + cnt.evals(), "gain": float(best[1] - ll_hat), "ll_ref": float(ll_ref),
            "best_z": [float(x) for x in best[2]], "box_image": image, "probe": probe, "bisections": n_bisect,
            "reconciled": list(history)}


def contradictions(key, w, Z, LL, thr=THR, edge_dex=EDGE_DEX):
    """What the archive (Z, LL) says against the walk w: ({side: indices of the points beyond its stated end by more
    than edge_dex at deviance < thr from its reference, farthest first; "above": the points above that reference,
    best first}, log g of every point). The acceptance archive check (tools/acceptance_003_walker.py) is this test."""
    LL = np.asarray(LL, float)
    lg = np.array([K.log_g(key, z) for z in Z]) if LL.size else np.empty(0)
    dev = 2.0 * (w["ll_ref"] - LL)
    out = {}
    for side, end in (("lower", w["ci"]["ci_lo"]), ("upper", w["ci"]["ci_hi"])):
        if end is None or not math.isfinite(end):
            continue
        beyond = ((math.log(end) - lg) if side == "lower" else (lg - math.log(end))) / DEX
        idx = np.flatnonzero((dev < thr) & (beyond > edge_dex))
        if idx.size:
            out[side] = idx[np.argsort(-beyond[idx], kind="stable")]
    above = np.flatnonzero(LL > w["ll_ref"] + POLISH_GAIN_TOL)
    if above.size:
        out["above"] = above[np.argsort(-LL[above], kind="stable")]
    return out, lg


def reconcile(obj, key, w, Z, LL, *, z_hat, ll_hat, bounds, box4, seed_z=None, archive=None, hazard_h=()):
    """One round of the final pass: (walk, record); the walk unchanged and record None when no archived point
    contradicts it. A contradicted side gets nodes at the log g of up to RECONCILE_SEEDS of the contradicting points,
    spread from the end to the farthest, and of the best of them, each solved from the point itself (it already
    satisfies the constraint, so the node is at least as high); if its outermost node is then inside, the side walks
    on from there. A point above the reference becomes a node too. Then the walk is finished again (polish, probe,
    repair, bisection, interval)."""
    con, lg = contradictions(key, w, Z, LL)
    if not con:
        return w, None
    cnt = _Counter(obj)
    Z, LL = np.asarray(Z, float), np.asarray(LL, float)
    ctx, image = _ctx(key, z_hat, bounds, box4)
    nodes = [(float(c), float(ll), np.asarray(z, float)) for c, ll, z in w["nodes"]]
    stops = {-1: w["stops"]["lower"], +1: w["stops"]["upper"]}
    rec = {}

    def add(j):
        c = min(max(float(lg[j]), ctx["c_lo"]), ctx["c_hi"])
        r = _best(obj, key, c, [Z[j]], bounds)
        if r is not None:
            nodes.append((c, r[1], r[0]))

    for side, sgn in (("lower", -1), ("upper", +1)):
        idx = con.get(side)
        if idx is None:
            continue
        end = math.log(w["ci"]["ci_lo"] if sgn < 0 else w["ci"]["ci_hi"])
        far = int(idx[0])
        picks = {int(idx[np.argmin(np.abs(lg[idx] - t))]) for t in np.linspace(end, lg[far], RECONCILE_SEEDS + 1)[1:]}
        picks.add(int(idx[np.argmax(LL[idx])]))
        # the old nodes beyond the farthest point are all outside (they lie beyond the end) and on a lower branch
        # than that point: the walk from it replaces them
        nodes[:] = [n for n in nodes if sgn * (n[0] - float(lg[far])) <= 0]
        for j in sorted(picks, key=lambda j: sgn * lg[j]):
            add(j)
        nodes.sort(key=lambda n: n[0])
        ll_ref = max(n[1] for n in nodes)
        c_o, ll_o, z_o = nodes[0] if sgn < 0 else nodes[-1]
        walked = 2.0 * (ll_ref - ll_o) <= THR
        if walked:
            ns, stops[sgn] = _walk_side(obj, key, sgn, c_o, z_o, 2.0 * (ll_ref - ll_o), ll_ref, **ctx)
            nodes += ns
        rec[side] = {"points": int(idx.size), "farthest_dex": float(sgn * (lg[far] - end) / DEX),
                     "seeds": len(picks), "walked_on": walked}
    if "above" in con:
        j = int(con["above"][0])
        add(j)
        rec["above"] = {"points": int(con["above"].size), "nats": float(LL[j] - w["ll_ref"])}
    hist = list(w.get("reconciled") or []) + [rec]
    return _finish(obj, key, nodes, stops, ll_hat, seed_z, archive, hazard_h, ctx, image, cnt, evals0=w["evals"],
                   history=hist), rec


def _polish(obj, key, nodes, stops, z_hat, seed_z, archive, hazard_h, ctx, thr=THR):
    """jax2 `_polish_edges` on log g, with one reference (the profile's maximum, re-read after every improvement;
    when it rises, both sides are polished again, at most twice)."""
    bounds = ctx["bounds"]
    for _round in range(3):
        ll_ref = max(n[1] for n in nodes)
        for sgn in (-1, +1):
            moves, extended = 0, False
            while moves <= POLISH_MAX_MOVES:
                nodes.sort(key=lambda n: n[0])
                jmax = int(np.argmax([n[1] for n in nodes]))
                order = list(range(jmax - 1, -1, -1)) if sgn < 0 else list(range(jmax + 1, len(nodes)))
                j_out = next((j for j in order if 2.0 * (ll_ref - nodes[j][1]) >= thr), None)
                if j_out is None:
                    if moves == 0 or extended or not order or stops[sgn] in ("box_edge", "beta_floor"):
                        break
                    c_l, ll_l, z_l = nodes[order[-1]]
                    ns, stops[sgn] = _walk_side(obj, key, sgn, c_l, z_l, 2.0 * (ll_ref - ll_l), ll_ref, **ctx)
                    extended = True
                    nodes += ns
                    continue
                k_pos = order.index(j_out)
                c_j, ll_j, z_j = nodes[j_out]
                extra = [nodes[order[k_pos + 1]][2]] if k_pos + 1 < len(order) else []
                if k_pos > 0:
                    extra.append(nodes[order[k_pos - 1]][2])
                starts = all_starts(obj, key, c_j, z_hat, seed_z, archive, hazard_h, node_z=z_j, bounds=bounds) + extra
                r = _best(obj, key, c_j, starts, bounds)
                if r is None or r[1] <= ll_j + POLISH_GAIN_TOL:
                    inner = order[k_pos - 1] if k_pos > 0 else jmax       # the node just inside the edge
                    c_i, ll_i, z_i = nodes[inner]
                    ri = _best(obj, key, c_i, all_starts(obj, key, c_i, z_hat, seed_z, archive, hazard_h,
                                                         node_z=z_i, bounds=bounds) + [z_j], bounds)
                    if ri is not None and ri[1] > ll_i + POLISH_GAIN_TOL:
                        nodes[inner] = (c_i, ri[1], ri[0])
                    break
                nodes[j_out] = (c_j, r[1], r[0])
                if 2.0 * (ll_ref - r[1]) >= thr:
                    break                                          # improved, but still places the edge
                moves += 1
        if max(n[1] for n in nodes) <= ll_ref + POLISH_GAIN_TOL:
            break
    nodes.sort(key=lambda n: n[0])
    return nodes, max(n[1] for n in nodes)


def _repair(obj, key, nodes, bounds):
    """Monotone deviance outward from the maximum (jax2 `_repair`): an inner node lower than its outer neighbour by
    more than 0.2 nats is re-solved from the neighbour's z."""
    for _ in range(REPAIR_PASSES):
        changed = False
        i_max = int(np.argmax([n[1] for n in nodes]))
        for i in range(i_max - 1, -1, -1):              # below the maximum, the profile must not rise going outward
            if nodes[i][1] > nodes[i + 1][1] + REPAIR_NATS:
                r = _best(obj, key, nodes[i + 1][0], [nodes[i][2]], bounds)
                if r is not None and r[1] > nodes[i + 1][1]:
                    nodes[i + 1] = (nodes[i + 1][0], r[1], r[0])
                    changed = True
        for i in range(i_max + 1, len(nodes)):
            if nodes[i][1] > nodes[i - 1][1] + REPAIR_NATS:
                r = _best(obj, key, nodes[i - 1][0], [nodes[i][2]], bounds)
                if r is not None and r[1] > nodes[i - 1][1]:
                    nodes[i - 1] = (nodes[i - 1][0], r[1], r[0])
                    changed = True
        if not changed:
            break
    return nodes
