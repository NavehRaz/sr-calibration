"""`mle_profile` with EXACT-GRADIENT optimisers on a JAX likelihood.

A vendored copy of `SRK.benchmark.methods.mle_profile` / `_walk_side` / `_profile_ll_at` (as pinned
by the reference, `mle_profile` v1.0.0), called the way `SRK.fit_cohort.fit` calls it. EVERYTHING
statistical is the same: the likelihood, the data-sized grid and its dt-ceiling rule, the seed, the
two starts, the prior box, the adaptive walk and its step rules, the monotonicity repair, the
re-centring check, and `fp_gof.profile_ci`. What differs, each marked `# JAX:`:

  1. the likelihood is `fp_jax.make_loglik`, the JAX twin of `fp_loglik_physical(scheme="bdf2")`;
  2. every Nelder-Mead search -- the two 4-D MLE starts and every 3-D profile node -- is bounded
     L-BFGS-B on the exact `jax.value_and_grad`, instead of a derivative-free simplex;
  3. (v0.2) a second start from the seed where a node lands in a DEGENERATE corner (`_degenerate`);
  4. (v0.3) an EDGE POLISH (`_polish_edges`): the node that places each interval edge is re-solved
     from fresh starts, because a warm-start chain that stalls once stays stalled outward;
  5. (v0.3) a memo of exact repeats and an adaptive rule for when the corner restart is worth running;
  6. (v0.3.3) the walk's node budget counts ACCEPTED nodes only (`_walk_side`, P1). This one is a
     change to the reference's walk, not a port: the reference shares the defect.

SAFETY NET. A gradient method cannot leave a region where every term sits on the -700 log floor:
the gradient there is exactly zero, and L-BFGS-B stops where it started. Nelder-Mead can wander off
such a plateau. The reference guards the MLE against it with a high-epsilon second start; here,
additionally, a search whose starting gradient is exactly zero, or whose result is not finite, first
tries the seed start and then falls back to Nelder-Mead on the same JAX function. MEASURED (v0.2):
the fallback rescued 30/53 and 21/23 plateau nodes to near-MLE loglik, so it is load-bearing.

Checkpointing is dropped: the tester calls `fit_cohort.fit`, which never passes `ckpt`.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize

PARAMS = ("eta", "beta", "epsilon", "kappa")

# Walk constants, verbatim from SRK/benchmark/methods.py
_DEV_STOP = 10.0
_DEV_MAX_STEP = 4.0
_STEP0_DEX = 0.05
_STEP_MIN_DEX = 1e-4
_STEP_MAX_DEX = 2.0
_REPAIR_MIN_NATS = 0.2

# JAX: L-BFGS-B stopping rules. The reference's simplex stops at fatol=1e-4 (MLE) / 1e-3 (nodes)
# in NATS; these are at least as tight. `ftol` is relative, and |nll| is O(1e3-1e5) here.
# MEASURED: loosening ftol or gtol lost 0.13-0.32 nats at flat nodes -- keep them tight.
_LBFGS_MLE = dict(maxiter=1000, maxfun=2000, ftol=1e-13, gtol=1e-6)
_LBFGS_NODE = dict(maxiter=400, maxfun=800, ftol=1e-13, gtol=1e-6)

# v0.3: adaptive corner restarts (`_want_restart`)
_RESTART_GAIN_TOL = 1e-3     # nats: a restart gaining less than this "gained nothing"
_RESTART_BACKOFF_N = 2       # consecutive zero-gain restarts on one walk side before backing off

# v0.3: edge polish (`_polish_edges`)
_POLISH_GAIN_TOL = 1e-3      # nats: smaller improvements are optimiser noise, not a stall
_POLISH_MAX_MOVES = 12       # per side: how far outward the polish may push the edge, in nodes

# v0.3.3 (P1): walk budget (`_walk_side`). A REJECTED step (deviance jump -> step/3) no longer uses
# up one of a side's `max_nodes`; rejections get their own cap, so a side cannot loop forever.
_MAX_REJECTIONS = 24         # per walk side; the same number as the node budget it was eating into


def box_for(kappa_max):
    return {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3),
            "kappa": (1e-3, float(kappa_max))}


class _Objective:
    """-loglik and its gradient over z = log(theta), with the reference's box rule, a memo and counters.

    MEMO (v0.3). The compiled likelihood is a deterministic function of the exact float64 bytes of z,
    so a repeated z is answered from a cache with a BIT-IDENTICAL result and every optimiser follows
    exactly the path it would have followed. MEASURED on the quick tier: ~11% of value+grad calls and
    ~62% of value-only calls (Nelder-Mead re-evaluating vertices) are exact repeats. Two caches,
    because `f` and `vg` are different compiled programs whose values may differ in the last ulp.
    `n_f` / `n_g` count REAL evaluations; `n_f_calls` / `n_g_calls` count requests.
    """

    def __init__(self, fj, t, e, L, solver_kw, box):
        self.box = box
        self.f, self.vg = fj.make_loglik(t, e, L, N=solver_kw["N"], dt=solver_kw["dt"],
                                         tau_max=solver_kw["tau_max"], Gtol=solver_kw["Gtol"],
                                         sampling_dt=solver_kw.get("sampling_dt"))
        self.L = float(L)
        self.seed_theta = None   # set by `fit`; the second start for degenerate nodes
        self._f_cache = {}
        self._vg_cache = {}
        self.pending = None      # a restart the adaptive rule skipped at the node just solved
        self.n_f = 0             # real value-only evaluations (the Nelder-Mead fallback)
        self.n_g = 0             # real value+gradient evaluations
        self.n_f_calls = 0
        self.n_g_calls = 0
        self.n_fallback = 0      # Nelder-Mead actually run
        self.n_fallback_improved = 0
        self.n_alt_start = 0     # zero-gradient start rescued by the seed start instead
        self.n_restart = 0       # degenerate node re-solved from the seed start
        self.n_restart_won = 0
        self.n_restart_skipped = 0     # adaptive rule: not run (back-off)
        self.n_restart_deferred = 0    # ... but then run because the walk saw a deviance jump
        self.n_restart_plateau = 0     # walk restart whose seed start is a plateau: not run
        self.n_polish = 0              # edge-polish node solves
        self.n_polish_improved = 0
        self.n_polish_moved = 0        # an edge moved outward past a polished node
        self.n_polish_extend = 0       # a side's walk continued past its last node
        self.p1_rejections = 0         # v0.3.3: rejected walk steps (they no longer use a node slot)

    def inside(self, th):
        return all(self.box[p][0] <= v <= self.box[p][1] for p, v in zip(PARAMS, th))

    def nll(self, z):
        """Value only, box-checked exactly as `methods._nll` (inf outside)."""
        zz = np.asarray(z, float)
        th = np.exp(np.clip(zz, math.log(1e-14), math.log(1e6)))
        if not self.inside(th):
            return np.inf
        self.n_f_calls += 1
        key = zz.tobytes()
        v = self._f_cache.get(key)
        if v is None:
            self.n_f += 1
            v = float(self.f(zz))
            self._f_cache[key] = v
        return np.inf if not np.isfinite(v) else -v

    def nll_grad(self, z):
        zz = np.asarray(z, float)
        self.n_g_calls += 1
        key = zz.tobytes()
        hit = self._vg_cache.get(key)
        if hit is None:
            self.n_g += 1
            v, g = self.vg(zz)
            hit = (float(v), np.asarray(g, float))
            self._vg_cache[key] = hit
        v, g = hit
        if not np.isfinite(v):
            return np.inf, np.zeros_like(g)
        return -v, -g                    # a fresh array: the cached one is never handed out


def _lbfgs(obj, fun_grad, fun_val, z0, bounds, opts, nm_opts, z_alt=None, no_nm=False):
    """Bounded L-BFGS-B from z0, with the counted fallbacks (see module docstring).

    Returns (nll, z, plateau). `fun_grad(z) -> (nll, grad)`, `fun_val(z) -> nll` (inf outside the
    box). On a zero-gradient start: first L-BFGS-B from `z_alt` if one is given and IS resolved;
    only then Nelder-Mead from z0 -- unless `no_nm`, which returns the start unchanged instead (used
    for walk restarts, see `_want_restart`). A fallback can never return something worse than its start.
    """
    lo_b = [b[0] for b in bounds]
    hi_b = [b[1] for b in bounds]

    def run(z):
        r = minimize(fun_grad, z, jac=True, method="L-BFGS-B", bounds=bounds, options=opts)
        return float(r.fun), np.asarray(r.x, float)

    z0 = np.clip(np.asarray(z0, float), lo_b, hi_b)
    v0, g0 = fun_grad(z0)
    plateau = (not np.isfinite(v0)) or (not np.any(g0))
    if not plateau:
        try:
            v, z = run(z0)
            if np.isfinite(v):
                return v, z, False
        except Exception:  # noqa: BLE001 - fall through
            pass
    if z_alt is not None:
        za = np.clip(np.asarray(z_alt, float), lo_b, hi_b)
        va, ga = fun_grad(za)
        if np.isfinite(va) and np.any(ga):
            obj.n_alt_start += 1
            try:
                v, z = run(za)
                if np.isfinite(v) and (not np.isfinite(v0) or v <= v0):
                    return v, z, True
            except Exception:  # noqa: BLE001
                pass
    if no_nm:
        obj.n_restart_plateau += 1
        return (float(v0) if np.isfinite(v0) else float("inf")), z0, True
    obj.n_fallback += 1
    r = minimize(fun_val, z0, method="Nelder-Mead", options=nm_opts)
    if np.isfinite(r.fun) and (not np.isfinite(v0) or r.fun < v0 - 1e-6):
        obj.n_fallback_improved += 1
    return float(r.fun), np.asarray(r.x, float), True


def _triggers(obj, z_full, fidx, bounds):
    """Which `_degenerate` rules fire, as a tuple of labels (empty iff not degenerate).

    MEASURED on pure-k0.01-p00-c1-s1 (v0.1.0). The exact-gradient MLE is BETTER than the simplex's
    (+0.087 nats) but sits in a corner: beta -> 2e-8, so the removal term vanishes and kappa rails
    to its box edge (300), where kappa no longer enters the model. In log-space the gradient along
    beta there is beta * dL/dbeta ~ 0, so every warm-started profile node stays in the corner --
    at +0.01 dex in eta the warm start read deviance 722 while a restart found 0.05, and eta's
    interval collapsed from 0.85 dex to 0.001 dex. The simplex escaped only because it stopped short
    of the corner. So: removal switched off (rho_beta < 1e-3), or a free parameter on its box edge,
    triggers a second start.
    """
    trig = []
    if 1 in fidx and math.exp(z_full[1]) * obj.L < 1e-3:
        trig.append("rho_beta<1e-3")
    for j, k in enumerate(fidx):
        lo, hi = bounds[j]
        if z_full[k] <= lo + 1e-6:
            trig.append(f"{PARAMS[k]}@lo")
        elif z_full[k] >= hi - 1e-6:
            trig.append(f"{PARAMS[k]}@hi")
    return tuple(trig)


def _want_restart(trig, side):
    """v0.3 adaptive rule for a WALK node: (run it?, does its gain feed the back-off counter?).

    MEASURED on the quick tier (v0.2): corner restarts were 42% of all value+grad evaluations; 32 of
    32 sampled restarts on one cohort gained < 2e-7 nats. The rule, verified to leave every verdict and
    every CI edge unchanged (largest edge change 4.5e-9 dex over 30 cohorts, and 4.8e-11 dex on 9
    further pairs incl. het/window/censor/realfull cohorts) while cutting ~20% of CPU:
      * removal switched off (rho_beta < 1e-3) ALWAYS restarts -- those restarts gained up to 6,517 nats;
      * otherwise, after _RESTART_BACKOFF_N consecutive restarts on this side gaining < _RESTART_GAIN_TOL,
        stop; re-arm when the trigger set changes or the walk sees a deviance jump (and then run the
        restart that was skipped at THAT node before the walk judges the jump -- `obj.pending`).
    NOT adopted: skipping restarts whose only trigger is kappa at its floor -- 5 of 259 such restarts
    improved accepted nodes by 0.009-0.25 nats and moved a CI edge by 0.083 dex.
    """
    if "rho_beta<1e-3" in trig:
        return True, False
    if trig != side["last_trig"]:
        side["n_zero"] = 0
    side["last_trig"] = trig
    if side["n_zero"] >= _RESTART_BACKOFF_N:
        return False, True
    return True, True


def _restart(obj, fg, fv, z_alt, bounds, opts, nm, v, z, side, ruled):
    """The second start from the seed. Returns the better (v, z).

    On a walk side (`side` given) a restart whose seed start is itself a zero-gradient plateau is not
    run (`no_nm`): it could only become a Nelder-Mead wander, and MEASURED, 212/212 such restarts on the
    quick tier never improved an accepted node while costing ~47% of all value-only evaluations.
    """
    obj.n_restart += 1
    v2, z2, _ = _lbfgs(obj, fg, fv, z_alt, bounds, opts, nm, no_nm=side is not None)
    gain = float(v - v2) if np.isfinite(v2) else float("-inf")
    if np.isfinite(v2) and v2 < v:
        obj.n_restart_won += 1
        v, z = v2, z2
    if ruled and side is not None:
        side["n_zero"] = 0 if gain >= _RESTART_GAIN_TOL else side["n_zero"] + 1
    return v, z


def _profile_ll_at(obj, theta, i, value, fidx, z0=None, maxiter=400, side=None):
    """Maximise the nuisance parameters with parameter `i` pinned at `value`. Returns (logL, z).

    JAX: warm-started L-BFGS-B, plus a SECOND start from the seed's nuisance values when the warm
    start ended in a degenerate corner (`_triggers`); the better of the two is kept, so the second
    start can only raise a node's loglik. `side` (a walk side's state) enables the adaptive rule;
    repair and polish solves (side=None) always restart.
    """
    box = obj.box
    zi = math.log(float(value))
    fl = list(fidx)

    def full(zf):
        z = np.empty(4)
        z[i] = zi
        z[fl] = zf
        return z

    def fg(zf):
        v, g = obj.nll_grad(full(zf))
        return v, g[fl]

    def fv(zf):
        return obj.nll(full(zf))

    if z0 is None:
        z0 = np.log([max(theta[k], 1e-12) for k in fidx])
    bounds = [(math.log(box[PARAMS[k]][0]), math.log(box[PARAMS[k]][1])) for k in fidx]
    opts = dict(_LBFGS_NODE, maxiter=int(maxiter))
    nm = dict(maxiter=maxiter, xatol=1e-4, fatol=1e-3)
    z_alt = None
    if obj.seed_theta is not None:
        z_alt = np.log([max(obj.seed_theta[k], 1e-12) for k in fidx])
    obj.pending = None
    try:
        v, z, plateau = _lbfgs(obj, fg, fv, z0, bounds, opts, nm, z_alt=z_alt)
        if z_alt is not None and not plateau:
            trig = _triggers(obj, full(z), fidx, bounds)
            if trig:
                if side is None:
                    v, z = _restart(obj, fg, fv, z_alt, bounds, opts, nm, v, z, None, False)
                else:
                    do, ruled = _want_restart(trig, side)
                    if do:
                        v, z = _restart(obj, fg, fv, z_alt, bounds, opts, nm, v, z, side, ruled)
                    else:
                        obj.n_restart_skipped += 1
                        vv, zz_ = v, z
                        obj.pending = lambda: _restart(obj, fg, fv, z_alt, bounds, opts, nm,
                                                       vv, zz_, side, ruled)
        return float(-v), z
    except Exception:  # noqa: BLE001
        return float("nan"), None


def _walk_side(obj, theta, i, p, sgn, ll_ref, fidx, z_start, max_nodes=12, step=_STEP0_DEX,
               start_value=None):
    """Verbatim `methods._walk_side` (raw-parameter path; no combinations), JAX inner optimiser.

    v0.3: carries the side state the adaptive restart rule needs; on a deviance jump it re-arms the
    rule and first runs the restart it skipped at that node.

    v0.3.3 (P1): `max_nodes` counts ACCEPTED nodes only. In the reference (and v0.3.2) every loop
    pass used a slot, including a REJECTED step -- a deviance jump > _DEV_MAX_STEP that only divides
    the step by 3 and retries -- so a side that hit a few jumps could run out of slots BEFORE its
    deviance crossed the threshold, and its claim silently became one-sided or "none". Whether a
    step is rejected depends on the last digits of a node's loglik, so this is what made v0.3.2
    path-sensitive. MEASURED (v0.3.2 review, ages x 1 and x (1 - 2^-52)): v0.3.2 flipped
    pure-k0.01-p05-c0-s0 eta lower_bound <-> none and beta none <-> upper_bound, and
    pure-k0.01-p01-c0-s0 beta none <-> upper_bound, each flipping side ending "max_nodes"; with this
    rule both runs of both cohorts give the same shapes (eta closed, beta upper_bound), at 0.95-1.10x
    v0.3.2's CPU. Rejections get their own cap (_MAX_REJECTIONS); a side that ends on either cap
    keeps the reason "max_nodes".
    MEASURED on the quick tier (30 pure cohorts, v0.3.3 vs v0.3.2, 2026-09-25): 11 new edges on 10
    cohorts -- 9 beta (none / one-sided -> a bound), 1 eta lower_bound -> closed, 1 kappa none ->
    lower_bound -- and no existing edge moved (bit-identical). Coverage eta 28/30 (was 27/29),
    epsilon 28/30 (28/30), beta two-sided 1/4 (1/2): the two new closed beta claims both miss, on
    cv-sharp p05 cohorts, where the generator is biased. 706 rejected steps over the 30 cohorts.
    """
    lo, hi = obj.box[p]
    cur = float(start_value if start_value is not None else theta[i])
    z = z_start
    nodes = []
    dev_prev = 0.0
    reason = "max_nodes"
    side = dict(n_zero=0, last_trig=None)
    n_accepted, n_rejected = 0, 0
    while n_accepted < max_nodes and n_rejected < _MAX_REJECTIONS:   # v0.3.3 (was: range(max_nodes))
        nxt = cur * (10.0 ** (sgn * step))
        if nxt <= lo * 1.01 or nxt >= hi * 0.99:
            nxt = lo * 1.01 if sgn < 0 else hi * 0.99
            if abs(math.log10(max(nxt, 1e-300) / max(cur, 1e-300))) < 1e-9:
                reason = "box_edge"
                break
        th_mid = list(theta)
        th_mid[i] = cur
        llv, zz = _profile_ll_at(obj, th_mid, i, nxt, fidx, z0=z, maxiter=400, side=side)
        dev = 2.0 * (ll_ref - llv) if np.isfinite(llv) else float("inf")
        jump = (not np.isfinite(dev)) or (dev - dev_prev) > _DEV_MAX_STEP
        if jump:
            side["n_zero"] = 0                     # re-arm the adaptive restart rule
            if obj.pending is not None:
                obj.n_restart_deferred += 1
                obj.n_restart_skipped -= 1
                try:
                    v_, z_ = obj.pending()
                    llv, zz = float(-v_), z_
                except Exception:  # noqa: BLE001
                    pass
                dev = 2.0 * (ll_ref - llv) if np.isfinite(llv) else float("inf")
                jump = (not np.isfinite(dev)) or (dev - dev_prev) > _DEV_MAX_STEP
        obj.pending = None
        if jump:
            if step > _STEP_MIN_DEX:
                step = max(step / 3.0, _STEP_MIN_DEX)
                n_rejected += 1                    # v0.3.3: a rejection is not a node
                obj.p1_rejections += 1
                continue
            nodes.append((float(nxt), float(llv), None))
            reason = "cliff"
            break
        n_accepted += 1
        nodes.append((float(nxt), float(llv), None if zz is None else [float(x) for x in zz]))
        if zz is not None:
            z = zz
        if dev >= _DEV_STOP:
            reason = "dev_stop"
            break
        if dev < 1.0:
            step = min(step * 3.0, _STEP_MAX_DEX)
        elif (dev - dev_prev) < 0.3:
            step = min(step * 2.0, _STEP_MAX_DEX)
        dev_prev = dev
        cur = float(nxt)
        if nxt <= lo * 1.02 or nxt >= hi * 0.98:
            reason = "box_edge"
            break
    return nodes, z, step, reason


def _repair(obj, theta, i, fidx, pts):
    """The reference's monotonicity repair, verbatim. Returns (n_violations, n_repaired)."""
    n_repaired, n_violations = 0, 0
    for _pass in range(3):
        lls = [p_[1] for p_ in pts]
        jmax = int(np.nanargmax(lls))
        fixed_any = False
        for rng_ in (range(jmax - 1, -1, -1), range(jmax + 1, len(pts))):
            idxs = list(rng_)
            for k_pos, j in enumerate(idxs):
                outer = idxs[k_pos + 1:]
                if not outer:
                    continue
                jb = max(outer, key=lambda q: pts[q][1])
                if pts[jb][1] > pts[j][1] + _REPAIR_MIN_NATS:
                    n_violations += 1
                    llv, zz = _profile_ll_at(obj, theta, i, pts[j][0], fidx,
                                             z0=(np.asarray(pts[jb][2], float)
                                                 if pts[jb][2] is not None else None),
                                             maxiter=600)
                    if np.isfinite(llv) and llv > pts[j][1]:
                        pts[j][1] = float(llv)
                        pts[j][2] = None if zz is None else [float(x) for x in zz]
                        n_repaired += 1
                        fixed_any = True
        if not fixed_any:
            break
    return n_violations, n_repaired


def _kappa_corner_starts(obj, fidx, zf):
    """Starts for the same node in the OTHER kappa basin: kappa at its floor, at 1 and at its ceiling,
    with beta rescaled so the removal rate at the threshold, beta/(1+kappa) at X = xc = 1, is kept.

    WHY. kappa's two box edges are two different limits of the model -- kappa -> 0: removal saturates
    at beta; kappa -> large: removal is linear, (beta/kappa) X -- and a profile can have a basin at
    each. MEASURED on mice_F epsilon (upper side): basin kappa=1e-3 (beta ~0.1-0.4) gives deviance
    3.90 / 5.44 / 7.04 at epsilon = 0.0103 / 0.0145 / 0.0205; basin kappa=300 (beta ~80-350) gives
    2.29 / 3.26 / 4.32. Every warm start, the seed and the MLE sit in the first, so without these
    starts the upper edge is 0.0102; the reference reached 0.0177 only because one node's warm start
    happened to jump basins and the monotonicity repair carried it inward.
    """
    if 3 not in fidx or zf is None:
        return []
    zf = np.asarray(zf, float)
    k_pos = fidx.index(3)
    b_pos = fidx.index(1) if 1 in fidx else None
    kap = math.exp(zf[k_pos])
    klo, khi = obj.box["kappa"]
    blo, bhi = obj.box["beta"]
    out = []
    for kv in (klo, 1.0, khi):
        if abs(math.log(kv / kap)) < 1e-6:
            continue                                     # that corner IS the current solution
        z = zf.copy()
        z[k_pos] = math.log(kv)
        if b_pos is not None:
            r1 = math.exp(zf[b_pos]) / (1.0 + kap)
            z[b_pos] = math.log(min(max(r1 * (1.0 + kv), blo), bhi))
        out.append(z)
    return out


def _polish_edges(obj, theta, i, p, fidx, pts, ll_ref, thr, z_mle, sides, per_side):
    """v0.3: re-solve the node that PLACES each interval edge from fresh starts; move outward while it
    keeps improving past the threshold.

    WHY. Every walk node warm-starts from its predecessor, so one stalled node puts the rest of the
    side in the wrong basin: the walk's deviance is too high from there on and the interval too
    narrow. MEASURED: mice_F epsilon, at the SAME node (0.01029) one backend's chain read deviance
    3.90 and the other's 2.29 -- every later node stayed in the worse basin, and the upper edge came
    out 0.0102 instead of 0.0177. On the 30 quick cohorts, a harder search at the reported edges found
    both methods' walks too narrow at some edges (reference: 18 of 117 eta/epsilon edges clearly;
    v0.2: 8 of 118).

    HOW. On each side, the first node outward whose deviance is >= `thr` places the edge. Re-solve it
    from the MLE's nuisance values, the seed's, the next node outward's solution, and (v0.3.2, when
    kappa is free) the opposite kappa corners (`_kappa_corner_starts`); keep the best.
    If it now falls below `thr`, the edge has moved: repeat at the next node. If that pushes the edge
    past the side's last node (and the walk had not stopped at the box edge), walk on from the last
    node, once. A polish can only RAISE node logliks (a max over more starts).
    """
    seed = obj.seed_theta
    z_seed = None if seed is None else np.log([max(seed[k], 1e-12) for k in fidx])
    for side_name, sgn in (("down", -1.0), ("up", +1.0)):
        moves = 0
        extended = False
        while moves <= _POLISH_MAX_MOVES:
            pts.sort(key=lambda x: x[0])
            jmax = int(np.nanargmax([q[1] for q in pts]))
            order = list(range(jmax - 1, -1, -1)) if sgn < 0 else list(range(jmax + 1, len(pts)))
            j_out = next((j for j in order if 2.0 * (ll_ref - pts[j][1]) >= thr), None)
            if j_out is None:
                # No node on this side crosses. Extend the walk ONLY if the polish itself pushed the
                # edge past the last node (moves > 0). A side that never crossed -- a flat kappa that
                # ran out of nodes, say -- keeps the reference's verdict: extending it would change
                # what "flat" means, which is not an edge polish.
                if (moves == 0 or extended or not order
                        or sides[side_name]["reason"] == "box_edge"):
                    break
                last = pts[order[-1]]
                if last[2] is None:
                    break
                nodes, _zl, _st, reason = _walk_side(obj, theta, i, p, sgn, ll_ref, fidx,
                                                     np.asarray(last[2], float), max_nodes=per_side,
                                                     start_value=last[0])
                obj.n_polish_extend += 1
                extended = True
                sides[side_name]["reason"] = reason
                for v, l_, zz in nodes:
                    pts.append([float(v), float(l_), zz])
                continue
            k_pos = order.index(j_out)
            starts = [z_mle] + ([z_seed] if z_seed is not None else [])
            if k_pos + 1 < len(order) and pts[order[k_pos + 1]][2] is not None:
                starts.append(np.asarray(pts[order[k_pos + 1]][2], float))
            starts += _kappa_corner_starts(obj, fidx, pts[j_out][2])   # v0.3.2: the other basin
            best_ll, best_z = pts[j_out][1], pts[j_out][2]
            for z0 in starts:
                obj.n_polish += 1
                llv, zz = _profile_ll_at(obj, theta, i, pts[j_out][0], fidx, z0=z0, maxiter=600)
                if np.isfinite(llv) and llv > best_ll + _POLISH_GAIN_TOL:
                    best_ll, best_z = float(llv), [float(x) for x in zz]
            if best_ll <= pts[j_out][1] + _POLISH_GAIN_TOL:
                break                                   # the edge node holds: done on this side
            obj.n_polish_improved += 1
            pts[j_out][1], pts[j_out][2] = best_ll, best_z
            if 2.0 * (ll_ref - best_ll) >= thr:
                break                                   # improved, but still places the edge
            obj.n_polish_moved += 1
            moves += 1


def solver_grid(t, e, delta):
    """(L, solver_kw, dt_ceiling): `mle_profile`'s grid under `fit_cohort.fit`'s dt-ceiling rule."""
    from SRK import fp_likelihood as FPL
    L = float(FPL.kaplan_meier(t, e)[2])
    tm, dt = FPL.auto_grid(t, L)
    ceiling = None
    if delta:
        dl = float(delta) / L
        if 0 < dl < dt:
            ceiling = dl / 3.0
            dt = min(dt, ceiling)
    kw = dict(N=900, dt=dt, tau_max=tm, Gtol=1e-11, scheme="bdf2")
    if delta:
        kw["sampling_dt"] = float(delta)
    return L, kw, ceiling


def fit(fj, ages, events, delta=None, levels=(0.95,), n_profile=11, kappa_max=300.0,
        solver_kw=None, polish=True):
    """The `fit_cohort.fit` -> `mle_profile` path. `fj` is the loaded `fp_jax` module.

    `solver_kw` overrides the grid (used only by the step-size study); None = production grid.
    `polish=False` disables the v0.3 edge polish (for A/B measurement only).
    """
    from scipy.stats import chi2
    from SRK import fp_gof as GOF
    from SRK.fp_curvefit import fp_curvefit_seed
    t = np.asarray(ages, float)
    e = np.asarray(events, int)
    delta = delta if delta else None
    L, kw, ceiling = solver_grid(t, e, delta)
    if solver_kw is not None:
        kw = dict(kw, **solver_kw)
    box = box_for(kappa_max)
    t0 = time.perf_counter()
    tc = time.perf_counter()
    obj = _Objective(fj, t, e, L, kw, box)
    obj.f(np.zeros(4)).block_until_ready()           # compile once, timed separately
    obj.vg(np.zeros(4))[0].block_until_ready()
    compile_s = time.perf_counter() - tc

    ts = time.perf_counter()
    s = fp_curvefit_seed(t, e, xc=1.0)
    seed_theta = [s["eta"], s["beta"], s["epsilon"], s["k"]]
    seed_s = time.perf_counter() - ts
    obj.seed_theta = list(seed_theta)

    starts = [np.log(np.clip(seed_theta, 1e-12, None)),
              np.log(np.clip([seed_theta[0], seed_theta[1], seed_theta[2] * 4.0, seed_theta[3]],
                             1e-12, None))]
    bounds4 = [(math.log(box[p][0]), math.log(box[p][1])) for p in PARAMS]
    best, best_v = None, np.inf
    tm_ = time.perf_counter()
    for z0 in starts:
        try:
            # JAX: was minimize(nll, z0, method="Nelder-Mead", maxiter=1200, xatol=1e-5, fatol=1e-4)
            v, z, _ = _lbfgs(obj, obj.nll_grad, obj.nll, z0, bounds4, _LBFGS_MLE,
                             dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(v) and v < best_v:
            best, best_v = z, v
    mle_s = time.perf_counter() - tm_
    if best is None:
        return dict(theta=seed_theta, ci={}, loglik=float("nan"), wall_s=time.perf_counter() - t0,
                    diagnostics=dict(error="all starts failed"), dt_ceiling=ceiling)
    theta = list(np.exp(best))
    ll = -best_v
    thr = float(chi2.ppf(max(levels), 1))            # the deviance threshold that places an edge

    ci, verdicts, profiles, grid_diag = {}, {}, {}, {}
    per_side = max(24, (int(n_profile) - 1) // 2 + 3)
    for p in PARAMS:
        i = PARAMS.index(p)
        fidx = [k for k in range(4) if k != i]
        centre = float(theta[i])
        ll_ref = ll
        z_mle = np.log([max(theta[k], 1e-12) for k in fidx])
        sides = {}
        for side, sgn in (("down", -1.0), ("up", +1.0)):
            nodes, zl, st_, reason = _walk_side(obj, theta, i, p, sgn, ll_ref, fidx, z_mle,
                                                max_nodes=per_side)
            sides[side] = dict(nodes=[[v, l_, zz] for v, l_, zz in nodes], reason=reason)
        pts = [[centre, ll_ref, list(np.asarray(z_mle, float))]]
        for side in ("down", "up"):
            for v, l_, zz in sides[side]["nodes"]:
                if l_ is not None:
                    pts.append([float(v), float(l_), zz])
        pts.sort(key=lambda x: x[0])

        n_violations, n_repaired = _repair(obj, theta, i, fidx, pts)
        if polish:
            p0 = (obj.n_polish_improved, obj.n_polish_moved)
            _polish_edges(obj, theta, i, p, fidx, pts, ll_ref, thr, z_mle, sides, per_side)
            pts.sort(key=lambda x: x[0])
            if (obj.n_polish_improved, obj.n_polish_moved) != p0:
                v2, r2 = _repair(obj, theta, i, fidx, pts)   # a raised node can expose a stall inward
                n_violations += v2
                n_repaired += r2
        rows = [dict(value=float(v), loglik=float(l_)) for v, l_, _z in pts]
        lls_arr = np.array([r["loglik"] for r in rows], float)
        if np.isfinite(lls_arr).any() and np.nanmax(lls_arr) > ll_ref + 0.01:
            jbest = int(np.nanargmax(lls_arr))
            grid_diag[p] = dict(recentred=True, old_centre=centre, new_centre=rows[jbest]["value"],
                                gain_nats=float(np.nanmax(lls_arr) - ll_ref),
                                down=sides["down"]["reason"], up=sides["up"]["reason"],
                                n_nodes=len(rows), n_violations=n_violations, n_repaired=n_repaired)
        else:
            grid_diag[p] = dict(recentred=False, down=sides["down"]["reason"],
                                up=sides["up"]["reason"], n_nodes=len(rows),
                                n_violations=n_violations, n_repaired=n_repaired)
        profiles[p] = rows
        for lv in levels:
            c = GOF.profile_ci(rows, level=lv)
            ci[f"{p}@{lv}"] = (c["ci_lo"], c["ci_hi"])
            if lv == max(levels):
                verdicts[p] = c["verdict"]
    wall = time.perf_counter() - t0
    counters = {k: getattr(obj, k) for k in (
        "n_f", "n_g", "n_f_calls", "n_g_calls", "n_fallback", "n_fallback_improved", "n_alt_start",
        "n_restart", "n_restart_won", "n_restart_skipped", "n_restart_deferred", "n_restart_plateau",
        "n_polish", "n_polish_improved", "n_polish_moved", "n_polish_extend", "p1_rejections")}
    return dict(theta=theta, ci=ci, loglik=ll, wall_s=wall, dt_ceiling=ceiling,
                diagnostics=dict(verdicts=verdicts, n_profile=n_profile, profiles=profiles,
                                 grid=grid_diag, n_eval=obj.n_f, n_grad=obj.n_g,
                                 compile_s=compile_s, seed_s=seed_s, mle_s=mle_s, solver=dict(kw),
                                 polish=bool(polish), **counters))
