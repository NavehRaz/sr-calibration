"""`mle_profile` with EXACT-GRADIENT optimisers on a JAX likelihood.

mle_profile_jax2 PACKAGE COPY (Phase B). This file is the jax2 rc1 candidate's `estimator.py` (md5 db46040b,
frozen at tests/fixtures/candidate_rc1/code/estimator.py) GENERALISED (PHASE_B_DESIGN.txt section 4):
  * the parameter list is the objective's, `obj.params` = the four SR parameters plus "h" (index 4) when the
    hazard is fitted, and every hard-coded `PARAMS` / `range(4)` / `np.empty(4)` / `np.zeros(4)` /
    `[1, 2, 3]` / `zb[1:]` is written in terms of it. There is no global `PARAMS`, so a missed site is a
    NameError, not a silent four-parameter loop. With the SR parameters only, every site computes what rc1
    computed;
  * `_Objective.inside` / `nll` / `nll_grad` raise ValueError on a z or theta of the wrong length
    (`zip` would truncate silently, and JAX clamps out-of-bounds indexing);
  * EXCEPTION POLICY (`_note_exc`): every `except Exception` site counts what it would swallow in
    `obj.exc_counts` (returned as `diagnostics.exc_counts`), re-raises under SRB_STRICT, and ALWAYS
    re-raises IndexError / ValueError / TypeError. rc1 swallowed all of them silently;
  * `fit` refuses to run when SRK_DT_CEILING is set (it changes SRK.auto_grid under every rule);
  * B2, `fit(..., hazard=True, window=(lo_eff, hi_eff))`: the external hazard h as a fifth, NUISANCE
    parameter (never claimed; `diagnostics.hazard` carries its point, interval and profile), and the fit
    window's left truncation in the likelihood. Every piece of it sits behind `if hazard:` /
    `obj.hazard`: with the hazard off no extra likelihood call, memo entry or counter increment happens,
    which is what keeps the off path bit-identical to rc1, counters included. The pieces:
      - h box (`_h_box`): h in [max(1e-6/L, 1e-12), min(10/L, 1e6)], searched in log h;
      - MLE starts (`_hazard_mle`), after the reference's two (each with h at its floor): the Makeham
        curve fit (`_makeham_seed`, status recorded), then an h ladder of 4-D nodes at hL in
        {1e-2, 1e-1, 0.5} whose climbing rungs become 5-D starts. NEVER fp_curvefit_seed(ndims=5), which
        stores ExtH = 1e-4, i.e. h ~ 1 (a source test enforces it);
      - `_triggers` skips 'h@lo' (the floor is the no-hazard limit, not a degenerate corner);
      - `_hazard_starts`: h at its floor, at the Makeham h, at the MLE h, and the Makeham (eps, h) pair --
        extra starts for polished edge nodes and the re-fit's eta ladder;
      - the h profile's first walk step is 0.5 dex when h-hat is within 1 dex of the floor (`_step0`);
      - `h_floor_kkt`: profile nodes at the h floor whose gradient pushes h up (dl/d(hL) > 1).
    `stages=("mle",)` stops after the MLE stage (truth-recovery rung L4).
  * rc2 I4, `fit(shadow=True)` (hazard only; False = dev3) -- _h's SR profiles may not fall below jax2's on the
    same data (L5 N2):
      - the JAX2-LIKE MLE (`_jax2like_mle`): the reference's two starts searched with h pinned at its floor
        (= jax2's MLE stage on this objective), then freed as a 5-D MLE start (`jax2like+h_lo`);
      - a SHADOW CHAIN per eta / beta / epsilon walk side (`_walk_side(shadow=)`, `_shadow_for`): every
        accepted node is also solved from a second warm chain, with h reset to its floor at each start, that
        lives in the OTHER kappa basin than the MLE (origin: the jax2-like MLE when it lies there, else the
        MLE's nuisance reflected into it). The shadow only RAISES node values: the walk's own chain, its node
        positions, steps, rejections and restarts are the shadow-free walk's exactly; only the stop rule reads
        the raised value;
      - `_polish_inner`: the node just inside each edge is polished as the edge node is (`profile_ci`
        interpolates the edge between the two); the polish also starts from the jax2-like MLE's nuisance.
    MEASURED (rc2 I4, reports/rc2/I4): B5's L5 N2 shortfalls (pure-k0.01-p00 eta 0.05-0.11 nats, p07 eps
    0.08-0.69) are kappa-basin path effects with h at its floor at EVERY node of both fits -- not h drift.
  * rc2 I4, `fit(exact_from_F=True)`: the exact-time density from the absorbed mass (fp_jax `_density_F`).
Bit-identity of the off path with rc1, counters included, is checked by tests/identity_rc1.py.
The rest of this docstring is rc1's.

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
  7. (BASIN-FIX PROTOTYPE, scratch only, not adopted) RE-FIT ON A WITNESSED BETTER POINT. After the
     profile stage, if any profile node beats the MLE by more than chi2_{0.95,1}/2 = 1.92 nats (the
     `recentred` gain -- the same event as "a claim point outside its own interval"), the MLE search
     stopped short. Re-run the 4-D L-BFGS-B from the best profile nodes and from a coarse eta ladder
     above the best node (`_basin_refit`), take the best, and redo ALL profiles from it; repeat while
     the signature fires (at most _REFIT_MAX re-fits). When it never fires nothing else runs, so the
     fit is the v0.3.3 fit exactly. See `_basin_refit` for why each start is there.
  8. (BASIN-FIX v2, scratch only) after an accepted re-fit, each polished edge node also starts from the
     _LIB_K library points (all nodes found so far, within deviance 20) nearest in the profiled parameter
     (`_library_starts`). The stock pass passes no library, so it is unchanged.
  9. (mle_profile_jax2 1.0.0rc1) A FLOORED dt CEILING (`solver_grid`, `floor_info`). Inside production's
     `0 < delta/L < dt_ord` guard, where the ceiling (delta/L)/3 would fall below _DT_FLOOR = 3e-4, use
     dt = min(dt_ord, 3e-4) and N = 1800 instead of dt = (delta/L)/3 at N = 900. Binds on exactly the 48
     E. coli / C. elegans cohorts of library v1 (their dl/3 <= 1.1e-4; every other ceiling cohort has
     dl/3 >= 3.18e-4); everywhere else the grid is bit-identical. Validated at the likelihood level on
     39 of the 48 (reports/dt_ceiling); `diagnostics.dt_floor` records the raw ceiling, dt_ord and N.
 10. (rc2 I3) THE FLOOR EARLY-TIME FIX (`fit(floor_early=True)`, `_grid`): where the floor binds, the march
     starts on a GRADED mesh (_FLOOR_EARLY: 20 BE steps of 1e-8, then x1.02 per step up to the floor dt,
     variable-step BDF2; `fp_jax.early_schedule`). The uniform 3e-4 grid mis-resolves deaths recorded in
     the first sampling bins (Phase A RO1: het-real-ecoli-c0-s1 1.9 nats, -c1-s0 0.12 nats, each from ONE
     early death); the graded start brings them to <= 0.006 nats for +7.4% steps, +8.6% CPU per evaluation
     (median; the 3-grid check passes 22/22 floor records; reports/rc2/I3-floor).
     Off (the default) is dev3's grid exactly; nothing changes where the floor does not bind.
 11. (rc2 I2) THE N RULE FOR LARGE n (`fit(n_rule=True)`, `_grid`, `_n_rule`): N = 900 * 2^k, the smallest k
     whose PREDICTED grid error of the claim-relevant deviances is <= 0.2 nats, capped at 3,600, composed with
     the floor as N = max(N_base, N_rule). The prediction is fitted from n and the Kaplan-Meier CV
     (log10 E900 = -3.034 + 0.428 log10 n - 0.900 log10 CV; error x0.6 per doubling of N) on 134 Phase A records
     (reports/rc2/I2: E900 up to 3 nats on n = 1e5 human cohorts and ~5 nats on sharp CV < 0.05 cohorts).
     Optional companion `fit(dt_rule=True)`: dt <= CV/80 (on sharp cohorts the TIME error dominates once N is
     refined). Off (the default) is dev3's grid exactly; on changes N (and with dt_rule dt) only.
 12. (rc2 D1, 1.0.0.dev5; rc2 E1, 1.0.0.dev6) THE GRADED EARLY MESH ON EVERY GRID (`fit(early_all=True)`,
     `_grid`): a graded start (_EARLY_ALL_MESH: 20 BE steps of 1e-8, then x1.05 per step up to the grid's dt, then
     dt) on EVERY grid that has a sampling_dt, floor-bound grids included -- ONE ratio everywhere (ratified R-8;
     the x1.05 growth is Naveh's 2026-10-03 addendum, reports/rc2/RATIFICATION.txt). dev5 used note 10's x1.02
     (_FLOOR_EARLY); x1.05 costs about half its overhead (reports/rc2/dev5 section 7, reports/rc2/dev6).
     WHY (reports/rc2/I3-floor section 8): the same initial-layer error sits on the production (non-floor) grids
     -- 5.7 / 5.3 nats on two n = 1e5 Denmark cohorts and 3.1 nats on a synthetic pure cohort, each from deaths in
     the first sampling bins; a larger N does not remove it. Exact-time data (no sampling_dt) keep the uniform
     grid: EXACT_F's `fp_jax._density_F` assumes it (`make_loglik` refuses the combination), and no library cohort
     is exact-time. dt_floor["early_all"] records whether the mesh was applied and the graded step count. Off (the
     default) is dev4's grid exactly (floor-bound grids keep note 10's x1.02 when floor_early is on); on, the
     floor-bound grids march _EARLY_ALL_MESH in place of _FLOOR_EARLY.

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
import os
import time

import numpy as np
from scipy.optimize import minimize

# jax2: the SR parameters, in z order. NOT a global parameter list: loops use the objective's `obj.params`.
PARAMS_SR = ("eta", "beta", "epsilon", "kappa")

# jax2 exception policy (`_note_exc`): these are programming errors, never a failed node solve.
_ALWAYS_RAISE = (IndexError, ValueError, TypeError)


def _strict():
    """SRB_STRICT set (to anything but "" or "0"): every swallowed exception is re-raised."""
    return os.environ.get("SRB_STRICT", "") not in ("", "0")


def _note_exc(obj, ex):
    """Count an exception an `except Exception` site is about to swallow; True = the caller must re-raise.

    Counted once per exception object (an inner site that re-raises does not count it again at an outer
    one), by type name, in `obj.exc_counts`, which `fit` returns as `diagnostics.exc_counts`."""
    if not getattr(ex, "_jax2_counted", False):
        k = type(ex).__name__
        obj.exc_counts[k] = obj.exc_counts.get(k, 0) + 1
        try:
            ex._jax2_counted = True
        except Exception:  # noqa: BLE001 - an exception type that refuses attributes: count it again later
            pass
    return _strict() or isinstance(ex, _ALWAYS_RAISE)

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

# BASIN-FIX PROTOTYPE (`_basin_refit`). The trigger tolerance is chi2.ppf(level, 1) / 2, computed in `fit`.
_REFIT_MAX = 3               # re-fits per cohort (each one redoes all four profiles)
_REFIT_RESEED = True         # after a re-fit the seed's second-start role passes to the re-fitted MLE
_LADDER_DEX = 1.0            # eta ladder above the best node: rung spacing ...
_LADDER_RUNGS = 3            # ... and at most this many rungs (it stops at the first rung that does not climb)
_LIB_K = 3                   # v2: library starts per polished edge node, after a re-fit only (`_library_starts`)

# rc2 I4 (fit(shadow=True), hazard only): the shadow chain's kappa basins and the inner-node polish
_SHADOW_KAPPA_SPLIT = 1.0    # a kappa below this is in the "low" basin, at or above it in the "high" one
_INNER_POLISH = True         # `_polish_inner`

# rc1: the floored dt ceiling (`solver_grid`). Applies only inside production's `0 < delta/L < dt_ord` guard.
_DT_FLOOR = 3e-4             # the ceiling (delta/L)/3 is floored here ...
_N_FLOOR = 1800              # ... and the space grid doubled, where the floor binds
# rc2 I3: the graded early-time mesh used where the floor binds and `floor_early` is on (fp_jax.early_schedule);
# with `early_all` off this is dev4's floor mesh exactly
_FLOOR_EARLY = dict(dt0=1e-8, ratio=1.02)
# rc2 E1 (dev6): the mesh `early_all` puts on EVERY grid with a sampling_dt, floor-bound grids included (one ratio
# everywhere; note 12). dev5 used _FLOOR_EARLY here.
_EARLY_ALL_MESH = dict(dt0=1e-8, ratio=1.05)
_EARLY_NSTART = 20           # the mesh's BE start steps: fp_jax.make_loglik's `nstart` default (record only)

# rc2 I2: the N RULE for large n (`_n_rule`, `fit(n_rule=True)`) and its optional companion, a CV-aware dt cap
# (`fit(dt_rule=True)`). Both off (the estimator default) = dev3's grid exactly. Fitted in reports/rc2/I2 on 134
# Phase A records (N 900-14,400 at auto dt; reference: the N-Richardson limit of each deviance).
_NR_A, _NR_B, _NR_C = -3.034, 0.428, -0.900   # log10 E900 = A + B log10 n + C log10 CV: the largest deviance error
                                               # (nats) at the claim-relevant points on the N = 900 grid
_NR_TARGET = 0.2             # nats: the wanted grid error of those deviances (~10% of chi2_1(0.95)/2 = 1.92)
_NR_Q = 0.6                  # error contraction per doubling of N (measured median 0.50, q75 0.65; humans 0.6-0.74)
_NR_N0 = 900                 # the ladder: N = 900 * 2^k ...
_NR_N_MAX = 3600             # ... capped: above it the value's round-off grows ~2.5x per doubling at box-corner
                             #     points (0.4-2 nats sd at N = 7,200 on human corner fits; reports/rc2/I2)
_DTR_K = 80.0                # dt_rule: dt <= CV / _DTR_K (the time error dominates on sharp cohorts, CV < 0.05)

# jax2 B2: the external hazard h (only when fitted). Box in units of 1/L, with absolute guards.
_H_LO_L = 1e-6               # h floor = max(_H_LO_L / L, _H_ABS_LO): the no-hazard limit
_H_HI_L = 10.0               # h ceiling = min(_H_HI_L / L, _H_ABS_HI)
_H_ABS_LO = 1e-12
_H_ABS_HI = 1e6
_H_BOX_RTOL = 1e-12          # `inside`: exp(log(h_lo)) may round one ulp below h_lo; L-BFGS-B sits exactly there
_H_LADDER_HL = (1e-2, 1e-1, 0.5)   # MLE-stage h ladder rungs, in hL
_H_STEP0_NEAR = 0.5          # first h-walk step (dex) when h-hat is within _H_NEAR_DEX of the floor ...
_H_NEAR_DEX = 1.0
_H_DEDUPE = 1e-6             # `_hazard_starts`: two log-h values closer than this are one start
_H_KKT_DLDHL = 1.0           # `h_floor_kkt`: a node at the floor with dl/d(hL) above this is flagged
_MK_KAPPA_CEIL = 5.0         # fp_curvefit_makeham's kappa ceiling (its KAPPA_CEIL); status kappa_at_5
_MK_LSR_BOUNDS = (0.3, 6.0)  # ... and its L_SR box in units of the data mean; status L_SR_at_bound


def box_for(kappa_max):
    return {"eta": (1e-12, 1e3), "beta": (1e-12, 1e3), "epsilon": (1e-12, 1e3),
            "kappa": (1e-3, float(kappa_max))}


def _h_box(L):
    """jax2 B2: the h box (h_lo, h_hi) for a cohort whose KM mean is L (physical time units)."""
    h_lo = max(_H_LO_L / float(L), _H_ABS_LO)
    h_hi = min(_H_HI_L / float(L), _H_ABS_HI)
    if not (np.isfinite(h_lo) and np.isfinite(h_hi) and h_lo < h_hi):
        raise ValueError(f"mle_profile_jax2: empty h box ({h_lo!r}, {h_hi!r}) for L = {L!r}")
    return h_lo, h_hi


class _Objective:
    """-loglik and its gradient over z = log(theta), with the reference's box rule, a memo and counters.

    MEMO (v0.3). The compiled likelihood is a deterministic function of the exact float64 bytes of z,
    so a repeated z is answered from a cache with a BIT-IDENTICAL result and every optimiser follows
    exactly the path it would have followed. MEASURED on the quick tier: ~11% of value+grad calls and
    ~62% of value-only calls (Nelder-Mead re-evaluating vertices) are exact repeats. Two caches,
    because `f` and `vg` are different compiled programs whose values may differ in the last ulp.
    `n_f` / `n_g` count REAL evaluations; `n_f_calls` / `n_g_calls` count requests.
    """

    def __init__(self, fj, t, e, L, solver_kw, box, hazard=False, window=None):
        self.box = box
        self.hazard = bool(hazard)
        # jax2: the parameter list every loop uses; h (index 4) only when the hazard is fitted
        self.params = PARAMS_SR + (("h",) if self.hazard else ())
        self.NP = len(self.params)
        if self.hazard and "h" not in box:
            raise ValueError("mle_profile_jax2 objective: the hazard is on but the box has no 'h'")
        self.exc_counts = {}     # jax2: exceptions an `except` site swallowed, by type (`_note_exc`)
        self.window = window
        self.f, self.vg = fj.make_loglik(t, e, L, N=solver_kw["N"], dt=solver_kw["dt"],
                                         tau_max=solver_kw["tau_max"], Gtol=solver_kw["Gtol"],
                                         sampling_dt=solver_kw.get("sampling_dt"), hazard=self.hazard,
                                         window=window,
                                         **({"exact_from_F": True} if solver_kw.get("exact_from_F") else {}),
                                         # rc2 I3: the graded early-time mesh; absent = dev3's call exactly
                                         **({"early": solver_kw["early"]} if solver_kw.get("early") else {}))
        self.L = float(L)
        if self.hazard:          # jax2 B2: hazard-only state (never created when the hazard is off)
            self.h_mk = None             # the Makeham seed's h and epsilon (clipped into the box)
            self.mk_eps = None
            self.h_mle = None            # h of the current MLE
            self.n_h_starts = 0          # `_hazard_starts` solves run (polish + re-fit ladder) ...
            self.n_h_starts_won = 0      # ... and how many of them gave the node its best value
            self.shadow_on = False       # rc2 I4: fit(shadow=True)
            self.z_shadow0 = None        # rc2 I4: the jax2-like MLE (full z, h at its floor) ...
            self.ll_shadow0 = None       # ... and its log-likelihood
            self.n_shadow = 0            # rc2 I4: shadow-chain node solves (`_shadow_solve`) ...
            self.n_shadow_won = 0        # ... accepted nodes whose value the shadow raised
            self.n_shadow_polish = 0     # polish solves started from the jax2-like MLE ...
            self.n_shadow_polish_won = 0 # ... that gave the polished node its best value
            self.n_inner_polish = 0      # `_polish_inner` solves ...
            self.n_inner_polish_improved = 0  # ... and inner nodes it raised
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

    def _check_z(self, zz):
        """jax2: a z of the wrong length is a programming error (ValueError, always re-raised)."""
        if zz.shape != (self.NP,):
            raise ValueError(f"mle_profile_jax2 objective: z must have shape ({self.NP},), got {zz.shape}")

    def inside(self, th):
        if len(th) != self.NP:          # jax2: zip() would silently skip a parameter's box check
            raise ValueError(f"mle_profile_jax2 objective: theta must have {self.NP} entries, got {len(th)}")
        if self.hazard:                 # jax2 B2: h sits ON its floor routinely; allow the exp(log) ulp
            hl, hh = self.box["h"]
            if not (hl * (1.0 - _H_BOX_RTOL) <= th[4] <= hh * (1.0 + _H_BOX_RTOL)):
                return False
            return all(self.box[p][0] <= v <= self.box[p][1] for p, v in zip(self.params[:4], th[:4]))
        return all(self.box[p][0] <= v <= self.box[p][1] for p, v in zip(self.params, th))

    def nll(self, z):
        """Value only, box-checked exactly as `methods._nll` (inf outside)."""
        zz = np.asarray(z, float)
        self._check_z(zz)
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
        self._check_z(zz)
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
        except Exception as ex:  # noqa: BLE001 - fall through (jax2: counted; strict re-raises)
            if _note_exc(obj, ex):
                raise
    if z_alt is not None:
        za = np.clip(np.asarray(z_alt, float), lo_b, hi_b)
        va, ga = fun_grad(za)
        if np.isfinite(va) and np.any(ga):
            obj.n_alt_start += 1
            try:
                v, z = run(za)
                if np.isfinite(v) and (not np.isfinite(v0) or v <= v0):
                    return v, z, True
            except Exception as ex:  # noqa: BLE001
                if _note_exc(obj, ex):
                    raise
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
            if obj.params[k] != "h":    # jax2 B2: the h floor is the no-hazard limit, not a degenerate corner
                trig.append(f"{obj.params[k]}@lo")
        elif z_full[k] >= hi - 1e-6:
            trig.append(f"{obj.params[k]}@hi")
    return tuple(trig)


def _step0(obj, p, theta):
    """jax2 B2: the first walk step (dex) of parameter p's profile. rc1's _STEP0_DEX for every SR parameter;
    for h, _H_STEP0_NEAR when h-hat (theta[4]) is within _H_NEAR_DEX of the floor (log-space gradients
    vanish there, and 0.05-dex steps would spend the node budget crawling off the floor)."""
    if p != "h":
        return _STEP0_DEX
    h_lo = obj.box["h"][0]
    if math.log10(max(float(theta[4]), 1e-300) / h_lo) < _H_NEAR_DEX:
        return _H_STEP0_NEAR
    return _STEP0_DEX


def _hazard_starts(obj, fidx, zf):
    """jax2 B2: extra starts for a node whose nuisance z is `zf` (h free, i.e. index 4 in fidx): h set to its
    floor, to the Makeham seed's h and to the MLE's h (deduplicated within _H_DEDUPE in log h, and against
    zf's own h), plus the Makeham (epsilon, h) PAIR when epsilon is free -- the h analogue of
    `_kappa_corner_starts` (h trades against epsilon on pure-like data). [] when the hazard is off."""
    if not obj.hazard or zf is None or 4 not in fidx:
        return []
    zf = np.asarray(zf, float)
    hp = fidx.index(4)
    lo, hi = (math.log(obj.box["h"][0]), math.log(obj.box["h"][1]))
    seen = [float(zf[hp])]
    out = []
    for hv in (obj.box["h"][0], obj.h_mk, obj.h_mle):
        if hv is None or not (hv > 0) or not np.isfinite(hv):
            continue
        lz = min(max(math.log(hv), lo), hi)
        if any(abs(lz - s) < _H_DEDUPE for s in seen):
            continue
        seen.append(lz)
        z = zf.copy()
        z[hp] = lz
        out.append(z)
    if 2 in fidx and obj.h_mk is not None and obj.mk_eps is not None:
        z = zf.copy()
        elo, ehi = (math.log(obj.box["epsilon"][0]), math.log(obj.box["epsilon"][1]))
        z[fidx.index(2)] = min(max(math.log(obj.mk_eps), elo), ehi)
        z[hp] = min(max(math.log(obj.h_mk), lo), hi)
        if not any(np.array_equal(z, o) for o in out) and not np.array_equal(z, zf):
            out.append(z)
    return out


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
        z = np.empty(obj.NP)
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
    bounds = [(math.log(box[obj.params[k]][0]), math.log(box[obj.params[k]][1])) for k in fidx]
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
    except Exception as ex:  # noqa: BLE001 - jax2: counted; strict / IndexError etc. re-raise
        if _note_exc(obj, ex):
            raise
        return float("nan"), None


def _shadow_solve(obj, th_mid, i, nxt, fidx, shadow):
    """rc2 I4 (hazard only): the SHADOW chain's solve at `nxt`, started from its last good nuisance with h RESET
    to its floor (`log(h_lo)`, the h bound's own float). It uses its own adaptive-restart state and leaves the
    walk's deferred restart (`obj.pending`, the main chain's) as it found it. Returns (ll, z)."""
    pend = obj.pending
    z0 = np.asarray(shadow["z"], float).copy()
    z0[shadow["hpos"]] = math.log(obj.box["h"][0])
    try:
        r = _profile_ll_at(obj, th_mid, i, nxt, fidx, z0=z0, maxiter=400, side=shadow["side"])
    finally:
        obj.pending = pend
    obj.n_shadow += 1
    return r


def _better(a, b):
    """rc2 I4: the better of two (ll, z) node solves; `a` (the walk's own) wins ties and a failed b."""
    if b[1] is not None and np.isfinite(b[0]) and (a[1] is None or not np.isfinite(a[0]) or b[0] > a[0]):
        return b
    return a


def _walk_side(obj, theta, i, p, sgn, ll_ref, fidx, z_start, max_nodes=12, step=_STEP0_DEX,
               start_value=None, shadow=None):
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
                except Exception as ex:  # noqa: BLE001
                    if _note_exc(obj, ex):
                        raise
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
        if shadow is not None:                     # rc2 I4 (hazard only): the shadow RAISES the node's value;
            sh = _shadow_solve(obj, th_mid, i, nxt, fidx, shadow)    # the walk's own chain, steps and
            best = _better((llv, zz), sh)                            # rejections stay the shadow-free walk's
            nodes.append((float(nxt), float(best[0]), None if best[1] is None else [float(x) for x in best[1]]))
            if best is sh:
                obj.n_shadow_won += 1
            if sh[1] is not None and np.isfinite(sh[0]):            # the shadow moves on unless its node
                d2 = 2.0 * (ll_ref - sh[0])                         # JUMPED against its own last one
                if not (d2 - shadow["dev_prev"]) > _DEV_MAX_STEP:
                    shadow["z"], shadow["dev_prev"] = np.asarray(sh[1], float), d2
            dev_stop_at = 2.0 * (ll_ref - best[0]) if np.isfinite(best[0]) else float("inf")
        else:
            nodes.append((float(nxt), float(llv), None if zz is None else [float(x) for x in zz]))
            dev_stop_at = dev
        if zz is not None:
            z = zz
        if dev_stop_at >= _DEV_STOP:
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


def _library_starts(library, i, fidx, value, ll_ref, k=None):
    """BASIN-FIX v2: nuisance starts for a node of parameter `i` at `value` from the node LIBRARY -- every full-z
    point found so far (all profiles, all rounds) -- the `k` whose i-coordinate is nearest in log.
    WHY (e3_denmark_kappa): after the re-fit the MLE can sit far out on a flat ridge (Denmark kappa 1.9,
    rho_eta 4e5); the kappa walk warm-chained from it lost the ridge (walk edge at kappa 0.293, where the
    profile's true deviance is <= 0.37), and the polish's own starts -- the MLE's nuisance, the seed's (== the
    MLE after the reseed), the next node's -- are all on the lost chain. Nodes of the OTHER profiles lie on
    the ridge at many kappa values; the nearest one in kappa starts the node back on it. Only points within
    deviance 2*_DEV_STOP of the current MLE are candidates: the pre-re-fit (trap) nodes are thousands of nats
    lower and would otherwise be the nearest (the stock pass's kappa walk samples kappa densely)."""
    k = _LIB_K if k is None else k
    if not library:
        return []
    zi = math.log(float(value))
    library = [q for q in library if q[0] >= ll_ref - _DEV_STOP]
    order = sorted(range(len(library)), key=lambda j: (abs(library[j][1][i] - zi), -library[j][0]))
    out, seen = [], set()
    for j in order:
        zf = np.asarray(library[j][1], float)[fidx]
        key = zf.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(zf)
        if len(out) >= k:
            break
    return out


def _full_points(pts_all, params):
    """v2: [(ll, full z)] of every profile node that carries its nuisance z. jax2: `params` = obj.params."""
    out = []
    NP = len(params)
    for p, pts in pts_all.items():
        i = params.index(p)
        fidx = [k for k in range(NP) if k != i]
        for v, l_, zz in pts:
            if zz is None or not np.isfinite(l_):
                continue
            z = np.empty(NP)
            z[i] = math.log(float(v))
            z[fidx] = np.asarray(zz, float)
            out.append((float(l_), z))
    return out


def _polish_edges(obj, theta, i, p, fidx, pts, ll_ref, thr, z_mle, sides, per_side, library=None):
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
                                                     step=_step0(obj, p, theta), start_value=last[0])
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
            h_from = len(starts)
            h_starts = _hazard_starts(obj, fidx, pts[j_out][2])        # jax2 B2: [] when the hazard is off
            starts += h_starts
            sh_at = None
            if obj.hazard and obj.z_shadow0 is not None and 4 in fidx:  # rc2 I4: the jax2-like MLE's nuisance
                sh_at = len(starts)
                starts.append(np.asarray(obj.z_shadow0, float)[fidx])
                obj.n_shadow_polish += 1
            if library is not None:                                     # BASIN-FIX v2: only after a re-fit
                starts += _library_starts(library, i, fidx, pts[j_out][0], ll_ref)
            best_ll, best_z = pts[j_out][1], pts[j_out][2]
            won = None
            for jz, z0 in enumerate(starts):
                obj.n_polish += 1
                llv, zz = _profile_ll_at(obj, theta, i, pts[j_out][0], fidx, z0=z0, maxiter=600)
                if np.isfinite(llv) and llv > best_ll + _POLISH_GAIN_TOL:
                    best_ll, best_z = float(llv), [float(x) for x in zz]
                    won = jz
            if h_starts:
                obj.n_h_starts += len(h_starts)
                if won is not None and h_from <= won < h_from + len(h_starts):
                    obj.n_h_starts_won += 1
            if sh_at is not None and won == sh_at:
                obj.n_shadow_polish_won += 1
            if best_ll <= pts[j_out][1] + _POLISH_GAIN_TOL:
                break                                   # the edge node holds: done on this side
            obj.n_polish_improved += 1
            pts[j_out][1], pts[j_out][2] = best_ll, best_z
            if 2.0 * (ll_ref - best_ll) >= thr:
                break                                   # improved, but still places the edge
            obj.n_polish_moved += 1
            moves += 1
        if obj.hazard and obj.shadow_on and p != "h" and _INNER_POLISH:   # rc2 I4: and the INNER edge node
            _polish_inner(obj, theta, i, fidx, pts, ll_ref, thr, z_mle, z_seed, sgn, library)


def _polish_inner(obj, theta, i, fidx, pts, ll_ref, thr, z_mle, z_seed, sgn, library):
    """rc2 I4 (fit(shadow=True), hazard only): re-solve the node just INSIDE the edge -- the last node on this
    side whose deviance is below `thr` -- from the polish's start family (MLE, seed, the edge node's solution,
    the kappa corners, the h starts, the jax2-like MLE, the library). It can only RAISE that node, which then
    stays inside.

    WHY. `profile_ci` places an edge by interpolating between the edge node and this inner node, so the inner
    node's value moves the claim as much as the edge node's; v0.3.3's polish re-solves only the edge node.
    MEASURED (rc2 I4, pure-k0.01-p07-c0-s0): when a node's value is raised below `thr` (here by a second
    chain) it is no longer the edge node and is not polished, while jax2's walk, whose node stayed above `thr`,
    polished it into the kappa ~ 3 basin, 0.046 nats higher: _h's eta upper edge came out 0.006 dex
    narrower than jax2's (eta 0.00118 node)."""
    pts.sort(key=lambda x: x[0])
    jmax = int(np.nanargmax([q[1] for q in pts]))
    order = list(range(jmax - 1, -1, -1)) if sgn < 0 else list(range(jmax + 1, len(pts)))
    j_out = next((j for j in order if 2.0 * (ll_ref - pts[j][1]) >= thr), None)
    if j_out is None:
        return
    k_pos = order.index(j_out)
    if k_pos == 0 or pts[order[k_pos - 1]][2] is None:
        return                                          # the inner bracket is the MLE row / has no nuisance
    j_in = order[k_pos - 1]
    starts = [z_mle] + ([z_seed] if z_seed is not None else [])
    if pts[j_out][2] is not None:
        starts.append(np.asarray(pts[j_out][2], float))
    starts += _kappa_corner_starts(obj, fidx, pts[j_in][2])
    starts += _hazard_starts(obj, fidx, pts[j_in][2])
    if obj.z_shadow0 is not None and 4 in fidx:
        starts.append(np.asarray(obj.z_shadow0, float)[fidx])
    if library is not None:
        starts += _library_starts(library, i, fidx, pts[j_in][0], ll_ref)
    best_ll, best_z = pts[j_in][1], pts[j_in][2]
    for z0 in starts:
        obj.n_inner_polish += 1
        llv, zz = _profile_ll_at(obj, theta, i, pts[j_in][0], fidx, z0=z0, maxiter=600)
        if np.isfinite(llv) and llv > best_ll + _POLISH_GAIN_TOL:
            best_ll, best_z = float(llv), [float(x) for x in zz]
    if best_ll > pts[j_in][1] + _POLISH_GAIN_TOL:
        obj.n_inner_polish_improved += 1
        pts[j_in][1], pts[j_in][2] = best_ll, best_z


def _basin_refit(obj, pts_all, ll, bounds4, nm_opts, tol):
    """BASIN-FIX PROTOTYPE: a new 4-D MLE after the profile stage witnessed a point > `tol` above `ll`.

    Returns (nll, z, info). Starts, each run through the MLE stage's own `_lbfgs` / _LBFGS_MLE:
      * the best profile node over all four profiles (full z), and the best node of every other
        profile that also beats `ll` by > tol. MEASURED (root-cause task): on 49 of the 52 truth-
        witnessed failures a node reaches >= 90% of the truth's gap, so this start alone is in the
        right basin;
      * an eta LADDER above the best node: eta x 10^(_LADDER_DEX k), k = 1.._LADDER_RUNGS, nuisance
        solved by `_profile_ll_at` warm-chained rung to rung, stopping at the first rung that does not
        climb. WHY: on the cliff-trapped Sweden_F_1910_homo-c0-s0 path (hgn16) every node sits in the
        round-off trap next to the MLE, and 4-D L-BFGS-B from the MLE, from the best node or from their
        kappa corners all stay there (Mac, e1); from the x10 rung it climbs to the truth basin
        (-449,839.0 vs truth -449,840.6; e2). The seed cannot return rho above 300 and starts at
        rho <= 6, so the trap is always BELOW the good basin: the ladder only climbs.
    A start is run only if it is already better than `ll` (a ladder rung on the all-floor plateau is
    not); a search result is never worse than its own start.
    """
    nodes = []
    NP = obj.NP
    fidx0 = [k for k in range(NP) if k != 0]          # jax2: was [1, 2, 3] / zb[1:] / z[1:]
    for p, pts in pts_all.items():
        i = obj.params.index(p)
        fidx = [k for k in range(NP) if k != i]
        for v, l_, zz in pts:
            if zz is None or not np.isfinite(l_):
                continue
            z = np.empty(NP)
            z[i] = math.log(float(v))
            z[fidx] = np.asarray(zz, float)
            nodes.append((float(l_), z, p))
    nodes.sort(key=lambda q: -q[0])
    starts, seen = [], set()

    def add(lab, z, l0):
        key = np.asarray(z, float).tobytes()
        if key not in seen and np.isfinite(l0) and l0 > ll:
            seen.add(key)
            starts.append((lab, np.asarray(z, float), float(l0)))

    info = dict(n_nodes=len(nodes), starts=[], ladder=[])
    if not nodes:
        return float("inf"), None, info
    add(f"node:{nodes[0][2]}", nodes[0][1], nodes[0][0])
    for p in obj.params:
        bp = next((q for q in nodes if q[2] == p), None)
        if bp is not None and bp[0] > ll + tol:
            add(f"node:{p}", bp[1], bp[0])
    zb = nodes[0][1]
    thb = list(np.exp(zb))
    chain, l_prev = zb[fidx0].copy(), nodes[0][0]
    for k in range(1, _LADDER_RUNGS + 1):
        val = math.exp(zb[0]) * 10.0 ** (_LADDER_DEX * k)
        if val >= obj.box["eta"][1] * 0.99:
            break
        llv, zz = _profile_ll_at(obj, thb, 0, val, fidx0, z0=chain, maxiter=400)
        if obj.hazard:                                    # jax2 B2: the rung also starts from the h starts
            h_starts = _hazard_starts(obj, fidx0, chain)
            won = False
            for z0h in h_starts:
                llh, zh = _profile_ll_at(obj, thb, 0, val, fidx0, z0=z0h, maxiter=400)
                if zh is not None and np.isfinite(llh) and (not np.isfinite(llv) or llh > llv):
                    llv, zz, won = llh, zh, True
            obj.n_h_starts += len(h_starts)
            obj.n_h_starts_won += int(won)
        info["ladder"].append(dict(k=k, eta=val, ll=float(llv)))
        if zz is None or not np.isfinite(llv) or llv <= l_prev:
            break
        z = np.empty(NP)
        z[0] = math.log(val)
        z[fidx0] = np.asarray(zz, float)
        add(f"ladder:{k}", z, llv)
        chain, l_prev = np.asarray(zz, float), float(llv)
    best_v, best_z, best_lab = float("inf"), None, None
    for lab, z0, l0 in starts:
        try:
            v, z, plat = _lbfgs(obj, obj.nll_grad, obj.nll, z0, bounds4, _LBFGS_MLE, nm_opts)
        except Exception as ex:  # noqa: BLE001
            if _note_exc(obj, ex):
                raise
            v, z, plat = float("inf"), z0, True
        if not (np.isfinite(v) and v <= -l0):
            v, z = -l0, z0                                   # never worse than its own start
        info["starts"].append(dict(label=lab, ll0=l0, ll=float(-v), plateau=bool(plat),
                                   theta=[float(x) for x in np.exp(z)]))
        if v < best_v:
            best_v, best_z, best_lab = float(v), np.asarray(z, float), lab
    info["chosen"] = best_lab
    return best_v, best_z, info


def _clip_theta(obj, th):
    """jax2 B2: a theta clipped into the objective's box, parameter by parameter."""
    return [min(max(float(v), obj.box[p][0]), obj.box[p][1]) for p, v in zip(obj.params, th)]


def _makeham_seed(obj, t, e):
    """jax2 B2: SRK's Makeham curve fit (SR survival x e^{-ht} least-squares on the KM curve), ndims=5, as a
    5-D MLE start. Returns a record: status (a list; ["ok"] when nothing is off), theta (clipped into the
    box) or None, the raw values, nfev and wall time.

    Its h is a rate per DATA time unit (the same units as obj's h). Its own boxes: kappa <= 5, h in
    [1e-9, 1e2], L_SR in [0.3, 6] x the data mean -- a value ON one of them is recorded (kappa_at_5,
    h_at_floor / h_at_ceil, L_SR_at_bound), and h is additionally compared with THIS box. Never
    `fp_curvefit_seed(ndims=5)`: that returns the fixed ExtH = 1e-4, i.e. h ~ 1, not a fit."""
    from SRK.fp_curvefit import fp_curvefit_makeham
    rec = dict(status=[], theta=None, raw=None, nfev=None, wall_s=None, cpu_s=None)
    w0, c0 = time.perf_counter(), time.process_time()
    try:
        mk = fp_curvefit_makeham(t, e, xc=1.0, ndims=5)
        if int(mk.get("ndims", -1)) != 5:
            raise RuntimeError(f"fp_curvefit_makeham returned ndims={mk.get('ndims')!r}, not 5")
        h = float(mk["h"])
        if not (np.isfinite(h) and h > 0):
            raise RuntimeError(f"fp_curvefit_makeham returned h={h!r}")
        if not int(mk.get("nfev", 0)) > 0:
            raise RuntimeError(f"fp_curvefit_makeham returned nfev={mk.get('nfev')!r}")
        raw = [float(mk["eta"]), float(mk["beta"]), float(mk["epsilon"]), float(mk["k"]), h]
        if not all(np.isfinite(raw)):
            raise RuntimeError(f"fp_curvefit_makeham returned a non-finite theta {raw!r}")
    except Exception as ex:  # noqa: BLE001 - jax2: counted; strict / IndexError etc. re-raise
        rec.update(status=[f"exception:{type(ex).__name__}"], error=str(ex)[:300],
                   wall_s=time.perf_counter() - w0, cpu_s=time.process_time() - c0)
        if _note_exc(obj, ex):
            raise
        return rec
    h_lo, h_hi = obj.box["h"]
    st = []
    if h <= 1e-9 * (1 + 1e-6) or h <= h_lo:
        st.append("h_at_floor")
    if h >= 1e2 * (1 - 1e-6) or h >= h_hi:
        st.append("h_at_ceil")
    if raw[3] >= _MK_KAPPA_CEIL * (1 - 1e-6):
        st.append("kappa_at_5")
    dm = float(mk.get("data_mean", float("nan")))
    lsr = float(mk.get("L_SR", float("nan")))
    if np.isfinite(dm) and np.isfinite(lsr) and dm > 0 and (
            lsr <= _MK_LSR_BOUNDS[0] * dm * (1 + 1e-6) or lsr >= _MK_LSR_BOUNDS[1] * dm * (1 - 1e-6)):
        st.append("L_SR_at_bound")
    rec.update(status=st or ["ok"], raw=raw, theta=_clip_theta(obj, raw), nfev=int(mk["nfev"]),
               hL=h * obj.L, L_SR=lsr, data_mean=dm, residual=float(mk.get("residual", float("nan"))),
               arm=mk.get("arm"), wall_s=time.perf_counter() - w0, cpu_s=time.process_time() - c0)
    return rec


def _hazard_mle(obj, t, e, seed4, bounds, nm_opts):
    """jax2 B2: the MLE stage with the hazard on. Returns (best_z, best_v, record, seed_theta).

    Starts, each a 5-D bounded L-BFGS-B through `_lbfgs` / _LBFGS_MLE, in this order:
      1. seed4 + h_lo              (the reference's first start, h at its floor);
      2. seed4 with eps x 4 + h_lo (the reference's second start);
      3. the Makeham seed (`_makeham_seed`), when it returned a theta;
      4. the h LADDER: from the best of 1-3, 4-D nodes with h pinned at hL = 1e-2, 1e-1, 0.5, warm-chained
         rung to rung; a rung that climbs (ll above its predecessor's; the best of 1-3 for the first) becomes
         a 5-D start, and the ladder stops at the first rung that does not climb (`_basin_refit`'s rule).
         WHY: at the h floor the log-space gradient h dl/dh vanishes, so a search that starts there cannot
         see a hazard; the ladder looks.
    seed_theta (the second start of degenerate nodes) = whichever of start 1 and the Makeham seed has the
    better log-likelihood. A start result is never worse than its own start."""
    h_lo = obj.box["h"][0]
    lo_b = [b[0] for b in bounds]
    hi_b = [b[1] for b in bounds]
    s = list(seed4)
    z1 = np.log(np.clip([s[0], s[1], s[2], s[3], h_lo], 1e-300, None))
    z2 = np.log(np.clip([s[0], s[1], s[2] * 4.0, s[3], h_lo], 1e-300, None))
    mk = _makeham_seed(obj, t, e)
    cand = [("seed4+h_lo", z1), ("seed4_eps4+h_lo", z2)]
    if mk["theta"] is not None:
        obj.h_mk = float(mk["theta"][4])
        obj.mk_eps = float(mk["theta"][2])
        cand.append(("makeham", np.log(mk["theta"])))
    # seed_theta: start 1 or the Makeham seed, by log-likelihood (value+grad: the same program as the searches)
    zc1 = np.clip(z1, lo_b, hi_b)
    v1 = obj.nll_grad(zc1)[0]
    seed_theta, seed_from = list(np.exp(zc1)), "seed4+h_lo"
    vmk = None
    if mk["theta"] is not None:
        zcm = np.clip(np.log(mk["theta"]), lo_b, hi_b)
        vmk = obj.nll_grad(zcm)[0]
        if np.isfinite(vmk) and (not np.isfinite(v1) or vmk < v1):
            seed_theta, seed_from = list(np.exp(zcm)), "makeham"
    obj.seed_theta = list(seed_theta)
    rows = []
    best, best_v, best_lab = None, np.inf, None

    def run(lab, z0, l0=None):
        nonlocal best, best_v, best_lab
        zc = np.clip(np.asarray(z0, float), lo_b, hi_b)
        try:
            v, z, plat = _lbfgs(obj, obj.nll_grad, obj.nll, zc, bounds, _LBFGS_MLE, nm_opts)
        except Exception as ex:  # noqa: BLE001
            if _note_exc(obj, ex):
                raise
            rows.append(dict(label=lab, theta0=[float(x) for x in np.exp(zc)], error=type(ex).__name__))
            return
        if l0 is not None and not (np.isfinite(v) and v <= -l0):
            v, z = -l0, zc                                   # never worse than its own start
        rows.append(dict(label=lab, theta0=[float(x) for x in np.exp(zc)],
                         ll0=(None if l0 is None else float(l0)), ll=float(-v) if np.isfinite(v) else None,
                         plateau=bool(plat), theta=[float(x) for x in np.exp(z)]))
        if np.isfinite(v) and v < best_v:
            best, best_v, best_lab = np.asarray(z, float), float(v), lab

    for lab, z0 in cand:
        run(lab, z0)
    best13 = best_lab
    ladder = []
    if best is not None:
        thb = list(np.exp(best))
        fidx4 = [0, 1, 2, 3]
        chain, l_prev = best[:4].copy(), -best_v
        climbed = []
        for hl in _H_LADDER_HL:
            hv = hl / obj.L
            if not (obj.box["h"][0] < hv < obj.box["h"][1]):
                ladder.append(dict(hL=hl, h=hv, skipped="outside the h box"))
                continue
            llv, zz = _profile_ll_at(obj, thb, 4, hv, fidx4, z0=chain, maxiter=400)
            climbs = bool(zz is not None and np.isfinite(llv) and llv > l_prev)
            ladder.append(dict(hL=hl, h=hv, ll=float(llv), climbs=climbs, ll_prev=float(l_prev)))
            if not climbs:
                break
            z5 = np.empty(5)
            z5[:4] = np.asarray(zz, float)
            z5[4] = math.log(hv)
            climbed.append((f"ladder:hL={hl:g}", z5, float(llv)))
            chain, l_prev = np.asarray(zz, float), float(llv)
        for lab, z0, l0 in climbed:
            run(lab, z0, l0)
    # rc2 I4 (fit(shadow=True)): the JAX2-LIKE MLE -- the reference's own two starts (seed4; seed4 with eps x 4)
    # searched in 4-D with h PINNED at its floor, i.e. jax2's MLE stage on this objective (which differs from
    # jax2's by -h_lo sum(t) and O(h_lo) per death). It is (a) a 5-D start (h freed), so this stage cannot end
    # below what jax2's would reach on these data, (b) the shadow chains' origin when it lies in the other kappa
    # basin than the MLE (`_shadow_for`), and (c) a polish start. WHY (rc2 I4, pure-k0.01-p00-c0-s0): the Makeham
    # start won at kappa = 300, 0.11 nats above jax2's MLE at kappa 0.0044, and the eta walk along that basin
    # fell 0.05-0.11 nats below jax2's profile where jax2's low-kappa basin is higher.
    j2 = None
    if obj.shadow_on:
        j2 = _jax2like_mle(obj, [("seed4", z1), ("seed4_eps4", z2)], bounds, nm_opts)
        if j2["z"] is not None:
            z5 = np.append(j2["z"], math.log(h_lo))
            obj.z_shadow0, obj.ll_shadow0 = z5.copy(), float(j2["ll"])
            run("jax2like+h_lo", z5, j2["ll"])
    for r in rows:
        r["won"] = (r["label"] == best_lab)
    rec = dict(starts=rows, winner=best_lab, best_of_1_3=best13, ladder=ladder, makeham=mk,
               **({"jax2like": j2["rows"]} if j2 is not None else {}),
               seed_from=seed_from, seed_ll=dict(seed4_h_lo=float(-v1) if np.isfinite(v1) else None,
                                                 makeham=(None if vmk is None or not np.isfinite(vmk)
                                                          else float(-vmk))))
    return best, best_v, rec, seed_theta


def _jax2like_mle(obj, starts5, bounds, nm_opts):
    """rc2 I4 (hazard only): the reference's MLE stage with h PINNED at its floor -- each start's first four
    entries searched by `_lbfgs` / _LBFGS_MLE on the 5-D objective at z[4] = log(h_lo), as jax2 searches its 4-D
    objective. Returns dict(z = the best 4-vector or None, ll, rows)."""
    lzlo = math.log(obj.box["h"][0])
    b4 = list(bounds[:4])

    def fg(z4):
        v, g = obj.nll_grad(np.append(z4, lzlo))
        return v, g[:4]

    def fv(z4):
        return obj.nll(np.append(z4, lzlo))

    best_v, best_z, rows = np.inf, None, []
    for lab, z5 in starts5:
        z0 = np.asarray(z5, float)[:4]
        try:
            v, z, plat = _lbfgs(obj, fg, fv, z0, b4, _LBFGS_MLE, nm_opts)
        except Exception as ex:  # noqa: BLE001 - jax2 policy: counted; strict / IndexError etc. re-raise
            if _note_exc(obj, ex):
                raise
            rows.append(dict(label=lab, error=type(ex).__name__))
            continue
        rows.append(dict(label=lab, ll=float(-v) if np.isfinite(v) else None, plateau=bool(plat),
                         theta=[float(x) for x in np.exp(z)]))
        if np.isfinite(v) and v < best_v:
            best_v, best_z = float(v), np.asarray(z, float)
    return dict(z=best_z, ll=(-best_v if best_z is not None else None), rows=rows)


def _shadow_for(obj, i, fidx, ll_ref, theta):
    """rc2 I4: the shadow-chain state for one walk side of eta, beta or epsilon (None without the hazard, and
    for kappa and h, whose walks pass through kappa's basins or have no jax2 counterpart).

    The shadow lives in the OTHER kappa basin than the MLE (split at _SHADOW_KAPPA_SPLIT). Its origin: the
    jax2-like MLE (`_jax2like_mle`) when that lies in the other basin; else the MLE's nuisance REFLECTED into it
    (kappa at its box ceiling or floor, beta rescaled so the removal rate at the threshold, beta/(1+kappa), is
    kept -- `_kappa_corner_starts`' map). h is reset to its floor at every shadow start. MEASURED (rc2 I4):
      * pure-k0.01-p00-c0-s0: MLE kappa = 300, jax2's walk in the kappa ~ 0.004 basin, 0.05-0.11 nats higher on
        three eta nodes; a shadow from the jax2-like MLE recovers them, one reflected from kappa = 300 does not;
      * pure-k0.01-p07-c0-s0: MLE kappa = 0.003, and jax2's eps walk jumped inside one node solve to kappa = 269
        and stayed there, 0.08-0.69 nats higher; a shadow reflected to kappa = 300 recovers it, a jax2-like one
        (kappa 0.003, h free or pinned) does not -- jax2's jump is not reproducible by following its basin.
    A shadow node that JUMPED against the shadow's own previous node does not advance it."""
    if not obj.hazard or obj.params[i] not in ("eta", "beta", "epsilon") or 4 not in fidx or 3 not in fidx:
        return None
    hi_side = float(theta[3]) >= _SHADOW_KAPPA_SPLIT
    lzlo = math.log(obj.box["h"][0])
    if obj.z_shadow0 is not None and (math.exp(obj.z_shadow0[3]) >= _SHADOW_KAPPA_SPLIT) != hi_side:
        zf = np.asarray(obj.z_shadow0, float)[fidx].copy()
        origin = "jax2like"
        dev0 = 2.0 * (ll_ref - obj.ll_shadow0) if obj.ll_shadow0 is not None else 0.0
    else:
        zf = np.log([max(theta[k], 1e-12) for k in fidx])
        k_pos = fidx.index(3)
        kap = math.exp(zf[k_pos])
        klo, khi = obj.box["kappa"]
        kv = klo if hi_side else khi
        zf[k_pos] = math.log(kv)
        if 1 in fidx:
            b_pos = fidx.index(1)
            blo, bhi = obj.box["beta"]
            r1 = math.exp(zf[b_pos]) / (1.0 + kap)
            zf[b_pos] = math.log(min(max(r1 * (1.0 + kv), blo), bhi))
        origin = "reflected"
        dev0 = 0.0
    hpos = fidx.index(4)
    zf[hpos] = lzlo
    return dict(z=zf, hpos=hpos, dev_prev=max(0.0, float(dev0)), origin=origin,
                side=dict(n_zero=0, last_trig=None))


def _kkt_scan(obj, pts_all, theta):
    """jax2 B2: nodes (and the MLE) whose h sits at its floor while dl/d(hL) > _H_KKT_DLDHL, i.e. the data
    push h up but the log-space gradient h dl/dh is too small for the bounded search to see it. Uses the
    value+grad memo (each node's final z was evaluated by its search), so it costs ~no likelihood calls."""
    z_floor = math.log(obj.box["h"][0]) + 1e-6
    flagged, n_at_floor = [], 0

    def check(lab, value, z):
        nonlocal n_at_floor
        if z[4] > z_floor:
            return
        n_at_floor += 1
        v, g = obj.nll_grad(z)
        if not np.isfinite(v):
            return
        dl_dhL = float(-g[4]) / (math.exp(z[4]) * obj.L)      # g = d(-ll)/d(log h)
        if dl_dhL > _H_KKT_DLDHL:
            flagged.append(dict(node=lab, value=value, dl_dhL=dl_dhL))

    check("mle", None, np.log(np.asarray(theta, float)))
    for p, pts in pts_all.items():
        if p == "h":
            continue
        i = obj.params.index(p)
        fidx = [k for k in range(obj.NP) if k != i]
        for v, l_, zz in pts:
            if zz is None or not np.isfinite(l_):
                continue
            z = np.empty(obj.NP)
            z[i] = math.log(float(v))
            z[fidx] = np.asarray(zz, float)
            check(p, float(v), z)
    return dict(n=len(flagged), n_nodes_at_floor=n_at_floor, threshold_dl_dhL=_H_KKT_DLDHL,
                nodes=flagged[:40])


def _grid(t, e, delta, floor_early=False, n_rule=False, dt_rule=False, early_all=False):
    """(L, solver_kw, dt_ceiling, dt_floor): the shared body of `solver_grid` and `floor_info`.

    Production's rule (`fit_cohort.fit`), op for op, with ONE change inside its guard: where the ceiling
    (delta/L)/3 would fall below _DT_FLOOR, the ceiling becomes _DT_FLOOR and N becomes _N_FLOOR, i.e.
    dt = min(dt_ord, max(dl/3, _DT_FLOOR)). Outside the guard, or where dl/3 >= _DT_FLOOR, nothing changes:
    the floor never coarsens a grid whose ceiling production leaves off (the unguarded formula would change
    dt on 51 more library cohorts). `dt_ceiling` is the ceiling actually applied (3e-4 where the floor binds).
    rc2 I3: `floor_early=True` adds solver_kw["early"] = _FLOOR_EARLY (the graded early-time mesh,
    `fp_jax.make_loglik(early=)`) and dt_floor["early"] -- ONLY where the floor binds. Off, or where the floor
    does not bind, both dicts are dev3's exactly.
    rc2 I2: `n_rule=True` raises N to max(N, `_n_rule(n, KM CV)`) -- the floor composes (N_base = _N_FLOOR where
    it binds); `dt_rule=True` caps dt at CV/_DTR_K. Either one adds dt_floor["grid_rule"] (n, cv, N_base,
    dt_base, the rule's prediction, the final N and dt). Both off: both dicts are dev3's exactly.
    rc2 D1 (dev5) / E1 (dev6): `early_all=True` puts the graded mesh _EARLY_ALL_MESH (solver_kw["early"],
    dt_floor["early"]) on EVERY grid with a sampling_dt, floor-bound grids included, where it REPLACES floor_early's
    _FLOOR_EARLY (note 12), and adds dt_floor["early_all"] = {applied, scope} (exact-time data: not applied, the
    uniform grid kept for EXACT_F). Off: both dicts are dev4's exactly.
    """
    from SRK import fp_likelihood as FPL
    L = float(FPL.kaplan_meier(t, e)[2])
    tm, dt = FPL.auto_grid(t, L)
    dt_ord = dt
    ceiling = None
    N = 900
    dl = None
    binds = False
    if delta:
        dl = float(delta) / L
        if 0 < dl < dt:
            ceiling = dl / 3.0
            if ceiling < _DT_FLOOR:                  # rc1: the floor
                binds = True
                ceiling = _DT_FLOOR
                N = _N_FLOOR
            dt = min(dt, ceiling)
    rule = None
    if n_rule or dt_rule:                            # rc2 I2: the N rule / the CV-aware dt cap
        tk, Sk, Lk = FPL.kaplan_meier(t, e)
        cv = _km_cv(tk, Sk, Lk)
        rule = dict(n=int(np.asarray(t).size), cv=cv, N_base=N, dt_base=dt, n_rule=bool(n_rule),
                    dt_rule=bool(dt_rule))
        if n_rule:
            N_r, rec = _n_rule(rule["n"], cv)
            rule.update(rec)
            N = max(N, N_r)                          # composes with the floor (N_base = _N_FLOOR where it binds)
        if dt_rule:
            cap = cv / _DTR_K if (math.isfinite(cv) and cv > 0) else None
            rule.update(dt_k=_DTR_K, dt_cap=cap)
            if cap is not None and cap < dt:
                dt = cap
        rule.update(N=N, dt=dt)
    kw = dict(N=N, dt=dt, tau_max=tm, Gtol=1e-11, scheme="bdf2")
    if delta:
        kw["sampling_dt"] = float(delta)
    floor = dict(binds=binds, dl=dl, raw_ceiling=(dl / 3.0 if (dl is not None and 0 < dl < dt_ord) else None),
                 dt_ord=dt_ord, dt_floor=_DT_FLOOR, n_floor=_N_FLOOR, dt=dt, N=N,
                 nsteps=int(tm / dt))
    if floor_early and binds:                       # rc2 I3: the graded early-time mesh, floor-bound only
        kw["early"] = dict(_FLOOR_EARLY)
        floor["early"] = dict(_FLOOR_EARLY)
    if early_all:                                   # rc2 D1/E1 (dev6): one mesh on every grid with a sampling_dt,
        applied = bool(delta)                       # floor-bound grids included (it replaces _FLOOR_EARLY there)
        if applied:
            kw["early"] = dict(_EARLY_ALL_MESH)
            floor["early"] = dict(_EARLY_ALL_MESH)
        floor["early_all"] = dict(applied=applied,
                                  scope=("every grid with a sampling_dt" if applied else
                                         "not applied: exact-time data keep the uniform grid (EXACT_F)"))
    if rule is not None:                             # rc2 I2: the rule's record (absent when both are off)
        floor["grid_rule"] = rule
    return L, kw, ceiling, floor


def solver_grid(t, e, delta, floor_early=False, n_rule=False, dt_rule=False, early_all=False):
    """(L, solver_kw, dt_ceiling): `mle_profile`'s grid under `fit_cohort.fit`'s dt-ceiling rule, floored
    (rc1, `_grid`). The 3-tuple is kept so every caller of the v0.3.3 signature still works; the floor's
    own record is `floor_info`. rc2 I3: `floor_early` as in `_grid`. rc2 I2: `n_rule` / `dt_rule` as in `_grid`.
    rc2 D1: `early_all` as in `_grid`."""
    L, kw, ceiling, _floor = _grid(t, e, delta, floor_early=floor_early, n_rule=n_rule, dt_rule=dt_rule,
                                   **({"early_all": True} if early_all else {}))
    return L, kw, ceiling


def floor_info(t, e, delta, floor_early=False, n_rule=False, dt_rule=False, early_all=False):
    """rc1: the floored ceiling's record for this cohort -- `diagnostics.dt_floor`. Keys: binds, dl
    (= delta/L), raw_ceiling (= dl/3 where production's guard fires, else None), dt_ord (auto_grid's dt),
    dt_floor, n_floor, and the grid actually used (dt, N, nsteps); rc2 I3: `early` where the graded start
    is on (`floor_early`) and the floor binds; rc2 I2: `grid_rule` where `n_rule` or `dt_rule` is on;
    rc2 D1: `early` and `early_all` where `early_all` is on (`_grid`)."""
    return _grid(np.asarray(t, float), np.asarray(e, int), delta if delta else None, floor_early=floor_early,
                 n_rule=n_rule, dt_rule=dt_rule, **({"early_all": True} if early_all else {}))[3]


def _km_cv(tk, Sk, L):
    """rc2 I2: the CV of the Kaplan-Meier lifetime distribution, from SRK.kaplan_meier's steps (tk, Sk) and its
    mean L, with the trapezoid SRK uses for L = int S dt (`fp_likelihood._trapz`): E[T^2] = int 2 t S dt and
    CV = sqrt(max(E[T^2] - L^2, 0)) / L. Censored mass past the last age is not extrapolated (restricted moments,
    as for L). 0.0 when L <= 0."""
    tk = np.asarray(tk, float)
    Sk = np.asarray(Sk, float)
    L = float(L)
    if not L > 0:
        return 0.0
    y = 2.0 * tk * Sk
    m2 = float(np.sum(np.diff(tk) * (y[1:] + y[:-1]) / 2.0))
    return math.sqrt(max(m2 - L * L, 0.0)) / L


def _n_rule(n, cv):
    """rc2 I2: (N_rule, record). The space grid whose PREDICTED grid error of the claim-relevant deviances meets
    _NR_TARGET: N = _NR_N0 * 2^k for the smallest k >= 0 with E900_pred * _NR_Q^k <= _NR_TARGET, stopping at
    _NR_N_MAX (`capped` = the target is not met there), E900_pred = 10^(A + B log10 n + C log10 CV). A CV that is
    not finite and > 0 (degenerate) takes the cap. The caller composes: N = max(N_base, N_rule)."""
    n = int(n)
    if n < 1:
        raise ValueError(f"mle_profile_jax2 _n_rule: n must be >= 1, got {n!r}")
    cv = float(cv)
    degenerate = not (math.isfinite(cv) and cv > 0)
    e900 = None if degenerate else 10.0 ** (_NR_A + _NR_B * math.log10(n) + _NR_C * math.log10(cv))
    N, e = _NR_N0, e900
    while (degenerate or e > _NR_TARGET) and N < _NR_N_MAX:
        N *= 2
        if not degenerate:
            e *= _NR_Q
    capped = bool(degenerate or e > _NR_TARGET)
    return N, dict(n=n, cv=cv, N_rule=N, e900_pred=e900, e_pred=e, capped=capped, degenerate_cv=degenerate,
                   target=_NR_TARGET, q=_NR_Q, n_max=_NR_N_MAX, coef=[_NR_A, _NR_B, _NR_C])


def _sc_archive(obj):
    """rc2 I1: every point this objective evaluated, as (Z, LL): log-parameters (m x NP) and log-likelihoods.

    Read from the two memo caches (`_f_cache`, `_vg_cache`), whose keys are the exact float64 bytes of z and
    whose values are the log-likelihood (the value+gradient cache holds (ll, grad)). Every MLE-stage start,
    L-BFGS-B iterate and line-search trial, walk / repair / polish / re-fit / ladder / KKT-scan solve goes
    through `nll` or `nll_grad`, so the caches ARE the fit's complete evaluation archive. No likelihood call.
    Not included: the two compile warm-up calls at z = 0 in `fit` (they bypass the memo)."""
    NP = obj.NP
    zs, lls = [], []
    for cache, val in ((obj._vg_cache, lambda h: h[0]), (obj._f_cache, lambda h: h)):
        for key, hit in cache.items():
            if len(key) != 8 * NP:               # a z of another length cannot be in this objective's memo
                raise ValueError(f"mle_profile_jax2 _sc_archive: memo key of {len(key)} bytes, NP = {NP}")
            zs.append(key)
            lls.append(float(val(hit)))
    if not zs:
        return np.empty((0, NP)), np.empty(0)
    Z = np.frombuffer(b"".join(zs), dtype=np.float64).reshape(-1, NP).copy()
    return Z, np.asarray(lls, float)


def sc_extents(Z, LL, params, box, ll_hat, thr, rtol=1e-9, stale_nats=0.01):
    """rc2 I1 (claim self-consistency): what every evaluated point says about every parameter's 95% set.

    A point theta with log-likelihood l bounds every parameter's profile from below at its own value:
    profile_q(theta_q) >= l. With ll_best = max(ll_hat, every evaluated l), a point with l >= ll_best - thr/2
    therefore lies inside q's likelihood-ratio set, for every q and for any reference <= ll_best (each
    profile's own reference -- `fp_gof.profile_ci` uses its profile's maximum -- is <= ll_best). So q's claim
    must contain the q-value of every such point: the claim builder widens the profile interval to the
    union with these extents (`__init__.sc_widen_claims`).

    Z (m x NP): log-parameters; a NaN entry means "this coordinate is unknown" (a stored node without its
    nuisance z -- the post-hoc use; the estimator's archive never has one): it is skipped for that
    parameter's extent and for the box check. Points outside `box` (relative tolerance `rtol`) are not
    part of the model and are ignored; so are non-finite log-likelihoods.

    Returns a JSON-ready dict: thr, ll_hat, ll_best, gap_nats = ll_best - ll_hat and its class ("none" <=
    stale_nats < "minor" <= thr/2 < "major": a point above the MLE by more than `stale_nats` means the MLE
    is stale; the re-fit (v1) acts only above thr/2, so "minor" is logged, not acted on), the best point,
    counts, and per parameter the smallest and largest value over the qualifying points with the full theta
    and log-likelihood of the point that sets each."""
    Z = np.asarray(Z, float).reshape(-1, len(params))
    LL = np.asarray(LL, float).reshape(-1)
    if Z.shape[0] != LL.shape[0]:
        raise ValueError(f"sc_extents: {Z.shape[0]} points but {LL.shape[0]} log-likelihoods")
    TH = np.exp(Z)
    ok = np.isfinite(LL)
    for k, p in enumerate(params):
        lo, hi = box[p]
        col = TH[:, k]
        known = ~np.isnan(Z[:, k])               # NaN = unknown coordinate; +-inf is simply out of the box
        ok &= ~known | ((col >= lo * (1.0 - rtol)) & (col <= hi * (1.0 + rtol)))
    n_out_box = int(np.isfinite(LL).sum() - ok.sum())
    ll_hat = float(ll_hat)
    ll_best = max(ll_hat, float(LL[ok].max())) if ok.any() else ll_hat
    gap = ll_best - ll_hat
    half = 0.5 * float(thr)
    stale = "none" if gap <= stale_nats else ("minor" if gap <= half else "major")
    q = ok & (LL >= ll_best - half)
    out = dict(thr=float(thr), half_thr=half, ll_hat=ll_hat, ll_best=ll_best, gap_nats=gap, stale=stale,
               stale_nats=float(stale_nats), rtol_box=float(rtol), n_points=int(LL.shape[0]),
               n_in_box=int(ok.sum()), n_out_of_box=n_out_box, n_qualifying=int(q.sum()), best=None, extents={})
    if ok.any() and gap > stale_nats:
        jb = int(np.flatnonzero(ok)[np.argmax(LL[ok])])
        out["best"] = dict(ll=float(LL[jb]), theta=[None if not np.isfinite(x) else float(x) for x in TH[jb]])
    for k, p in enumerate(params):
        idx = np.flatnonzero(q & ~np.isnan(Z[:, k]))
        if idx.size == 0:
            out["extents"][p] = dict(n=0, qmin=None, qmax=None)
            continue
        jmin = int(idx[np.argmin(TH[idx, k])])
        jmax = int(idx[np.argmax(TH[idx, k])])
        out["extents"][p] = dict(
            n=int(idx.size),
            qmin=float(TH[jmin, k]), ll_qmin=float(LL[jmin]),
            theta_qmin=[None if not np.isfinite(x) else float(x) for x in TH[jmin]],
            qmax=float(TH[jmax, k]), ll_qmax=float(LL[jmax]),
            theta_qmax=[None if not np.isfinite(x) else float(x) for x in TH[jmax]])
    return out


def fit(fj, ages, events, delta=None, levels=(0.95,), n_profile=11, kappa_max=300.0,
        solver_kw=None, polish=True, hazard=False, window=None, stages=None, self_consistent=False,
        floor_early=False, n_rule=False, dt_rule=False,
        shadow=False, exact_from_F=False, early_all=False):
    """The `fit_cohort.fit` -> `mle_profile` path. `fj` is the loaded `fp_jax` module.

    `solver_kw` overrides the grid (used only by the step-size study); None = production grid.
    `polish=False` disables the v0.3 edge polish (for A/B measurement only).
    jax2: `hazard=True` fits h = exp(z[4]) as a fifth (nuisance) parameter; `window=(lo_eff, hi_eff)` adds
    the left truncation (the observation model has snapped the window and censored at hi_eff already).
    Both off is rc1's fit exactly. `stages=("mle",)` returns after the MLE stage (no profiles); None = all.
    rc2 I1: `self_consistent=True` adds `diagnostics.self_consistency` (`sc_extents` over the complete
    evaluation archive, `_sc_archive`), from which the adapter widens the claims. It makes no likelihood call
    and changes nothing else; False (the default) is the dev3 fit exactly.
    rc2 I3: `floor_early=True` marches the graded early-time mesh where the floor binds (`_grid`, note 10);
    False (the default), or a cohort where the floor does not bind, is the dev3 fit exactly.
    rc2 I2: `n_rule=True` solves on the N rule's grid (`_grid`, note 11), `dt_rule=True` caps dt at CV/80; both
    are recorded in diagnostics.dt_floor.grid_rule. False (the default) is the dev3 fit exactly.
    Refuses to run when SRK_DT_CEILING is set.
    rc2 I4: `shadow=True` (hazard only) adds the jax2-like MLE start, the shadow chains and the inner-edge polish;
    `exact_from_F=True` scores exact-time deaths by the density from the absorbed mass (fp_jax `_density_F`).
    Both False (the defaults) run dev3's fit exactly.
    rc2 D1 (dev5) / E1 (dev6): `early_all=True` marches the graded early-time mesh _EARLY_ALL_MESH (x1.05) on EVERY
    grid with a sampling_dt, floor-bound grids included (`_grid`, note 12), and records the graded step count in
    diagnostics.dt_floor.early_all.nsteps; False (the default) is the dev4 fit exactly.
    """
    if "SRK_DT_CEILING" in os.environ:
        raise RuntimeError("REFUSING: SRK_DT_CEILING is set; it changes SRK.auto_grid under every grid "
                           "rule, so this fit would not be mle_profile_jax2's")
    if stages is not None and not set(stages) <= {"mle", "profile"}:
        raise ValueError(f"mle_profile_jax2: unknown stages {stages!r}")
    from scipy.stats import chi2
    from SRK import fp_gof as GOF
    from SRK.fp_curvefit import fp_curvefit_seed
    t = np.asarray(ages, float)
    e = np.asarray(events, int)
    delta = delta if delta else None
    L, kw, ceiling, dt_floor = _grid(t, e, delta, floor_early=floor_early, n_rule=n_rule,
                                     dt_rule=dt_rule, **({"early_all": True} if early_all else {}))
    if solver_kw is not None:
        kw = dict(kw, **solver_kw)
        dt_floor = dict(dt_floor, overridden=dict(solver_kw))
    if early_all and kw.get("early"):                # rc2 D1: the graded march's step count (record only)
        dt_floor["early_all"] = dict(dt_floor["early_all"], nsteps=int(fj.early_schedule(
            kw["dt"], kw["tau_max"], kw["early"].get("nstart", _EARLY_NSTART), kw["early"]["dt0"],
            kw["early"]["ratio"])[0].size))
    box = box_for(kappa_max)
    if hazard:                                       # jax2 B2
        box = dict(box, h=_h_box(L))
    t0 = time.perf_counter()
    cpu0 = time.process_time()                       # BASIN FIX: measurement only (stock-pass CPU)
    tc = time.perf_counter()
    if exact_from_F:                                 # rc2 I4: absent = dev3's make_loglik call exactly
        kw = dict(kw, exact_from_F=True)
    obj = _Objective(fj, t, e, L, kw, box, hazard=hazard, window=window)
    if hazard:
        obj.shadow_on = bool(shadow)                 # rc2 I4
    obj.f(np.zeros(obj.NP)).block_until_ready()      # compile once, timed separately
    obj.vg(np.zeros(obj.NP))[0].block_until_ready()
    compile_s = time.perf_counter() - tc

    ts = time.perf_counter()
    s = fp_curvefit_seed(t, e, xc=1.0)
    seed_theta = [s["eta"], s["beta"], s["epsilon"], s["k"]]
    seed_s = time.perf_counter() - ts
    obj.seed_theta = list(seed_theta)

    bounds4 = [(math.log(box[p][0]), math.log(box[p][1])) for p in obj.params]   # jax2: NP bounds
    best, best_v = None, np.inf
    tm_ = time.perf_counter()
    hz_mle = None
    if not hazard:
        starts = [np.log(np.clip(seed_theta, 1e-12, None)),
                  np.log(np.clip([seed_theta[0], seed_theta[1], seed_theta[2] * 4.0, seed_theta[3]],
                                 1e-12, None))]
        for z0 in starts:
            try:
                # JAX: was minimize(nll, z0, method="Nelder-Mead", maxiter=1200, xatol=1e-5, fatol=1e-4)
                v, z, _ = _lbfgs(obj, obj.nll_grad, obj.nll, z0, bounds4, _LBFGS_MLE,
                                 dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
            except Exception as ex:  # noqa: BLE001
                if _note_exc(obj, ex):
                    raise
                continue
            if np.isfinite(v) and v < best_v:
                best, best_v = z, v
    else:                                            # jax2 B2: seed4 + h_lo (x2), Makeham, h ladder
        seed4 = list(seed_theta)
        best, best_v, hz_mle, seed_theta = _hazard_mle(obj, t, e, seed4, bounds4,
                                                       dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
        hz_mle["seed4"] = [float(x) for x in seed4]
    mle_s = time.perf_counter() - tm_
    if best is None:
        dg = dict(error="all starts failed", dt_floor=dt_floor, exc_counts=dict(obj.exc_counts))
        if hazard:
            dg["hazard"] = dict(mle=hz_mle, box=list(box["h"]))
        return dict(theta=seed_theta, ci={}, loglik=float("nan"), wall_s=time.perf_counter() - t0,
                    diagnostics=dg, dt_ceiling=ceiling)
    theta = list(np.exp(best))
    ll = -best_v
    if hazard:
        obj.h_mle = float(theta[4])
    if stages is not None and "profile" not in stages:      # jax2: the MLE stage only (rung L4)
        dg = dict(stages=list(stages), n_eval=obj.n_f, n_grad=obj.n_g, compile_s=compile_s, seed_s=seed_s,
                  mle_s=mle_s, solver=dict(kw), dt_floor=dt_floor, exc_counts=dict(obj.exc_counts))
        if hazard:
            dg["hazard"] = _hazard_diag(obj, theta, L, None, None, None, hz_mle, None, levels)
        return dict(theta=theta, ci={}, loglik=ll, wall_s=time.perf_counter() - t0, dt_ceiling=ceiling,
                    diagnostics=dg)
    thr = float(chi2.ppf(max(levels), 1))            # the deviance threshold that places an edge
    per_side = max(24, (int(n_profile) - 1) // 2 + 3)

    def profile_stage(theta, ll, library=None):
        """BASIN FIX: v0.3.3's profile stage, verbatim, as a function of the MLE (theta, ll), so that it
        can be re-run after a re-fit. Also returns every parameter's nodes WITH their z (`pts_all`)."""
        ci, verdicts, profiles, grid_diag, pts_all = {}, {}, {}, {}, {}
        for p in obj.params:
            i = obj.params.index(p)
            fidx = [k for k in range(obj.NP) if k != i]
            centre = float(theta[i])
            ll_ref = ll
            z_mle = np.log([max(theta[k], 1e-12) for k in fidx])
            sides = {}
            for side, sgn in (("down", -1.0), ("up", +1.0)):
                nodes, zl, st_, reason = _walk_side(obj, theta, i, p, sgn, ll_ref, fidx, z_mle,
                                                    max_nodes=per_side, step=_step0(obj, p, theta),
                                                    shadow=(_shadow_for(obj, i, fidx, ll_ref, theta)
                                                            if (hazard and shadow) else None))
                sides[side] = dict(nodes=[[v, l_, zz] for v, l_, zz in nodes], reason=reason)
            pts = [[centre, ll_ref, list(np.asarray(z_mle, float))]]
            for side in ("down", "up"):
                for v, l_, zz in sides[side]["nodes"]:
                    if l_ is not None:
                        pts.append([float(v), float(l_), zz])
            pts.sort(key=lambda x: x[0])

            n_violations, n_repaired = _repair(obj, theta, i, fidx, pts)
            if polish:
                p0 = (obj.n_polish_improved, obj.n_polish_moved, getattr(obj, "n_inner_polish_improved", 0))
                _polish_edges(obj, theta, i, p, fidx, pts, ll_ref, thr, z_mle, sides, per_side,
                              library=library)
                pts.sort(key=lambda x: x[0])
                if (obj.n_polish_improved, obj.n_polish_moved, getattr(obj, "n_inner_polish_improved", 0)) != p0:
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
            pts_all[p] = pts
            if library is not None:                  # v2: this profile's nodes serve the next profiles
                library.extend(_full_points({p: pts}, obj.params))
            for lv in levels:
                c = GOF.profile_ci(rows, level=lv)
                ci[f"{p}@{lv}"] = (c["ci_lo"], c["ci_hi"])
                if lv == max(levels):
                    verdicts[p] = c["verdict"]
        return ci, verdicts, profiles, grid_diag, pts_all

    ci, verdicts, profiles, grid_diag, pts_all = profile_stage(theta, ll)

    # BASIN FIX: re-fit while a profile node beats the reported MLE by more than thr / 2.
    tol = 0.5 * thr
    nm_opts = dict(maxiter=1200, xatol=1e-5, fatol=1e-4)
    stock = dict(theta=[float(x) for x in theta], loglik=float(ll), ci=dict(ci), verdicts=dict(verdicts),
                 grid={p: dict(g) for p, g in grid_diag.items()},
                 cpu_s=time.process_time() - cpu0, wall_s=time.perf_counter() - t0, n_g=obj.n_g)
    rounds, stop = [], "no_signature"
    lib_hist = []
    while True:
        gains = {p: float(g.get("gain_nats", 0.0)) for p, g in grid_diag.items() if g.get("recentred")}
        gmax = max(gains.values(), default=0.0)
        if not gmax > tol:
            stop = "no_signature" if not rounds else "converged"
            break
        if len(rounds) >= _REFIT_MAX:
            stop = "cap"
            break
        c_r, w_r, g_r = time.process_time(), time.perf_counter(), obj.n_g
        v_new, z_new, info = _basin_refit(obj, pts_all, ll, bounds4, nm_opts, tol)
        info.update(round=len(rounds) + 1, ll_before=float(ll), trigger_gain=gmax,
                    trigger_param=max(gains, key=gains.get))
        info["refit_cpu_s"] = time.process_time() - c_r
        if not (z_new is not None and np.isfinite(v_new) and -v_new > ll + tol):
            info.update(accepted=False, cpu_s=time.process_time() - c_r,
                        wall_s=time.perf_counter() - w_r, n_g=obj.n_g - g_r)
            rounds.append(info)
            stop = "no_gain"
            break
        theta, ll = list(np.exp(z_new)), float(-v_new)
        if hazard:
            obj.h_mle = float(theta[4])              # jax2 B2: `_hazard_starts` / `_step0` follow the MLE
        if _REFIT_RESEED:
            obj.seed_theta = list(theta)
        lib_hist.extend(_full_points(pts_all, obj.params))   # v2: every node of every round so far
        lib_hist.extend((float(q["ll"]), np.log(q["theta"])) for q in info["starts"])
        ci, verdicts, profiles, grid_diag, pts_all = profile_stage(theta, ll, library=list(lib_hist))
        info.update(accepted=True, ll_after=float(ll), theta_after=[float(x) for x in theta],
                    cpu_s=time.process_time() - c_r, wall_s=time.perf_counter() - w_r,
                    n_g=obj.n_g - g_r)
        rounds.append(info)
    basin_fix = dict(fired=bool(rounds), n_refit=sum(1 for r in rounds if r["accepted"]), stop=stop,
                     tol=tol, rounds=rounds, stock=stock if rounds else None,
                     reseed=bool(_REFIT_RESEED), ladder_dex=_LADDER_DEX, ladder_rungs=_LADDER_RUNGS, lib_k=_LIB_K,
                     refit_max=_REFIT_MAX)

    hz_diag = None
    if hazard:                                       # jax2 B2: before `counters` (the scan may call vg)
        kkt = _kkt_scan(obj, pts_all, theta)
        hz_diag = _hazard_diag(obj, theta, L, ci, verdicts, profiles, hz_mle, kkt, levels)
    wall = time.perf_counter() - t0
    counters = {k: getattr(obj, k) for k in (
        "n_f", "n_g", "n_f_calls", "n_g_calls", "n_fallback", "n_fallback_improved", "n_alt_start",
        "n_restart", "n_restart_won", "n_restart_skipped", "n_restart_deferred", "n_restart_plateau",
        "n_polish", "n_polish_improved", "n_polish_moved", "n_polish_extend", "p1_rejections")}
    diagnostics = dict(verdicts=verdicts, n_profile=n_profile, profiles=profiles,
                       grid=grid_diag, n_eval=obj.n_f, n_grad=obj.n_g,
                       compile_s=compile_s, seed_s=seed_s, mle_s=mle_s, solver=dict(kw),
                       polish=bool(polish), basin_fix=basin_fix, dt_floor=dt_floor, **counters,
                       exc_counts=dict(obj.exc_counts))
    if hazard:
        diagnostics["hazard"] = hz_diag
    if window is not None:
        diagnostics["window"] = dict(lo_eff=float(window[0]), hi_eff=float(window[1]))
    if self_consistent:                              # rc2 I1: no likelihood call; read from the memo caches
        Z_sc, LL_sc = _sc_archive(obj)
        diagnostics["self_consistency"] = sc_extents(Z_sc, LL_sc, obj.params, obj.box, ll, thr)
    return dict(theta=theta, ci=ci, loglik=ll, wall_s=wall, dt_ceiling=ceiling, diagnostics=diagnostics)


def _hazard_diag(obj, theta, L, ci, verdicts, profiles, hz_mle, kkt, levels):
    """jax2 B2: `diagnostics.hazard` -- everything about h, the nuisance parameter the tester does not score
    yet. The would-be claim (`h_claim`) is added by the adapter, which owns VERDICT_TO_SHAPE."""
    h = float(theta[4])
    h_lo, h_hi = obj.box["h"]
    lv = max(levels)
    out = dict(h=h, hL=h * L, ExtH=-math.log(h), L=float(L),
               at_floor=bool(math.log(h) <= math.log(h_lo) + 1e-6),
               box=[h_lo, h_hi], box_hL=[h_lo * L, h_hi * L],
               ci=None, verdict=None, level=lv, profile=None,
               mle=hz_mle, n_h_starts=obj.n_h_starts, n_h_starts_won=obj.n_h_starts_won,
               h_floor_kkt=kkt,
               **({"shadow": dict(n_solves=obj.n_shadow, n_won=obj.n_shadow_won, n_polish=obj.n_shadow_polish,
                                  n_polish_won=obj.n_shadow_polish_won, n_inner_polish=obj.n_inner_polish,
                                  n_inner_polish_improved=obj.n_inner_polish_improved,
                                  jax2like=(None if obj.z_shadow0 is None
                                            else [float(x) for x in np.exp(obj.z_shadow0)]),
                                  jax2like_ll=obj.ll_shadow0)} if obj.shadow_on else {}))
    if ci is not None:
        c = ci.get(f"h@{lv}")
        out["ci"] = None if c is None else [None if x is None else float(x) for x in c]
        out["verdict"] = verdicts.get("h")
        out["profile"] = profiles.get("h")
    return out
