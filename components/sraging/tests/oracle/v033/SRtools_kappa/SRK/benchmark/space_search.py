"""Is there a space in which only ONE coordinate is commonly unidentifiable?

Naveh's recommendation, made testable. If the flat direction points the same way in every cohort, then
rotating into it turns "four coupled, partly-unidentifiable parameters" into "three well-determined
coordinates plus one flat one you report as a bound" -- which is a far easier thing to publish and to
reason about than the current situation.

The flat direction is measured from the POSTERIOR itself rather than from a Fisher matrix at a point:
the covariance of the pooled log-parameter draws, whose largest eigenvector is the direction the
posterior is least able to constrain. Log space because every parameter here is a positive scale
parameter and the degeneracies (beta/kappa, beta*x/(x+kappa)) are multiplicative -- a power-law
combination is a straight line in log coordinates, so a LINEAR eigenvector there corresponds to a
product `prod_i theta_i^{v_i}`, which is exactly the kind of quantity the project already reports
(beta/kappa, sqrt(eta)/eps, beta/(kappa+1)).

WHAT IS REPORTED
  * per cohort: the eigen-spectrum, the flat eigenvector's composition, and the condition number
  * across cohorts: whether the flat directions ALIGN (mean |cos| between them). If they do, one fixed
    rotation works library-wide; if they do not, the flat direction is cohort-specific and no single
    reparameterisation can help -- which is itself the answer.
  * the best fixed candidate, and how much of each cohort's flat direction it captures
  * a comparison against the spaces already in use: production `[xc/eta, beta/eta, xc^2/eps, kappa]`,
    plain log space, and the existing named combinations

    python -m SRK.benchmark.space_search --ckpt prod/ckpt --manifest manifest.json --out prod/space.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

from . import cohorts as C
from .chain_sweep import REGIMES

PARAMS = ("eta", "beta", "epsilon", "kappa")


# Calibrated negligibility floors from `negligible.py`: KL is exactly quadratic in the perturbation
# (fitted exponent 2.01 / 1.97), so the detectability floor falls as 1/sqrt(n).
BETA_NEG_COEF = 3.03      # beta_neg / eta  = BETA_NEG_COEF / sqrt(n_eff)
KAPPA_NEG_COEF = 0.364    # kappa_neg / xc  = KAPPA_NEG_COEF / sqrt(n_eff)

# *** n IS CAPPED AT 2000 FOR THE FLOOR. Naveh's call, and it is the conservative direction. ***
#
# The 1/sqrt(n) law is derived from KL between two EXACT model densities, so it assumes the data really
# is drawn from the SR model with no contamination. Real cohorts are not that clean -- heterogeneity,
# recording resolution, a stray long-lived individual, mixed sub-populations -- and those act as a
# SYSTEMATIC floor on detectability that does not shrink with sample size. Past a few thousand animals
# you are limited by model misspecification, not by counting noise, so continuing to shrink the floor
# as 1/sqrt(n) would claim a resolving power the data does not have.
#
# Capping n_eff makes the floor LARGER for big cohorts, i.e. it declines to claim detection of very
# small removal even with 24,000 animals. That is the safe side of the error: it can under-claim
# detectability, never over-claim it.
#
# Effect: floors stop improving at beta/eta = 0.068 and kappa = 0.0081. Only 1 of the 67 library
# cohorts exceeds n=2000, so the benchmark barely moves -- but the real sets it matters for are exactly
# the big ones (yeast n=24512, humans n~20000+, mice n=3701).
N_EFF_CAP = 2000


# REGIME UNCERTAINTY ON `BETA_NEG_COEF`. Read this before treating the floor as a threshold.
#
# 3.03 is a single closed-form fit across the four regimes `negligible.REGIMES` sweeps. Their OWN
# measured coefficients (floor * sqrt(n), from `negligible.py --summarise`) are not close together:
#
#     sharp       (rho_eps 0.10, kappa 0.05)   c = 0.93
#     mid         (rho_eps 0.30, kappa 0.15)   c = 2.21
#     kappa_large (rho_eps 0.30, kappa 1.00)   c = 5.07
#     diffuse     (rho_eps 0.50, kappa 0.50)   c > 16.8   (right-censored at the top of the grid)
#     PRODUCTION                               c = 3.03
#
# an 18-fold spread, so the floor carries roughly -0.5 to >+0.75 dex of regime uncertainty. Against
# each regime's own value the production floor is up to **3.3x too LENIENT** in the sharp corner
# (~11x in nats, since KL is quadratic in the perturbation) and at least **5.5x too STRICT** in the
# diffuse corner. `negligible.py`'s own summary already says as much: median measured floor 0.173 at
# n=130 against the closed form's 0.269.
#
# Worse for the cohorts we actually fit: the sweep covered rho_eps in {0.10, 0.30, 0.50}, while the
# six real species fitted here sit at rho_eps = 0.113, 0.42, 1.36, 15.7, 117 and **6242**. Four of
# six are OUTSIDE the calibrated range, Denmark by ~10^4. And the sweep varies rho_eps and kappa
# TOGETHER, so their effects cannot be separated and the DIRECTION of the error outside the box is
# not known.
#
# DECISION (owner, 2026-09-10): keep the single coefficient and label the uncertainty, rather than
# re-calibrating now. Consequence for callers: **a "negligible" verdict within a factor of ~3 of the
# floor is approximate, not a test.** Do not build a hard accept/reject on it, and do not compare
# clamped accuracies across cohorts in different regimes as if the bar were the same. The principled
# fix -- computing each cohort's floor at its OWN (rho_eps, kappa) -- is logged as an open item in
# `benchmark_reports/REMAINING_WORK.md`; it is affordable (a mean-one solve is 12 ms) and would
# remove the question rather than bounding it.
BETA_NEG_COEF_REGIME_RANGE = (0.93, 16.8)     # measured across negligible.REGIMES


def negligibility_floors(n, n_cap=N_EFF_CAP):
    n_eff = min(max(float(n), 1.0), float(n_cap))
    return BETA_NEG_COEF / math.sqrt(n_eff), KAPPA_NEG_COEF / math.sqrt(n_eff)


def beta_floor_regime_band(n, n_cap=N_EFF_CAP):
    """The floor's plausible range over the calibrated regimes, as (lo, hi) on the RATIO.

    Use this to decide whether a cohort is *safely* inside or outside the negligible region rather
    than reading the single value as a boundary. A cohort between `lo` and `hi` is one the
    calibration cannot classify.
    """
    n_eff = min(max(float(n), 1.0), float(n_cap))
    lo, hi = BETA_NEG_COEF_REGIME_RANGE
    return lo / math.sqrt(n_eff), hi / math.sqrt(n_eff)


def beta_floor_physical(n, eta_phys, L_phys, n_cap=N_EFF_CAP):
    """`beta`'s negligibility floor in PHYSICAL units. THE `L` FACTOR IS NOT OPTIONAL.

    The calibration in `negligible.py` ran at tau-mean = 1, i.e. in dimensionless time, so what it
    measured is a bound on the DIMENSIONLESS ratio

        rho_beta / rho_eta  <  BETA_NEG_COEF / sqrt(n_eff)

    and `rho_beta/rho_eta = beta/(eta*L)`, so the physical floor carries an `L`:

        beta_neg = BETA_NEG_COEF / sqrt(n_eff) * eta * L

    Physical `beta/eta` has units of TIME, so comparing it to a dimensionless coefficient is only
    correct at L = 1 -- and 57% of the benchmark library and 100% of the manifold arm have L != 1.
    Dropping the `L` has now caused this bug twice in opposite directions: once 926x too STRICT in
    `fit_cohort` (making "negligible" unreachable on any real cohort), and once L times too LENIENT
    in `interval_estimators._floors_for`, where it flagged 0 of 2400 manifold cohorts as
    negligible-beta against 185 under the correct rule. Convert to the dimensionless ratio and the
    ambiguity disappears, which is why `negligible_beta` below is the preferred entry point.
    """
    bcoef, _k = negligibility_floors(n, n_cap=n_cap)
    if eta_phys is None or not (eta_phys > 0) or not math.isfinite(float(eta_phys)):
        return float("nan")
    # NO `L = 1` FALLBACK. Defaulting a missing L to 1 returns a finite floor that is wrong by
    # exactly the factor this function exists to carry -- the same silent-factor-of-L bug in a new
    # costume. nan is the honest answer and every caller already treats it as "no floor applies",
    # which under-claims negligibility rather than mis-scoring it.
    try:
        Lv = float(L_phys)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(Lv) or Lv <= 0:
        return float("nan")
    return bcoef * float(eta_phys) * Lv


def kappa_floor(n, n_cap=N_EFF_CAP):
    """`kappa`'s floor. `kappa` is dimensionless already, so there is no `L` and no ratio."""
    _b, k = negligibility_floors(n, n_cap=n_cap)
    return k


def negligible_beta(n, rho_beta, rho_eta, n_cap=N_EFF_CAP):
    """The unit-free form of the same test: is removal indistinguishable from zero at this n?

    Prefer this wherever the dimensionless parameters are in hand -- it cannot be got wrong by an
    `L`, because no `L` appears.
    """
    bcoef, _k = negligibility_floors(n, n_cap=n_cap)
    try:
        return bool(float(rho_beta) / float(rho_eta) < bcoef)
    except Exception:
        return False


def pooled_log_draws(ckpt, name, k=50, burn_frac=0.4, thin=20, max_draws=200000,
                     floor_n=None):
    """Pooled PHYSICAL draws in log space. Thinned hard -- covariance needs shape, not sample size.

    `floor_n` applies the calibrated negligibility floors for a cohort of that size, BEFORE the log.
    This is not cosmetic and it may change the whole eigenstructure. The covariance is of LOG
    parameters, so draws with beta -> 0 contribute unboundedly large log-variance; if a sizeable share
    of the posterior sits below the detectability floor, `log beta` is dominated by values that are all
    physically the same thing (zero), and beta will appear to own the flat direction for a reason that
    has nothing to do with the likelihood's geometry. Flooring collapses that spurious spread to a
    point, so `beta` stops being inflated by the representation.

    Both the floored and unfloored analyses are worth running: if the flat direction survives
    flooring, it is geometry; if it moves, the original answer was an artifact of treating
    "negligible" and "negligible/1000" as different values.
    """
    import emcee
    out = []
    for p in sorted(glob.glob(os.path.join(ckpt, f"{name}.c*.h5")))[:k]:
        try:
            b = emcee.backends.HDFBackend(p, read_only=True)
            it = int(b.iteration)
            if it <= 10:
                continue
            ch = b.get_chain(discard=int(burn_frac * it), thin=thin)
        except Exception:
            continue
        z = ch.reshape(-1, ch.shape[-1])
        eta = 1.0 / z[:, 0]
        phys = np.column_stack([eta, z[:, 1] * eta, 1.0 / z[:, 2], z[:, 3]])
        if floor_n:
            bcoef, kfloor = negligibility_floors(floor_n)
            np.maximum(phys[:, 1], bcoef * phys[:, 0], out=phys[:, 1])   # beta floored at ratio*eta
            np.maximum(phys[:, 3], kfloor, out=phys[:, 3])               # kappa floored absolutely
        out.append(phys)
    if not out:
        return None
    A = np.concatenate(out, 0)
    A = A[np.all(np.isfinite(A) & (A > 0), axis=1)]
    if A.shape[0] > max_draws:
        A = A[:: int(np.ceil(A.shape[0] / max_draws))]
    return np.log(A) if A.shape[0] > 50 else None


def spectrum(logdraws):
    """Eigen-decomposition of the log-posterior covariance, largest (flattest) first."""
    Csig = np.cov(logdraws, rowvar=False)
    w, V = np.linalg.eigh(Csig)
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def as_power_law(v, params=PARAMS, tol=0.08):
    """Render an eigenvector as a readable product prod theta_i^{v_i}, sign-normalised."""
    v = np.asarray(v, float)
    v = v / (np.max(np.abs(v)) or 1.0)
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    num, den = [], []
    for p, c in zip(params, v):
        if abs(c) < tol:
            continue
        (num if c > 0 else den).append(f"{p}^{abs(c):.2f}")
    s = " * ".join(num) if num else "1"
    if den:
        s += " / (" + " * ".join(den) + ")"
    return s


def analyse(ckpt, manifest, k=50, out=None, floor=False):
    specs = [C.CohortSpec(**d) for d in json.load(open(manifest))]
    rows = []
    for sp in specs:
        L = pooled_log_draws(ckpt, sp.name, k=k, floor_n=(sp.n if floor else None))
        if L is None:
            print(f"{sp.name:<26s} SKIP", flush=True)
            continue
        w, V = spectrum(L)
        flat = V[:, 0]
        # ALL FOUR eigenvectors are persisted, not just the flat one. The earlier version kept only
        # V[:,0] and discarded V[:,1:4] -- which threw away the answer to the question that actually
        # matters. The saved eigenvalues say directions 2 and 3 have median 95% widths of 0.40 and 0.11
        # dex, i.e. 18.6x TIGHTER than the flat direction, so those are the well-determined
        # combinations. They could not be recovered from the old artifact at all.
        rows.append(dict(name=sp.name, n=sp.n, rho_beta=sp.rho_beta, kappa=sp.kappa,
                         omega=sp.omega,
                         eigenvalues=[float(x) for x in w],
                         eigenvectors=[[float(x) for x in V[:, j]] for j in range(V.shape[1])],
                         flat_vector=[float(x) for x in flat],
                         cond=float(w[0] / max(w[-1], 1e-300)),
                         flat_frac=float(w[0] / max(w.sum(), 1e-300)),
                         power_law=as_power_law(flat),
                         power_laws=[as_power_law(V[:, j]) for j in range(V.shape[1])],
                         # 95% width of each eigendirection, in decades: 2*1.96*sd/ln(10)
                         widths_dex=[float(2 * 1.959963984540054 * math.sqrt(max(x, 0.0))
                                           / math.log(10.0)) for x in w]))
        print(f"{sp.name:<26s} cond={rows[-1]['cond']:.3g}  flat={rows[-1]['power_law']}", flush=True)
    if out:
        from . import methods as M
        M._atomic_json(out, rows)
        print(f"\nwrote {out} ({len(rows)} cohorts)")
    return rows


def summarise(rows):
    L = []
    A = L.append
    Vs = np.array([r["flat_vector"] for r in rows], float)
    # sign-normalise: an eigenvector and its negation are the same direction
    for i in range(Vs.shape[0]):
        j = int(np.argmax(np.abs(Vs[i])))
        if Vs[i, j] < 0:
            Vs[i] *= -1
    A("IS THERE ONE SPACE IN WHICH A SINGLE COORDINATE IS THE FLAT ONE?")
    A("=" * 100)
    A(f"{len(rows)} cohorts. Flat direction = top eigenvector of the log-posterior covariance, i.e. the")
    A("direction the data constrains least. In log space a linear direction IS a power-law combination.")
    A("")
    Cm = np.abs(Vs @ Vs.T)
    iu = np.triu_indices(len(rows), 1)
    al = Cm[iu]
    A(f"ALIGNMENT between cohorts' flat directions: mean |cos| = {al.mean():.3f}, "
      f"median {np.median(al):.3f}, frac > 0.9 = {np.mean(al > 0.9):.2f}")
    A("")
    if al.mean() > 0.85:
        A("=> The flat direction is essentially the SAME in every cohort, so ONE fixed rotation works")
        A("   library-wide and the reparameterisation Naveh suggested is available.")
    elif al.mean() > 0.6:
        A("=> Partially aligned. One rotation captures most cohorts but not all; expect a residual set")
        A("   where the flat direction differs and must be handled per cohort.")
    else:
        A("=> NOT aligned. The flat direction is COHORT-SPECIFIC, so no single reparameterisation can")
        A("   make one coordinate the flat one everywhere. That is a real answer to the question, and")
        A("   it means the per-cohort identifiability verdict cannot be replaced by a fixed basis.")
    A("")
    # the best single fixed direction = leading eigenvector of the mean outer product
    Msum = (Vs[:, :, None] * Vs[:, None, :]).mean(axis=0)
    ww, VV = np.linalg.eigh(Msum)
    best = VV[:, int(np.argmax(ww))]
    cap = np.abs(Vs @ best)
    A(f"BEST SINGLE FIXED DIRECTION: {as_power_law(best)}")
    A(f"   captures |cos| mean {cap.mean():.3f}, median {np.median(cap):.3f}, "
      f"worst {cap.min():.3f}; >0.9 on {np.mean(cap > 0.9):.0%} of cohorts")
    A("")
    A("FLAT-DIRECTION COMPOSITION (mean |component|, sign-normalised)")
    for p, m in zip(PARAMS, np.abs(Vs).mean(axis=0)):
        A(f"   {p:<9s} {m:.3f}")
    A("")
    A("CONDITIONING (eigenvalue ratio, log-posterior covariance)")
    cd = np.array([r["cond"] for r in rows])
    ff = np.array([r["flat_frac"] for r in rows])
    A(f"   condition number: median {np.median(cd):.3g}, worst {cd.max():.3g}")
    A(f"   fraction of total variance in the flat direction: median {np.median(ff):.2f}")
    A("")
    A("BY REGIME -- does the flat direction change with the regime?")
    A(f"{'regime':<15s} {'n':>4s} {'mean|cos| to best':>18s} {'median cond':>13s}   dominant component")
    for lab, fn in REGIMES:
        idx = [i for i, r in enumerate(rows) if fn(r)]
        if not idx:
            continue
        sub = Vs[idx]
        c = np.abs(sub @ best)
        dom = PARAMS[int(np.argmax(np.abs(sub).mean(axis=0)))]
        A(f"{lab:<15s} {len(idx):>4d} {c.mean():>18.3f} "
          f"{np.median(cd[idx]):>13.3g}   {dom}"
          + ("  (thin)" if len(idx) < 6 else ""))
    return "\n".join(L)


# Physically-named candidate combinations, as exponent vectors in log(eta, beta, epsilon, kappa).
# The point of testing these rather than just reporting eigenvectors: an eigenvector is a number, a
# named quantity is something you can put in a paper and interpret. If one of these aligns with a
# well-determined eigendirection, THAT is the removal quantity to quote.
NAMED = {
    "beta/(kappa+1)":      None,       # not a pure power law -- handled separately below
    "beta/kappa":          (0.0, 1.0, 0.0, -1.0),
    "beta/eta":            (-1.0, 1.0, 0.0, 0.0),      # removal / production, dimensionless
    "beta*xc/eps":         (0.0, 1.0, -1.0, 0.0),      # removal Peclet
    "beta*kappa":          (0.0, 1.0, 0.0, 1.0),
    "beta*kappa^0.35":     (0.0, 1.0, 0.0, 0.35),      # the measured FLAT direction, for reference
    "beta/(eta*kappa)":    (-1.0, 1.0, 0.0, -1.0),
    "eta/eps":             (1.0, 0.0, -1.0, 0.0),      # omega-like, known well determined
    "sqrt(eta)/eps":       (0.5, 0.0, -1.0, 0.0),      # s, known well determined
}


def _unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def orthogonal_analysis(rows, out=None):
    """WHAT IS MEASURABLE ABOUT REMOVAL. The direct answer to 'nothing about removal is measurable'.

    Three questions, in order, because each only matters if the previous one passed:

      1. How TIGHT is each eigendirection? (from the eigenvalues -- already known: 2.31 / 1.15 / 0.40 /
         0.11 dex at the median)
      2. Is each direction STABLE across cohorts? A direction pinned to 0.11 dex that points somewhere
         different in every cohort is not a reportable quantity, it is a per-cohort curiosity. Measured
         the same way direction 0 was: mean |cos| between cohorts.
      3. Does any PHYSICALLY NAMED quantity lie along a tight, stable direction? This is what turns an
         eigenvector into something publishable.
    """
    if not rows or "eigenvectors" not in rows[0]:
        return "no eigenvectors saved -- rerun `analyse` with the current code"
    L = []
    A = L.append
    nd = 4
    # `eigenvectors[j]` is stored as V[:, j], so the axes are (cohort, DIRECTION, component).
    # Indexing this as Vs[i, :, j] reads "component j across all directions", which is not a direction
    # at all -- it silently produced a plausible-looking table with the wrong compositions and a
    # cross-cohort alignment of 0.588 where the correct value is 0.908.
    Vs = np.array([r["eigenvectors"] for r in rows], float)      # (cohort, direction, component)
    Ws = np.array([r["eigenvalues"] for r in rows], float)
    widths = np.array([r["widths_dex"] for r in rows], float)

    A("WHAT IS MEASURABLE ABOUT REMOVAL?")
    A("=" * 100)
    A(f"{len(rows)} cohorts. Eigendirections of the log-posterior covariance, WIDEST first. Direction 0")
    A("is the flat/unidentifiable one; directions 2-3 are the tight ones. Units: widths in decades")
    A("(95% = 2*1.96*sd/ln10); |cos| dimensionless.")
    A("")

    A("1. TIGHTNESS AND STABILITY PER DIRECTION")
    A(f"{'dir':>4s} {'median width':>13s} {'worst':>8s} {'mean|cos| across cohorts':>26s} "
      f"{'frac>0.9':>9s}   dominant")
    A("-" * 92)
    stab = []
    for j in range(nd):
        Vj = np.array([_unit(Vs[i, j, :]) for i in range(len(rows))])
        # sign-normalise before comparing: v and -v are the same direction
        for i in range(Vj.shape[0]):
            k = int(np.argmax(np.abs(Vj[i])))
            if Vj[i, k] < 0:
                Vj[i] *= -1
        Cm = np.abs(Vj @ Vj.T)
        iu = np.triu_indices(len(rows), 1)
        al = Cm[iu]
        stab.append((float(al.mean()), Vj))
        dom = PARAMS[int(np.argmax(np.abs(Vj).mean(axis=0)))]
        A(f"{j:>4d} {np.median(widths[:, j]):>13.3f} {widths[:, j].max():>8.3f} "
          f"{al.mean():>26.3f} {np.mean(al > 0.9):>9.2f}   {dom}")
    A("")
    A("A direction is REPORTABLE only if it is BOTH tight and stable. Tight-but-unstable means the")
    A("data constrains something different in every cohort -- real, but not a quantity you can name.")
    A("")

    A("2. COMPOSITION OF EACH DIRECTION (mean over cohorts, sign-normalised)")
    A(f"{'dir':>4s} " + " ".join(f"{p:>9s}" for p in PARAMS) + "   power law (median cohort)")
    for j in range(nd):
        Vj = stab[j][1]
        mean_v = _unit(Vj.mean(axis=0))
        A(f"{j:>4d} " + " ".join(f"{x:>9.3f}" for x in np.abs(Vj).mean(axis=0))
          + f"   {as_power_law(mean_v)}")
    A("")

    A("3. DO ANY PHYSICALLY NAMED QUANTITIES ALIGN WITH A TIGHT DIRECTION?")
    A("|cos| between each named combination and each eigendirection, averaged over cohorts.")
    A("A named quantity aligned with direction 2 or 3 is a REPORTABLE removal quantity.")
    A("")
    A(f"{'named combination':<22s} " + " ".join(f"{'dir' + str(j):>8s}" for j in range(nd))
      + f" {'best':>6s} {'its width':>10s}   verdict")
    A("-" * 92)
    best_named = []
    for name, expo in NAMED.items():
        if expo is None:
            continue
        u = _unit(expo)
        cosines = []
        for j in range(nd):
            Vj = stab[j][1]
            cosines.append(float(np.mean(np.abs(Vj @ u))))
        jb = int(np.argmax(cosines))
        w = float(np.median(widths[:, jb]))
        # a named quantity is only usable if it aligns well AND lands on a tight direction
        if cosines[jb] >= 0.8 and w < 1.0:
            verdict = "REPORTABLE"
        elif cosines[jb] >= 0.8:
            verdict = "aligned but WIDE (a bound)"
        else:
            verdict = "spans several directions"
        best_named.append((name, jb, cosines[jb], w, verdict))
        A(f"{name:<22s} " + " ".join(f"{c:>8.3f}" for c in cosines)
          + f" {'dir' + str(jb):>6s} {w:>10.3f}   {verdict}")
    A("")
    A("`beta/(kappa+1)` is not a pure power law so it has no fixed exponent vector; it is scored")
    A("directly in the coverage tables instead. Its measured width was 11.02 dex at 0.84 coverage.")
    A("")

    A("4. THE NET-DRIFT ARGUMENT, ANSWERED DIRECTLY")
    A("Naveh's objection: production and removal jointly set the net drift, so if removal is not")
    A("negligible it MUST leave a signature and MUST be measurable. That argument is correct, and the")
    A("compositions above show how it is satisfied -- and why it does not give you `beta`.")
    A("")
    bidx = PARAMS.index("beta")
    bw = [float(np.abs(stab[j][1]).mean(axis=0)[bidx]) for j in range(nd)]
    A(f"   beta's mean weight per direction:  "
      + ",  ".join(f"dir{j} {bw[j]:.3f}" for j in range(nd)))
    A("   Every tight direction is a MIXTURE that beta participates in, alongside larger eta and")
    A("   epsilon weights. So the data DOES constrain quantities involving beta -- exactly as the drift")
    A("   argument requires. What it does not do is separate beta's contribution from production's.")
    A("")
    A("   That is the precise resolution: **the net drift is measurable; its decomposition into")
    A("   production and removal is not.** A single curve fixes how fast damage approaches threshold,")
    A("   not how much of that rate is eta pushing up versus beta failing to pull down. Both raise the")
    A("   same curve, which is why beta appears only in mixed combinations and dominates the FLAT")
    A("   direction rather than a tight one.")
    A("")
    A("   Breaking it requires changing the net drift without changing the parameters -- i.e. an")
    A("   intervention, a covariate, or a direct damage measurement. It is not a sampling problem and")
    A("   no amount of compute addresses it.")
    A("")

    A("VERDICT")
    rep = [b for b in best_named if b[4] == "REPORTABLE"]
    tight_stable = [j for j in range(nd)
                    if np.median(widths[:, j]) < 1.0 and stab[j][0] > 0.85]
    if rep:
        A("   A NAMED removal quantity IS measurable:")
        for name, jb, c, w, _v in sorted(rep, key=lambda b: b[3]):
            A(f"     {name:<20s} aligns with direction {jb} at |cos| {c:.3f}, "
              f"median 95% width {w:.3f} dex")
        A("   Quote these as intervals. The claim that nothing about removal is measurable was an")
        A("   artifact of only ever scoring `beta`, `beta/kappa` and `beta/(kappa+1)`.")
    elif tight_stable:
        A(f"   Directions {tight_stable} are tight AND stable, so measurable combinations exist -- but")
        A("   none of the named candidates aligns with them. Report the eigen-combination itself, and")
        A("   look for a physical reading of the composition above.")
    else:
        A("   No direction is both tight and stable, so there is genuinely no transferable removal")
        A("   quantity on this library. State this against the net-drift argument explicitly.")
    txt = "\n".join(L)
    if out:
        with open(out, "w") as fh:
            fh.write(txt + "\n")
    return txt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summarise", default=None)
    ap.add_argument("--orthogonal", default=None,
                    help="space.json -> the orthogonal/named-quantity analysis (answers #22)")
    ap.add_argument("--floor", action="store_true",
                    help="apply the calibrated negligibility floors to the draws before the "
                         "eigen-decomposition (tests whether the flat direction is geometry or a "
                         "beta->0 tail artifact)")
    ap.add_argument("-k", type=int, default=50)
    a = ap.parse_args(argv)
    if a.orthogonal:
        print(orthogonal_analysis(json.load(open(a.orthogonal))))
        return 0
    if a.summarise:
        rows = json.load(open(a.summarise))
        print(summarise(rows))
        print()
        print(orthogonal_analysis(rows))
        return 0
    if not a.ckpt:
        ap.error("--ckpt is required unless --summarise or --orthogonal is given")
    rows = analyse(a.ckpt, a.manifest, k=a.k, out=a.out, floor=a.floor)
    print()
    print(summarise(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
