"""The gates themselves.

Every one of these caught a real error in this project's history. The `motivating_failure` string
on each is not decoration -- it is why the gate exists, and it is printed with the outcome so that
someone reading a report knows what the check is protecting them from.

Ids mirror the predecessor's G1..G14 where the check is the same, so the two harnesses can be
compared check by check.
"""
from __future__ import annotations

import math

from .registry import Gate, GateOutcome, outcome, register

# --------------------------------------------------------------------------- G1 bounds


def _g1(specs=None, bounds=None, **_) -> GateOutcome:
    bad = []
    for s in specs or []:
        truth = s.physical_truth() if hasattr(s, "physical_truth") else s
        for p, (lo, hi) in (bounds or {}).items():
            v = truth[p]
            if not (lo < v < hi):
                bad.append(f"{getattr(s, 'cohort_id', '?')}:{p}={v:g}")
    return outcome(G1, not bad,
                   f"{len(specs or [])} cohorts checked against the bounds in use; "
                   + (f"{len(bad)} OUTSIDE OR ON a bound: {bad[:5]}" if bad
                      else "all truths strictly inside"))


G1 = register(Gate(
    gate_id="G1", title="every planted truth lies strictly inside the bounds in use",
    motivating_failure=("a truth ON a boundary cannot be covered by a bounded interval, so it "
                        "scores a guaranteed miss that reads as a method failure"),
    check=_g1,
    failing_fixture=lambda: {"specs": [{"eta": 1e3, "beta": 1.0, "epsilon": 1.0, "kappa": 0.1}],
                             "bounds": {"eta": (1e-12, 1e3)}},
    passing_fixture=lambda: {"specs": [{"eta": 1.0, "beta": 1.0, "epsilon": 1.0, "kappa": 0.1}],
                             "bounds": {"eta": (1e-12, 1e3)}},
))


# --------------------------------------------------------------------------- G2 cells

def _g2(stratum_counts=None, minimum=10, **_) -> GateOutcome:
    thin = {k: v for k, v in (stratum_counts or {}).items() if v < minimum}
    return outcome(G2, not thin,
                   f"{len(stratum_counts or {})} strata; {len(thin)} below {minimum} cells"
                   + (f": {dict(list(thin.items())[:4])}" if thin else ""),
                   warn_instead=True)


G2 = register(Gate(
    gate_id="G2", title="no per-stratum figure rests on too few cells",
    motivating_failure=("a stratum of 3 cohorts produced a confident-looking coverage that was "
                        "noise"),
    check=_g2, blocking=False,
    failing_fixture=lambda: {"stratum_counts": {"a": 3, "b": 40}},
    passing_fixture=lambda: {"stratum_counts": {"a": 20, "b": 40}},
))


# --------------------------------------------------------------------------- G3 monotone

def _g3(profiles=None, **_) -> GateOutcome:
    bad = []
    for name, (subset_ll, superset_ll) in (profiles or {}).items():
        if superset_ll < subset_ll - 1e-9:
            bad.append(f"{name}: superset {superset_ll:.4g} < subset {subset_ll:.4g}")
    return outcome(G3, not bad,
                   f"{len(profiles or {})} nested profiles; {len(bad)} violations"
                   + (f" (worst: {bad[0]})" if bad else ""))


G3 = register(Gate(
    gate_id="G3", title="a profile over a superset is never worse than over a subset",
    motivating_failure=("a 59,872-nat optimiser failure that had already been written into a "
                        "headline; a violation means the optimiser failed, not that the "
                        "likelihood is interesting"),
    check=_g3,
    failing_fixture=lambda: {"profiles": {"x": (10.0, 5.0)}},
    passing_fixture=lambda: {"profiles": {"x": (5.0, 10.0)}},
))


# --------------------------------------------------------------------------- G4 dimensions

def _g4(n=1000, rho_beta=1.0, rho_eta=1.0, L=100.0, **_) -> GateOutcome:
    from ..score.floors import beta_floor_dimensionless, beta_floor_physical, negligible_beta
    eta = rho_eta / (L * L)
    dimensionless = beta_floor_dimensionless(n)
    physical = beta_floor_physical(n, eta, L)
    # The identity: the physical floor is the dimensionless one times eta*L.
    expected = dimensionless * eta * L
    ok = abs(physical - expected) <= 1e-12 * max(1.0, abs(expected))
    # And the unit-free entry point must agree with the physical comparison.
    agree = negligible_beta(n, rho_beta, rho_eta) == ((rho_beta / L) < physical)
    return outcome(G4, ok and agree,
                   f"dimensionless {dimensionless:.6g}; physical {physical:.6g}; "
                   f"identity holds={ok}; unit-free entry point agrees={agree}")


G4 = register(Gate(
    gate_id="G4", title="the negligibility floor is applied to the quantity it was calibrated on",
    motivating_failure=("the scale factor has been dropped TWICE in opposite directions -- once "
                        "926x too strict, making 'negligible' unreachable, and once a factor of L "
                        "too lenient, calling 0 of 2,400 cohorts negligible against 185"),
    check=_g4,
    failing_fixture=lambda: {"n": 1000, "L": 0.0},     # a zero scale makes the physical form raise
    passing_fixture=lambda: {"n": 1000, "L": 100.0},
))


# --------------------------------------------------------------------------- G5 fixed floor

def _g5(accuracy_results=None, **_) -> GateOutcome:
    moving = [a for a in (accuracy_results or []) if not a.get("floor_is_fixed", True)]
    return outcome(G5, not moving,
                   f"{len(accuracy_results or [])} accuracy figures; {len(moving)} computed at a "
                   "MOVING floor")


G5 = register(Gate(
    gate_id="G5", title="accuracy compared across sample size uses a FIXED floor",
    motivating_failure=("the floor shrinks 3.65x across the n range, which alone turned a flat "
                        "~0.60 dex into an apparent degradation to 0.892 dex -- the floor's own "
                        "n-dependence read as estimator behaviour"),
    check=_g5,
    failing_fixture=lambda: {"accuracy_results": [{"floor_is_fixed": False}]},
    passing_fixture=lambda: {"accuracy_results": [{"floor_is_fixed": True}]},
))


# --------------------------------------------------------------------------- G6 paired

def _g6(comparisons=None, **_) -> GateOutcome:
    unpaired = [c for c in (comparisons or []) if not c.get("paired", False)]
    return outcome(G6, not unpaired,
                   f"{len(comparisons or [])} cross-group comparisons; {len(unpaired)} unpaired")


G6 = register(Gate(
    gate_id="G6", title="comparisons across sample size or method use a paired cohort set",
    motivating_failure=("the fit declines less often as n grows, so the scored population "
                        "changes: 62% of one apparent coverage decline was composition, not "
                        "behaviour"),
    check=_g6,
    failing_fixture=lambda: {"comparisons": [{"paired": False}]},
    passing_fixture=lambda: {"comparisons": [{"paired": True}]},
))


# --------------------------------------------------------------------------- G7 null shape

def _g7(agreement_tests=None, **_) -> GateOutcome:
    wrong = [t for t in (agreement_tests or [])
             if t.get("null") != "se_of_difference_of_means"]
    return outcome(G7, not wrong,
                   f"{len(agreement_tests or [])} agreement tests; {len(wrong)} compared against "
                   "the wrong null")


G7 = register(Gate(
    gate_id="G7", title="an agreement test's null matches the statistic being tested",
    motivating_failure=("comparing a difference of MEANS against a single-observation spread is "
                        "lenient by sqrt(2/k); a test that passed at ratio 0.95 under the wrong "
                        "null FAILS at 1.88 under the right one"),
    check=_g7,
    failing_fixture=lambda: {"agreement_tests": [{"null": "single_observation_sd"}]},
    passing_fixture=lambda: {"agreement_tests": [{"null": "se_of_difference_of_means"}]},
))


# --------------------------------------------------------------------------- G8 resolution

def _g8(recomputations=None, **_) -> GateOutcome:
    bad = [r for r in (recomputations or []) if not r.get("recomputed", False)]
    return outcome(G8, not bad,
                   f"{len(recomputations or [])} stored results; {len(bad)} whose recording-"
                   "resolution treatment was taken from the code rather than RECOMPUTED")


G8 = register(Gate(
    gate_id="G8", title="the recording-resolution treatment is verified by recomputation",
    motivating_failure=("omitting it fits interval-censored data with a density and costs "
                        "0.15-0.20 of coverage; the resulting gradient was very nearly reported "
                        "as a finding about the parameter regime"),
    check=_g8,
    failing_fixture=lambda: {"recomputations": [{"recomputed": False}]},
    passing_fixture=lambda: {"recomputations": [{"recomputed": True}]},
))


# --------------------------------------------------------------------------- G9 solver failure

def _g9(profile_values=None, drop_nats=100.0, **_) -> GateOutcome:
    vals = list(profile_values or [])
    dips = []
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] - drop_nats and vals[i] < vals[i + 1] - drop_nats:
            dips.append((i, vals[i]))
    return outcome(G9, not dips,
                   f"{len(vals)} profile nodes; {len(dips)} sit more than {drop_nats:g} nats "
                   "below BOTH neighbours (a failed evaluation, not data)"
                   + (f": {dips[:3]}" if dips else ""))


G9 = register(Gate(
    gate_id="G9", title="evaluations far below their neighbours are rejected, not interpolated",
    motivating_failure=("interpolating an interval edge into a 3,874-nat cliff turns a solver "
                        "failure into a confident likelihood bound"),
    check=_g9,
    failing_fixture=lambda: {"profile_values": [-10.0, -11.0, -4000.0, -12.0, -13.0]},
    passing_fixture=lambda: {"profile_values": [-10.0, -11.0, -12.0, -13.0, -14.0]},
))


# ----------------------------------------------------- G12 the truth witness, BOTH tails

#: chi2(k) at 0.95, HALVED -- the deviance threshold as a log-likelihood difference, keyed by the
#: number of FITTED parameters k. Hard-coded so the constants are auditable and this module keeps
#: its single `math` dependency.
WITNESS_THRESHOLD_BY_DF = {1: 1.921, 2: 2.996, 3: 3.907, 4: 4.744}

#: The methods here fit eta, beta, epsilon and kappa, so k = 4.
WITNESS_DF = 4

#: Kept as a name because it reads in the detail string; it is now the df-correct value, not 1.92.
WITNESS_NATS = WITNESS_THRESHOLD_BY_DF[WITNESS_DF]

#: Under the null, |W| exceeds the threshold on this fraction of cohorts BY CHANCE. A per-cohort
#: 95% cut applied to 1,500 cohorts flags ~75 of them with nothing whatever wrong.
WITNESS_TAIL_RATE = 0.05

#: How far above the chance count the negative tail must sit before it means anything: 3 sigma of
#: the binomial, normal-approximated. Deliberately crude and deliberately explicit -- the point is
#: that SOME allowance for multiplicity exists, not that this is the sharpest possible test.
WITNESS_SIGMAS = 3.0


def _chance_ceiling(n: int, rate: float = WITNESS_TAIL_RATE, sigmas: float = WITNESS_SIGMAS):
    """The largest tail count attributable to chance on `n` cohorts."""
    if n <= 0:
        return 0.0
    return rate * n + sigmas * math.sqrt(n * rate * (1.0 - rate))


def _g12(truth_witness=None, threshold=None, df=WITNESS_DF, **_) -> GateOutcome:
    """Count BOTH tails of W = loglik(truth) - loglik(fit).

    WHICH log-likelihood each tail uses, and it is NOT the same one (jax2, 2026-10-18, asked that
    this be stated here rather than left to whoever builds the witness):

      POSITIVE tail -> the best point the method EVALUATED (`self_consistency.ll_best`).
      NEGATIVE tail -> the point it REPORTED (`self_consistency.ll_hat` == `diagnostics.loglik`).

    Because `mle_profile_jax2` 1.0.x may report a point below the best it found -- a "minor stale"
    point, kept deliberately, with the claims still covering the best one. Measured on the v5
    reference: 1,300 of 1,776 records (73%) are stale, median 0.0013 nats, worst 2.2648. Using the
    reported point for the POSITIVE tail would subtract a too-low number and manufacture a search
    failure out of staleness; at 2.26 nats that is half of this gate's 4.74 threshold, enough to
    flip a cohort sitting near it. The negative tail is safe either way, and the reported point is
    the better choice there because it describes the claim as actually stored.

    ⚑ The 1.92 cap is per CLASS, not overall, and the distinction matters when sizing. The method
    labels each record `stale`: none / minor / major. Measured across the v5 reference:

        none    1,291   worst gap 0.0100      minor   484   worst gap 1.8393
        major       1   gap 2.2648            (hazard-real-Denmark_F_1890_hetro-c0-s1)

    So 1.92 bounds MINOR exactly, and the single record above it is classified `major` -- the rule
    fired and classified correctly rather than failing. Use the gap, never the class name, when a
    bound is needed: a `major` record can sit anywhere.

    W > +threshold  the optimizer missed a point better than the truth -- a SEARCH failure.
    W < -threshold  the likelihood REJECTS the planted truth. THREE causes, not one, and they are
                    easy to confuse because the gate cannot tell them apart:
                      (a) misspecification -- the model being fitted is not the model that
                          generated the data;
                      (b) the DATA -- a generator defect, which is what this gate was written for;
                      (c) the METHOD'S OWN LIKELIHOOD GRID -- the fitter evaluates the truth on a
                          time step too coarse for the cohort, so loglik(truth) is computed too
                          low and the truth is rejected by arithmetic rather than by evidence.

    (c) IS NOT HYPOTHETICAL, and it is large. sraging measured (2026-10-15) that
    mle_profile_jax2 1.0.0 -- the current REFERENCE -- evaluates the likelihood at the TRUE
    parameters up to 0.214 nats per individual too low at sharp, small-kappa points, because its
    dt is 0.002 at delta 1, L~100. That is 2,140 nats of bias at n=10,000. On jax2's own DT_RULE
    grid (dt <= CV/80, built but switched OFF) the same gap is <= 1.8e-5. Their fit-free prediction
    is a witness of D = 20-23 at n = 10,000 from the method alone, i.e. W of -10 to -11 nats, which
    clears this gate's 4.74 threshold on its own.

    SO: a negative tail on sharp cohorts is NOT evidence against the generator until the fitter's
    time step has been ruled out. Reading it that way would have blamed library v6 for a defect
    that lives in the method, which is exactly the misattribution this note exists to prevent.

    CONFIRMED END TO END, 2026-10-16, on real fits rather than on this reasoning. sraging ran
    spec 003's censored coverage on sraging-gen-1 cohorts and reported 67 truth witnesses:
    9 beyond the cut (0.134 against 0.082 allowed) -- the rule FAILS. They then traced 8 of the 9
    to mle_profile_jax2 1.0.0's time step, and closed SC-011 with NO GENERATOR CAUSE: the
    generator's law matches an independent refined likelihood to <= 1.2e-6 nats per individual at
    every failing configuration. Remove the 8 and it is 1 of 59 against 2.95 expected.

    Their numbers and this gate agree exactly, which is the useful part:
      * their cut D = 9.49 is this threshold doubled (D = -2W), 9.49/2 = 4.745 against our 4.744;
      * their chance allowance, 2.95 of 59 "at 4 df", is WITNESS_TAIL_RATE on the same df;
      * their clean median D 2.88 is W = -1.44, inside the -1.2 to -1.7 predicted above;
      * run through this check, their full 67 FAILS (9 against a ceiling of 8.70) and their
        8-removed 59 PASSES (1 against 7.97) -- the same verdict they reached independently.

    Note what the OLD 1.92-nat cut would have done to this set: it is D = 3.84, just above their
    median of 3.085, so it would have flagged roughly half of a sample whose generator is now
    proven sound to 1.2e-6 nats per individual.

    WHY BOTH, and why this gate exists at all. Every rule in this project looked only at the
    positive tail. On the v1 validity audit that tail holds 3 cohorts; the negative tail holds 266,
    with a MEDIAN W of -22.4 nats and 176 cohorts below -20. The median cohort rejected its own
    planted truth by 22 nats and nothing said a word.

    That silence is how the first-bin defect survived: the generator clamped sub-delta deaths to
    delta/2 while every likelihood scored a death at `a` over `[a, a+delta)`, so the truth was
    rejected by 20-41 nats on affected cohorts -- a huge, systematic, entirely invisible signal.
    Requested by the mle_profile_jax2 session on 2026-10-08 for exactly this reason.

    A one-sided witness is not a weaker check than a two-sided one; it is a check pointed away
    from the failure mode that actually occurred.

    CORRECTED 2026-10-14, on a defect reported by the sraging generator session: this gate had TWO
    statistical errors, and both made it fire on data that was fine.

      * the cut was 1.92 nats, which is chi2 at ONE degree of freedom. The witness compares the
        truth against a fit that optimised 3-4 free parameters, so the null is chi2(3) or chi2(4)
        and the cut belongs at 3.91 or 4.74. sraging measured the consequence directly: the old cut
        would fail 28-43% of a PERFECT generator's cohorts. Their independent check that the model
        is right: a perfect generator's median W should be -1.2 to -1.7 nats, and the chi2 medians
        at df 3 and 4 are 1.18 and 1.68.
      * ANY cohort in either tail failed the gate. But the negative tail is a draw from the null --
        ~5% of cohorts sit beyond a 95% cut with nothing wrong -- so on a 1,500-cohort run roughly
        75 are expected and the gate could essentially never pass. It now asks whether the tail is
        bigger than chance, while the positive tail still fails on one, because a search failure is
        not a draw from anything.

    WHAT SURVIVES THE CORRECTION, and it is the whole point: v1's median witness of -22.4 nats is
    nowhere near explained by the cut. A perfect generator medians around -1.5. The first-bin
    defect this gate was written to catch is still enormous and still real; what changed is that
    the gate no longer cries wolf on the other 95%.
    """
    thr = threshold if threshold is not None else WITNESS_THRESHOLD_BY_DF.get(
        int(df), WITNESS_NATS)
    w = [float(x) for x in (truth_witness or []) if x is not None]
    hi = [x for x in w if x > thr]
    lo = [x for x in w if x < -thr]
    worst = min(w) if w else None

    # The two tails are NOT the same kind of event, so they are not judged the same way.
    #
    # W > +thr means the optimizer returned a point worse than one it was handed. That is a search
    # failure, not a draw from the null -- it does not happen by chance, so ONE is enough.
    #
    # W < -thr IS a draw from the null: 2|W| ~ chi2(df) when the fitted model generated the data,
    # so ~5% of cohorts sit beyond the 95% point with nothing wrong at all. Failing on the first
    # such cohort, as this gate did until 2026-10-14, makes it unpassable on any large run.
    ceiling = _chance_ceiling(len(w))
    lo_excess = len(lo) > ceiling

    detail = (f"{len(w)} cohorts with a truth witness (chi2 df={df}, threshold {thr:.2f} nats); "
              f"{len(hi)} above +{thr:.2f} (search missed a better point), {len(lo)} below "
              f"-{thr:.2f} (the likelihood rejects the truth) against {ceiling:.1f} attributable "
              f"to chance")
    if lo:
        detail += f"; worst {worst:.1f} nats"
    return outcome(G12, not (hi or lo_excess), detail)


G12 = register(Gate(
    gate_id="G12", title="the truth witness is counted in BOTH tails",
    motivating_failure=("every rule looked only at the POSITIVE tail. On the v1 audit that tail "
                        "holds 3 cohorts and the other holds 266, median -22.4 nats -- which is "
                        "how the delta/2 first-bin defect stayed hidden for the whole project"),
    check=_g12,
    # The failing fixture is the SHAPE of the real defect: a run whose positive tail is clean and
    # whose negative tail is not. A gate that only failed on a two-sided mess would have passed
    # the thing it was written for.
    failing_fixture=lambda: {"truth_witness": [0.1, -0.4, 0.8, -22.4, -35.0, -41.6]},
    passing_fixture=lambda: {"truth_witness": [0.1, -0.4, 0.8, 1.2, -1.1, 0.0]},
))


# --------------------------------------------------------------------------- G10 document

def _g10(markdown=None, **_) -> GateOutcome:
    import re
    bad = []
    for block in (markdown or []):
        rows = [ln for ln in block.split("\n") if ln.strip().startswith("|")]
        counts = set()
        for ln in rows:
            s = re.sub(r"\\\|", "", ln)                     # escaped pipes are not separators
            s = re.sub(r"`[^`]*`", lambda m: m.group(0).replace("|", "\x00"), s)
            counts.add(s.strip().strip("|").count("|") + 1)
        if len(counts) > 1:
            bad.append(sorted(counts))
    return outcome(G10, not bad,
                   f"{len(markdown or [])} tables; {len(bad)} with mismatched column counts"
                   + (f" {bad[:3]}" if bad else ""))


G10 = register(Gate(
    gate_id="G10", title="generated tables have matching column counts",
    motivating_failure=("pipes inside inline code AND absolute-value bars both act as column "
                        "separators; a row rendered with 14 columns against an 8-column header "
                        "and looked fine in a diff"),
    check=_g10,
    failing_fixture=lambda: {"markdown": ["| a | b |\n|---|---|\n| 1 | 2 | 3 |"]},
    passing_fixture=lambda: {"markdown": ["| a | b |\n|---|---|\n| 1 | 2 |"]},
))


# --------------------------------------------------------------------------- G11 combos

def _g11(n_draws=200, tol=1e-9, **_) -> GateOutcome:
    import numpy as np
    from ..score import combos as C
    box = {"eta": (1e-6, 1e2), "beta": (1e-6, 1e2), "epsilon": (1e-6, 1e2), "kappa": (1e-3, 300.0)}
    order = ("eta", "beta", "epsilon", "kappa")
    rng = np.random.default_rng(20260922)
    worst, worst_nm, checked = 0.0, None, 0
    for name, spec in C.COMBOS.items():
        si = C.IDX[spec["solve"]]
        for _ in range(n_draws):
            th = [float(10 ** rng.uniform(math.log10(box[p][0]), math.log10(box[p][1])))
                  for p in order]
            g = spec["formula"](th)
            if not math.isfinite(g) or g <= 0:
                continue
            th2 = list(th)
            for k in range(4):
                if k == si:
                    continue
                lo, hi = box[order[k]]
                th2[k] = float(min(max(th[k] * 10 ** rng.uniform(-1, 1), lo * 1.001), hi * 0.999))
            try:
                th2[si] = float(spec["inv"](g, th2))
            except (ZeroDivisionError, ValueError, OverflowError):
                continue
            lo, hi = box[order[si]]
            if not (math.isfinite(th2[si]) and lo <= th2[si] <= hi):
                continue
            back = spec["formula"](th2)
            if not math.isfinite(back) or back <= 0:
                continue
            checked += 1
            rel = abs(back - g) / abs(g)
            if rel > worst:
                worst, worst_nm = rel, name
    return outcome(G11, checked > 0 and worst < tol,
                   f"{checked} round trips over {len(C.COMBOS)} combinations; worst relative "
                   f"error {worst:.2e}" + (f" on {worst_nm}" if worst_nm else ""))


G11 = register(Gate(
    gate_id="G11", title="combination substitutions round-trip after moving along the surface",
    motivating_failure=("an inverted substitution yields a smooth, plausible, completely wrong "
                        "interval and NOTHING downstream notices: the walk converges, the "
                        "deviance rises, the threshold is crossed, a number is reported"),
    check=_g11,
    failing_fixture=lambda: {"n_draws": 200, "tol": 0.0},   # no tolerance -> must fail
    passing_fixture=lambda: {"n_draws": 200, "tol": 1e-9},
))


# --------------------------------------------------------------------------- G13 components

def _g13(claims=None, **_) -> GateOutcome:
    unrecorded = [c for c in (claims or []) if c.get("n_components") is None]
    return outcome(G13, not unrecorded,
                   f"{len(claims or [])} claims; {len(unrecorded)} did not record how many "
                   "components their region has")


G13 = register(Gate(
    gate_id="G13", title="the claimed region is the whole region, with components recorded",
    motivating_failure=("20 of 72 profiles had a DISJOINT 95% set and on 4 of 24 the truth sat in "
                        "a component other than the optimum's; returning only the optimum's "
                        "component is narrower always in the same direction, so it depresses "
                        "coverage most on the flat parameters where it matters"),
    check=_g13,
    failing_fixture=lambda: {"claims": [{"n_components": None}]},
    passing_fixture=lambda: {"claims": [{"n_components": 1}]},
))


# --------------------------------------------------------------------------- G14 density

def _g14(points_per_kappa=None, minimum=20, **_) -> GateOutcome:
    """ADVISORY, like G2, and for the same reason: it constrains a CLAIM, not a run.

    A run smaller than the full tier legitimately carries fewer points per kappa. That does not
    make the run wrong -- it makes a cross-kappa comparison unlicensed, which the report states
    rather than the gate forbidding the run outright. Blocking here would stop a perfectly good
    medium-tier result for failing to support a claim it never makes.
    """
    thin = {k: v for k, v in (points_per_kappa or {}).items() if v < minimum}
    detail = (f"{len(points_per_kappa or {})} kappa values; "
              f"{len(thin)} below {minimum} distinct manifold points"
              + (f": {thin}" if thin else ""))
    if thin:
        detail += ("  -> this run does NOT license a cross-kappa comparison; the report must not "
                   "make one")
    return outcome(G14, not thin, detail, warn_instead=True)


G14 = register(Gate(
    gate_id="G14", title="every kappa carries enough distinct manifold points for a comparison",
    motivating_failure=("the predecessor's map holds 6 points at kappa=0.05 against 21 at "
                        "kappa=120, so reading 'regime dependence' there mostly reads WHICH "
                        "manifold positions that kappa happens to contain"),
    check=_g14, blocking=False,
    failing_fixture=lambda: {"points_per_kappa": {0.01: 6, 120.0: 21}},
    passing_fixture=lambda: {"points_per_kappa": {0.01: 24, 10.0: 24}},
))


# --------------------------------------------------------------------------- G15 the null

def _g15(coverages=None, **_) -> GateOutcome:
    missing = [c.get("parameter") for c in (coverages or []) if c.get("null_coverage") is None]
    return outcome(G15, not missing,
                   f"{len(coverages or [])} coverage figures; {len(missing)} reported WITHOUT the "
                   "data-ignoring null beside them" + (f": {missing[:4]}" if missing else ""))


G15 = register(Gate(
    gate_id="G15", title="every coverage figure is reported beside the data-ignoring null",
    motivating_failure=("an estimator that ignores the data and returns the prior box covers the "
                        "flattest parameter at 1.0000 on all 462 cohorts at every level, beating "
                        "the selected estimator's 0.822 -- and it went unrun for months"),
    check=_g15,
    failing_fixture=lambda: {"coverages": [{"parameter": "kappa", "null_coverage": None}]},
    passing_fixture=lambda: {"coverages": [{"parameter": "kappa", "null_coverage": 1.0}]},
))


# --------------------------------------------------------------------------- G16 shapes

def _g16(coverages=None, **_) -> GateOutcome:
    pooled = [c.get("parameter") for c in (coverages or []) if c.get("shape_class") == "any"]
    return outcome(G16, not pooled,
                   f"{len(coverages or [])} coverage figures; {len(pooled)} POOLED one-sided with "
                   "two-sided claims" + (f": {pooled[:4]}" if pooled else ""))


G16 = register(Gate(
    gate_id="G16", title="one-sided and two-sided claims are never pooled into one coverage",
    motivating_failure=("pooled, the flattest parameter reads 0.676; its two-sided intervals "
                        "cover 0.250 -- a one-sided claim contains the truth almost for free"),
    check=_g16,
    failing_fixture=lambda: {"coverages": [{"parameter": "kappa", "shape_class": "any"}]},
    passing_fixture=lambda: {"coverages": [{"parameter": "kappa", "shape_class": "closed"}]},
))
