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
