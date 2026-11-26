"""Fit ONE cohort from a CSV with a registered tester method: `mle_profile_jax` v0.3.3 (the default) or
`mle_profile_jax2` (--method mle_profile_jax2), whatever version is registered: 1.0.1 since 2026-10-27.

The tester's reference is `methods/mle_profile_jax2` 1.0.1 on library v6 since 2026-10-27 (run jax2-v6-full-bign);
1.0.0 on library v5 from 2026-10-13 (run jax2-v5-full-bign), and from 2026-09-26 to 2026-10-13
`methods/mle_profile_jax` v0.3.3. This script's DEFAULT stays v0.3.3: flipping
it is a later, separate decision (jax2 RATIFICATION, decision 5). Both methods live here as tester adapters, not in
SRK. This script is the one-line entry point for fitting a REAL cohort with either: it calls the registered
adapter's own `calibrate()`, so the claims it prints are exactly what the tester scores -- same estimator, same
claim shapes, same prior box -- with no second copy of the logic.

usage (from code_5/):
    export PYTHONPATH="$PWD/SRtools_kappa:$PWD/sr_calibration_tester"
    python sr_calibration_tester/tools/fit_cohort_jax.py \\
        --csv my.csv --delta 1.0 [--json out.json] [--method mle_profile_jax2]

--method picks the registered method; the default stays mle_profile_jax v0.3.3. With --method mle_profile_jax2
(1.0.0 and later) the two v0.3.3 warnings below (CENSORING TIES, EXACT TIMES) are not printed: it relabels such ties
itself and computes the exact-time density from the absorbed mass.
NOT offered here, by decision: `mle_profile_jax2_h` (H 1.0.1, which also fits an external hazard and takes a fit
window) and the kfix variants. Call those from Python, as the sr-aging-model skill's "Fit ONE cohort" section shows.

CSV: header + two columns (death time, event; 1 = observed, 0 = right-censored), read with SRK's own
tolerant `fit_cohort.read_csv`. `--delta` is the recording resolution in the ages' own units (1.0 daily,
7 weekly); pass it whenever ages are rounded (the likelihood then scores each death as dying inside its
recording interval). It is REQUIRED: a run without it refuses unless --exact-times says the ages really are
continuous (see EXACT TIMES below).

What the claims mean (see sr_calibration_tester/STATUS.md): `closed` = two-sided 95% profile-likelihood
interval; `upper_bound` / `lower_bound` = only one side is constrained by the data; `none` = the data do
not constrain this parameter inside the prior box (a decline, reported without a point estimate). kappa
is declined on most cohorts, and one-sided kappa/beta bounds are poorly calibrated for every method
(~45% / ~63% coverage against a nominal >= 95% on the benchmark) -- do not quote them as 95% bounds.
The model has no external hazard, fit window, heterogeneity or censoring model beyond right-censoring.
Cost: typically ~10 CPU-min; hours when the recording resolution is very fine relative to the lifespan
(delta/L below ~1e-3 forces a fine solver step -- [T-DT-CEILING]).

EXACT TIMES (rc2 I4, 2026-10-03). Without --delta the likelihood scores each death by the density p(t) = -dS/dt,
which v0.3.3 computes as a difference of two survival values. Where S is within round-off of 1 -- the early deaths
of a large or sharp cohort -- that difference is 0 or noise, so those deaths sit on the -700 log floor or jump off
it under 1-ulp changes of the parameters: MEASURED on a 1,000-record sharp benchmark cohort with exact ages, the
value spreads by 1e4-1e5 nats under +-3 ulp (median 2.6e4) and its forward- and reverse-mode gradients disagree by
up to 46% (the exact-time twin of the interval-mass stall behind [T-JAX-BASIN]). mle_profile_jax2 (1.0.0.dev4+)
computes that density from the absorbed mass instead (<= 7.4e-10 nats spread). With v0.3.3, record ages at their
resolution and pass --delta; --exact-times runs anyway and prints a WARNING.

Two KNOWN DEFECTS of v0.3.3 are checked after its fits and printed as WARNINGs (claims unchanged). NOT CONVERGED
is checked for either method; CENSORING TIES only for v0.3.3:
  * NOT CONVERGED -- a profile walk found a point more than 1.92 nats better than the reported best
    fit, so the point is not the maximum and the intervals are centred wrongly. Measured on the
    benchmark: all such failures sit on large, sharp cohorts (humans, n >= 25,000), where the stored
    intervals missed the truth by up to 3 decades ([T-JAX-BASIN]). Do not use those numbers.
  * CENSORING TIES -- with --delta, deaths recorded in the same bin as an end-of-study censoring age
    are scored over the whole bin although they died before the study ended. That biases eta upward
    once they are >= 0.5% of the cohort (benchmark censor arm: eta coverage 0.50 there vs 0.88 below).
    Relabelling those deaths as censored at that age removed the bias in the audit ([T-TESTER-DEFECTS] 7).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


#: Half the 95% deviance threshold: a gain this large moves an interval edge.
_GAIN_NATS = 1.92
_TIE_FRACTION = 0.005


def convergence_warnings(cs) -> list[str]:
    """Truth-free signs that the reported best fit is not the maximum (see the module docstring)."""
    out = []
    grid = (cs.diagnostics or {}).get("grid") or {}
    gains = {p: g.get("gain_nats", 0.0) for p, g in grid.items() if g.get("recentred")}
    worst = max(gains.values(), default=0.0)
    if worst > _GAIN_NATS:
        out.append(f"NOT CONVERGED: the profile walk found a point {worst:.1f} nats better than the "
                   f"reported best fit ({', '.join(p for p, g in gains.items() if g > _GAIN_NATS)}). "
                   f"The point is not the maximum and the intervals are centred wrongly -- do not use "
                   f"these numbers ([T-JAX-BASIN]).")
    for c in cs.claims:
        if (c.shape == "closed" and None not in (c.point, c.lower, c.upper)
                and not c.lower * (1 - 1e-9) <= c.point <= c.upper * (1 + 1e-9)):
            out.append(f"NOT CONVERGED: the {c.parameter} point {c.point:.5g} lies outside its own "
                       f"interval [{c.lower:.5g}, {c.upper:.5g}].")
    return out


def censoring_tie_warning(ages, events, delta) -> str | None:
    """Deaths recorded in the bin of an end-of-study censoring age (see the module docstring)."""
    if not delta or np.all(events == 1):
        return None
    tc = float(np.max(ages[events == 0]))
    if tc < float(np.max(ages)):
        return None                            # censoring is not at the end of follow-up
    tied = int(np.sum((events == 1) & np.isclose(ages, tc)))
    if tied < _TIE_FRACTION * len(ages):
        return None
    return (f"CENSORING TIES: {tied} deaths ({tied / len(ages):.1%} of the cohort) are recorded at the "
            f"censoring age {tc:g}. They are scored as dying anywhere in [{tc:g}, {tc + delta:g}) "
            f"although they died before follow-up ended, which biases eta upward. If follow-up ended "
            f"inside that bin, refit with those deaths relabelled as censored at {tc:g} "
            f"([T-TESTER-DEFECTS] 7).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="two columns: death time, event (1 death, 0 censored)")
    ap.add_argument("--delta", type=float, default=None, help="recording resolution in the ages' units")
    ap.add_argument("--exact-times", action="store_true",
                    help="the ages are continuous (no --delta): run anyway (with v0.3.3, prints the EXACT TIMES "
                         "warning)")
    ap.add_argument("--method", default="mle_profile_jax", choices=("mle_profile_jax", "mle_profile_jax2"),
                    help="registered method to fit with. Default: mle_profile_jax (frozen v0.3.3; flipping the "
                         "default is a separate, later decision). mle_profile_jax2 = the registered version, 1.0.1 (SR only; "
                         "fix B, the censoring-tie relabel). The external-hazard / fit-window variant "
                         "(mle_profile_jax2_h) is not offered here: call it from Python.")
    ap.add_argument("--name", default=None, help="cohort label (default: the CSV file name)")
    ap.add_argument("--json", default=None, help="also write the full result (claims + diagnostics) here")
    args = ap.parse_args(argv)
    if not args.delta and not args.exact_times:
        if args.method == "mle_profile_jax":
            ap.error("--delta is required (the recording resolution of the ages). Exact ages are scored by a "
                     "round-off density in mle_profile_jax v0.3.3 (see EXACT TIMES in --help); pass --exact-times "
                     "to run without --delta anyway.")
        ap.error("--delta is required (the recording resolution of the ages): rounded ages scored as exact times "
                 "are the wrong likelihood. mle_profile_jax2 computes the exact-time density from the absorbed "
                 "mass, so if the ages really are continuous pass --exact-times.")

    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    from SRK.fit_cohort import read_csv                       # SRK's tolerant two-column reader
    from srbench.contract.method import discover, get
    from srbench.contract.types import Observation

    ages, events = read_csv(args.csv)
    if args.delta:
        off = np.abs(ages / args.delta - np.round(ages / args.delta))
        frac = float(np.mean(off > 1e-6))
        if frac > 0:
            print(f"warning: {frac:.1%} of ages are not multiples of --delta {args.delta:g}; "
                  f"check the recording resolution")
    discover()
    method = get(args.method)
    v033 = args.method == "mle_profile_jax"     # the EXACT TIMES and CENSORING TIES warnings describe v0.3.3
    obs = Observation(cohort_id=args.name or os.path.basename(args.csv),
                      ages=tuple(float(a) for a in ages), events=tuple(int(e) for e in events),
                      delta=args.delta)
    d = method.descriptor
    print(f"method  {d.name} v{d.version}   n={len(ages)} ({int(np.sum(events))} deaths)   "
          f"delta={args.delta}")
    t0 = time.time()
    cs = method.calibrate(obs)
    if cs.status != "ok":
        print(f"FAILED: {cs.status}: {cs.diagnostics.get('exception')}")
        return 1
    print(f"done in {time.time() - t0:.0f} s wall, {cs.cpu_seconds:.0f} CPU-s;  "
          f"loglik {cs.diagnostics.get('loglik'):.3f}")
    print(f"{'parameter':10s} {'claim':12s} {'point':>12s} {'lower':>12s} {'upper':>12s}")
    for c in cs.claims:
        f = lambda v: "" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.5g}"
        print(f"{c.parameter:10s} {c.shape:12s} {f(c.point):>12s} {f(c.lower):>12s} {f(c.upper):>12s}")
    tie = censoring_tie_warning(np.asarray(ages, float), np.asarray(events, int), args.delta) if v033 else None
    exact = None if (args.delta or not v033) else (
        "EXACT TIMES: no --delta, so every death is scored by mle_profile_jax v0.3.3's exact-time density, which is "
        "round-off for deaths where S(t) is within ~1e-13 of 1 (early deaths of a large or sharp cohort): their terms "
        "sit on the -700 floor or jump under 1-ulp changes, and the fit can stall. Prefer --delta.")
    for w in convergence_warnings(cs) + ([tie] if tie else []) + ([exact] if exact else []):
        print(f"WARNING  {w}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(cs.to_dict(), fh, indent=1, default=float)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
