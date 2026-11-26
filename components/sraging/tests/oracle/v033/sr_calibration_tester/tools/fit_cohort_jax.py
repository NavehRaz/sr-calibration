"""Fit ONE cohort with the tester's reference method, `mle_profile_jax`, from a CSV.

The reference calibration method since 2026-09-26 is `methods/mle_profile_jax` (v0.3.3). It lives here,
as a tester adapter, not in SRK. This script is the one-line entry point for fitting a REAL cohort with
it: it calls the registered adapter's own `calibrate()`, so the claims it prints are exactly what the
tester scores -- same estimator, same claim shapes, same prior box -- with no second copy of the logic.

usage (from code_5/):
    export PYTHONPATH="$PWD/SRtools_kappa:$PWD/sr_calibration_tester"
    /Users/navehr/SRvenv/bin/python sr_calibration_tester/tools/fit_cohort_jax.py \\
        --csv my.csv --delta 1.0 [--json out.json]

CSV: header + two columns (death time, event; 1 = observed, 0 = right-censored), read with SRK's own
tolerant `fit_cohort.read_csv`. `--delta` is the recording resolution in the ages' own units (1.0 daily,
7 weekly); pass it whenever ages are rounded (the likelihood then scores each death as dying inside its
recording interval). Omit it only for genuinely continuous times.

What the claims mean (see sr_calibration_tester/STATUS.md): `closed` = two-sided 95% profile-likelihood
interval; `upper_bound` / `lower_bound` = only one side is constrained by the data; `none` = the data do
not constrain this parameter inside the prior box (a decline, reported without a point estimate). kappa
is declined on most cohorts, and one-sided kappa/beta bounds are poorly calibrated for every method
(~45% / ~63% coverage against a nominal >= 95% on the benchmark) -- do not quote them as 95% bounds.
The model has no external hazard, fit window, heterogeneity or censoring model beyond right-censoring.
Cost: typically ~10 CPU-min; hours when the recording resolution is very fine relative to the lifespan
(delta/L below ~1e-3 forces a fine solver step -- [T-DT-CEILING]).

Two KNOWN DEFECTS of v0.3.3 are checked after every fit and printed as WARNINGs (claims unchanged):
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
    ap.add_argument("--name", default=None, help="cohort label (default: the CSV file name)")
    ap.add_argument("--json", default=None, help="also write the full result (claims + diagnostics) here")
    args = ap.parse_args(argv)

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
    method = get("mle_profile_jax")
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
    tie = censoring_tie_warning(np.asarray(ages, float), np.asarray(events, int), args.delta)
    for w in convergence_warnings(cs) + ([tie] if tie else []):
        print(f"WARNING  {w}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(cs.to_dict(), fh, indent=1, default=float)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
