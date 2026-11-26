"""Generate the small synthetic test cohort `tests/unit/data/small_cohort.csv` (task T027), reproducibly.

Deaths are drawn by inverse-CDF sampling of the FP survival of the ported v0.3.3 solver at stated
dimensionless parameters, scaled by a time unit L, floored to the recording grid delta = 1 (as the
tester's library floors ages), with ~5% independent (not end-of-study) right-censoring, so neither the
censor-tie nor the off-grid check should fire. Seeded; the CSV is committed so tests never depend on it.

    .venv/bin/python tools/make_test_cohort.py
"""
from __future__ import annotations

import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")

PARAMS = dict(rho_eta=2.0, rho_beta=1.0, rho_eps=0.1, kappa=1.0)   # dimensionless, xc = 1
L_UNIT = 50.0     # time unit: ages ~ tau * L_UNIT
N = 300
DELTA = 1.0
SEED = 20261003


def main():
    from sraging.compat.v033.fp_curvefit import fpt_full_survival

    tau = np.linspace(0.0, 6.0, 6001)
    G, mean, cv, s_end = fpt_full_survival(PARAMS["rho_eta"], PARAMS["rho_beta"], PARAMS["rho_eps"],
                                           PARAMS["kappa"], tau)
    F = 1.0 - np.asarray(G, float)                     # CDF on the tau grid, non-decreasing
    F = np.maximum.accumulate(F)
    rng = np.random.default_rng(SEED)
    ages, events = [], []
    while len(ages) < N:
        u = rng.uniform(F[1], F[-1])
        t = float(np.interp(u, F, tau)) * L_UNIT
        a = np.floor(t / DELTA) * DELTA
        if a <= 0:
            continue
        if rng.uniform() < 0.05:                        # independent censoring, strictly before death
            c = np.floor(rng.uniform(0.2, 0.9) * t / DELTA) * DELTA
            if c <= 0:
                continue
            ages.append(float(c)); events.append(0)
        else:
            ages.append(float(a)); events.append(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("death times,events\n")
        for a, e in zip(ages, events):
            fh.write(f"{a!r},{e}\n")
    print(f"wrote {OUT}: n={len(ages)}, deaths={sum(events)}, FP mean tau={mean:.4f}, cv={cv:.3f}")


if __name__ == "__main__":
    main()
