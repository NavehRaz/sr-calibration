"""Spec 004's pilot P2, S5's power (tasks T087, T112; research R19, R20; gate C4, C5). The phase `p2-s5-power` in
tools/acceptance_004.py runs it; its result sizes S5 at freeze-L.

On S5's configuration (gen-1's spec at xc sd 0.2; tools/presets_004.py ROLES["s5"], Naveh's to decide):
- **the plant:** gen-1's law at a perturbed sd whose error is 1.92 nats per 1e5 individuals: n KL(P_sd || P_sd') =
  1.92 n / 1e5, solved for sd' > sd (`plant_sd`);
- **R** for the score, any component (Bonferroni over 4), at 90 % power, from the law alone: the plant's mean shift
  n P_sd . d log P_sd' and the per-replicate SD sqrt(n Var_P(d log P)) are exact for a multinomial; R for KS from the
  plant's sup|dF| (`r_needed`);
- **the instrument's step bias at that R, in SE units**, by a coupled halving on HALVING_COHORTS cohorts (the dt rung
  against its dt/2 rung, the same streams), converted to the bias at dt as an upper bound, (|mean| + 2 SE) x
  1/(1 - 2^-p) with p from S5's p1-pooled cell (Naveh 2026-11-03): for clamp, and for reflect when gate C5's second
  branch is in play;
- **the per-cohort CPU, the projection against the 86 CPU-h cap, and the branch** (sampler_rules.s5_wall_branch).
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import s5_gen1_004 as S5  # noqa: E402

PLANT_NATS_PER_1E5 = 1.92
ALPHA, POWER = 0.05, 0.90
HALVING_COHORTS = 100
CAP_S5_HET = 86.0


def kl(P, Q) -> float:
    P, Q = np.asarray(P, float), np.maximum(np.asarray(Q, float), 1e-300)
    m = P > 0
    return float(np.sum(P[m] * np.log(P[m] / Q[m])))


def plant_sd(spec, nats_per_1e5=PLANT_NATS_PER_1E5, hi_step=0.2, tol=1e-5) -> dict:
    """sd' > sd with KL(P_sd || P_sd') = nats_per_1e5 / 1e5 per individual (bisection)."""
    base = S5.law(spec)
    target = nats_per_1e5 / 1e5

    def k(sd2):
        return kl(base["P"], S5.on_cells(S5.law(S5.with_fields(spec, het_var=sd2), C=base["C"]), base["ages"],
                                         base["events"]))
    lo, hi = spec.het_var, spec.het_var + hi_step
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if k(mid) < target else (lo, mid)
    sd2 = 0.5 * (lo + hi)
    return {"sd": spec.het_var, "plant_sd": sd2, "kl_per_individual": k(sd2), "target": target}


def r_needed(spec, plant, k=4, alpha=ALPHA, power=POWER) -> dict:
    """R for the score (any component) and for KS, from gen-1's laws alone."""
    from statistics import NormalDist
    base = S5.law(spec)
    pspec = S5.with_fields(spec, het_var=plant["plant_sd"])
    pl = S5.law(pspec, C=base["C"])
    P_on_plant = S5.on_cells(base, pl["ages"], pl["events"])
    d = S5.spec_gradient(pspec, pl)
    n = spec.n
    shift = S5.expected_score(P_on_plant, d, n)
    var = n * (P_on_plant @ (d * d) - (P_on_plant @ d) ** 2)
    sd = np.sqrt(np.maximum(var, 1e-300))
    zc, zp = NormalDist().inv_cdf(1 - alpha / (2 * k)), NormalDist().inv_cdf(power)
    R_comp = ((zc + zp) * sd / np.maximum(np.abs(shift), 1e-300)) ** 2
    F0 = np.cumsum(base["P"])
    F1 = np.cumsum(S5.on_cells(pl, base["ages"], base["events"]))
    dF = float(np.max(np.abs(F0 - F1)))
    R_ks = ((1.358 + zp) / dF) ** 2 / n
    return {"shift": shift.tolist(), "sd_per_replicate": sd.tolist(), "R_per_component": R_comp.tolist(),
            "R_score": int(math.ceil(float(np.min(R_comp)))), "R_ks": int(math.ceil(R_ks)), "sup_dF": dF, "zcrit": zc}


def coupled_halving(spec, dt, wall, key0, cohorts=HALVING_COHORTS, *, base=None, d=None) -> dict:
    """The instrument's step bias: per cohort the score at dt minus the score at its coupled dt/2 rung (both run to
    gen-1's horizon and recorded by gen-1's transform, S5.record), mean and SE over the cohorts, with the per-cohort
    CPU of a bare (error="ks") draw. A cohort that raises is kept in `errors` (the phase then reads incomplete) and
    the loop goes on; one progress line per cohort. `base` and `d` (gen-1's law and its gradient, as S5.run computes
    them) are reused when given: s5-het's re-check at its own step (A6) computes them once."""
    from sraging.langevin import sample as LS
    S5.plain(spec)
    base = S5.law(spec) if base is None else base
    t_end = S5.horizon(spec)                                 # while gen-1's cache still holds the spec's curves
    d = S5.spec_gradient(spec, base) if d is None else d
    p, het = S5.params_of(spec)
    diffs, cpu, errors, beyond = [], [], [], 0
    for i in range(cohorts):
        c0 = time.process_time()
        try:
            cfg = LS.LangevinConfig(params=p, n=spec.n, key=int(key0 + i), dt=float(dt), heterogeneity=het,
                                    delta=spec.delta, wall=wall, error="ks", t_max=t_end)
            st = LS.prepare(cfg)
            r1 = LS.run_rung(cfg, st["arrays"], st["t_max"], 1)
            cpu.append(time.process_time() - c0)
            r2 = LS.run_rung(cfg, st["arrays"], st["t_max"], 2)
            a1, e1, b1 = S5.record(r1[0], t_end, spec)
            a2, e2, b2 = S5.record(r2[0], t_end, spec)
            diffs.append(S5.score(S5.counts(a1, e1, base), d) - S5.score(S5.counts(a2, e2, base), d))
            beyond += b1 + b2
            what = "ok"
        except Exception as ex:                                # noqa: BLE001 -- kept as the cohort's error row
            import traceback
            errors.append({"cohort": i, "type": type(ex).__name__, "message": str(ex),
                           "traceback": traceback.format_exc().rstrip().splitlines()[-12:],
                           "cpu_s": time.process_time() - c0})
            what = f"error ({type(ex).__name__}: {ex})"
        print(f"[p2-s5-power] {wall} cohort {i + 1}/{cohorts}: {what}; CPU {time.process_time() - c0:.0f} s",
              flush=True)
    D = np.asarray(diffs)
    if len(D) < 2:
        return {"mean": None, "se": None, "cohorts": len(D), "cpu_per_cohort_s": float(np.median(cpu)) if cpu else None,
                "wall": wall, "dt": dt, "t_max": t_end, "n_beyond": beyond, "errors": errors}
    return {"mean": D.mean(axis=0).tolist(), "se": (D.std(axis=0, ddof=1) / math.sqrt(len(D))).tolist(),
            "cohorts": len(D), "cpu_per_cohort_s": float(np.median(cpu)), "wall": wall, "dt": dt, "t_max": t_end,
            "n_beyond": beyond, "errors": errors}


def a6_step(frozen_m, frozen_dt, m, text) -> dict:
    """S5's step under amendment A6 (prepared for Naveh's T119 answer): m a power-of-two multiple of the frozen m,
    dt = frozen dt x frozen m / m. Refused above gate C2's m_max (2 x the frozen m) unless the amendment's text says
    C2's m_max does not bind S5's step (A6_C2_CLAUSE)."""
    frozen_m, m = int(frozen_m), int(m)
    k = m // frozen_m
    if m <= frozen_m or m % frozen_m or k & (k - 1):
        raise ValueError(f"A6: m {m} is not a power-of-two multiple (> 1) of S5's frozen m {frozen_m}")
    m_max = 2 * frozen_m
    if m > m_max and A6_C2_CLAUSE.lower() not in str(text).lower():
        raise ValueError(f"A6: m {m} is above gate C2's m_max {m_max} (2 x the frozen m {frozen_m}); the amendment's "
                         f"text must say \"{A6_C2_CLAUSE}\"")
    return {"m": m, "dt": float(frozen_dt) / k, "factor": k, "m_max": m_max, "above_m_max": m > m_max}


A6_C2_CLAUSE = "C2's m_max does not bind S5's step"


def bias_factor(p_hat) -> float:
    """1 / (1 - 2^-p): the step bias of a sample at dt from the coupled halving's difference b(dt) - b(dt/2) (gate
    C1); 3.41 at p 1/2, 2 at p 1."""
    return 1.0 / (1.0 - 2.0 ** -float(p_hat))


def step_bias_in_se(halving, R, sd_per_replicate, p_hat) -> list:
    """S5's step bias at dt, an upper bound, in units of the mean score's SE at R (Naveh 2026-11-03, sr-naveh finding
    5): (|mean| + 2 SE) of the halving's difference x 1/(1 - 2^-p), p from S5's p1-pooled cell; the halving's
    difference alone is b(dt) - b(dt/2), not b(dt)."""
    f = bias_factor(p_hat)
    return [(abs(m) + 2.0 * e) * f / (s / math.sqrt(R))
            for m, e, s in zip(halving["mean"], halving["se"], sd_per_replicate)]


def smallest_detectable(R_needed, R_cap, nats=PLANT_NATS_PER_1E5) -> float:
    """At R_cap the detectable KL scales as 1/R (a mean shift detectable at 1/sqrt R, KL quadratic in it)."""
    return nats * R_needed / max(R_cap, 1)
