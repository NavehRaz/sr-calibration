"""Spec 004's S5, heterogeneous half: an independent check of gen-1's heterogeneous law (tasks T085, T088; research
R20; spec HR-7; gate C4, C5; OD-9). The phase `s5-het` in tools/acceptance_004.py runs it under SR (cap 86 CPU-h).

Langevin cohorts at gen-1's configuration (xc sd 0.2), run to gen-1's horizon (`horizon`: its support's end) and
recorded by gen-1's own transform (`record`: `_record`, a survivor at the horizon at gen-1's last node), are tested
one-sample against gen-1's exact binned law (`generator.binned_distribution`, read-only) with two
statistics, at P2's R:
- **the score:** sum_cells counts x d log P_cell / d log theta, with gen-1's law differentiated once per configuration
  by central differences in log (rho_eta, rho_beta, rho_eps, kappa); mean / SE over the R replicates per component;
- **KS** on the pooled raw times against gen-1's continuous survival.
The replicates draw no coupled rung (error="ks"): they cite P2's measured step bias. **The plant** scores the same
cohorts against gen-1's law at P2's perturbed sd; **the twin** against the true law. On a fail, OD-9's response record.
"""
from __future__ import annotations

import functools
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sraging import generator as GEN  # noqa: E402  (gen-1, read-only)
from sraging.errors import UnsupportedOptionError  # noqa: E402

COMPONENTS = ("rho_eta", "rho_beta", "rho_eps", "kappa")
CD_STEP = 1e-4
_FLOOR = 1e-300


# --------------------------------------------------------------------------------------------- gen-1's law
def with_fields(spec, **changes):
    """Another CohortSpec, the same fields but `changes` (CohortSpec is frozen and keyword-only)."""
    return GEN.CohortSpec(**{**{f: getattr(spec, f) for f in GEN.FIELDS}, **changes})


def law(spec, C=None) -> dict:
    """gen-1's exact binned law of the recorded outcome: {ages, events, P, C}."""
    return GEN.binned_distribution(spec, dict(GEN.GRID), C=C)


def on_cells(lw, ages, events) -> np.ndarray:
    """A law's P on another law's cells (the cells beyond its own support read the floor)."""
    idx = {(round(float(a), 9), int(e)): i for i, (a, e) in enumerate(zip(lw["ages"], lw["events"]))}
    return np.array([lw["P"][idx[(round(float(a), 9), int(e))]] if (round(float(a), 9), int(e)) in idx else 0.0
                     for a, e in zip(ages, events)])


def log_law_gradient(law_fn, x0, h=1e-5) -> np.ndarray:
    """(cells x K) central differences of log P (floored at 1e-300) in each x_k."""
    x0 = np.asarray(x0, float)
    cols = []
    for k in range(x0.size):
        e = np.zeros_like(x0)
        e[k] = h
        up, dn = np.maximum(law_fn(x0 + e), _FLOOR), np.maximum(law_fn(x0 - e), _FLOOR)
        cols.append((np.log(up) - np.log(dn)) / (2 * h))
    return np.column_stack(cols)


def spec_gradient(spec, base=None, h=CD_STEP) -> np.ndarray:
    """d log P_cell / d log (rho_eta, rho_beta, rho_eps, kappa) of gen-1's law, on the base law's cells (its C)."""
    base = base or law(spec)
    x0 = np.log([getattr(spec, c) for c in COMPONENTS])

    def P(x):
        s = with_fields(spec, **{c: float(math.exp(v)) for c, v in zip(COMPONENTS, x)})
        return on_cells(law(s, C=base["C"]), base["ages"], base["events"])
    return log_law_gradient(P, x0, h)


def score(counts, dlogP) -> np.ndarray:
    return np.asarray(counts, float) @ np.asarray(dlogP, float)


def expected_score(P, dlogP, n) -> np.ndarray:
    """The mean score of n draws from the law P (a planted law's shift; 0 for the true law)."""
    return float(n) * (np.asarray(P, float) @ np.asarray(dlogP, float))


@functools.lru_cache(maxsize=8)
def horizon(spec) -> float:
    """gen-1's support end in days, tau[-1] x L: where its march stopped (S below Gtol, ~1e-12), the time `_inverse`
    places a draw beyond its CDF's end at, and the end of its law's cells (171.12 days at S5's spec). Memoised per
    spec (a pure cache: gen-1's 161 node curves cost minutes once evicted from its own cache)."""
    return float(GEN._model(spec, dict(GEN.GRID))["tau"][-1]) * float(spec.L_phys)


def plain(spec):
    """S5 compares with gen-1's law before censoring and windows: a spec with either is refused by name."""
    if spec.censor_frac > 0:
        raise UnsupportedOptionError("censor_frac", "S5 compares with gen-1's law before censoring: censor_frac=0")
    if spec.time_range is not None:
        raise UnsupportedOptionError("time_range", "S5 compares with gen-1's law before windows: time_range=None")


def record(times, t_end, spec) -> tuple:
    """gen-1's record of Langevin raw times run to gen-1's horizon t_end: (ages, events, n_beyond), sorted as gen-1
    sorts. Every time is a death at floor(t / delta) delta (`_record` with no censoring: gen-1's law here has death
    cells only); a survivor at t_end sits at gen-1's last node, as `_inverse` places a draw beyond its CDF's end, and
    is counted (gen-1's n_beyond_cdf). A time past t_end is refused: the paths were run to another horizon.
    P2's run of 2026-11-03 (31ec891) ran to the sampler's default t_max (S_obs < 0.01 / n, 140.57 days here), where a
    survivor is recorded censored, a cell gen-1's law does not have: cohorts 53 and 69 raised in `counts`."""
    t = np.asarray(times, float)
    if t.size and float(t.max()) > t_end:
        raise ValueError(f"a raw time {float(t.max())} is past gen-1's horizon {t_end}: run the paths to t_max = "
                         f"horizon(spec)")
    ages, events = GEN._record(t, None, spec.delta)
    order = np.lexsort((events, ages))
    return ages[order], events[order], int(np.count_nonzero(t >= t_end))


def counts(ages, events, lw) -> np.ndarray:
    """Each recorded (age, event) on its cell of the law; refused when one is not a cell."""
    idx = {(round(float(a), 9), int(e)): i for i, (a, e) in enumerate(zip(lw["ages"], lw["events"]))}
    c = np.zeros(len(lw["ages"]))
    for a, e in zip(ages, events):
        k = (round(float(a), 9), int(e))
        if k not in idx:
            raise ValueError(f"record (age {a}, event {e}) is not a cell of the law")
        c[idx[k]] += 1
    return c


# ---------------------------------------------------------------------------------------------- the Langevin
def params_of(spec):
    """(SRParams, Heterogeneity or None) of a gen-1 spec: the tester's conversion at xc = 1 (srbench physical_truth)."""
    from sraging import SRParams
    from sraging.engine.inputs import Heterogeneity
    L = float(spec.L_phys)
    p = SRParams(eta=spec.rho_eta / (L * L), beta=spec.rho_beta / L, epsilon=spec.rho_eps / L, kappa=spec.kappa,
                 h_ext=math.exp(-spec.external_hazard) if spec.external_hazard is not None else 0.0)
    return p, (Heterogeneity(parameter="xc", sd=spec.het_var) if spec.heterogeneous else None)


def replicate(spec, key, dt, wall) -> dict:
    """One Langevin cohort at gen-1's configuration, run to gen-1's horizon and recorded by gen-1's transform
    (`record`); no coupled rung."""
    from sraging.langevin import simulate
    plain(spec)
    t_end = horizon(spec)
    p, het = params_of(spec)
    s = simulate(p, n=spec.n, key=int(key), dt=float(dt), heterogeneity=het, delta=spec.delta, wall=wall, error="ks",
                 t_max=t_end)
    ages, events, n_beyond = record(s.times, t_end, spec)
    return {"ages": ages, "events": events, "times": s.times, "t_max": t_end, "n_beyond": n_beyond,
            "cpu_s": s.cost_record["cpu_s"], "clean_but_for_ks": [w.code for w in s.warnings]}


def ks_gen1(times, t_max, spec) -> tuple:
    """(D, p) of raw times against gen-1's continuous observed survival."""
    from sraging.langevin.sample import ks_one_sample
    m = GEN._model(spec, dict(GEN.GRID))
    kinds = (np.asarray(times) < t_max).astype(np.int8)
    return ks_one_sample(np.asarray(times), kinds, lambda t: GEN._S_obs(m, spec, np.asarray(t, float)))


def zstats(G) -> dict:
    G = np.asarray(G, float)
    mean, sd = G.mean(axis=0), G.std(axis=0, ddof=1)
    se = sd / math.sqrt(G.shape[0])
    return {"mean": mean.tolist(), "se": se.tolist(), "z": dict(zip(COMPONENTS, (mean / se).tolist()))}


def run(spec, R, dt, wall, key0, plant_sd, work=None, *, base=None, d_true=None) -> dict:
    """S5's arms at R: the twin (gen-1's law) and the plant (gen-1's law at plant_sd), on the same R cohorts; KS
    pooled. `work(R) -> replicates`, to run under the phase's budget (default: in process); an error row among them
    (a replicate that raised) is kept in "errors" and left out of the arms, and HR-7 then reads incomplete. `base` and
    `d_true` (gen-1's law and its gradient) are reused when given (s5-het computes them once for its re-check, A6)."""
    base = law(spec) if base is None else base
    horizon(spec)                                            # while gen-1's cache still holds the spec's curves
    d_true = spec_gradient(spec, base) if d_true is None else d_true
    plant_spec = with_fields(spec, het_var=float(plant_sd))
    pbase = law(plant_spec)
    d_plant = spec_gradient(plant_spec, pbase)
    reps = work(R) if work is not None else [replicate(spec, key0 + i, dt, wall) for i in range(R)]
    errors = [{"id": r.get("id"), **r["error"]} for r in reps if r.get("error")]
    reps = [r for r in reps if not r.get("error")]           # a replicate that raised adds nothing to the arms
    Gt = [score(counts(r["ages"], r["events"], base), d_true) for r in reps]
    Gp = [score(counts(r["ages"], r["events"], pbase), d_plant) for r in reps]
    times = np.concatenate([r["times"] for r in reps])
    D, pv = ks_gen1(times, min(r["t_max"] for r in reps), spec)
    return {"R": len(reps), "twin": zstats(Gt), "plant": zstats(Gp), "ks": {"D": D, "p_value": pv, "n": int(times.size)},
            "plant_sd": float(plant_sd), "cpu_s": [r["cpu_s"] for r in reps], "errors": errors}


def od9_response(spec, verdict) -> dict:
    """OD-9 (a)'s response to a fail (Naveh 2026-10-29), recorded for T129: a finding against gen-1, not a gen-2."""
    return {"finding": f"S5 fails against gen-1's heterogeneous law at {spec.name} (xc sd {spec.het_var})",
            "verdict": verdict, "actions": ["record it in OPEN_TASKS and the register",
                                            "send it to the tester's owner by sesync",
                                            "state a limit on the v6 heterogeneous cohorts", "no gen-2 in spec 004"],
            "done": False}
