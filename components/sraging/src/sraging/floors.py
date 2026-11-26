"""FLOORED_DEATHS: which deaths the fit scored by its arithmetic or its time grid rather than by the model, and
why (spec 002 FR-026 to FR-028; research R17, revised at review T064: sr-naveh N4, N6, N7, N12; register
D-004, D-105).

After a fit, the deaths are evaluated at theta-hat on the fit's own grid with v0.3.3's arithmetic (the fit's
numerics "v033") and with the corrected arithmetic ("v1"). A death is a CANDIDATE when
- its v033 term is on the -700 floor, or
- its term moves by more than 1 nat between the two, or
- its corrected mass is below 1e-12 (robust to how v033 happened to round: one ulp takes the ITP example's
  added age-0 death from -700 to -30.5, D-105).
Every candidate that lies inside the march is REFINED: the time step is divided by 4, 16, ... with N fixed and
the march stopping just after the last candidate, until its term moves by less than 1 nat between two
refinements (at most REFINE_MAX refinements and REFINE_BUDGET node-steps; the step, not N, is what an early
death's term depends on: at the ITP theta-hat the age-0 death is -36.8 on the fit's grid and -114.0 at dt/4^6,
while N = 900/1800/3600 agree to 0.2 nats). Each candidate gets one category, from its refined value:
- horizon: beyond the march's early stop on the fit's grid (t/L >= tau_k): the grid, not the data, floored it;
- impossible: its refined mass is below 1e-12: the model at theta-hat leaves no room for it;
- grid: otherwise, if the refined term differs from the corrected one by more than 1 nat: the fit's time step
  is too coarse for this death and the fit's value for it is the grid's;
- cancellation: otherwise: v0.3.3's arithmetic misscored it, and the corrected value is resolved.
A candidate whose refinement did not converge within the budget is reported with converged = False.

The corrected-numerics gradient at theta-hat is kept in the report (not in the message). Nothing here changes
the fit, its claims or the other checks.
"""
from __future__ import annotations

import math
import time

import numpy as np

LIST_NATS = 1.0
IMPOSSIBLE_MASS = 1e-12
CANDIDATE_MASS = 1e-12
SCREEN_MASS = 1e-6             # jax2: deaths below this fit-grid mass are refined too (sr-naveh impl. review 11)
REFINE_MAX = 6
REFINE_BUDGET = 4e9            # node-steps over all refinements
CATEGORIES = ("horizon", "impossible", "grid", "cancellation")
_FLOOR = -700.0 + 1e-9

ADVICE = {
    "horizon": ("they lie beyond the point where the solver stopped (the survival had fallen below 1e-11): the "
                "grid, not the data, put them on the floor. Treat this fit as unreliable for them; a longer "
                "horizon is a grid-policy change (spec 003)."),
    "impossible": ("at theta-hat the model gives these deaths essentially zero probability; they are not ageing "
                   "deaths it describes. Check the rows (recording errors, a different population); if they are "
                   "real, refit without the deaths before a cut-off age you choose and state (fit-cohort cannot fit a "
                   "window until spec 006), and compare the fits with and without them."),
    "grid": ("the fit's time step is too coarse for these deaths: their term moves by more than 1 nat when the "
             "step is refined, so the fit's value for them is the grid's. Refit without the deaths before a cut-off "
             "age you choose and state (fit-cohort cannot fit a window until spec 006), and compare the fits with "
             "and without them."),
    "cancellation": ("v0.3.3's arithmetic scored these by round-off (register D-004), so the fit could neither use "
                     "nor ignore them properly; the corrected value is resolved. If they are early deaths the "
                     "model does not describe, refit without the deaths before a cut-off age you choose and state "
                     "(fit-cohort cannot fit a window until spec 006), and compare the fits with and without them."),
}


def _mass(cohort, aux, rows, step_phys):
    """Death masses of `rows`; for exact times, the density times the fit grid's step (a fixed reference)."""
    m = np.asarray(aux["mass"], float)[rows]
    return m * step_phys if cohort.continuous else m


def report(cohort, theta_hat, *, time_unit=None) -> dict:
    """The FloorReport (data-model.md) for one fit. `theta_hat` is the fit's SRParams (xc = 1, h_ext = 0)."""
    from sraging.cohort import Cohort
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.params import SRParams

    c0 = time.process_time()
    p = SRParams(eta=theta_hat.eta, beta=theta_hat.beta, epsilon=theta_hat.epsilon, kappa=theta_hat.kappa,
                 time_unit=cohort.time_unit if time_unit is None else time_unit)
    e33 = Evaluator(cohort, numerics="v033", grid=GridPolicy("v033"))
    e1 = Evaluator(cohort, numerics="v1", grid=GridPolicy("v033"))
    t33, _ = e33.terms(p)
    t1, a1 = e1.terms(p)
    _, grad = e1.value_and_grad(p)
    g = e1.grid
    step_phys = g.dt * g.L
    died = np.asarray(cohort.events) == 1
    m1 = np.zeros(cohort.n)
    m1[died] = _mass(cohort, a1, died, step_phys)
    cand = died & ((t33 <= _FLOOR) | (np.abs(t1 - t33) > LIST_NATS) | (m1 < CANDIDATE_MASS))
    rows = np.flatnonzero(cand)
    beyond = np.asarray(a1["beyond_march"], bool)
    inside = rows[~beyond[rows]]
    ref_t = {int(i): float(t1[i]) for i in rows}
    ref_m = {int(i): float(m1[i]) for i in rows}
    conv = {int(i): None for i in rows}
    steps = []
    if inside.size:
        ages = np.asarray(cohort.ages, float)[inside]
        d = cohort.delta
        dd = None if d is None else (float(d) if np.ndim(d) == 0 else np.asarray(d, float)[inside])
        sub = Cohort.from_arrays(ages, np.ones(inside.size, int), delta=dd, continuous=cohort.continuous,
                                 time_unit=cohort.time_unit)
        reach = (ages + (0.0 if dd is None else dd)).max() / g.L
        prev = np.array([ref_t[int(i)] for i in inside])
        budget = REFINE_BUDGET
        for k in range(1, REFINE_MAX + 1):
            dt = g.dt / 4 ** k
            tau_end = reach + 4 * dt
            cost = g.N * np.ceil(tau_end / dt)
            if cost > budget:
                break
            budget -= cost
            er = Evaluator(sub, numerics="v1", grid=GridPolicy("fixed", N=g.N, dt=dt, tau_max=tau_end, Gtol=g.Gtol,
                                                              L=g.L))
            tk, ak = er.terms(p)
            mk = _mass(cohort, ak, np.arange(inside.size), step_phys)
            moved = np.abs(tk - prev)
            steps.append({"dt": dt, "node_steps": float(cost), "max_moved": float(moved.max())})
            for j, i in enumerate(inside):
                ref_t[int(i)], ref_m[int(i)] = float(tk[j]), float(mk[j])
                conv[int(i)] = bool(moved[j] < LIST_NATS)
            prev = tk
            if np.all(moved < LIST_NATS):
                break
    deaths = []
    for i in rows:
        i = int(i)
        if beyond[i]:
            cat = "horizon"
        elif ref_m[i] < IMPOSSIBLE_MASS or (m1[i] < IMPOSSIBLE_MASS and abs(ref_t[i] - t1[i]) <= LIST_NATS):
            cat = "impossible"
        elif abs(ref_t[i] - t1[i]) > LIST_NATS:
            cat = "grid"
        else:
            cat = "cancellation"
        deaths.append({"row": i + 1, "age": float(cohort.ages[i]), "term_v033": float(t33[i]), "term_v1": float(t1[i]),
                       "term_refined": None if beyond[i] else ref_t[i], "mass_refined": None if beyond[i] else ref_m[i],
                       "converged": None if beyond[i] else conv[i], "category": cat})
    counts = {k: sum(x["category"] == k for x in deaths) for k in CATEGORIES}
    return {"deaths": deaths, "counts": counts, "grad_v1_at_theta_hat": grad, "grid": g.summary(),
            "refinement": {"N": g.N, "steps": steps, "refine_max": REFINE_MAX, "budget_node_steps": REFINE_BUDGET},
            "rules": {"list_nats": LIST_NATS, "candidate_mass": CANDIDATE_MASS, "impossible_mass": IMPOSSIBLE_MASS},
            "cost_s": time.process_time() - c0}


CATEGORIES_JAX2 = ("horizon", "impossible", "grid")      # spec 003 FR-011: no "cancellation" under fix B


def _graded_start(dt, early) -> float:
    """The length of jax2's graded early mesh at step `dt` (compat/jax2/fp_jax.py `early_schedule`: nstart steps of
    dt0, then dt0 * ratio^j while below dt); 0 without one. A refinement must march past it: `early_schedule` refuses a
    tau_max that leaves no uniform part, which a group of only the earliest deaths would ask for (measured
    2026-10-05: the ITP example's age-0 death alone, at the theta-hat fitted without it)."""
    if not early:
        return 0.0
    from sraging.compat.jax2 import estimator as je
    t = float(early.get("nstart", je._EARLY_NSTART)) * float(early["dt0"])    # make_loglik's default nstart, 20
    d = float(early["dt0"]) * float(early["ratio"])
    while d < dt:
        t += d
        d *= float(early["ratio"])
    return t + 2.0 * dt


def report_jax2(cohort, theta_hat, name: str, port=None) -> dict:
    """The FloorReport of a jax2 fit (spec 003 FR-011; research R12): terms from the port's own likelihood on the fit's
    own grid (compat.jax2 fp_jax make_loglik(with_terms=True)), refined by dt/4^k with N fixed as `report` does.
    Candidates: deaths whose mass is below CANDIDATE_MASS or whose term is on the -700 floor; and, refined as a
    second group with its own budget (so the first group's refinement is unchanged), deaths whose fit-grid mass is
    below SCREEN_MASS -- FR-011 is about the REFINED mass, and a coarse grid can smear mass above the cut (sr-naveh
    implementation review 11). Categories: horizon (no mass on any grid tried: the march stops first, or the mass
    underflows), impossible (refined mass below IMPOSSIBLE_MASS), grid (the refined term moved by more than
    LIST_NATS); a screened death in none of them is not listed. Each listed death gets its share of the
    log-likelihood: under fix B such deaths have finite terms that pull the fit (D-105). The share is not leverage:
    how far one individual moves theta-hat is the leverage check's (spec 003f, `sraging.leverage`). `port`: the fit's
    jax2 port (spec 003b; default 1.0.0's), whose likelihood and grid are used."""
    from sraging.calibrators import profile_jax2 as PJ
    port = PJ.V100 if port is None else port
    fj = port.fp_jax

    c0 = time.process_time()
    inp = PJ.fit_inputs(cohort, name, port)
    t, e, L, kw = inp["t"], inp["e"], inp["L"], inp["kw"]
    z = [math.log(theta_hat.eta), math.log(theta_hat.beta), math.log(theta_hat.epsilon), math.log(theta_hat.kappa)]
    if inp["hazard"]:
        z.append(math.log(theta_hat.h_ext))
    z = np.asarray(z, float)

    def terms_on(tt, ee, dt, tau_max):
        _f, _vg, tf = fj.make_loglik(tt, ee, L, N=kw["N"], dt=dt, tau_max=tau_max, Gtol=kw["Gtol"],
                                     sampling_dt=kw.get("sampling_dt"), hazard=inp["hazard"], window=inp["window"],
                                     with_terms=True, exact_from_F=bool(kw.get("exact_from_F")), early=kw.get("early"))
        d = tf(z)
        return (np.asarray(d["term"], float), np.asarray(d["core"], float), float(np.asarray(d["cond"])),
                float(np.asarray(d["S_lo"])))

    step_phys = kw["dt"] * L
    term, core, cond, s_lo = terms_on(t, e, kw["dt"], kw["tau_max"])
    mass = core * step_phys if cohort.continuous else core
    died = e == 1
    total = float(np.sum(term) + cond)
    cand = died & ((term <= _FLOOR) | (mass < CANDIDATE_MASS))
    screen = died & ~cand & (mass < SCREEN_MASS)
    rows_c, rows_s = np.flatnonzero(cand), np.flatnonzero(screen)
    steps = []

    def refine(rows, group):
        ref_t = {int(i): float(term[i]) for i in rows}
        ref_m = {int(i): float(mass[i]) for i in rows}
        zero = {int(i): mass[i] == 0.0 for i in rows}
        conv = {int(i): None for i in rows}
        if rows.size:
            tt, ee = t[rows], np.ones(rows.size, int)
            reach = (tt.max() + (inp["delta"] or 0.0)) / L
            prev, budget = term[rows].copy(), REFINE_BUDGET
            for k in range(1, REFINE_MAX + 1):
                dt = kw["dt"] / 4 ** k
                tau_end = max(reach + 4 * dt, _graded_start(dt, kw.get("early")) + 4 * dt)
                cost = kw["N"] * np.ceil(tau_end / dt)
                if cost > budget:
                    break
                budget -= cost
                tk, ck, _, _ = terms_on(tt, ee, dt, tau_end)
                mk = ck * step_phys if cohort.continuous else ck
                moved = np.abs(tk - prev)
                steps.append({"dt": dt, "node_steps": float(cost), "max_moved": float(moved.max()), "group": group})
                for j, i in enumerate(rows):
                    ref_t[int(i)], ref_m[int(i)] = float(tk[j]), float(mk[j])
                    zero[int(i)] = zero[int(i)] and mk[j] == 0.0
                    conv[int(i)] = bool(moved[j] < LIST_NATS)
                prev = tk
                if np.all(moved < LIST_NATS):
                    break
        return ref_t, ref_m, zero, conv

    deaths = []
    for rows, group in ((rows_c, "candidates"), (rows_s, "screened")):
        ref_t, ref_m, zero, conv = refine(rows, group)
        for i in rows:
            i = int(i)
            if zero[i]:
                cat = "horizon"
            elif ref_m[i] < IMPOSSIBLE_MASS or (mass[i] < IMPOSSIBLE_MASS and abs(ref_t[i] - term[i]) <= LIST_NATS):
                cat = "impossible"
            elif group == "candidates" or abs(ref_t[i] - term[i]) > LIST_NATS:
                cat = "grid"
            else:
                continue                                   # screened, and its refined mass is not in question
            deaths.append({"row": i + 1, "age": float(cohort.ages[i]), "term_fit": float(term[i]),
                           "term_refined": ref_t[i], "mass_refined": ref_m[i], "converged": conv[i], "category": cat,
                           "ll_share": float(term[i] / total) if total else None, "screened": group == "screened"})
    deaths.sort(key=lambda x: x["row"])
    counts = {k: sum(x["category"] == k for x in deaths) for k in CATEGORIES_JAX2}
    return {"calibrator": name, "deaths": deaths, "counts": counts, "loglik_terms_total": total,
            "grid": {"N": kw["N"], "dt": kw["dt"], "tau_max": kw["tau_max"], "early": kw.get("early"), "L": L},
            "refinement": {"N": kw["N"], "steps": steps, "refine_max": REFINE_MAX, "budget_node_steps": REFINE_BUDGET},
            "rules": {"list_nats": LIST_NATS, "candidate_mass": CANDIDATE_MASS, "impossible_mass": IMPOSSIBLE_MASS,
                      "screen_mass": SCREEN_MASS}, "n_screened": int(rows_s.size),
            "cost_s": time.process_time() - c0,
            **({"S_lo": s_lo} if inp["window"] is not None else {})}     # spec 006 R7: S(lo) at theta-hat, windows only


def message(rep: dict) -> str:
    """The FLOORED_DEATHS warning text: per death its age and three values, the category, and what to do."""
    d = rep["deaths"]
    if "calibrator" in rep:                    # spec 003: a jax2 fit's report (its own likelihood; no v0.3.3 value)
        parts = []
        for cat in CATEGORIES_JAX2:
            xs = [x for x in d if x["category"] == cat]
            if not xs:
                continue
            shown = "; ".join(f"age {x['age']:g}: {x['term_fit']:.1f} on the fit's grid, {x['term_refined']:.1f} refined"
                              f"{'' if x['converged'] in (None, True) else ' (not converged)'}, "
                              f"{x['ll_share']:.2%} of the log-likelihood" for x in xs[:8]) + \
                (f"; ... ({len(xs)} in all)" if len(xs) > 8 else "")
            parts.append(f"{len(xs)} {cat} ({shown}): {ADVICE[cat]}")
        return (f"{len(d)} death(s) carry essentially no probability under the fit's own likelihood, or are scored by "
                f"its time grid: each shows its term on the fit's grid, on a refined grid, and its share of the "
                f"log-likelihood (register D-201, D-105). " + " ".join(parts) +
                " The share is not leverage: how far one individual moves theta-hat is the leverage check's "
                "(spec 003f; its own line and warning).")
    parts = []
    for cat in CATEGORIES:
        xs = [x for x in d if x["category"] == cat]
        if not xs:
            continue

        def one(x):
            ref = "" if x["term_refined"] is None else (f", {x['term_refined']:.1f} refined"
                                                        + ("" if x["converged"] else " (not converged)"))
            return f"age {x['age']:g}: {x['term_v033']:.1f} in this fit, {x['term_v1']:.1f} corrected{ref}"
        shown = "; ".join(one(x) for x in xs[:8]) + (f"; ... ({len(xs)} in all)" if len(xs) > 8 else "")
        parts.append(f"{len(xs)} {cat} ({shown}): {ADVICE[cat]}")
    return (f"{len(d)} death(s) are scored by the arithmetic or the time grid rather than by the model: each "
            f"shows its log-likelihood term with v0.3.3's arithmetic (this fit), with the corrected arithmetic, "
            f"and on a refined time grid (register D-004, D-105). " + " ".join(parts))
