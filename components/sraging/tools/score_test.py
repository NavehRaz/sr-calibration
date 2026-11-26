"""Simulator-anchored score test (spec 002 SC-006; research R13; review T064: sr-manager M5 and b, sr-naveh N10).

For each feature, at a true theta0: simulate R cohorts of n = 2,000 with the frozen SR simulator (the
fp_piecewise snapshot's SR_hetro.death_times_euler_brownian_bridge, Euler with a Brownian-bridge crossing test,
through the oracle worker, numba's generator seeded per cohort), apply the observation transform written here
(recording to delta; a common window; left truncation at per-row entry ages), and evaluate the engine's score
(the gradient of the log-likelihood in log parameters) at theta0 per cohort. A likelihood that is right for the
data has mean score zero; the test is mean / SE per component against Bonferroni bounds at family-wise
alpha = 0.01 over every (feature, component) pair.

- "pure" is the control: the simulator and the Fokker-Planck model are not identical (the simulator clamps x at
  0 where the model reflects), and a biased pure score would make every feature's test uninformative.
- Power: each feature with an alternative also scores that alternative on the same cohorts, and the alternative
  must be detected; otherwise the feature is reported "not informative at this R", never as a pass:
  window -> numerics "v033" (SRK's window close, register D-107); entry -> conditioning on S_SR(e) instead of
  S_obs(e) (with a hazard); heterogeneity (with a window) -> conditioning node by node instead of on the
  mixture.
- R comes from a 5-cohort pilot per feature: the R at which the alternative's measured mean shift is detected
  with 95% power, within [R_MIN, R_MAX]; what R = 40 would have detected is reported too.
- The engine's grid is certified by halving its step and by doubling N on the same cohorts (the mean score moves
  < 0.5 SE each); the simulator's by halving its step on 20 cohorts (the difference is reported as a z).
All cohorts of one feature are concatenated into ONE evaluator (every term is per row, including the
conditioning), and a cohort's score is the weighted gradient with weight 1 on its rows: one compilation.
"""
from __future__ import annotations

import math
import os
import sys
import time
import zlib

import numpy as np

import acceptance_002 as A

THETA0_COHORT = "pure-k0.01-p00-c0-s0"          # theta0 = its library truth (L ~ 100 days)
N_PEOPLE = 2000
SIM_DT, SIM_STEPS = 0.0125 / 8, 24000 * 8         # 300 days; measured, not assumed: at dt = 0.0125 the pure control's
# mean score was biased by ~5 SE at R = 200, shrinking ~0.39x per halving of the simulator's step (evidence:
# specs/002-fp-engine/review/evidence/score_sim_step.json); at dt/8 the bias extrapolates to ~0.3 SE
DELTA = 0.5                                       # recording resolution (days)
GRID = dict(N=7200, tau_max=3.0, L=100.0)         # engine: dtau = delta / (5 L); N measured, not assumed:
# at N = 1800 the pure control's mean score moved by ~1 SE when N was doubled (2026-10-04), so N = 7200 is used
# and certified by doubling N as well as by halving the step
ALPHA = 0.01
R_MIN, R_MAX, R_DEFAULT = 40, 400, 200
FEATURES = {
    "pure": {},
    "hazard": {"ext": 5.5},
    "window": {"window": (80.0, 100.0), "plant": "window_close_v033"},
    # t_hi at the median lifetime (amended 2026-10-04 after the run at 42282a1): with t_hi = 120 (lifetimes
    # ~100 +- 4) no death was recorded in the bin at t_hi, so v1's window close (D-107) and SRK's never differed
    # and the alternative could not be detected (not informative at R = 400, its z identical to the main test's)
    "entry": {"ext": 5.5, "entry_max": 80.0, "plant": "condition_on_S_SR"},
    "heterogeneity": {"het": ("xc", 0.2, 41), "window": (60.0, 120.0), "plant": "node_by_node_conditioning",
                      "N": 3600, "R": 200, "cert_cohorts": 20},
    # Naveh, 2026-10-04: the smaller run (M = 41 costs 24.5 CPU-s per cohort score at N = 3600, measured): N = 3600
    # certified by doubling, R = 200, the engine-grid certification on the first 20 cohorts (a grid change moves
    # each cohort's score almost deterministically) -- ~4 CPU-h instead of ~10
    # M measured, not assumed: at M = 11 the mixture's log-likelihood was 3-7 nats off and its epsilon score off by
    # 16-26 per 2,000-person cohort on this tight theta0 (cv ~4%); M = 41 and 81 agree to ~0.01
    # (specs/002-fp-engine/review/evidence/mixture_M.json)
}


def theta0():
    tr = dict(A.O.library().truth(THETA0_COHORT))
    return {k: tr[k] for k in A.PARAMS}


def simulate(srk, feat, r, *, dt=SIM_DT, steps=SIM_STEPS, seed_base=0):
    """One cohort: (ages, events, entry or None) after the observation transform."""
    th = theta0()
    f = FEATURES[feat]
    kw = dict(steps=steps, dt=dt, npeople=N_PEOPLE, seed=1_000_003 * (seed_base + 1) + 7919 * r + zlib.crc32(feat.encode()) % 997,
              eta=th["eta"], beta=th["beta"], kappa=th["kappa"], epsilon=th["epsilon"], xc=1.0)
    if "ext" in f:
        kw["external_hazard"] = f["ext"]
    if "het" in f:
        kw[f["het"][0] + "_var"] = f["het"][1]
    out = srk.call("simulate_sr_hetro", **kw)
    T, E = np.asarray(out["death_times"], float), np.asarray(out["events"], int)
    rec = np.floor(T / DELTA + 1e-9) * DELTA                   # [k delta, (k+1) delta) contains the crossing step
    en = None
    if "entry_max" in f:
        rng = np.random.default_rng(424242 + r)
        e = rng.uniform(0.0, f["entry_max"], T.size)
        keep = T >= e                                          # left truncation: seen only if alive at entry
        rec, E, en = rec[keep], E[keep], e[keep]
    return rec, E, en


def build(feat, cohorts, numerics="v1", dt=None, N=None):
    """One evaluator over the concatenated cohorts; returns (evaluator, row -> cohort index)."""
    from sraging import Cohort
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.engine.inputs import Heterogeneity
    f = FEATURES[feat]
    t = np.concatenate([c[0] for c in cohorts])
    e = np.concatenate([c[1] for c in cohorts])
    idx = np.concatenate([np.full(c[0].size, i) for i, c in enumerate(cohorts)])
    en = None if cohorts[0][2] is None else np.concatenate([c[2] for c in cohorts])
    coh = Cohort.from_arrays(t, e, delta=DELTA, window=f.get("window"), entry=en)
    dtau = (dt or DELTA / (5 * GRID["L"]))
    grid = GridPolicy("fixed", N=N or f.get("N", GRID["N"]), dt=dtau, tau_max=GRID["tau_max"], L=GRID["L"])
    free = A.PARAMS + (("h_ext",) if "ext" in f else ())
    fixed = {} if "ext" in f else {"h_ext": 0.0}
    het = (Heterogeneity(parameter=f["het"][0], definition="normal_frac_sd", v=f["het"][1], M=f["het"][2])
           if "het" in f else None)
    ev = Evaluator(coh, numerics=numerics, grid=grid, free=free, fixed=fixed, heterogeneity=het)
    return ev, idx


def params(feat):
    from sraging import SRParams
    f = FEATURES[feat]
    return SRParams(**theta0(), h_ext=math.exp(-f["ext"]) if "ext" in f else 0.0)


def scores(ev, idx, p, R):
    """(R x components) per-cohort scores in log parameters, in the layout's order."""
    G = np.zeros((R, len(ev.layout.names)))
    for r in range(R):
        g = ev.weighted_grad(p, (idx == r).astype(float))
        G[r] = [g[n] for n in ev.layout.names]
    return G


def plant_scores(feat, cohorts, ev, idx, p, G):
    """The alternative's per-cohort scores on the same cohorts."""
    f = FEATURES[feat]
    R = len(cohorts)
    if f["plant"] == "window_close_v033":
        ev33, _ = build(feat, cohorts, numerics="v033")
        return scores(ev33, idx, p, R)
    if f["plant"] == "condition_on_S_SR":
        # log S_obs(e) = log S_SR(e) - h e: conditioning on S_SR adds -h e to each row; its derivative in log h is
        # -h e, so only the h_ext component moves
        h = p.h_ext
        en = np.concatenate([c[2] for c in cohorts])
        Gp = G.copy()
        j = ev.layout.names.index("h_ext")
        for r in range(R):
            Gp[r, j] -= h * float(np.sum(en[idx == r]))
        return Gp
    if f["plant"] == "node_by_node_conditioning":
        return _node_by_node(feat, cohorts, ev, idx, p)
    raise ValueError(f["plant"])


def _node_by_node(feat, cohorts, ev, idx, p):
    """term_i = log sum_k w_k exp(t_k,i), t_k,i node k's own window-conditioned term: each node conditioned on its
    own S(t_lo) instead of the mixture on the mixture's. One evaluator, its mixture assembly replaced (vmapped over
    the nodes, differentiated by JAX); scores as weighted gradients, as for the engine."""
    import jax
    import jax.numpy as jnp
    ep, _ = build(feat, cohorts)
    asm, par = ep._asm, ep.het.parameter

    def node_terms_of(march, vals, diagnostics=False):
        sc, wt = ep._node_scales(vals)

        def one(si):
            vk = dict(vals)
            vk[par] = (vals.get(par, 1.0) if par == "xc" else vals[par]) * si
            t, _ = asm.assemble(asm.points(march(asm.theta(vk)), diagnostics), vk)
            return t
        T = jax.vmap(one)(sc)                                       # (M, n)
        lw = jnp.where(wt > 0, jnp.log(jnp.where(wt > 0, wt, 1.0)), -jnp.inf)
        return jax.scipy.special.logsumexp(T + lw[:, None], axis=0), {}
    ep._terms_of = node_terms_of                                    # before the first (lazy) trace
    return scores(ep, idx, p, len(cohorts))


def _grid_policy(ev):
    from sraging.engine.grid import GridPolicy
    g = ev.grid
    return GridPolicy("fixed", N=g.N, dt=g.dt, tau_max=g.tau_max, L=g.L)


def zstats(G):
    R = G.shape[0]
    mean, sd = G.mean(axis=0), G.std(axis=0, ddof=1)
    return mean, sd / math.sqrt(R), mean / (sd / math.sqrt(R))


def run_feature(feat, zcrit, smoke):
    """One feature, in its own process with its own oracle worker."""
    import oracle_fp_piecewise as OP
    A.O.load()

    class _Args:
        only = smoke
    args = _Args()
    with OP.session() as srk:
        f = FEATURES[feat]
        p = params(feat)
        c0 = time.process_time()
        # pilot: 5 cohorts -> the per-cohort score SD and the alternative's mean shift -> R
        pilot = [simulate(srk, feat, r, seed_base=1) for r in range(5)]
        ev, idx = build(feat, pilot)
        Gp = scores(ev, idx, p, 5)
        R = f.get("R", R_DEFAULT)
        pilot_rec = {"sd": Gp.std(axis=0, ddof=1).tolist()}
        if "plant" in f:
            Ga = plant_scores(feat, pilot, ev, idx, p, Gp)
            shift = (Ga - Gp).mean(axis=0)
            sd = np.maximum(Ga.std(axis=0, ddof=1), 1e-300)
            need = ((zcrit + 1.645) * sd / np.maximum(np.abs(shift), 1e-300)) ** 2
            R = f.get("R") or int(np.clip(np.ceil(np.min(need)), R_MIN, R_MAX))
            pilot_rec.update(plant_shift=shift.tolist(), plant_sd=sd.tolist(), R_needed=need.tolist())
        print(f"  [{feat}] pilot: R = {R}  ({pilot_rec})", flush=True)
        if args.only:                                      # a smoke run: small counts, never the acceptance run
            R = min(R, 10)
        cohorts = [simulate(srk, feat, r) for r in range(R)]
        ev, idx = build(feat, cohorts)
        G = scores(ev, idx, p, R)
        mean, se, zz = zstats(G)
        rec = {"R": R, "n": N_PEOPLE, "components": list(ev.layout.names), "mean": mean.tolist(), "se": se.tolist(),
               "z": zz.tolist(), "pass": bool(np.all(np.abs(zz) <= zcrit)), "pilot": pilot_rec}
        # the engine's grid: halve its step, and double N, on the same cohorts (all R, or the first cert_cohorts)
        nc = min(f.get("cert_cohorts", R), R)
        sub = cohorts[:nc]
        e1c, ic = (ev, idx) if nc == R else build(feat, sub)
        base_c = G[:nc].mean(axis=0)
        ev2, _ = build(feat, sub, dt=DELTA / (10 * GRID["L"]))
        move = np.abs(scores(ev2, ic, p, nc).mean(axis=0) - base_c) / se
        ev3, _ = build(feat, sub, N=2 * f.get("N", GRID["N"]))
        move_N = np.abs(scores(ev3, ic, p, nc).mean(axis=0) - base_c) / se
        rec["engine_grid_cert_cohorts"] = nc
        rec["engine_grid_move_in_se"] = {"step_halved": move.tolist(), "N_doubled": move_N.tolist()}
        rec["engine_grid_certified"] = bool(np.all(move < 0.5) and np.all(move_N < 0.5))
        if "plant" in f:
            Ga = plant_scores(feat, cohorts, ev, idx, p, G)
            _, sea, za = zstats(Ga)
            det = bool(np.any(np.abs(za) > zcrit))
            za40 = zstats(Ga[:40])[2] if R >= 40 else za
            rec["plant"] = {"name": f["plant"], "z": za.tolist(), "detected": det,
                            "detected_at_R40": bool(np.any(np.abs(za40) > zcrit))}
            rec["verdict"] = ("pass" if rec["pass"] and det else
                              "fail" if not rec["pass"] else "not informative at this R")
        else:
            rec["verdict"] = "pass" if rec["pass"] else "fail"
        # the simulator's step: halve it on 20 cohorts
        nh = 5 if args.only else 20
        half = [simulate(srk, feat, r, dt=SIM_DT / 2, steps=2 * SIM_STEPS, seed_base=2) for r in range(nh)]
        full = [simulate(srk, feat, r, seed_base=2) for r in range(nh)]
        eh, ih = build(feat, half)
        ef, iff = build(feat, full)
        Gh, Gf = scores(eh, ih, p, nh), scores(ef, iff, p, nh)
        d = Gh.mean(axis=0) - Gf.mean(axis=0)
        sed = np.sqrt(Gh.var(axis=0, ddof=1) / nh + Gf.var(axis=0, ddof=1) / nh)
        rec["simulator_halving"] = {"diff": d.tolist(), "z": (d / sed).tolist(), "diff_in_main_se": (d / se).tolist()}
        rec["cpu_s"] = time.process_time() - c0
        print(f"  [{feat}] R {R}: z {np.round(zz, 2).tolist()} -> {rec['verdict']}; engine-grid move "
              f"{np.round(move, 2).tolist()} / N {np.round(move_N, 2).tolist()} SE; simulator halving z "
              f"{np.round(d / sed, 2).tolist()}"
              + (f"; plant z {np.round(rec['plant']['z'], 2).tolist()}" if "plant" in rec else ""), flush=True)

    return rec


def phase_score(args):
    from statistics import NormalDist
    import oracle_fp_piecewise as OP
    sel = args.only or getattr(args, "features", None)
    feats = list(FEATURES) if not sel else [f for f in FEATURES if f in sel]
    n_pairs = sum(len(A.PARAMS) + (1 if "ext" in FEATURES[f] else 0) for f in FEATURES)
    zcrit = NormalDist().inv_cdf(1 - ALPHA / (2 * n_pairs))
    run = A.Run("score", [], args.dev, f"<= 2 CPU-h ({len(feats)} features; a 5-cohort pilot each, then R cohorts of "
                f"{N_PEOPLE}; Bonferroni |z| <= {zcrit:.3f} over {n_pairs} pairs at family-wise alpha {ALPHA})")
    A.preflight_or_exit(run)
    out = {"theta0": theta0(), "theta0_cohort": THETA0_COHORT, "zcrit": zcrit, "n_pairs": n_pairs, "features": {}}
    import concurrent.futures as cf
    import multiprocessing as mp
    import acceptance_002_t1 as T1
    with cf.ProcessPoolExecutor(len(feats), mp_context=mp.get_context("spawn"), initializer=T1._worker_init) as ex:
        futs = {ex.submit(run_feature, feat, zcrit, bool(args.only)): feat for feat in feats}
        for f_ in cf.as_completed(futs):
            feat = futs[f_]
            try:
                out["features"][feat] = f_.result()
            except Exception as exn:                       # noqa: BLE001 -- recorded, never a pass
                out["features"][feat] = {"verdict": "error", "error": repr(exn), "engine_grid_certified": False}
            print(f"  [{feat}] done: {out['features'][feat].get('verdict')}", flush=True)
    ok = all(v["verdict"] == "pass" and v["engine_grid_certified"] for v in out["features"].values())
    missing = [f for f in FEATURES if f not in out["features"]]
    out["features_not_run"] = missing                   # a subset is never the full SC-006 pass
    status = ("smoke-" if args.only else "") + ("pass" if ok and not missing else "partial-pass" if ok else "fail")
    return run.write({"status": status, **out})
