"""Spec 002 `speed` phase (SC-009; research R16; sr-manager gate M8). See tests/acceptance/panel_002_speed.toml.

Process CPU time per evaluation on one core (PJRT_NPROC=1, set by importing sraging.compat.v033), compilation
excluded (one warm-up call per program). For each cohort, `replicates` rounds of `evaluations_per_replicate`
calls, alternating the arms within a round to cancel drift; a cohort's ratio is the ratio of the arms' median
per-call times; the panel ratio is the geometric mean over cohorts, with a 95% bootstrap CI that resamples the
cohorts and, within each, the rounds. Per-step costs (us per march step actually taken) are reported against
PHASE_PLAN's cost model.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

import acceptance_002 as A
import acceptance_002_t1 as T

PANEL_SPEED = "tests/acceptance/panel_002_speed.toml"


def _time(fn, n):
    c0 = time.process_time()
    for _ in range(n):
        fn()
    return (time.process_time() - c0) / n


def _block(x):
    """Wait for a JAX result (process_time counts this thread's work, including the dispatch)."""
    import jax
    return jax.block_until_ready(x)


def _boot(per_cohort, B, rng):
    """per_cohort: list of (arm_a rounds, arm_b rounds) arrays of per-call times. Returns (estimate, lo, hi)."""
    def gm(pc):
        return math.exp(np.mean([math.log(np.median(a) / np.median(b)) for a, b in pc]))
    est = gm(per_cohort)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, len(per_cohort), len(per_cohort))
        pc = []
        for i in idx:
            a, b = per_cohort[i]
            j = rng.integers(0, len(a), len(a))
            pc.append((a[j], b[j]))
        stats.append(gm(pc))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return est, float(lo), float(hi)


def phase_speed(args):
    import jax.numpy as jnp
    from engine_cases import arrays, cohort, panel_ids, theta_points
    from sraging import SRParams
    from sraging.compat.v033 import estimator as port_est
    from sraging.compat.v033 import fp_jax as V
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    sp = A.load_toml(PANEL_SPEED)
    load0 = os.getloadavg()
    run = A.Run("speed", [PANEL_SPEED, A.PANEL_V1], args.dev,
                f"~60 CPU-min (40 cohorts x 3 arms x {sp['replicates']} x {sp['evaluations_per_replicate']} "
                f"value and value-and-gradient calls; {len(sp['feature'])} feature configurations against NumPy); "
                f"load average now {load0[0]:.2f}")
    if load0[0] > sp["max_load"] and not args.dev:
        sys.exit(f"[speed] REFUSED: the 1-minute load average is {load0[0]:.2f} > {sp['max_load']} (run it on an idle "
                 f"machine, after every other phase)")
    A.preflight_or_exit(run)
    R, E = sp["replicates"], sp["evaluations_per_replicate"]
    rng = np.random.default_rng(20261004)
    rows = []
    ids = panel_ids() if not args.only else [c for c in panel_ids() if any(t in c for t in args.only)]
    for cid in ids:
        t, e, d = arrays(cid)
        L, kw, _ = port_est.solver_grid(t, e, d)
        vf, vg = V.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                               sampling_dt=kw.get("sampling_dt"))
        e33 = Evaluator(cohort(cid), numerics="v033", grid=GridPolicy("v033"))
        e1 = Evaluator(cohort(cid), numerics="v1", grid=GridPolicy("v033"))
        lab, th = next((lab, th) for lab, th in theta_points(cid) if lab == sp["theta"])
        z = jnp.asarray(np.log(np.array([th[k] for k in A.PARAMS], float)))
        arms = {"V": (vf, vg), "v033": e33.objective(), "v1": e1.objective()}
        for f, g in arms.values():                       # compile and warm up
            _block(f(z))
            _block(g(z))
        k = int(e1.survival_on_grid(SRParams(**th))[3])
        tv = {a: [] for a in arms}
        tg = {a: [] for a in arms}
        for _ in range(R):
            for a, (f, g) in arms.items():
                tv[a].append(_time(lambda: _block(f(z)), E))
                tg[a].append(_time(lambda: _block(g(z)), E))
        row = {"cohort": cid, "steps": k, "N": kw["N"],
               "value_s": {a: float(np.median(v)) for a, v in tv.items()},
               "value_and_grad_s": {a: float(np.median(v)) for a, v in tg.items()},
               "rounds_value": {a: v for a, v in tv.items()}, "rounds_vg": {a: v for a, v in tg.items()}}
        row["us_per_step"] = {"value": {a: 1e6 * row["value_s"][a] / max(k, 1) for a in arms},
                              "value_and_grad": {a: 1e6 * row["value_and_grad_s"][a] / max(k, 1) for a in arms}}
        rows.append(row)
        print(f"  {cid:44s} k {k:5d}  value us/step V {row['us_per_step']['value']['V']:6.2f} v033 "
              f"{row['us_per_step']['value']['v033']:6.2f} v1 {row['us_per_step']['value']['v1']:6.2f}  vg us/step V "
              f"{row['us_per_step']['value_and_grad']['V']:6.2f} v1 {row['us_per_step']['value_and_grad']['v1']:6.2f}",
              flush=True)
    ratios = {}
    for name, a, b, lim in (("v033/V", "v033", "V", 1.03), ("v1/V", "v1", "V", 1.20)):
        for what, key in (("value", "rounds_value"), ("value_and_grad", "rounds_vg")):
            pc = [(np.asarray(r[key][a]), np.asarray(r[key][b])) for r in rows]
            est, lo, hi = _boot(pc, sp["bootstrap"], rng)
            ratios[f"{name} {what}"] = {"estimate": est, "ci95": [lo, hi], "limit": lim, "ok": hi <= lim}
            print(f"  ratio {name:7s} {what:15s} {est:.4f} [{lo:.4f}, {hi:.4f}] limit {lim} "
                  f"{'ok' if hi <= lim else 'FAIL'}", flush=True)
    # features against SRK's NumPy code
    import oracle_fp_piecewise as OP
    t1p = A.load_toml(T.PANEL_T1)
    feats = []
    with OP.session() as srk:
        for f in sp["feature"]:
            feats.append(_feature_speed(f, t1p, srk, R, max(3, E // 3), rng, sp["bootstrap"]))
            x = feats[-1]
            print(f"  feature {f['name']:32s} engine/NumPy {x['estimate']:.4f} [{x['ci95'][0]:.4f}, {x['ci95'][1]:.4f}] "
                  f"{'ok' if x['ok'] else 'FAIL'}", flush=True)
    load1 = os.getloadavg()
    cm = sp["cost_model"]
    per_step = {"value_v1_median_us": float(np.median([r["us_per_step"]["value"]["v1"] for r in rows])),
                "value_and_grad_v1_median_us": float(np.median([r["us_per_step"]["value_and_grad"]["v1"] for r in rows])),
                "value_V_median_us": float(np.median([r["us_per_step"]["value"]["V"] for r in rows])),
                "value_and_grad_V_median_us": float(np.median([r["us_per_step"]["value_and_grad"]["V"] for r in rows])),
                "cost_model": cm}
    ok = all(r["ok"] for r in ratios.values()) and all(x["ok"] for x in feats)
    status = ("smoke-" if args.only else "") + ("pass" if ok else "fail")
    return run.write({"status": status, "load_average": {"start": load0, "end": load1}, "ratios": ratios,
                      "features": feats, "per_step": per_step, "results": rows})


def _feature_speed(f, t1p, srk, R, E, rng, B):
    import jax
    from sraging import SRParams
    from sraging.engine.evaluator import Evaluator
    case = f["case"]
    if case.startswith("segments:"):
        unit_kind = "segments"
        sp = t1p["segments"]
        cfg = next(x for x in sp["configs"] if f"segments:{x['name']}" == case)
        coh, kw, th, h = T.guard_feature({"kind": "segments", "config": cfg["name"]}, t1p)
        g = sp["grid"]
        t, e = T.seg_cohort()
        win = list(cfg["window"]) if cfg.get("window") else None
        iv = ([{"kind": "ExtHazard", "level": cfg["level"], "window": win}] if cfg["parameter"] == "h_ext" else
              [{"kind": "Intervention", "param": cfg["parameter"], "effect": cfg["effect"], "window": win}])
        call = dict(op="piecewise_loglik", death_times=t, events=e, theta=dict(**th, xc=1.0), interventions=iv,
                    L=g["L"], t_end=float(t.max()), external_hazard=sp["external_hazard_level"], sampling_dt=1.0,
                    N=g["N"], dt=g["dt"], tau_max=g["tau_max"], Gtol=1e-11, scheme="bdf2")
    else:
        unit_kind = "library"
        cs = T.case_setup(case, "library", "bdf2", t1p)
        coh, kw, th, h = cs["coh"], cs["kw"], cs["truth"], cs["h"]
        g0 = cs["g0"]
        gk = dict(N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, scheme="bdf2")
        if cs["het"] is None:
            call = dict(op="fp_loglik_physical", death_times=cs["t"], events=cs["e"], xc=1.0, L=g0.L,
                        external_hazard=math.inf if cs["ext"] is None else cs["ext"],
                        time_range=list(cs["win"]) if cs["win"] else None, sampling_dt=cs["d"], **th, **gk)
        else:
            call = dict(op="fp_loglik_het_physical", death_times=cs["t"], events=cs["e"], xc=1.0, L=g0.L,
                        het_param=cs["het"][0], het_var=cs["het"][1], M=cs["het"][2], time_range=None,
                        sampling_dt=cs["d"], **th, **gk)
    ev = Evaluator(coh, numerics="v1", **kw)
    p = SRParams(**th, h_ext=h)
    zf = ev._z(p)
    fval = ev.objective()[0]
    jax.block_until_ready(fval(zf))
    op = call.pop("op")
    srk.call(op, **call)                               # warm the worker
    a_r, b_r = [], []
    for _ in range(R):
        a_r.append(_time(lambda: jax.block_until_ready(fval(zf)), E))
        b_r.append(float(np.mean([float(srk.call(op, **call)["cpu_s"]) for _ in range(E)])))
    est, lo, hi = _boot([(np.asarray(a_r), np.asarray(b_r))], B, rng)
    return {"name": f["name"], "case": case, "kind": unit_kind, "engine_s": float(np.median(a_r)),
            "numpy_s": float(np.median(b_r)), "estimate": est, "ci95": [lo, hi], "limit": 1.0, "ok": hi <= 1.0,
            "rounds_engine": a_r, "rounds_numpy": b_r}
