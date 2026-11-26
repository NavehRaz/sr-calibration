"""Spec 002 `mpref` phase (SC-007; research R11, R12; reviews T064 sr-naveh N5 and sr-manager M9): numerics "v1"
against the discrete scheme in mpmath (tests/reference/mp_scheme.py, 60 digits), term by term.

Two parts:
1. The low-mass terms `t1` routed (its committed result names them per evaluation) where v1 takes the F route
   (use_F: the head, where the round-off lives; the cheaper option the sr-manager gate stated in advance, M9,
   taken because the full estimate was 2.27 CPU-h): the engine's v1 terms of those rows against mpmath's terms
   by definition, on the same grid; per evaluation sum |dl| <= 1e-5 nats. The mpmath march stops just past the
   last row it needs. Routed rows outside use_F are v033's own expression in both numerics. Segment configurations are NOT APPLICABLE (the mpmath
   scheme has no segments; their T1 reference is fp_piecewise, and D-113's convergence is measured there).
2. The round-off panel (tests/acceptance/panel_002_roundoff.toml), every term: the v1 total within 1e-5 nats of
   mpmath's per evaluation; the +-3-ulp scan of theta (theta x (1 + k 2^-52), k = -3..3) moves the v1 total by
   at most 1e-5 nats (v033's spread reported).
The cost is printed before anything runs and capped at MPREF_CAP_CPU_H (sr-manager M9: above it, stop and ask).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

import acceptance_002 as A
import acceptance_002_t1 as T

PANEL_ROUNDOFF = "tests/acceptance/panel_002_roundoff.toml"
MPREF_CAP_CPU_H = 2.0
PER_EVAL = 1e-5
SPREAD = 1e-5
DPS = 60


def _rho(th, L):
    return (L ** 2 * th["eta"], L * th["beta"], L * th["epsilon"], th["kappa"])


def _node_marches(MP, th, L, het, g, scheme, tau_stop):
    """[(weight, MPMarch)] for the cohort's model: one march, or the heterogeneity mixture's nodes (dropped
    nodes as the engine drops them, register D-116)."""
    kw = dict(scheme=scheme, nstart=20 if scheme == "bdf2" else 10 ** 9, Gtol=g["Gtol"], dps=DPS, announce=False)
    if het is None:
        return [(1.0, MP.march(_rho(th, L), g["N"], g["dt"], tau_stop, **kw))]
    from numpy.polynomial.hermite_e import hermegauss
    par, v, M = het
    z, w = hermegauss(M)
    w = w / np.sqrt(2 * np.pi)
    sc = 1.0 + v * z
    keep = sc > 0
    w = np.where(keep, w, 0.0)
    w = w / w.sum()
    out = []
    for sk, wk in zip(sc, w):
        if wk == 0:
            continue
        thk = dict(th)
        if par == "xc":                             # x_c scales every dimensionless group (xc = 1 at the base)
            r = (L ** 2 * th["eta"] / sk, L * th["beta"] / sk, L * th["epsilon"] / sk ** 2, th["kappa"] / sk)
        else:
            thk[par] = th[par] * sk
            r = _rho(thk, L)
        out.append((float(wk), MP.march(r, g["N"], g["dt"], tau_stop, **kw)))
    return out


def _needed_tau(cs, rows, L):
    """The furthest age (units of L) the requested rows read: their interval ends, entries and the window."""
    t = np.asarray(cs["t"], float)[rows]
    d = 0.0 if cs["d"] is None else np.broadcast_to(np.asarray(cs["d"], float), np.asarray(cs["t"]).shape)[rows]
    x = [float(np.max(t + d))] if len(rows) else [0.0]
    if cs.get("entry") is not None:
        x.append(float(np.max(np.asarray(cs["entry"], float)[rows])))
    if cs.get("win") is not None:
        x += list(cs["win"])
    return max(x) / L


def phase_mpref(args):
    import tomllib
    sys.path.insert(0, os.path.join(A.REPO, "tests"))
    from reference import mp_scheme as MP
    from sraging import SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    t1_path = os.path.join(A.RESULTS, "t1_002.json")
    if not os.path.exists(t1_path):
        sys.exit("[mpref] REFUSED: results/t1_002.json does not exist (run t1 first)")
    t1_res = json.load(open(t1_path))
    if t1_res.get("status_for_acceptance") != "acceptance" and not args.dev:
        sys.exit("[mpref] REFUSED: results/t1_002.json is a development run")
    panel = A.load_toml(T.PANEL_T1)
    rpanel = A.load_toml(PANEL_ROUNDOFF)
    rate = MP.estimate_seconds(100, 100, DPS) / 1e4                    # seconds per node-step
    # -- price everything first ----------------------------------------------------------------------------
    jobs, na, setups, skipped_tail = [], [], {}, 0
    for r in t1_res["results"]:
        if "error" in r or not r.get("n_routed"):
            continue
        if r["kind"] == "segments":
            na.append({"case": r["case"], "scheme": r["scheme"], "theta": r["theta"], "n_routed": r["n_routed"],
                       "reason": "the mpmath scheme has no segments"})
            continue
        key = (r["case"], r["kind"], r["scheme"])
        if key not in setups:
            cs = T.case_setup(r["case"], r["kind"], r["scheme"], panel)
            setups[key] = (cs, Evaluator(cs["coh"], numerics="v1", **cs["kw"]))
        cs, ev1 = setups[key]
        # the cheaper option the gate stated in advance (sr-manager M9): only the routed rows where v1 takes the
        # F route (use_F: the head, where round-off lives), the mpmath march stopping just past the last of them;
        # routed rows outside use_F are v033's own expression in both numerics
        _, a1 = ev1.terms(SRParams(**r["theta_values"], h_ext=cs["h"]))
        uf = np.asarray(a1.get("use_F", np.zeros(cs["coh"].n, bool)), bool)
        routed = [i for i in r["routed_rows"] if uf[i]]
        if not routed:
            skipped_tail += 1
            continue
        r = {**r, "routed_rows": routed, "n_routed_all": r["n_routed"]}
        tau_stop = min(_needed_tau(cs, routed, cs["L"]) + 4 * cs["g0"].dt, cs["g0"].tau_max)
        nodes = 1 if cs["het"] is None else int(np.sum(1 + cs["het"][1] * np.polynomial.hermite_e.hermegauss(cs["het"][2])[0] > 0))
        cost = rate * cs["g0"].N * math.ceil(tau_stop / cs["g0"].dt) * nodes
        jobs.append((r, cs, tau_stop, cost))
    roundoff = []
    for case in rpanel["case"]:
        for label, th in _roundoff_points(case):
            roundoff.append((case, label, th))
    ro_cost = 0.0
    ro_setups = []
    for case, label, th in roundoff:
        cs = _roundoff_setup(case)
        cost = rate * cs["g0"].N * cs["g0"].nsteps
        ro_cost += cost
        ro_setups.append((case, label, th, cs))
    total_h = (sum(j[3] for j in jobs) + ro_cost) / 3600
    run = A.Run("mpref", [T.PANEL_T1, PANEL_ROUNDOFF], args.dev,
                f"{total_h:.2f} CPU-h at {DPS} digits ({len(jobs)} t1 evaluations with routed terms, "
                f"{len(na)} not applicable, {skipped_tail} with no routed use_F rows; {len(roundoff)} round-off "
                f"evaluations); cap {MPREF_CAP_CPU_H} CPU-h")
    if total_h > MPREF_CAP_CPU_H and not args.dev:
        sys.exit(f"[mpref] STOPPED: the estimate {total_h:.2f} CPU-h is above the cap {MPREF_CAP_CPU_H} CPU-h "
                 f"(sr-manager M9: ask before running)")
    A.preflight_or_exit(run)
    # -- part 1: t1's routed terms -------------------------------------------------------------------------
    rows_out = []
    for r, cs, tau_stop, cost in jobs:
        t_c = time.process_time()
        th = r["theta_values"]
        p = SRParams(**th, h_ext=cs["h"])
        ev = setups[(r["case"], r["kind"], r["scheme"])][1]
        ev33 = Evaluator(cs["coh"], numerics="v033", **cs["kw"])
        t1v, _ = ev.terms(p)
        t33, _ = ev33.terms(p)
        g = {"N": cs["g0"].N, "dt": cs["g0"].dt, "Gtol": cs["g0"].Gtol}
        marches = _node_marches(MP, th, cs["L"], cs["het"], g, r["scheme"], tau_stop)
        rows = r["routed_rows"]
        ref = MP.terms(marches, cs["t"], np.asarray(cs["e"]) == 1, cs["d"], cs["L"], h=cs["h"], entry=cs["entry"],
                       window=cs["win"], rows=rows)
        ref = np.array([float(x) for x in ref])
        d1 = np.abs(np.asarray(t1v)[rows] - ref)
        d33 = np.abs(np.asarray(t33)[rows] - ref)
        out = {"case": r["case"], "kind": r["kind"], "scheme": r["scheme"], "theta": r["theta"], "n_rows": len(rows),
               "n_routed_all": r["n_routed_all"],
               "sum_abs_dl_v1": float(d1.sum()), "max_abs_dl_v1": float(d1.max()),
               "sum_abs_dl_v033": float(d33.sum()), "max_abs_dl_v033": float(d33.max()), "tau_stop": tau_stop,
               "cpu_s": time.process_time() - t_c}
        out["ok"] = out["sum_abs_dl_v1"] <= PER_EVAL
        rows_out.append(out)
        print(f"  {r['case'][:44]:44s} {r['scheme']:4s} {r['theta']:12s} rows {len(rows):5d} v1 sum|dl| "
              f"{out['sum_abs_dl_v1']:.2e} (v033 {out['sum_abs_dl_v033']:.2e}) {'ok' if out['ok'] else 'FAIL'}",
              flush=True)
    # -- part 2: the round-off panel -----------------------------------------------------------------------
    ro_out = []
    for case, label, th, cs in ro_setups:
        t_c = time.process_time()
        p = SRParams(**th, h_ext=cs["h"])
        ev = Evaluator(cs["coh"], numerics="v1", **cs["kw"])
        ev33 = Evaluator(cs["coh"], numerics="v033", **cs["kw"])
        t1v, _ = ev.terms(p)
        t33, _ = ev33.terms(p)
        g = {"N": cs["g0"].N, "dt": cs["g0"].dt, "Gtol": cs["g0"].Gtol}
        marches = _node_marches(MP, th, cs["L"], None, g, "bdf2", cs["g0"].tau_max)
        ref = MP.terms(marches, cs["t"], np.asarray(cs["e"]) == 1, cs["d"], cs["L"], h=cs["h"], window=cs["win"])
        tot_mp = float(sum(ref))
        spread = {}
        for name, e_ in (("v1", ev), ("v033", ev33)):
            vals = []
            for k in range(-3, 4):
                pk = SRParams(**{q: th[q] * (1.0 + k * 2.0 ** -52) for q in A.PARAMS}, h_ext=cs["h"])
                vals.append(e_.value(pk))
            spread[name] = float(max(vals) - min(vals))
        out = {"cohort": case["cohort"], "theta": label, "n": int(cs["coh"].n), "total_v1": float(np.sum(t1v)),
               "total_v033": float(np.sum(t33)), "total_mp": tot_mp, "dl_v1": float(np.sum(t1v)) - tot_mp,
               "dl_v033": float(np.sum(t33)) - tot_mp, "spread_v1": spread["v1"], "spread_v033": spread["v033"],
               "cpu_s": time.process_time() - t_c}
        out["ok"] = abs(out["dl_v1"]) <= PER_EVAL and out["spread_v1"] <= SPREAD
        ro_out.append(out)
        print(f"  {case['cohort'][:40]:40s} {label:18s} v1 - mp {out['dl_v1']:+.2e}  v033 - mp {out['dl_v033']:+.3e}  "
              f"ulp spread v1 {out['spread_v1']:.1e} v033 {out['spread_v033']:.1e}  {'ok' if out['ok'] else 'FAIL'}",
              flush=True)
    ok = all(x["ok"] for x in rows_out) and all(x["ok"] for x in ro_out)
    return run.write({"status": "pass" if ok else "fail", "dps": DPS, "thresholds": {"per_evaluation": PER_EVAL,
                      "ulp_spread": SPREAD}, "t1_source": {"repo_head": t1_res.get("repo_head")},
                      "routed": rows_out, "not_applicable": na, "no_routed_use_F": skipped_tail, "roundoff": ro_out,
                      "estimate_cpu_h": total_h})


def _roundoff_points(case):
    """[(label, physical theta)] for a round-off panel case."""
    from engine_cases import theta_points
    th = case["theta"]
    if th == "panel_v1":
        return [(lab, t) for lab, t in theta_points(case["cohort"])]
    if th == "truth":
        tr = dict(A.O.library().truth(case["cohort"]))
        return [("truth", {k: tr[k] for k in A.PARAMS})]
    if th == "t1_labels":                           # T1's five points: truth and the scaled ones
        import acceptance_002_t1 as T1
        tr = dict(A.O.library().truth(case["cohort"]))
        base = {k: tr[k] for k in A.PARAMS}
        return [(lab, T1._scale(base, lab)) for lab in A.load_toml(T1.PANEL_T1)["theta_labels"]]
    if th == "stored":
        d = json.load(open(os.path.join(A.REPO, "tests", "reference", "data", "t7_mpref.json")))
        key = f"{case['cohort']}|stored"
        rho, L = d["entries"][key]["rho"], d["provenance"]["cohort_and_grid"][key]["L_days"]
        return [("stored", {"eta": rho[0] / L ** 2, "beta": rho[1] / L, "epsilon": rho[2] / L, "kappa": rho[3]})]
    return [(case.get("theta_label", "given"), dict(th))]


def _roundoff_setup(case):
    from sraging import Cohort
    from sraging.engine.grid import GridPolicy
    c = T.library_case(case["cohort"], A.load_toml(T.PANEL_T1))
    cont = bool(case.get("exact"))
    d = None if cont else c["d"]
    win = c["window"] if case.get("window") else None
    g0 = GridPolicy("v033").grid(Cohort.from_arrays(c["t"], c["e"], delta=d, continuous=cont))
    coh = Cohort.from_arrays(c["t"], c["e"], delta=d, continuous=cont, window=win)
    h = 0.0 if c["ext"] is None else math.exp(-c["ext"])
    grid = GridPolicy("fixed", N=g0.N, dt=g0.dt, tau_max=g0.tau_max, Gtol=g0.Gtol, L=g0.L)
    return dict(coh=coh, kw=dict(grid=grid, scheme="bdf2", fixed={"h_ext": h}), h=h, g0=g0, L=g0.L, t=c["t"],
                e=c["e"], d=d, win=win)
