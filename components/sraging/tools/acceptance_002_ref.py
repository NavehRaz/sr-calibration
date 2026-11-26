"""Spec 002 phases against references beyond the old code (research R11, R12; review T064 sr-naveh N4, N11):

`early` (SC-005): every early-death term (S_obs(t) > 1 - 1e-3 at the evaluation theta) of the cohorts in
  panel_002_roundoff.toml [[early]], at the v033 grid ("v1" and "v033"), against a converged reference: a graded
  start (dtau_0 = 1e-10, ratio 1.005) then uniform dtau = delta/(10 L), at N = 1800, 3600, 7200, extrapolated in N
  at the observed order. A term's reference is accepted as converged when halving the uniform step moves it by
  < 1e-4 nats and the N = 3600/7200 pair by < 1e-3 nats. The residuals (fit grid minus converged) are written
  whatever their size; they go into register D-004.
`anchor` (R11 item 3, N11): (a) pure diffusion (eta = beta = 0) at three rho_eps against the closed form of
  tests/reference/diffusion.py: S on the grid and early-death interval masses, refining N and the step together
  on a graded start (N = 150..1200); the errors must fall monotonically along the ladder and S's observed order
  at the finest pair must lie in [0.5, 3]. (b) Self-convergence where no closed form exists, at a stiff human
  truth and at kappa ~ h (the ITP example's theta-hat, kappa = 0.001 ~ 0.9 h): the total log-likelihood under
  the same joint refinement up to N = 7200, reported (not judged): at the stiff truth N = 900 is pre-asymptotic
  (the spec's known limitation), so the measurement is the grid error it leaves.
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
EARLY_S = 1.0 - 1e-3
NS_EARLY = (1800, 3600, 7200)
HALVING_TOL, NPAIR_TOL = 1e-4, 1e-3
MAX_STAGE = 4                     # unconverged rows are refined up to four times (N up to 115,200)


def stored_theta_hat(cid, cohort):
    """The production theta-hat of a library cohort: the stored v0.3.3 record's claim points; where a 'none' claim
    stores no point, the port's own fit (byte-identical to the reference's, spec 001) supplies it."""
    cl = A.O.stored_reference_record(cid)["claim_set"]["claims"]
    th = {c["parameter"]: c.get("point") for c in cl}
    if any(v is None for v in th.values()):
        from sraging.calibrators import profile_v033
        fit = profile_v033.fit(cohort)
        th = {k: getattr(fit.theta_hat, k) for k in A.PARAMS}
    return th


def _early_case(case):
    """(cohort, theta dict) for an [[early]] entry."""
    from sraging import Cohort
    if case["cohort"] == "itp_plus_age0":
        c = Cohort.from_csv(os.path.join(A.REPO, "tests", "data", "itp_plus_age0.csv"), delta=1.0, time_unit="days")
        d = json.load(open(os.path.join(A.REPO, "examples", "results", "itp_c2016_het3_control_female.json")))
        return c, {k: d["theta_hat"][k] for k in A.PARAMS}
    lib = T.library_case(case["cohort"], A.load_toml(T.PANEL_T1))
    c = Cohort.from_arrays(lib["t"], lib["e"], delta=lib["d"])
    if case["theta"] == "stored_fit":
        return c, stored_theta_hat(case["cohort"], c)
    if case["theta"] == "stored":
        d = json.load(open(os.path.join(A.REPO, "tests", "reference", "data", "t7_mpref.json")))
        key = f"{case['cohort']}|stored"
        rho, L = d["entries"][key]["rho"], d["provenance"]["cohort_and_grid"][key]["L_days"]
        return c, {"eta": rho[0] / L ** 2, "beta": rho[1] / L, "epsilon": rho[2] / L, "kappa": rho[3]}
    raise ValueError(f"unknown early theta {case['theta']!r}")


def phase_early(args):
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy, graded_start
    panel = A.load_toml(PANEL_ROUNDOFF)
    run = A.Run("early", [PANEL_ROUNDOFF], args.dev, f"~30 CPU-min ({len(panel['early'])} cohorts; graded-start "
                f"references at N = {NS_EARLY} plus a step halving, each marched just past the last early death)")
    A.preflight_or_exit(run)
    out = []
    for case in panel["early"]:
        coh, th = _early_case(case)
        p = SRParams(**th, time_unit=coh.time_unit)
        e1, e33 = Evaluator(coh, numerics="v1"), Evaluator(coh, numerics="v033")
        g = e1.grid
        t1, a1 = e1.terms(p)
        t33, _ = e33.terms(p)
        died = np.asarray(coh.events) == 1
        rows = np.flatnonzero(died & (np.asarray(a1["S_i"]) > EARLY_S))
        rec = {"cohort": case["cohort"], "theta": th, "grid": g.summary(), "n_early": int(rows.size)}
        if not rows.size:
            rec.update(ok=True, note="no early deaths at this theta")
            out.append(rec)
            continue
        delta = float(coh.delta) if np.ndim(coh.delta) == 0 else None
        ages = np.asarray(coh.ages, float)[rows]
        c0 = time.process_time()
        stages = []
        n = rows.size
        rich, conv = np.full(n, np.nan), np.zeros(n, bool)
        half_move, npair = np.full(n, np.nan), np.full(n, np.nan)
        orders = []

        def terms_at(idx, N, div):
            sub_ = Cohort.from_arrays(ages[idx], np.ones(idx.size, int), delta=delta, time_unit=coh.time_unit)
            tau_end = float((ages[idx] + delta).max() / g.L) * 1.01 + 4 * delta / (div * g.L)
            nodes = graded_start(1e-10, 1.005, delta / (div * g.L), tau_end)
            ev = Evaluator(sub_, numerics="v1", grid=GridPolicy("fixed", N=N, nodes=nodes, tau_max=tau_end,
                                                               Gtol=g.Gtol, L=g.L))
            return np.asarray(ev.terms(p)[0])
        # stage s: N = 1800, 3600, 7200 (x 2^s) at the uniform step delta/(10 2^s L), and the step halved at the
        # largest N. The pre-registered test is applied at each stage; rows that pass keep that stage's
        # reference, the others are refined at the next stage (up to MAX_STAGE).
        for s_ in range(MAX_STAGE + 1):
            idx = np.flatnonzero(~conv)
            if not idx.size:
                break
            div = 10 * 2 ** s_
            Ns = [m * 2 ** s_ for m in NS_EARLY]
            a, b, c = (terms_at(idx, N, div) for N in Ns)
            half = terms_at(idx, Ns[-1], 2 * div)
            d1, d2 = np.abs(b - a), np.abs(c - b)
            ok_ord = (d1 > 0) & (d2 > 0)
            order = float(np.median(np.log2(d1[ok_ord] / d2[ok_ord]))) if ok_ord.any() else float("nan")
            p_use = order if np.isfinite(order) and order > 0.25 else 1.0
            orders.append(order)
            rich[idx] = c + (c - b) / (2 ** p_use - 1)
            half_move[idx], npair[idx] = np.abs(half - c), d2
            conv[idx] = (np.abs(half - c) < HALVING_TOL) & (d2 < NPAIR_TOL)
            stages.append({"stage": s_, "rows": int(idx.size), "N": Ns, "step_divisor": div, "observed_order_N": order,
                           "n_converged_now": int(conv[idx].sum()), "max_halving_move": float(np.abs(half - c).max()),
                           "max_N_pair": float(d2.max())})
            print(f"  {case['cohort'][:40]:40s} stage {s_} ({idx.size} rows) N {Ns} step delta/({div} L): converged "
                  f"{int(conv.sum())}/{n}, halving {np.abs(half - c).max():.2e}, N pair {d2.max():.2e}", flush=True)
        res_v1, res_v033 = t1[rows] - rich, t33[rows] - rich
        rec.update(stages=stages, observed_orders_N=orders, n_converged=int(conv.sum()),
                   max_halving_move_unconverged=float(np.nanmax(np.where(conv, np.nan, half_move))) if not conv.all() else 0.0,
                   max_N_pair_unconverged=float(np.nanmax(np.where(conv, np.nan, npair))) if not conv.all() else 0.0,
                   residual_v1={"max_abs": float(np.abs(res_v1).max()), "sum": float(res_v1.sum())},
                   residual_v033={"max_abs": float(np.abs(res_v033).max()), "sum": float(res_v033.sum())},
                   terms=[{"row": int(i) + 1, "age": float(coh.ages[i]), "v1": float(t1[i]), "v033": float(t33[i]),
                           "reference": float(rich[j]), "converged": bool(conv[j]),
                           "halving_move": float(half_move[j]), "N_pair": float(npair[j])} for j, i in enumerate(rows)],
                   cpu_s=time.process_time() - c0)
        rec["ok"] = bool(conv.all())
        out.append(rec)
        print(f"  {case['cohort'][:40]:40s} early {rows.size:4d} converged {int(conv.sum())}/{rows.size} order {order:.2f}"
              f"  residual v1 max {rec['residual_v1']['max_abs']:.3g} sum {rec['residual_v1']['sum']:+.3g} nats  "
              f"(v033 max {rec['residual_v033']['max_abs']:.3g})", flush=True)
    ok = all(r["ok"] for r in out)
    return run.write({"status": "pass" if ok else "fail", "criteria": {"early_S": EARLY_S, "N": NS_EARLY,
                      "halving_tol": HALVING_TOL, "N_pair_tol": NPAIR_TOL}, "results": out})


# --------------------------------------------------------------------------------------------- anchor
ANCHOR_RHO = (0.05, 0.2, 1.0)
ANCHOR_LADDER = ((150, 4e-3), (300, 2e-3), (600, 1e-3), (1200, 5e-4))   # (N, uniform dtau): refined together
SELF_LADDER = ((450, 1), (900, 2), (1800, 4), (3600, 8), (7200, 16))    # (N, divisor of the v033 grid's dtau)


def phase_anchor(args):
    """(a) pure diffusion against its closed form, refining N and the step together on a graded start (the
    early terms are set by the step, not N: smoke run 2026-10-04, fixed step, the earliest term's error grew
    with N); (b) self-convergence of the total log-likelihood under the same joint refinement."""
    sys.path.insert(0, os.path.join(A.REPO, "tests"))
    from reference import diffusion as DF
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy, graded_start
    run = A.Run("anchor", [], args.dev, f"~20 CPU-min (pure diffusion at {len(ANCHOR_RHO)} rho_eps x "
                f"{len(ANCHOR_LADDER)} grids; self-convergence at 2 points x {len(SELF_LADDER)} grids)")
    out = {"diffusion": [], "self_convergence": []}
    for rho in ANCHOR_RHO:
        L, tau_max = 1.0, 4.0 / rho
        ages = np.array([0.01, 0.02, 0.05, 0.1, 0.3]) / rho
        dl = 0.005 / rho
        exact_m = np.array([DF.interval_mass(a, a + dl, rho) for a in ages])
        coh = Cohort.from_arrays(ages, np.ones(ages.size, int), delta=dl)
        p = SRParams(eta=1e-300, beta=1e-300, epsilon=rho, kappa=1.0)
        errS, errT = [], []
        for N, du in ANCHOR_LADDER:
            nodes = graded_start(1e-10, 1.005, du, tau_max)
            ev = Evaluator(coh, numerics="v1", grid=GridPolicy("fixed", N=N, nodes=nodes, tau_max=tau_max, Gtol=1e-11,
                                                              L=L))
            tau, S, _, k = ev.survival_on_grid(p)
            Sx = DF.survival(tau[:k + 1], rho)
            errS.append(float(np.max(np.abs(S[:k + 1] - Sx))))
            terms = np.asarray(ev.terms(p)[0])
            errT.append(np.abs(terms - np.log(exact_m)).tolist())
            print(f"  diffusion rho {rho:5.2f} N {N:5d} dtau {du:g}: max|dS| {errS[-1]:.3e}  early-term errors "
                  f"{np.array2string(np.asarray(errT[-1]), precision=2)}", flush=True)
        eT = np.asarray(errT)
        order_S = math.log2(errS[-2] / errS[-1]) if errS[-1] > 0 else float("inf")
        orders_T = np.log2(eT[-2] / np.maximum(eT[-1], 1e-300)).tolist()
        mono = all(errS[i + 1] < errS[i] for i in range(len(errS) - 1)) and bool(np.all(np.diff(eT, axis=0) < 0))
        out["diffusion"].append({"rho_eps": rho, "ladder": ANCHOR_LADDER, "max_dS": errS, "early_ages": ages.tolist(),
                                 "early_term_errors": errT, "exact_masses": exact_m.tolist(),
                                 "observed_order_S": order_S, "observed_orders_terms": orders_T, "monotone": mono,
                                 "ok": mono and 0.5 <= order_S <= 3.0})
    pts = []
    cid = "pure-real-Denmark_F_1900_hetro-c0-s0"
    lib = T.library_case(cid, A.load_toml(T.PANEL_T1))
    pts.append(("stiff human truth: " + cid, Cohort.from_arrays(lib["t"], lib["e"], delta=lib["d"]),
                SRParams(**lib["truth"])))
    itp = Cohort.from_csv(os.path.join(A.REPO, "examples", "data", "itp_c2016_het3_control_female.csv"), delta=1.0,
                          time_unit="days")
    d = json.load(open(os.path.join(A.REPO, "examples", "results", "itp_c2016_het3_control_female.json")))
    pts.append(("kappa ~ h: ITP example theta-hat", itp,
                SRParams(**{k: d["theta_hat"][k] for k in A.PARAMS}, time_unit="days")))
    for name, coh, p in pts:
        g = Evaluator(coh, numerics="v1").grid
        vals = []
        for N, div in SELF_LADDER:
            ev = Evaluator(coh, numerics="v1", grid=GridPolicy("fixed", N=N, dt=g.dt / div, tau_max=g.tau_max,
                                                              Gtol=g.Gtol, L=g.L))
            vals.append(ev.value(p))
        dif = np.abs(np.diff(vals))
        orders = [math.log2(dif[i] / dif[i + 1]) if dif[i + 1] > 0 else float("inf") for i in range(len(dif) - 1)]
        asym = bool(dif[-1] < dif[-2])                  # the last pair falls: the asymptotic regime is reached
        out["self_convergence"].append({"point": name, "ladder": SELF_LADDER, "loglik": vals,
                                        "differences": dif.tolist(), "observed_orders": orders,
                                        "asymptotic_at_finest": asym,
                                        "production_grid_error_estimate": float(vals[-1] - vals[1])})
        print(f"  self-convergence {name}: differences {np.array2string(dif, precision=3)} orders "
              f"{np.round(orders, 2).tolist()}", flush=True)
    ok = all(r["ok"] for r in out["diffusion"])         # (b) is a measurement, reported (review T064 N11)
    return run.write({"status": "pass" if ok else "fail", "criteria": {"order_range_S": [0.5, 3.0],
                      "monotone": True, "self_convergence": "reported, not judged"}, **out})
