"""Spec 003 `walker` phase (SC-004; Clarifications; research R8; task T032; sr-manager gate 3; sr-naveh impl 8).

Two steps, so the brute force (the expensive part) runs only after Naveh has decided the combinations' cost:

  step 1  on the panel's [walker] cohorts (pure under profile_jax2, hazard under profile_jax2_h, kappa-hat at 300; the
          [pilot] cohorts under profile_jax2 when the panel has no [walker] table): the port's fit and every
          combination walk, with their CPU and their objective evaluations per key (the cost decision is put on
          evaluations, which do not depend on the machine's load), the closed claims' widths by stratum, and a free
          independent check: no archived point -- every point the fit and the walks evaluated -- lies more than
          0.01 dex beyond an unwidened walk endpoint at deviance below 3.8415 from the walk's reference;
  step 2  for every stated endpoint, brute-force constrained maximisation at the two points that decide SC-004:
            outward 0.01 dex: every start must give deviance >= 3.8415 there (it stops at the first that does not);
            inward 0.05 dex:  some start must give deviance <= 3.8415 there (it stops at the first that does) --
                              or at the profile's maximum, when that is nearer (Naveh, 2026-10-08: on an interval
                              narrower than 0.05 dex the point lay beyond the other end, so 6 ends of 3 such intervals
                              failed where an inside point within 0.05 dex is what SC-004 asks);
          the starts: `n_sobol` scrambled Sobol points in the +-1.5 dex cube around theta-hat (clipped to the box),
          `n_sobol` more over the whole box in beta and kappa with eta, eps and h in that cube (amended 2026-10-06,
          Naveh: the cube cannot reach a branch 9-10 dex away in beta), the ridge (eta, beta, eps) x
          10^{+-0.5, +-1, +-1.5} from theta-hat, the kappa corners of theta-hat and of the walk node nearest the point,
          and the walk's nodes within 1 dex; the fraction of starts that converge is reported.
          It refuses to start without step 1's result for the same panel, and -- when the combinations cost more
          than 2x the fit -- without Naveh's recorded decision (tests/acceptance/decisions_003.toml
          [combination_cost]); and when the default lead pair alone costs more than 2x, unless [walker_fix] records
          that Naveh accepted it (`lead_cost_accepted`): his walker decision brings that cost back to him.
Step 1's archive is every captured objective's (the walks' starts and the widening use the same), and since the
walker's final pass re-walks a contradicted side, its check is that pass's convergence check; the brute force is the
independent test.
Step 2 rebuilds the objective from the port's own fit inputs; it must reproduce the fit's log-likelihood at theta-hat.
The deviance reference is each walk's own (its profile maximum, at least the fit's), the reference of its interval.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
import statistics as st
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR, OUT_DEX, IN_DEX, COST_LIMIT, CUBE_DEX = 3.8415, 0.01, 0.05, 2.0, 1.5
RIDGE_BF_DEX = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
LL_REBUILD_TOL = 1e-3          # nats
DECISIONS = os.path.join(REPO, "tests", "acceptance", "decisions_003.toml")
DEX = math.log(10.0)


def _init():
    sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
    for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def port_observation(cid):
    """The library's observation as the port's Observation (window included), the oracle's own input."""
    import oracle_jax2 as OJ
    from sraging.calibrators import profile_jax2 as PJ
    o = OJ.observation(cid)
    return PJ.Observation(cohort_id=o.cohort_id, ages=tuple(o.ages), events=tuple(o.events), delta=o.delta,
                          window=o.window)


def _evals(objs):
    return int(sum(o.n_g + o.n_f for o in objs))


def archive_check(key, walk, Z, LL):
    """Per stated endpoint of the unwidened walk: the farthest archived point beyond it (dex) at deviance < THR."""
    import numpy as np
    from sraging.claims import combinations as K
    out = []
    lg = np.array([K.log_g(key, z) for z in Z]) if len(Z) else np.empty(0)
    inside = 2.0 * (walk["ll_ref"] - np.asarray(LL)) < THR
    for side, end in (("lower", walk["ci"]["ci_lo"]), ("upper", walk["ci"]["ci_hi"])):
        if end is None or not math.isfinite(end):
            continue
        beyond = (math.log(end) - lg) if side == "lower" else (lg - math.log(end))
        m = beyond[inside] / DEX if inside.any() else np.empty(0)
        far = float(m.max()) if m.size else None
        out.append({"key": key, "side": side, "end": end, "max_dex_beyond": far,
                    "ok": far is None or far <= OUT_DEX})
    return out


def step1_worker(cid, name):
    try:
        import numpy as np
        import oracle_jax2 as OJ
        from sraging.calibrators import profile_jax2 as PJ
        obs = port_observation(cid)
        c0 = time.process_time()
        out = PJ.run_port(obs, name, capture=True)
        fit_cpu = time.process_time() - c0
        fit_evals = _evals(out["objectives"])
        c1 = time.process_time()
        cc = PJ.combination_claims(out, name, None)
        combo_cpu = time.process_time() - c1
        combo_evals = int(cc["evals"])
        objs = [o for o in out["objectives"] if o.NP == PJ.main_objective(out, name).NP]
        Z, LL = PJ.fit_archive(objs)                 # every captured objective's, as the walks and the widening
        checks = [ch for k, w in cc["walks"].items() for ch in archive_check(k, w, Z, LL)]
        widths = {k: (math.log10(w["ci"]["ci_hi"] / w["ci"]["ci_lo"]) if w["ci"]["verdict"] == "identified" else None)
                  for k, w in cc["walks"].items()}
        res = out["result"]
        return {"cohort_id": cid, "name": name, "stratum": OJ.library().spec(cid).stratum,
                "theta": [float(x) for x in res["theta"]], "ll_hat": float(res["loglik"]),
                "fit_cpu_s": fit_cpu, "combo_cpu_s": combo_cpu, "fit_evals": fit_evals, "combo_evals": combo_evals,
                "ratio_cpu": combo_cpu / fit_cpu if fit_cpu else None,
                "ratio_evals": combo_evals / fit_evals if fit_evals else None,
                "evals_by_key": {k: int(w["evals"]) for k, w in cc["walks"].items()},
                "walks": {k: {"ci": w["ci"], "stops": w["stops"], "gain": w["gain"], "ll_ref": w["ll_ref"],
                              "nodes": w["nodes"], "box_image": list(w["box_image"]), "probe": w.get("probe"),
                              "bisections": w.get("bisections"), "reconciled": w.get("reconciled")}
                          for k, w in cc["walks"].items()},
                "lead_group": cc["lead_group"], "combination_claims_contradicted": cc["combination_claims_contradicted"],
                "claims": [c.to_dict() for c in cc["claims"]],
                "param_claims": out["claims"], "archive_checks": checks, "widths_dex": widths,
                "gap_nats": cc["gap_nats"], "archive_points": int(np.asarray(LL).size)}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "name": name, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-1500:]}


def bf_starts(z_hat, bounds, n_sobol, nodes_near, seed):
    import numpy as np
    from scipy.stats import qmc
    from sraging.claims import combinations as K
    from sraging.claims import walker as W
    lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    z_hat = np.asarray(z_hat, float)
    a, b = np.maximum(z_hat - CUBE_DEX * DEX, lo), np.minimum(z_hat + CUBE_DEX * DEX, hi)
    out = [a + u * (b - a) for u in qmc.Sobol(len(z_hat), scramble=True, seed=seed).random(n_sobol)]
    a2, b2 = a.copy(), b.copy()                    # beta and kappa over the whole box, the rest in the cube
    a2[[K.BETA, K.KAP]], b2[[K.BETA, K.KAP]] = lo[[K.BETA, K.KAP]], hi[[K.BETA, K.KAP]]
    out += [a2 + u * (b2 - a2) for u in qmc.Sobol(len(z_hat), scramble=True, seed=seed + 1).random(n_sobol)]
    for d in RIDGE_BF_DEX:
        s = z_hat.copy()
        s[:3] += d * DEX
        out.append(s)
    out += W.kappa_corner_starts(z_hat)
    for z in nodes_near:
        out += [np.asarray(z, float)] + W.kappa_corner_starts(z)
    return out


def step2_worker(row, n_sobol):
    try:
        import numpy as np
        from sraging.calibrators import profile_jax2 as PJ
        from sraging.claims import walker as W
        from sraging.compat.jax2 import estimator as est, fp_jax as fj
        name = row["name"]
        inp = PJ.fit_inputs(port_observation(row["cohort_id"]), name)
        obj = est._Objective(fj, inp["t"], inp["e"], inp["L"], inp["kw"], inp["box"], hazard=inp["hazard"],
                             window=inp["window"])
        z_hat = np.log(np.asarray(row["theta"], float))
        ll_check = -float(obj.nll_grad(z_hat)[0])
        if abs(ll_check - row["ll_hat"]) > LL_REBUILD_TOL:      # log(theta) is 1-4 ulp off the fit's z, and fp_jax
            #                                                       is discontinuous at ulp scale at box corners
            #                                                       (J2 reports/rc2/RATIFICATION.txt:75): recorded
            raise RuntimeError(f"the rebuilt objective gives {ll_check!r} at theta-hat, the fit {row['ll_hat']!r}")
        bounds = [(math.log(obj.box[p][0]), math.log(obj.box[p][1])) for p in obj.params]
        c0, n0 = time.process_time(), obj.n_g + obj.n_f
        checks, tried, landed = [], 0, 0
        for key, w in row["walks"].items():
            for side, end in (("lower", w["ci"]["ci_lo"]), ("upper", w["ci"]["ci_hi"])):
                if end is None or not math.isfinite(end):
                    continue
                sgn = -1 if side == "lower" else 1
                rec = {"key": key, "side": side, "end": end}
                am = w["ci"].get("argmax_value")
                in_dex = min(IN_DEX, abs(math.log10(am / end))) if am is not None and am > 0 else IN_DEX
                for label, dex, need in (("outward", OUT_DEX, "all_outside"), ("inward", -in_dex, "one_inside")):
                    c = math.log(end) + sgn * dex * DEX
                    near = [z for cn, _, z in w["nodes"] if abs(cn - c) < DEX]
                    starts = bf_starts(z_hat, bounds, n_sobol, near, seed=20261005)
                    best_dev, decided, n_ok = None, None, 0
                    for z0 in starts:
                        r = W._solve(obj, key, c, z0, bounds)
                        tried += 1
                        if r is None:
                            continue
                        n_ok += 1
                        landed += 1
                        dev = 2.0 * (w["ll_ref"] - r[1])
                        best_dev = dev if best_dev is None else min(best_dev, dev)
                        if need == "all_outside" and dev < THR:
                            decided = False                     # a start inside: the walk is too narrow here
                            break
                        if need == "one_inside" and dev <= THR:
                            decided = True                      # a start inside: the walk is not too wide here
                            break
                    ok = (decided is None and n_ok > 0) if need == "all_outside" else bool(decided)
                    rec[label] = {"best_dev": best_dev, "ok": ok, "starts": len(starts), "converged": n_ok,
                                  "dex": abs(dex)}
                rec["ok"] = rec["outward"]["ok"] and rec["inward"]["ok"]
                checks.append(rec)
        return {"cohort_id": row["cohort_id"], "name": name, "checks": checks, "cpu_s": time.process_time() - c0,
                "evals": obj.n_g + obj.n_f - n0, "starts_tried": tried, "starts_converged": landed,
                "ll_rebuilt_minus_fit": ll_check - row["ll_hat"]}
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": row["cohort_id"], "name": row.get("name"), "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-1500:]}


def _pool(fn, jobs, workers, label):
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                j = futs[f]
                r = {"cohort_id": j[0] if isinstance(j[0], str) else j[0]["cohort_id"], "error": f"crash: {e}"}
            out.append(r)
            print(f"  [{label}] {r['cohort_id']} {r.get('name', '')}: "
                  f"{'ERROR ' + r['error'][:90] if 'error' in r else 'ok'}", flush=True)
    return out


def cohorts(panel):
    w = panel.get("walker", {}).get("cohorts")
    if w:
        return [(c["id"], c["name"]) for c in w]
    return [(c, "profile_jax2") for c in panel["pilot"]["cohorts"]]


def decision(table="combination_cost"):
    import tomllib
    if not os.path.exists(DECISIONS):
        return None
    return tomllib.load(open(DECISIONS, "rb")).get(table)


def phase_walker(args):
    import json
    import acceptance_003 as A
    step = int(getattr(args, "walker_step", 1) or 1)
    n_sobol = int(getattr(args, "n_sobol", 32) or 32)
    jobs = cohorts(A.load_toml(A.PANEL))
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    if step == 1:
        run = A.Run("walker", manifests=[A.PANEL], dev=args.dev or bool(args.only),
                    cost=f"step 1: {len(jobs)} port fits with every combination walk; the fits alone cost a median "
                         f"~270-1330 CPU-s each on this Mac (fits smoke, the ITP example), the walks are what step 1 "
                         f"measures; {args.workers} workers")
        A.preflight_or_exit(run)
        rows = _pool(step1_worker, jobs, args.workers, "step 1")
        good = [r for r in rows if "error" not in r]
        errors = [r for r in rows if "error" in r]
        checks = [ch for r in good for ch in r["archive_checks"]]
        failures = [ch for ch in checks if not ch["ok"]]
        widths = {}
        for r in good:
            for k, v in r["widths_dex"].items():
                if v is not None:
                    widths.setdefault(k, {}).setdefault(r["stratum"], []).append(v)
        ratio_e = st.median([r["ratio_evals"] for r in good if r.get("ratio_evals")]) if good else None
        ratio_c = st.median([r["ratio_cpu"] for r in good if r.get("ratio_cpu")]) if good else None
        by_key = {k: st.median([r["evals_by_key"][k] / r["fit_evals"] for r in good if r["fit_evals"]])
                  for k in (good[0]["evals_by_key"] if good else {})}
        ratio_lead = st.median([sum(r["evals_by_key"][k] for k in r["lead_group"]) / r["fit_evals"]
                                for r in good if r["fit_evals"]]) if good else None
        run.write({"status": "step1-written" if not errors else "step1-errors", "step": 1,
                   "archive_checks": {"passed": len(checks) - len(failures), "total": len(checks)},
                   "archive_failures": failures, "errors": errors, "cost_ratio_median": ratio_e,
                   "cost_ratio_cpu_median": ratio_c, "cost_ratio_by_key_median": by_key,
                   "cost_ratio_lead_median": ratio_lead, "cost_limit": COST_LIMIT,
                   "cost_basis": "objective evaluations (combinations / fit), median over cohorts; CPU beside",
                   "cost_decision": decision(),
                   "median_widths_dex_by_stratum": {k: {s: st.median(v) for s, v in d.items()} for k, d in widths.items()},
                   "rows": good}, name="walker_step1")
        return
    # step 2
    p1 = os.path.join(A.RESULTS, "walker_step1_003.json")
    if not os.path.exists(p1):
        sys.exit("walker step 2 refused: results/walker_step1_003.json does not exist (run --walker-step 1 first)")
    s1 = json.load(open(p1))
    dec = decision()
    if s1["cost_ratio_median"] is not None and s1["cost_ratio_median"] > COST_LIMIT and not dec:
        sys.exit(f"walker step 2 refused: the combinations cost {s1['cost_ratio_median']:.2f}x the fit (evaluations), "
                 f"above {COST_LIMIT}x, and Naveh's decision is not recorded in {os.path.relpath(DECISIONS, REPO)}")
    lead = s1.get("cost_ratio_lead_median")
    if lead is not None and lead > COST_LIMIT and not (decision("walker_fix") or {}).get("lead_cost_accepted"):
        sys.exit(f"walker step 2 refused: the default lead pair costs {lead:.2f}x the fit (evaluations), above "
                 f"{COST_LIMIT}x; Naveh's walker decision brings that back to him ([walker_fix] lead_cost_accepted)")
    rows = [r for r in s1["rows"] if (r["cohort_id"], r["name"]) in set(jobs)]
    run = A.Run("walker", manifests=[A.PANEL, p1, *([DECISIONS] if os.path.exists(DECISIONS) else [])],
                dev=args.dev or bool(args.only),
                cost=f"step 2: brute force on {len(rows)} cohorts, {n_sobol} cube + {n_sobol} whole-box (beta, kappa) "
                     f"Sobol + {len(RIDGE_BF_DEX)} ridge + kappa corner + node starts per point, 2 points per stated "
                     f"endpoint, early exit; price it on 2 cohorts with --only first; {args.workers} workers")
    A.preflight_or_exit(run)
    out = _pool(step2_worker, [(r, n_sobol) for r in rows], args.workers, "step 2")
    good = [r for r in out if "error" not in r]
    errors = [r for r in out if "error" in r]
    checks = [ch for r in good for ch in r["checks"]]
    failures = [ch for ch in checks if not ch["ok"]]
    a_fail = s1.get("archive_failures") or []
    tried = sum(r["starts_tried"] for r in good)
    conv = sum(r["starts_converged"] for r in good)
    cost_ok = s1["cost_ratio_median"] is not None and (s1["cost_ratio_median"] <= COST_LIMIT or bool(dec))
    ok = not failures and not a_fail and not errors and cost_ok and checks
    status = ("smoke-pass" if ok else "smoke-fail") if args.only else ("pass" if ok else "fail")
    run.write({"status": status, "step": 2, "checks": {"passed": len(checks) - len(failures), "total": len(checks)},
               "failures": failures, "archive_checks": s1.get("archive_checks"), "archive_failures": a_fail,
               "errors": errors, "cost_ratio_median": s1["cost_ratio_median"],
               "cost_ratio_cpu_median": s1.get("cost_ratio_cpu_median"), "cost_limit": COST_LIMIT,
               "cost_decision": dec, "median_widths_dex_by_stratum": s1.get("median_widths_dex_by_stratum"),
               "n_sobol": n_sobol, "starts_converged_fraction": (conv / tried) if tried else None,
               "step1_repo_head": s1.get("repo_head"), "rows": good,
               "rule": {"outward_dex": OUT_DEX, "inward_dex": IN_DEX,
                        "inward_cap": "the profile's maximum, when nearer", "thr": THR, "cube_dex": CUBE_DEX,
                        "box_sobol": ["beta", "kappa"],
                        "ridge_dex": list(RIDGE_BF_DEX)}})
