"""Spec 003c P1, the walker check: is the eta/eps walk's upper end short at the truth on the 11 coverage misses?

The rule is docs/planning/spec003c_seed.md, "P1's rule, fixed before any refit" (committed at d5a2cae, before any
refit). Per miss of spec 003's coverage run (rows of d903979, read-only from .cache/out/wexac_rows/<sha>/):
  refit      on this Mac, from the observation the coverage fit used (library v2 through the v033 snapshot's loader,
             acceptance_003_coverage.library()): the port's fit and the lead combination walks; the refit counts only
             if its eta/eps claim's two ends each equal the row's within MATCH_DEX;
  reference  the eta/eps walk's own ll_ref (its profile maximum, at least the fit's), raised by any higher value found
             at psi = truth, so a deviance is never negative;
  step A     the deviance of the full truth (h = exp(-external_hazard) for profile_jax2_h, SRK benchmark/cohorts.py
             641-642): SHORT if below THR (sufficient: the profile at psi_truth is at least ll(theta_truth)); set aside
             when theta_truth is outside the box, or for hazard when |log10(h-hat / h_truth)| > UNITS_DEX;
  step B     constrained maximisation at psi = truth with the walker's own solver, from theta_truth and its kappa
             corners, then walker step 2's start set: SHORT at the first start below THR; NOT_SHORT when every start
             is at or above it and one converged; UNDECIDED when none converged.
--pilot runs censor-k0.1-p01-c0-s1 alone (results/walker_check_pilot_003c.json, with the price of the other 10 from
the rows' fit CPU); the full run does the other 10 and takes the pilot's verdict, refusing if src/ or this tool
changed since the pilot ran (results/walker_check_003c.json).
--outside-rule COHORT reruns one cohort the full run left UNDECIDED_REFIT_DIFFERS and runs step B on the Mac refit
anyway (Naveh, 2026-10-11): its verdict under the rule stays undecided; the step-B verdict is recorded beside it as
outside the rule (results/walker_check_outside_rule_003c.json), refusing if src/ or the two helper tools changed since
the full run.

    caffeinate -i -w <pid>; .venv/bin/python tools/walker_check_003c.py --pilot
    .venv/bin/python tools/walker_check_003c.py --workers 10
    .venv/bin/python tools/walker_check_003c.py --outside-rule censor-k0.1-p00-c1-s1
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

THR, MATCH_DEX, UNITS_DEX, N_SOBOL, BF_SEED = 3.8415, 1e-4, 1.0, 32, 20261005
LL_REBUILD_TOL = 1e-3          # nats: the objective at theta-hat against the fit's log-likelihood
Q = "eta_over_eps"
ROWS_HEAD = "d903979764194c5b7afa7b048710210c15e9179b"
ROWS_DIR = os.path.join(REPO, ".cache", "out", "wexac_rows", ROWS_HEAD)
ARMS = (("profile_jax2", "censor"), ("profile_jax2_h", "hazard"))
MISSES = {
    "censor-k0.01-p00-c1-s0": "profile_jax2", "censor-k0.01-p02-c0-s1": "profile_jax2",
    "censor-k0.01-p02-c1-s0": "profile_jax2", "censor-k0.0316-p00-c1-s0": "profile_jax2",
    "censor-k0.0316-p01-c1-s1": "profile_jax2", "censor-k0.1-p00-c0-s0": "profile_jax2",
    "censor-k0.1-p00-c1-s1": "profile_jax2", "censor-k0.1-p01-c0-s1": "profile_jax2",
    "hazard-k0.01-p00-c1-s0": "profile_jax2_h", "hazard-k0.01-p03-c0-s0": "profile_jax2_h",
    "hazard-k0.0316-p00-c1-s0": "profile_jax2_h",
}
PILOT = "censor-k0.1-p01-c0-s1"
SEED_DOC = "docs/planning/spec003c_seed.md"
DEX = math.log(10.0)


def _init():
    import acceptance_003_coverage as C
    C._init()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_rows(rows_dir=ROWS_DIR):
    files = sorted(glob.glob(os.path.join(rows_dir, "*.jsonl")))
    rows = [json.loads(line) for f in files for line in open(f)]
    return rows, {os.path.basename(f): sha256(f) for f in files}


def claim_of(row, q=Q):
    return next((c for c in row["claims"] if c["parameter"] == q), None)


def misses_from_rows(rows) -> dict:
    """{cohort_id: calibrator} of the closed, not box-limited eta/eps claims that exclude the truth, on the two arms."""
    out = {}
    for r in rows:
        if (r["calibrator"], r["arm"]) not in ARMS or r.get("status") != "ok":
            continue
        c = claim_of(r)
        if c is None or c["shape"] != "closed" or (c.get("label_info") or {}).get("kind") == "box_limited":
            continue
        t = r["truth"]["eta"] / r["truth"]["epsilon"]
        if not c["lower"] <= t <= c["upper"]:
            out[r["cohort_id"]] = r["calibrator"]
    return out


def ends_match(refit, row, tol=MATCH_DEX) -> dict:
    """Both ends' differences in dex and whether each is within tol (a missing or non-positive end never matches)."""
    out = {}
    for side in ("lower", "upper"):
        a, b = refit.get(side), row.get(side)
        out[f"{side}_dex"] = abs(math.log10(a / b)) if a and b and a > 0 and b > 0 else None
    out["match"] = all(out[f"{s}_dex"] is not None and out[f"{s}_dex"] <= tol for s in ("lower", "upper"))
    return out


def deviance(ll_ref, ll):
    """2 (ll_ref' - ll) with ll_ref' = max(ll_ref, ll): never negative."""
    return max(0.0, 2.0 * (ll_ref - ll))


def verdict(match, d_a, a_usable, d_b, n_converged, thr=THR) -> tuple[str, str | None]:
    """(verdict, step): the seed's rule. Step B's d_b is its best deviance (None if it did not run)."""
    if not match:
        return "UNDECIDED_REFIT_DIFFERS", None
    if a_usable and d_a is not None and d_a < thr:
        return "SHORT", "A"
    if d_b is not None and d_b < thr:
        return "SHORT", "B"
    if n_converged:
        return "NOT_SHORT", "B"
    return "UNDECIDED", "B"


def run_step_b(match, outside_rule, a_usable, d_a, thr=THR) -> bool:
    """Step B runs when the refit counts (or, outside the rule, when Naveh asked for it) and step A does not decide."""
    return (match or outside_rule) and not (a_usable and d_a is not None and d_a < thr)


def check_worker(cid, name, row_claim, outside_rule=False):
    try:
        import numpy as np
        import acceptance_003_coverage as C
        import acceptance_003_walker as WK
        from sraging.adapters.observation import cohort_from_observation
        from sraging.calibrators import profile_jax2 as PJ
        from sraging.claims import combinations as K
        from sraging.claims import walker as W
        from sraging.compat.v033 import runtime
        from sraging.params import SRParams
        lib = C.library()
        sp, truth = lib.spec(cid), lib.truth(cid)
        cohort = cohort_from_observation(lib.observation(cid))
        runtime.check()
        c0 = time.process_time()
        out = PJ.run_port(cohort, name, capture=True)
        fit_cpu = time.process_time() - c0
        theta = [float(x) for x in out["result"]["theta"]]
        hazard = PJ.CLASSES[name].HAZARD
        theta_hat = SRParams(**dict(zip(PJ.PARAMS, theta[:4])), h_ext=theta[4] if hazard else 0.0,
                             time_unit=cohort.time_unit)
        c1 = time.process_time()
        cc = PJ.combination_claims(out, name, theta_hat, mode="lead")
        walk_cpu = time.process_time() - c1
        c = next(x for x in cc["claims"] if x.parameter == Q)
        refit_claim = {"shape": c.shape, "lower": c.lower, "upper": c.upper, "point": c.point}
        m = ends_match(refit_claim, row_claim)
        w = cc["walks"][Q]
        ll_ref, ll_hat = float(w["ll_ref"]), float(out["result"]["loglik"])
        obj = PJ.main_objective(out, name)
        z_hat = np.log(np.asarray(theta, float))
        ll_at_hat = -float(obj.nll_grad(z_hat)[0])
        if abs(ll_at_hat - ll_hat) > LL_REBUILD_TOL:
            raise RuntimeError(f"the fit's objective gives {ll_at_hat!r} at theta-hat, the fit {ll_hat!r}")
        bounds = [(math.log(obj.box[p][0]), math.log(obj.box[p][1])) for p in obj.params]
        lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        th_t = [truth["eta"], truth["beta"], truth["epsilon"], truth["kappa"]]
        h_true = None
        if hazard:
            h_true = math.exp(-sp.external_hazard)
            th_t.append(h_true)
        z_t = np.log(np.asarray(th_t, float))
        c_t = K.log_g(Q, z_t)
        rec = {"cohort_id": cid, "name": name, "arm": sp.arm, "row_claim": row_claim, "refit_claim": refit_claim,
               "refit_match": m, "ll_hat": ll_hat, "ll_ref": ll_ref, "ll_at_theta_hat_minus_fit": ll_at_hat - ll_hat,
               "psi_truth": math.exp(c_t), "theta_hat": theta, "theta_truth": th_t, "fit_cpu_s": fit_cpu,
               "walk_cpu_s": walk_cpu, "walk_stops": w.get("stops"), "walk_ci": w.get("ci")}
        # step A: one evaluation at the full truth
        inside = bool(obj.inside(th_t))
        units_dex = math.log10(theta[4] / h_true) if hazard and theta[4] > 0 else None
        a_usable = inside and (units_dex is None or abs(units_dex) <= UNITS_DEX)
        d_a = None
        if inside:
            d_a = deviance(ll_ref, -float(obj.nll_grad(z_t)[0]))
        rec["step_a"] = {"truth_inside_box": inside, "h_units_dex": units_dex, "usable": a_usable, "deviance": d_a}
        # step B: brute force at psi = truth, only when the refit counts and A does not decide
        d_b, n_tried, n_conv, decided_at = None, 0, 0, None
        if run_step_b(m["match"], outside_rule, a_usable, d_a):
            c2, e0 = time.process_time(), obj.n_g + obj.n_f
            zt_in = np.clip(z_t, lo, hi)
            near = [z for cn, _, z in w["nodes"] if abs(cn - c_t) < DEX]
            starts = [zt_in] + W.kappa_corner_starts(zt_in) + WK.bf_starts(z_hat, bounds, N_SOBOL, near, BF_SEED)
            best = None
            for i, z0 in enumerate(starts):
                r = W._solve(obj, Q, c_t, z0, bounds)
                n_tried += 1
                if r is None:
                    continue
                n_conv += 1
                best = r[1] if best is None else max(best, r[1])
                if deviance(ll_ref, best) < THR:
                    decided_at = i
                    break
            d_b = None if best is None else deviance(ll_ref, best)
            rec["step_b"] = {"deviance": d_b, "best_ll": best, "above_ll_ref": best is not None and best > ll_ref,
                             "starts": len(starts), "tried": n_tried, "converged": n_conv,
                             "decided_at_start": decided_at, "cpu_s": time.process_time() - c2,
                             "evals": obj.n_g + obj.n_f - e0}
        rec["verdict"], rec["decided_by"] = verdict(m["match"], d_a, a_usable, d_b, n_conv)
        if outside_rule and not m["match"]:
            ov, ostep = verdict(True, d_a, a_usable, d_b, n_conv)
            rec["outside_rule"] = {"verdict": ov, "decided_by": ostep,
                                   "note": "step B on a refit whose ends differ from the row's by more than MATCH_DEX; "
                                           "not counted under the rule"}
        rec["cpu_s"] = time.process_time() - c0
        return rec
    except Exception as ex:  # noqa: BLE001
        return {"cohort_id": cid, "name": name, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-1500:]}


def _pool(jobs, workers):
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = {ex.submit(check_worker, *j): j for j in jobs}
        for f in cf.as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"cohort_id": futs[f][0], "name": futs[f][1], "error": f"crash: {e}"}
            out.append(r)
            print(f"  [walker-check] {r['cohort_id']}: {r.get('verdict') or 'ERROR ' + r['error'][:90]}", flush=True)
    return out


def summary(results) -> dict:
    out = {}
    for arm in ("censor", "hazard"):
        rs = [r for r in results if r["cohort_id"].startswith(arm + "-")]
        out[arm] = {v: sum(r.get("verdict") == v for r in rs)
                    for v in ("SHORT", "NOT_SHORT", "UNDECIDED", "UNDECIDED_REFIT_DIFFERS")}
        out[arm]["errors"] = sum("error" in r for r in rs)
        out[arm]["n"] = len(rs)
    s = out["censor"]
    out["branch"] = ("walker_short" if s["SHORT"] else
                     "misses_real" if s["NOT_SHORT"] == s["n"] else "undecided_to_naveh")
    return out


def main(argv=None):
    import acceptance_003 as A
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pilot", action="store_true", help=f"{PILOT} alone, with the price of the other 10")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--only", nargs="*", help="these cohorts (a development run that never counts)")
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--outside-rule", metavar="COHORT",
                    help="step B on one cohort the full run left UNDECIDED_REFIT_DIFFERS, recorded outside the rule")
    args = ap.parse_args(argv)
    rows, shas = load_rows()
    heads = sorted({r["head"] for r in rows})
    if heads != [ROWS_HEAD] or len(rows) != 215:
        sys.exit(f"walker check refused: the rows are not the 215 of {ROWS_HEAD[:7]} (heads {heads}, {len(rows)} rows)")
    found = misses_from_rows(rows)
    if found != MISSES:
        sys.exit(f"walker check refused: the rows' misses differ from the seed's list: {sorted(set(found) ^ set(MISSES))}")
    by_id = {(r["cohort_id"], r["calibrator"]): r for r in rows}
    pilot_path = os.path.join(A.RESULTS, "walker_check_pilot_003c.json")
    pilot = full = None
    if args.outside_rule:
        full_path = os.path.join(A.RESULTS, "walker_check_003c.json")
        if not os.path.exists(full_path):
            sys.exit("walker check refused: --outside-rule needs the full run (results/walker_check_003c.json)")
        full = json.load(open(full_path))
        if full.get("status_for_acceptance") != "acceptance" or full.get("status") != "written":
            sys.exit("walker check refused: the full run's result is not a written acceptance run")
        prior = {r["cohort_id"]: r.get("verdict") for r in full["results"]}
        if prior.get(args.outside_rule) != "UNDECIDED_REFIT_DIFFERS":
            sys.exit(f"walker check refused: {args.outside_rule} is {prior.get(args.outside_rule)!r} in the full run, "
                     "not UNDECIDED_REFIT_DIFFERS")
        moved = A.git("diff", "--name-only", full["repo_head"], "HEAD", "--", "src", "tools/acceptance_003_walker.py",
                      "tools/acceptance_003_coverage.py")
        if moved:
            sys.exit(f"walker check refused: changed since the full run: {moved.split()}")
        ids = [args.outside_rule]
    elif args.pilot:
        ids = [PILOT]
    elif args.only:
        ids = [c for c in args.only if c in MISSES]
    else:
        if not os.path.exists(pilot_path):
            sys.exit("walker check refused: run --pilot first (results/walker_check_pilot_003c.json)")
        pilot = json.load(open(pilot_path))
        if pilot.get("status_for_acceptance") != "acceptance":
            sys.exit("walker check refused: the pilot's result is a development run")
        moved = A.git("diff", "--name-only", pilot["repo_head"], "HEAD", "--", "src", "tools/walker_check_003c.py",
                      "tools/acceptance_003_walker.py", "tools/acceptance_003_coverage.py")
        if moved:
            sys.exit(f"walker check refused: changed since the pilot ran: {moved.split()}")
        ids = [c for c in MISSES if c != PILOT]
    jobs = [(c, MISSES[c], {k: claim_of(by_id[(c, MISSES[c])])[k] for k in ("shape", "lower", "upper", "point")},
             bool(args.outside_rule)) for c in ids]
    fit_cpu_rows = {c: by_id[(c, MISSES[c])]["cpu_s"] for c in MISSES}
    run = A.Run("walker-check", manifests=[SEED_DOC], dev=args.dev or bool(args.only), suffix="003c",
                cost=f"{len(jobs)} Mac refits (fit + lead walks) and a brute force at psi = truth where step A does not "
                     f"decide; the coverage fits took {sum(fit_cpu_rows[c] for c in ids) / 3600:.2f} EPYC CPU-h with "
                     f"all six walks; {min(args.workers, len(jobs))} workers")
    A.preflight_or_exit(run)
    results = _pool(jobs, min(args.workers, len(jobs)))
    rule = {"thr": THR, "match_dex": MATCH_DEX, "units_dex": UNITS_DEX, "n_sobol": N_SOBOL, "bf_seed": BF_SEED,
            "reference": "the eta/eps walk's own ll_ref, raised by any higher value at psi = truth",
            "seed_doc": SEED_DOC, "seed_doc_commit": A.last_commit(SEED_DOC)}
    common = {"rule": rule, "rows": {"dir": os.path.relpath(ROWS_DIR, REPO), "head": ROWS_HEAD, "n": len(rows),
                                     "sha256": shas}, "misses": MISSES}
    if args.outside_rule:
        r = results[0]
        run.write({"status": "outside-rule-written" if "error" not in r else "outside-rule-error", "result": r,
                   "full_run_repo_head": full["repo_head"],
                   "decision": "Naveh 2026-10-11 (AskUserQuestion): brute force on the Mac refit, recorded outside the "
                               "rule; the rule's verdict stays UNDECIDED_REFIT_DIFFERS and the rule's summary and "
                               "branch are unchanged", **common}, name="walker_check_outside_rule")
        return
    if args.pilot:
        r = results[0]
        price = None
        if "error" not in r:
            per_row_cpu = r["cpu_s"] / fit_cpu_rows[PILOT]     # Mac check CPU per EPYC coverage-fit CPU, this cohort
            price = {"pilot_cpu_s": r["cpu_s"], "pilot_row_cpu_s": fit_cpu_rows[PILOT], "ratio": per_row_cpu,
                     "other_10_cpu_h_inferred": per_row_cpu * sum(v for c, v in fit_cpu_rows.items() if c != PILOT) / 3600,
                     "basis": "the pilot's check CPU over its coverage fit's CPU, times the other 10 fits' CPU "
                              "(inferred: brute force runs only where step A does not decide)"}
        run.write({"status": "pilot-written" if "error" not in r else "pilot-error", "result": r, "price": price,
                   **common}, name="walker_check_pilot")
        return
    allr = results + ([pilot["result"]] if pilot else [])
    errors = [r for r in allr if "error" in r]
    sm = summary(allr)
    status = "written" if not errors else "errors"
    if args.only:
        status = "dev-" + status
    run.write({"status": status, "summary": sm, "results": sorted(allr, key=lambda r: r["cohort_id"]),
               "pilot_repo_head": pilot["repo_head"] if pilot else None, **common}, name="walker_check")


if __name__ == "__main__":
    main()
