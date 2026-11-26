"""Spec 003d T052 (Naveh 2026-10-15, "Wording + the DT_RULE test"): the sr-naveh review's falsifier for attributing
the re-test's large witness D values to mle_profile_jax2 1.0.0's time step.

    .venv/bin/python tools/dt_rule_test_003d.py [--workers 12]    # needs a clean committed tree (a .cache/wt worktree)

For each named re-test cohort (the arrays shipped to WEXAC, sha-checked against the manifest), the calibrator's own
fit stopped after its MLE stage (`acceptance_003d.theta_hat_only`) twice: on its production grid, and with jax2's own
DT_RULE (dt <= CV/80; built and off in 1.0.0) added. Each gives theta-hat, ll(theta-hat) and ll(truth) on the same
objective, so D = 2 (ll(theta-hat) - ll(truth)). The review's rule: if D on the DT_RULE grid stays above 9.49, the
attribution fails. Writes tests/acceptance/results/dt_rule_test_003d.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import multiprocessing as mp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

COHORTS = ("censor-k0.01-p05-c1-s0", "censor-k0.01-p05-c1-s1", "censor-k0.0316-p05-c1-s0", "censor-k0.0316-p05-c1-s1",
           "nlad-k0.1-p00-n10000-s0", "nlad-k0.1-p00-n10000-s1")
CUT = 9.49                        # panel_003d.toml [thresholds].witness_D_cut


def one(cid, dt_rule):
    import acceptance_003d as ACC
    import retest_003d as RT
    try:
        row = next(r for r in json.load(open(RT.MANIFEST))["cohorts"] if r["cohort_id"] == cid)
        ages, events = RT.load_checked(row)
        r = ACC.theta_hat_only(ages, events, row["delta"], row["calibrator"], truth=row["truth"], dt_rule=dt_rule)
        D = 2.0 * (r["loglik"] - r["loglik_truth"])
        t = row["truth"]
        return {"cohort_id": cid, "dt_rule": dt_rule, "D": D, "loglik": r["loglik"], "loglik_truth": r["loglik_truth"],
                "theta": r["theta"], "truth": t, "cpu_s": r["cpu_s"],
                "grid": {k: (r["diagnostics"].get("solver") or {}).get(k) for k in ("N", "dt", "tau_max")},
                "dex_from_truth": {p: math.log10(r["theta"][p] / t[p]) for p in t},
                "eta_over_eps_dex": math.log10((r["theta"]["eta"] / r["theta"]["epsilon"]) / (t["eta"] / t["epsilon"]))}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"cohort_id": cid, "dt_rule": dt_rule, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-1500:]}


def main(argv=None):
    import acceptance_003 as A
    import acceptance_003d as ACC
    import glob
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--rows", default=None, help="the WEXAC rows, for the full fits' D beside these (optional)")
    ap.add_argument("--cohorts", nargs="*", default=None, help="other manifest cohorts than COHORTS (Naveh 2026-10-16: "
                                                               "the two sharp pure witnesses with D above the cut)")
    ap.add_argument("--name", default="dt_rule_test", help="the result file's stem (default dt_rule_test)")
    args = ap.parse_args(argv)
    run = A.Run("dt-rule-test", manifests=[ACC.PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"{2 * len(args.cohorts or COHORTS)} theta-hat-only fits (production and DT_RULE grids), "
                     f"{args.workers} workers")
    cohorts = tuple(args.cohorts) if args.cohorts else COHORTS
    jobs = [(c, d) for c in cohorts for d in (False, True)]
    out = []
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                initializer=ACC._fit_init) as ex:
        for i, f in enumerate(cf.as_completed([ex.submit(one, *j) for j in jobs]), 1):
            r = f.result()
            out.append(r)
            print(f"  [dt-rule] {i}/{len(jobs)} {r['cohort_id']} dt_rule={r['dt_rule']}: "
                  f"{'ERROR ' + r['error'] if 'error' in r else 'D %.2f, cpu %.0f s' % (r['D'], r['cpu_s'])}",
                  flush=True)
    full = {}
    if args.rows:
        for p in glob.glob(os.path.join(args.rows, "shard_*.jsonl")):
            for ln in open(p):
                r = json.loads(ln)
                full[r["cohort_id"]] = (r.get("witness") or {}).get("D")
    by = {}
    for r in out:
        by.setdefault(r["cohort_id"], {})["dt_rule" if r["dt_rule"] else "production"] = r
    verdicts = {}
    for c in cohorts:
        d = by.get(c, {})
        rule = d.get("dt_rule", {})
        verdicts[c] = {"D_production": d.get("production", {}).get("D"), "D_dt_rule": rule.get("D"),
                       "D_full_fit_production": full.get(c),
                       "attribution_holds": (rule.get("D") is not None and rule["D"] <= CUT)}
    fails = [c for c, v in verdicts.items() if not v["attribution_holds"]]
    run.write({"status": "attribution holds" if not fails else "attribution fails", "fails": fails, "cut": CUT,
               "rule": "the sr-naveh review's falsifier: D on the DT_RULE grid above the cut means the attribution "
                       "to 1.0.0's time step fails", "cohorts": list(cohorts), "verdicts": verdicts, "fits": out},
              name=args.name)


if __name__ == "__main__":
    main()
