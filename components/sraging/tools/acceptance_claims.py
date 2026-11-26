"""The `claims` and `adapter` phases of tools/acceptance.py (tasks T044, T050, T053).

claims  per panel cohort, in ONE worker process holding both the port and the frozen oracle:
        value_bytes / grad_bytes on the 6 theta points; cohort_arrays_direct / _csv (+ the reference
        reader); claims_identity (research R7); warnings (counts per kind vs the reference entry point);
        role_reconfirmed (named cohorts); and, for quick-tier cohorts, the port's srbench record written
        through srbench's own run() into sraging's store, reusing the SAME port fit (no second fit).
adapter the tester's own comparison of those 30 records against the stored reference, plus a
        before/after fingerprint of the tester tree (SC-004).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import tomllib

import acceptance as A
import oracle_v033 as O

CSV_DIR = os.path.join(A.REPO, ".cache", "acceptance_csv")
CHECKS = ("value_bytes", "grad_bytes", "cohort_arrays_direct", "cohort_arrays_csv", "claims_identity",
          "warnings", "role_reconfirmed", "srbench_record", "theta_hat")


def panel_status():
    rel = os.path.relpath(A.PANEL, A.REPO)
    if not os.path.exists(A.PANEL):
        return False, "no panel: run `tools/acceptance.py pilot` first"
    tracked = subprocess.run(["git", "-C", A.REPO, "ls-files", "--error-unmatch", rel], capture_output=True).returncode == 0
    clean = subprocess.run(["git", "-C", A.REPO, "diff", "--quiet", "HEAD", "--", rel], capture_output=True).returncode == 0
    if not (tracked and clean):
        return False, f"{rel} is not committed (FR-021: the panel is frozen in git before the claims run)"
    commit = subprocess.run(["git", "-C", A.REPO, "log", "-1", "--format=%H", "--", rel], capture_output=True,
                            text=True).stdout.strip()
    return True, commit


def warn_counts_ref(o, cs, ages, events, delta):
    return {"basin": len(o.convergence_warnings(cs)),
            "tie": int(o.censoring_tie_warning(ages, events, delta) is not None),
            "grid": int(bool(delta) and o.delta_grid_fraction(ages, delta) > 0)}


def warn_counts_port(fit):
    codes = [w.code for w in fit.warnings]
    return {"basin": codes.count("BASIN_GAIN") + codes.count("BASIN_OUTSIDE"),
            "tie": codes.count("CENSOR_TIE"), "grid": codes.count("DELTA_GRID")}


CHECKED_ROLES = ("dt_ceiling", "basin_fires", "basin_blind", "censor_tie")   # the roles a fit can reconfirm

_SRB = {}   # per worker process: srbench connected and the port's method registered ONCE


def srbench_method(port, cid):
    """This worker's single srbench method, with `port` added as cohort `cid`'s precomputed fit. srbench
    refuses a second method object under the same name, so a worker's second quick-tier cohort must reuse
    the first registration (2026-10-03: both full claims runs broke on it)."""
    if not _SRB:
        from sraging.adapters import srbench as AD
        mods = AD.connect(O.TESTER)                    # SRAGING_TESTER_ROOT, as the oracle tool reads it
        pre = {}
        _SRB.update(AD=AD, mods=mods, pre=pre,
                    method=AD.register_with(mods["contract.method"], AD.make_method(mods["contract.types"], pre)))
    _SRB["pre"][cid] = port
    return _SRB


def claims_worker(item):
    import numpy as np
    from parity_checks import arrays_diff, bytes_diff, identity_diff, oracle_identity_view
    from sraging.adapters.observation import cohort_from_observation
    from sraging.calibrators import profile_v033
    from sraging.cohort import Cohort
    from sraging.compat.v033 import estimator as pest
    from sraging.compat.v033 import fp_jax as pfj

    cid, role, points, quick, store_root = item       # quick: compute the srbench record (False: not applicable)
    t0 = time.perf_counter()
    o = O.load()
    obs = O.observation(cid)
    out = {"cohort_id": cid, "role": role, "checks": {}}
    chk = out["checks"]
    ages, events = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int)
    delta = obs.delta if obs.delta else None

    # cohort arrays, both paths + the reference reader on the same CSV
    cohort = cohort_from_observation(obs)
    chk["cohort_arrays_direct"] = arrays_diff(cohort.ages, cohort.events, ages, events, "direct")
    os.makedirs(CSV_DIR, exist_ok=True)
    p = os.path.join(CSV_DIR, f"{cid}.csv")
    with open(p, "w") as fh:
        fh.write("death times,events\n")
        for a, e in zip(obs.ages, obs.events):
            fh.write(f"{float(a)!r},{int(e)}\n")
    c2 = Cohort.from_csv(p, delta=delta, continuous=delta is None)
    ra, re_ = o.read_csv(p)
    chk["cohort_arrays_csv"] = (arrays_diff(c2.ages, c2.events, ages, events, "csv")
                                + arrays_diff(c2.ages, c2.events, ra, re_, "reference reader"))

    # value / value_and_grad bytes on the 6 points
    pg, og = pest.solver_grid(ages, events, delta), o.estimator.solver_grid(ages, events, delta)
    vb, gb = [] if repr(pg) == repr(og) else ["solver_grid differs"], []

    def mk(fj, g):
        L, kw, _ = g
        return fj.make_loglik(ages, events, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                              sampling_dt=kw.get("sampling_dt"))
    (pf, pvg), (of, ovg) = mk(pfj, pg), mk(o.fp_jax, og)
    values = []
    for pt in points:
        z = np.log(np.array([pt[k] for k in A.PARAMS], float))
        vb += bytes_diff(pf(z), of(z), f"value@{pt['label']}")
        gb += bytes_diff(tuple(pvg(z)), tuple(ovg(z)), f"value_and_grad@{pt['label']}")
        v, g = pvg(z)                             # stored (review N2): the values the bytes were compared on
        values.append({"label": pt["label"], "value": float(pf(z)).hex(), "vg_value": float(v).hex(),
                       "grad": [float(x).hex() for x in np.asarray(g)]})
    chk["value_bytes"], chk["grad_bytes"] = vb, gb
    out["point_values"] = values
    out["n_points"] = len(points)

    # claims identity: the port's fit vs the reference adapter's calibrate(), same process
    port = profile_v033.fit(cohort)
    captured = {}                                 # the reference estimator's own return value (review N3), recorded
    real_fit = o.estimator.fit                    # in memory around the adapter's call; no file is touched

    def recording_fit(*a, **k):
        captured["res"] = real_fit(*a, **k)
        return captured["res"]
    o.estimator.fit = recording_fit
    try:
        ocs = o.method.calibrate(obs)
    finally:
        o.estimator.fit = real_fit
    if "res" not in captured:
        chk["theta_hat"] = ["the reference estimator's theta was not captured"]
    elif port.theta_hat is None:
        chk["theta_hat"] = ["port theta-hat is None"]
    else:
        pt_ = [getattr(port.theta_hat, k) for k in A.PARAMS]
        chk["theta_hat"] = bytes_diff(tuple(pt_), tuple(float(x) for x in captured["res"]["theta"]), "theta_hat")
    chk["claims_identity"] = identity_diff(port.identity_view(), oracle_identity_view(ocs))
    out["port_claims"] = [c.reference_dict() for c in port.claims]
    out["cpu_s"] = {"port": port.timing.get("cpu_s"), "oracle": ocs.cpu_seconds}

    # warnings: per-kind counts equal the reference entry point's
    wp, wr = warn_counts_port(port), warn_counts_ref(o, ocs, ages, events, obs.delta)
    out["warnings"] = {"port": wp, "reference": wr}
    chk["warnings"] = [] if wp == wr else [f"port {wp} != reference {wr}"]

    # role reconfirmed in the side-by-side run (named cohorts)
    rr = []
    if role == "basin_blind" and not (wp["basin"] == 0 and wr["basin"] == 0):
        rr.append(f"basin warning fired (port {wp['basin']}, reference {wr['basin']}) on a basin-blind cohort")
    if role == "dt_ceiling" and not (port.diagnostics.get("dt_ceiling") is not None
                                     and (ocs.diagnostics or {}).get("dt_ceiling") is not None):
        rr.append("dt_ceiling did not apply in both")
    if role == "basin_fires" and not (wp["basin"] > 0 and wr["basin"] > 0):
        rr.append(f"basin warning did not fire in both (port {wp['basin']}, reference {wr['basin']})")
    if role == "censor_tie" and not (wp["tie"] and wr["tie"]):
        rr.append("censor-tie warning did not fire in both")
    if role in CHECKED_ROLES:                  # quick-tier cohorts have no role: not counted as passes
        chk["role_reconfirmed"] = rr
    out["basin_fired"] = {"port": wp["basin"] > 0, "reference": wr["basin"] > 0}
    fr = (port.diagnostics or {}).get("floor_report")       # spec 002's sixth check, outside the identity view
    out["floored_deaths"] = None if fr is None else [[d["row"], d["category"]] for d in fr["deaths"]]

    # the port's srbench record (quick tier), through srbench's own run(), reusing `port`
    if quick:
        s = srbench_method(port, cid)
        man, path = s["AD"].run_records(s["mods"], s["method"], [cid], tester_root=O.TESTER, store_root=store_root,
                                        run_id=f"{os.path.basename(store_root)}/claims-{cid}")
        fresh = man["n_cohorts"] == man["n_computed"] == 1 and man["n_reused"] == man["n_failed"] == 0
        chk["srbench_record"] = [] if fresh else [f"run manifest {man} (the record must be computed in this run)"]
        out["srbench_manifest"] = os.path.relpath(path, A.REPO)
    out["status"] = "ok"
    out["wall_s"] = time.perf_counter() - t0
    import resource                            # macOS ru_maxrss is bytes: the worker's peak so far
    out["worker"] = {"pid": os.getpid(), "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}
    return out


def summarise(rows, expected):
    res = {}
    for name in CHECKS:
        applicable = [r for r in rows if name in r.get("checks", {})]
        fails = [{"cohort_id": r["cohort_id"], "diff": r["checks"][name][:5]} for r in applicable if r["checks"][name]]
        crashed = [r["cohort_id"] for r in rows if r.get("status") != "ok"]
        res[name] = {"passed": len(applicable) - len(fails), "total": len(applicable) + len(crashed),
                     "N": len(applicable) + len(crashed), "failures": fails, "crashed": crashed,
                     "expected_N": expected.get(name)}
    return res


def phase_claims(args):
    name = getattr(args, "result_name", None) or "claims"     # every write goes to this name (spec 003 re-runs it)
    A.preflight_or_exit(name)
    ok, info = panel_status()
    if not ok and not args.only:
        A.write_result(name, {"status": "not_applicable", "reasons": [info]})
        print(f"[claims] NOT APPLICABLE: {info}")
        return 3
    panel = tomllib.load(open(A.PANEL, "rb"))
    quick = list(panel["quick"])
    named = [(n["cohort_id"], n["role"]) for n in panel["named"]]
    points = {t["cohort_id"]: t["points"] for t in panel["theta"]}
    # a fresh srbench store per run: srbench reuses a stored record whose key matches, and the key ignores
    # the git SHA, so records from an earlier (failed) run would otherwise be scored as this run's
    store_root = os.path.join(A.REPO, "var", "srbench_store", "claims-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    srb = not getattr(args, "no_srbench", False)
    items = ([(c, "quick", points[c], srb, store_root) for c in quick]
             + [(c, r, points[c], False, store_root) for c, r in named])
    if args.only:
        items = [it for it in items if it[0] in args.only]
    A.print_cost("claims", [it[0] for it in items], 2, args.workers)
    from sraging.adapters import srbench as AD
    fp0 = AD.fingerprint(O.TESTER)                # the tester tree around the claims phase too (review N5)
    rows = A.run_pool(claims_worker, items, args.workers)
    fp1 = AD.fingerprint(O.TESTER)
    expected = {"theta_hat": len(items), "value_bytes": len(items), "grad_bytes": len(items), "cohort_arrays_direct": len(items),
                "cohort_arrays_csv": len(items), "claims_identity": len(items), "warnings": len(items),
                "role_reconfirmed": sum(it[1] in CHECKED_ROLES for it in items),
                "srbench_record": sum(it[3] for it in items)}
    summary = summarise(rows, expected)
    changed = [f for f in set(fp0[0]) ^ set(fp1[0])]
    summary["tester_tree_unchanged"] = {"passed": int(not changed and fp0[1] == fp1[1]), "total": 1,
                                        "N": len(fp0[0]), "expected_N": 1,       # one run-level comparison
                                        "failures": sorted(x[0] for x in changed)[:20]
                                        + (["git status changed"] if fp0[1] != fp1[1] else []), "crashed": []}
    all_pass = all(s["passed"] == s["total"] == s["expected_N"] for s in summary.values())
    smoke = bool(args.only) or not ok            # a subset is never the acceptance run, panel committed or not
    status = ("smoke-" if smoke else "") + ("pass" if all_pass else "fail")
    extra = {}
    if not srb:
        extra["srbench_record"] = {"status": "not_applicable", "reason": args.no_srbench}
    A.write_result(name,
                   {"status": status, "panel_commit": info if ok else None,
                    "panel_note": None if ok else info, "srbench_store": os.path.relpath(store_root, A.REPO),
                    "checks": summary, "results": rows, **extra})
    print(f"[claims] {status}")
    for name, s in summary.items():
        print(f"  {name:22s} {s['passed']}/{s['total']} (N={s['N']}, expected {s['expected_N']})")
    return 0 if all_pass else 1


def phase_adapter(args):
    A.preflight_or_exit("adapter")
    from parity_checks import tree_diff
    from sraging.adapters import srbench as AD
    tester = O.TESTER
    print("[adapter] fingerprinting the tester tree (read-only) ...", flush=True)
    fp0 = AD.fingerprint(tester)
    mods = AD.connect(tester)
    claims_res = json.load(open(os.path.join(A.RESULTS, "claims.json")))
    quick = [r["cohort_id"] for r in claims_res["results"] if r.get("srbench_manifest")]
    if args.only:
        quick = [c for c in quick if c in args.only]
    paths = {r["cohort_id"]: r["srbench_manifest"] for r in claims_res["results"] if r.get("srbench_manifest")}
    mans = [json.load(open(os.path.join(A.REPO, paths[c]))) for c in quick]
    port_manifest = dict(mans[0], cohort_ids=quick, keys={c: m["keys"][c] for c, m in zip(quick, mans)},
                         n_cohorts=len(quick))
    out, a, b = AD.compare(mods, port_manifest, quick, tester,
                           port_store_root=os.path.join(A.REPO, claims_res.get("srbench_store", "var/srbench_store")))
    per = []
    by_b = {cs.cohort_id: cs for cs in b}
    for cs in a:
        fields = ("parameter", "shape", "lower", "upper", "point", "level", "n_components")
        pa = [{k: getattr(c, k) for k in fields} for c in cs.claims]
        pb = [{k: getattr(c, k) for k in fields} for c in by_b[cs.cohort_id].claims]
        per.append({"cohort_id": cs.cohort_id, "diff": tree_diff(pa, pb)})
    claim_fail = [p for p in per if p["diff"]]
    axis_fail, cov_fail = [], []
    for p, d in out["parameters"].items():
        for ax, r in d["axes"].items():
            if ax != "cost" and r["outcome"] not in ("no difference", "not measured"):
                axis_fail.append(f"{p}.{ax}: {r['outcome']}")
        if d["coverage"]["port"] != d["coverage"]["reference"]:
            cov_fail.append(p)
    fp1 = AD.fingerprint(tester)
    changed = sorted(set(fp0[0]) ^ set(fp1[0]))
    git_changed = fp0[1] != fp1[1]
    checks = {
        "adapter_claims_vs_stored": {"passed": len(per) - len(claim_fail), "total": len(per), "N": len(per),
                                     "failures": claim_fail[:10]},
        "adapter_axes_no_difference": {"passed": int(not axis_fail and not cov_fail), "total": 1,
                                       "N": out["N"], "failures": axis_fail + [f"coverage differs: {c}" for c in cov_fail]},
        "tester_tree_unchanged": {"passed": int(not changed and not git_changed), "total": 1, "N": len(fp0[0]),
                                  "failures": [f"{x[0]}" for x in changed[:20]] + (["git status changed"] if git_changed else [])},
    }
    all_pass = (all(c["passed"] == c["total"] for c in checks.values()) and len(per) == len(quick)
                and (bool(args.only) or len(quick) == 30))
    status = ("smoke-" if args.only else "") + ("pass" if all_pass else "fail")
    A.write_result("adapter", {"status": status, "checks": checks, "comparison": out})
    print(f"[adapter] {status}")
    for k, c in checks.items():
        print(f"  {k:28s} {c['passed']}/{c['total']} (N={c['N']})")
    for p, d in out["parameters"].items():
        print(f"  {p:8s} " + "  ".join(f"{ax}={r['outcome']}" for ax, r in d["axes"].items()))
    return 0 if all_pass else 1
