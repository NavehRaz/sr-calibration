"""Spec 003g T087 (sr-naveh N13 on DV-3): the free gradient per individual |g|/n at theta-hat on 003b AC-11's small-n
null (library v6, 47 cohorts of n = 150, profile_jax2@1.0.1, combinations off).

DV-3's G_CRIT_PER_IND = 5e-5 was set with two fits on the assessed side (ITP+age0 n = 305, fit 1 n = 10,000). If a
converged fit's |g| is set by the optimiser's tolerance rather than by n, a converged fit of a small cohort crosses it.
This phase measures |g|/n at stored points through the existing leverage check (`sraging.leverage.check` on the 1.0.1
port, its own refits included): no fit, no calibration.

Where theta-hat comes from. AC-11 (`acceptance_003b.ac11_worker`) ran `sraging.fit_cohort(c, "profile_jax2@1.0.1",
combinations=False)` on library v6 and kept the leverage block's label, labels, R, delta_total, g0 and dropped row,
not theta-hat. Its g0 block's `theta` is where the estimator's L-BFGS-B from theta-hat on the full data ended
(`leverage.refit_warm`): exp(clip(log theta-hat)) when that warm start returned its start, which `g0 == 0.0` exactly
records. Only there is theta-hat recovered, and a row counts as measured only when (a) log then exp gives back the
stored point in every coordinate, and (b) the check re-run there reproduces AC-11's stored block bit for bit (g0's g0,
theta, ll_ref and ll_warm; R per set; delta_total; the labels; the dropped row), on AC-11's host, in AC-11's worker
set-up (`acceptance_003b._init101`: single-thread XLA and BLAS, one cohort per spawned process). A row whose warm
start moved (g0 != 0) kept no theta-hat and is not measured. jax2's W3 gate records of the same cohorts are another fit
of the same calibrator and version (WEXAC) and are not used.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import traceback

WORKERS = 4                      # spawned processes at a time (the Mac is shared with other sessions)
LIBRARY = "v6"                   # AC-11's library (ac11_003b.json provenance)
VERSION = "1.0.1"                # AC-11's calibrator version: profile_jax2@1.0.1, PJ.PORTS["1.0.1"]
PARAMS = ("eta", "beta", "epsilon", "kappa")         # sraging.leverage.PARAMS: a non-hazard g0.theta, in this order
CONVERGED_RULE = ("scipy_status == 0: the estimator's L-BFGS-B (_LBFGS_MLE: gtol 1e-6, ftol 1e-13) started at "
                  "theta-hat on the full data reports CONVERGENCE (leverage.warm_termination); the second reading, "
                  "n_fit_ok_above, takes every fit AC-11 returned with status ok as converged")
REPRO_G0 = ("g0", "theta", "ll_ref", "ll_warm")
REPRO = ("labels", "R", "delta_total", "dropped")


def _same(a, b) -> bool:
    """Bit-for-bit equality of JSON values (floats by repr, NaN included)."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------------------------------------------ theta-hat
def recover(row: dict) -> dict:
    """AC-11's own theta-hat for one row, from its g0 block, or why it cannot be recovered."""
    import numpy as np
    g = row.get("g0") or {}
    th = g.get("theta")
    if row.get("status") != "ok":
        return {"ok": False, "reason": f"AC-11's fit status is {row.get('status')!r}"}
    if g.get("status") != "ok" or th is None:
        return {"ok": False, "reason": "AC-11's g0 block failed: no stored point"}
    if g.get("g0") != 0.0:
        return {"ok": False, "reason": f"AC-11 stored no theta-hat; g0.theta is where the warm start from theta-hat "
                                       f"ended, and it moved (g0 = {g.get('g0'):.3g} nats), so theta-hat cannot be "
                                       f"recovered"}
    if len(th) != len(PARAMS):
        return {"ok": False, "reason": f"g0.theta has {len(th)} coordinates, not {len(PARAMS)}"}
    z = [math.log(float(t)) for t in th]                  # leverage._z: math.log of each coordinate
    back = np.exp(np.asarray(z, float))                   # leverage.refit_warm: np.exp of the array
    bad = [p for p, t, b in zip(PARAMS, th, back) if float(b) != float(t)]
    if bad:
        return {"ok": False, "reason": f"log then exp does not give back g0.theta in {bad}"}
    return {"ok": True, "theta": dict(zip(PARAMS, (float(t) for t in th))), "reason": None}


def reproduced(stored: dict, lev: dict) -> dict:
    """Whether the re-run check reproduces AC-11's stored block bit for bit (the evidence theta-hat is the same)."""
    sg, ng = stored.get("g0") or {}, lev.get("g0") or {}
    differs = [f"g0.{k}" for k in REPRO_G0 if not _same(sg.get(k), ng.get(k))]
    differs += [k for k in REPRO if not _same(stored.get(k), lev.get(k))]
    return {"equal": not differs, "differs": differs}


# --------------------------------------------------------------------------------------------------------- worker
def check_worker(cid: str, theta: dict) -> dict:
    """One leverage check at a stored theta-hat, on AC-11's cohort (the v6 observation through sraging's adapter, as
    `acceptance_003b.ac11_worker` built it) and AC-11's port. Plain values only (they cross the process boundary)."""
    try:
        import oracle_jax2_101 as OJ1
        from sraging import leverage as LEV
        from sraging.adapters.observation import cohort_from_observation
        from sraging.calibrators import profile_jax2 as PJ
        from sraging.params import SRParams
        c = cohort_from_observation(OJ1.observation(cid, LIBRARY))
        lev = LEV.check(c, "profile_jax2", SRParams(**theta, time_unit=c.time_unit), {}, library=LIBRARY,
                        port=PJ.PORTS[VERSION])
        a = lev.get("at_theta_hat") or {}
        free = ((lev.get("influence") or {}).get("hessian") or {}).get("free")
        keep = ("label", "labels", "reason", "R", "delta_total", "dropped", "g0", "warm_termination", "cost")
        return {"cohort_id": cid, "n": c.n, **{k: lev.get(k) for k in keep},
                "free_gradient_norm": a.get("free_gradient_norm"),
                "free_gradient_per_individual": a.get("free_gradient_per_individual"),
                "n_free": None if free is None else int(sum(bool(x) for x in free))}
    except Exception as ex:  # noqa: BLE001 -- recorded as not measured
        return {"cohort_id": cid, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-3000:]}


def run_checks(jobs: list, workers: int = WORKERS) -> dict:
    """{cid: check_worker result}, each cohort in its own spawned process set up as AC-11's workers were."""
    import concurrent.futures as cf
    import multiprocessing as mp
    import acceptance_003b as B
    out = {}
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=B._init101,
                                max_tasks_per_child=1) as ex:
        futs = {ex.submit(check_worker, cid, th): cid for cid, th in jobs}
        for i, f in enumerate(cf.as_completed(futs), 1):
            cid = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"cohort_id": cid, "error": f"crash: {type(e).__name__}: {e}"}
            out[cid] = r
            gpi = r.get("free_gradient_per_individual")
            print(f"[lev-small-n] {i}/{len(futs)} {cid}: "
                  + (r["error"][:120] if "error" in r else f"{r.get('label')}, |g|/n = {gpi:.3g}, "
                                                           f"{(r.get('cost') or {}).get('cpu_s', 0):.0f} CPU-s"),
                  flush=True)
    return out


# ------------------------------------------------------------------------------------------------------- the rows
def measured_row(stored: dict, rec: dict, res: dict | None, g_crit: float, provenance: dict) -> dict:
    """One output row: the stored AC-11 facts, then the check's measurement when theta-hat was recovered and the
    re-run reproduced AC-11's block; otherwise not measured, with the reason."""
    row = {"cohort_id": stored["cohort_id"], "n": stored.get("n"), "label_003b": stored.get("label"),
           "fit_status_003b": stored.get("status"), "g0_003b": (stored.get("g0") or {}).get("g0"),
           "measured": False, "not_measured_reason": None, "provenance": provenance}
    if not rec["ok"]:
        row["not_measured_reason"] = rec["reason"]
        return row
    row.update(theta_source="ac11_003b.json g0.theta (g0 == 0.0: the warm start's start, exp(log theta-hat))",
               theta=rec["theta"])
    if res is None or "error" in res:
        row["not_measured_reason"] = f"the check did not run: {(res or {}).get('error', 'no result')}"
        return row
    rep = reproduced(stored, res)
    wt = res.get("warm_termination") or {}
    meas = {"label": res.get("label"), "labels": res.get("labels"), "reason": res.get("reason"),
            "free_gradient_norm": res.get("free_gradient_norm"),
            "free_gradient_per_individual": res.get("free_gradient_per_individual"), "n_free": res.get("n_free"),
            "g0": (res.get("g0") or {}).get("g0"), "warm_termination": res.get("warm_termination"),
            "cost": res.get("cost"), "reproduced": rep}
    if not rep["equal"]:
        row.update(unverified=meas,
                   not_measured_reason=f"the re-run check did not reproduce AC-11's stored block "
                                       f"({', '.join(rep['differs'])}): not shown to be the same point")
        return row
    gpi = meas["free_gradient_per_individual"]
    row.update(meas, measured=True, above=gpi is not None and gpi > g_crit,
               converged=wt.get("scipy_status") == 0 if wt.get("status") == "ok" else None)
    return row


def reference_rows(results_dir: str) -> list:
    """The measured fits DV-3 was set on: lev_pilot_003g.json's rows (ITP+age0 and the 7 human hazard cohorts) and fit
    1 (fit1_003g.json, n = 10,000), as {source, cohort_id, n, label, |g|, |g|/n, scipy_status, message}."""
    out = []
    pilot = json.load(open(os.path.join(results_dir, "lev_pilot_003g.json")))
    for r in pilot["rows"]:
        wt = r.get("warm_termination") or {}
        out.append({"source": "lev_pilot_003g.json", "cohort_id": r["cohort_id"], "n": r["n"], "label": r["label"],
                    "free_gradient_norm": r["free_gradient_norm"],
                    "free_gradient_per_individual": r["free_gradient_per_individual"],
                    "scipy_status": wt.get("scipy_status"), "message": wt.get("message")})
    f1 = json.load(open(os.path.join(results_dir, "fit1_003g.json")))
    lev = f1["fit"]["leverage"]
    g = lev["at_theta_hat"]["free_gradient_norm"]
    wt = lev.get("warm_termination") or {}
    out.append({"source": "fit1_003g.json", "cohort_id": f1["cohort_id"], "n": f1["n"], "label": lev["label"],
                "free_gradient_norm": g, "free_gradient_per_individual": g / f1["n"],
                "scipy_status": wt.get("scipy_status"), "message": wt.get("message")})
    return out


# ---------------------------------------------------------------------------------------------------- the summary
def _dist(v):
    v = sorted(v)
    return {"min": v[0], "median": statistics.median(v), "max": v[-1], "n": len(v)} if v else None


def n_scaling(measured: list, refs: list, g_crit: float) -> dict | None:
    """Is |g| about constant across n (the optimiser's tolerance) or proportional to n (per individual)? The
    log-log slope of |g| against n from the small-n median to each assessed reference (0: constant; 1: per
    individual), and the n below which a fit with the small-n |g| would cross g_crit if |g| did not depend on n."""
    g = sorted(r["free_gradient_norm"] for r in measured if r.get("free_gradient_norm") is not None)
    if not g:
        return None
    n0 = statistics.median(r["n"] for r in measured)
    med = statistics.median(g)
    slopes = {r["cohort_id"]: (math.log(r["free_gradient_norm"] / med) / math.log(r["n"] / n0)
                               if r["n"] != n0 and med > 0 and r["free_gradient_norm"] > 0 else None)
              for r in refs if r["label"] != "not_assessed"}
    return {"n": n0, "median_norm": med, "spread_dex": math.log10(g[-1] / g[0]) if g[0] > 0 else None,
            "slope_to_assessed_reference": slopes,
            "n_cross_if_constant": {"median": med / g_crit, "max": g[-1] / g_crit},
            "reading": "slope 0: |g| set by the optimiser, independent of n; slope 1: |g| proportional to n, so |g|/n "
                       "does not depend on n"}


def summarise(rows: list, g_crit: float, refs: list = ()) -> dict:
    """The phase's summary: the |g|/n distribution over measured rows, the count above g_crit, the count converged
    (CONVERGED_RULE) and nonetheless above, the count above among fits AC-11 returned ok, labels now, and the small-n
    values beside the references."""
    m = [r for r in rows if r.get("measured")]
    above = [r for r in m if r.get("above")]
    conv = [r for r in m if r.get("converged")]
    labels_now, changed = {}, {}
    for r in m:
        labels_now[r["label"]] = labels_now.get(r["label"], 0) + 1
        if r["label"] != r["label_003b"]:
            k = f"{r['label_003b']} -> {r['label']}"
            changed[k] = changed.get(k, 0) + 1
    statuses = {}
    for r in m:
        s = str((r.get("warm_termination") or {}).get("scipy_status"))
        statuses[s] = statuses.get(s, 0) + 1
    beside = sorted([{k: r.get(k) for k in ("source", "cohort_id", "n", "label", "free_gradient_norm",
                                             "free_gradient_per_individual", "scipy_status")} for r in refs]
                    + [{"source": "ac11 small_n", "cohort_id": r["cohort_id"], "n": r["n"], "label": r["label"],
                        "free_gradient_norm": r["free_gradient_norm"],
                        "free_gradient_per_individual": r["free_gradient_per_individual"],
                        "scipy_status": (r.get("warm_termination") or {}).get("scipy_status")} for r in m],
                    key=lambda x: (x["n"], x["free_gradient_per_individual"]))
    return {"status": "written", "n_rows": len(rows), "n_measured": len(m), "n_not_measured": len(rows) - len(m),
            "not_measured": {r["cohort_id"]: r["not_measured_reason"] for r in rows if not r.get("measured")},
            "g_crit_per_ind": g_crit,
            "free_gradient_per_individual": _dist(r["free_gradient_per_individual"] for r in m),
            "free_gradient_norm": _dist(r["free_gradient_norm"] for r in m),
            "n_above": len(above), "above": sorted(r["cohort_id"] for r in above),
            "converged_rule": CONVERGED_RULE, "n_converged": len(conv),
            "n_converged_above": sum(1 for r in conv if r.get("above")),
            "n_fit_ok_above": sum(1 for r in above if r.get("fit_status_003b") == "ok"),
            "scipy_status_counts": statuses, "labels_now": labels_now, "label_changes": changed,
            "n_scaling": n_scaling(m, list(refs), g_crit), "beside": beside}


# ------------------------------------------------------------------------------------------------------ the phase
def phase_lev_small_n(args):
    """T087 (N13): |g|/n at AC-11's own theta-hats on its 47 small-n cohorts (n = 150); gradient evaluations through
    the leverage check, no fits. Writes phase `lev-small-n`."""
    import acceptance_003b as B
    import oracle_jax2_101 as OJ1
    from acceptance_003g import RESULTS_DIR, _run, row_provenance
    from sraging.leverage_rule import G_CRIT_PER_IND
    ac11 = json.load(open(os.path.join(RESULTS_DIR, "ac11_003b.json")))
    stored = [r for r in ac11["rows"] if r.get("mode") == "small_n"]
    want = B.panel()["ac11"]["small_n"]["sha256"]             # AC-11's registered arrays (its phase refused others)
    recs = {}
    for r in stored:
        rec = recover(r)
        if rec["ok"]:
            got = B._file_sha(os.path.join(OJ1.LIBRARY_V6, "cohorts", f"{r['cohort_id']}.npz"))
            if got != want.get(r["cohort_id"]):
                rec = {"ok": False,
                       "reason": f"the v6 array is {str(got)[:12]}, not AC-11's {want.get(r['cohort_id'])}"}
        recs[r["cohort_id"]] = rec
    jobs = [(cid, rec["theta"]) for cid, rec in recs.items() if rec["ok"]]
    run = _run("lev-small-n", args, f"{len(jobs)} leverage checks at AC-11's own theta-hats ({len(stored)} small-n "
                                    f"rows; {len(stored) - len(jobs)} kept no theta-hat), ~30-130 CPU-s each [M: "
                                    f"ac11_003b check_cost, lev_pilot_003g], {WORKERS} at a time; 0 fits")
    res = run_checks(jobs, WORKERS) if jobs else {}
    prov = row_provenance(calibrator="profile_jax2", version=VERSION, library=LIBRARY)
    rows = [measured_row(r, recs[r["cohort_id"]], res.get(r["cohort_id"]), G_CRIT_PER_IND, prov) for r in stored]
    refs = reference_rows(RESULTS_DIR)
    run.write({**summarise(rows, G_CRIT_PER_IND, refs), "rows": rows, "references": refs,
               "source": {"result": "tests/acceptance/results/ac11_003b.json", "repo_head": ac11["repo_head"],
                          "written_at": ac11["written_at"], "host": ac11["host"], "provenance": ac11["provenance"],
                          "fit": f"sraging.fit_cohort(c, 'profile_jax2@{VERSION}', combinations=False) on library "
                                 f"{LIBRARY} (acceptance_003b.ac11_worker)",
                          "theta_hat_stored": False,
                          "recovery": "g0.theta where g0 == 0.0, verified by log/exp round trip and a bit-for-bit "
                                      "reproduction of AC-11's stored check block",
                          "not_used": "jax2's W3 records (jax2-v6-sharp-nladder): another fit of the same calibrator "
                                      "and version, on WEXAC"},
               "workers": WORKERS})


if __name__ == "__main__":
    sys.exit("run it as: .venv/bin/python tools/acceptance_003g.py lev-small-n [--dev]")
