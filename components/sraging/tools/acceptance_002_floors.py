"""Spec 002 FLOORED_DEATHS acceptance (SC-004; research R17 as revised at review T064).

`floors-expect` precomputes, and the result is COMMITTED before `floors` runs, the firing set and categories at
stored theta-hats, by an independent route: the candidate rule re-implemented here from per-term values, and
each candidate's converged term from a graded-start mesh (`graded_start`, a different discretisation from
floors.py's uniform refinement). Cases: the ITP example plus an age-0 death at the example's theta-hat; the
round-off cohort Sweden_F_1910 at its stored theta-hat; spec 001's quick tier at the stored reference records'
theta-hats (where a 'none' claim stores no point, the port's fit supplies theta-hat: byte-identical to the
reference's, spec 001; the theta-hats are written into the committed file).
`floors` then runs the check for real: the ITP-plus-age-0 cohort is fitted through the CLI (one fit, ~25
CPU-min) and its FitResult's FLOORED_DEATHS must list the expected rows with the expected categories (its own
theta-hat differs from the example's; the shift is reported); the other cases through floors.report at their
stored theta-hats. Rows and categories must match exactly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

import acceptance_002 as A
import acceptance_002_t1 as T

EXPECTED = "tests/acceptance/expected_floors_002.json"
ITP_CSV = "tests/data/itp_plus_age0.csv"
ITP_RESULT = "examples/results/itp_c2016_het3_control_female.json"


def _cases():
    from sraging import Cohort
    from engine_cases import panel
    out = []
    d = json.load(open(os.path.join(A.REPO, ITP_RESULT)))
    out.append(("itp_plus_age0", Cohort.from_csv(os.path.join(A.REPO, ITP_CSV), delta=1.0, time_unit="days"),
                {k: d["theta_hat"][k] for k in A.PARAMS}, "days"))
    t7 = json.load(open(os.path.join(A.REPO, "tests", "reference", "data", "t7_mpref.json")))
    key = "pure-real-Sweden_F_1910_homo-c0-s0|stored"
    rho, L = t7["entries"][key]["rho"], t7["provenance"]["cohort_and_grid"][key]["L_days"]
    lib = T.library_case("pure-real-Sweden_F_1910_homo-c0-s0", A.load_toml(T.PANEL_T1))
    out.append(("pure-real-Sweden_F_1910_homo-c0-s0", Cohort.from_arrays(lib["t"], lib["e"], delta=lib["d"]),
                {"eta": rho[0] / L ** 2, "beta": rho[1] / L, "epsilon": rho[2] / L, "kappa": rho[3]}, None))
    for cid in panel()["quick"]:
        lib = T.library_case(cid, A.load_toml(T.PANEL_T1))
        out.append((cid, Cohort.from_arrays(lib["t"], lib["e"], delta=lib["d"]), None, None))
    return out


def _theta_worker(cid):
    """The production theta-hat of a quick-tier cohort (stored claim points, or the port's byte-identical fit)."""
    T._worker_init()
    import acceptance_002_ref as R
    from sraging import Cohort
    A.O.load()
    lib = T.library_case(cid, A.load_toml(T.PANEL_T1))
    return cid, R.stored_theta_hat(cid, Cohort.from_arrays(lib["t"], lib["e"], delta=lib["d"]))


def _quick_thetas(cids, workers):
    import concurrent.futures as cf
    import multiprocessing as mp
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return dict(ex.map(_theta_worker, cids))


def predict(coh, th, time_unit=None):
    """The independent prediction: candidates by the rule, categories from a graded-start reference."""
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy, graded_start
    if any(v is None for v in th.values()):
        return None, "a theta-hat component is not stored (a 'none' claim without a point)"
    kw = {} if time_unit is None else {"time_unit": time_unit}
    p = SRParams(**th, **kw)
    e33, e1 = Evaluator(coh, numerics="v033"), Evaluator(coh, numerics="v1")
    t33, _ = e33.terms(p)
    t1, a1 = e1.terms(p)
    g = e1.grid
    died = np.asarray(coh.events) == 1
    m1 = np.asarray(a1["mass"], float)
    cand = np.flatnonzero(died & ((t33 <= -700 + 1e-9) | (np.abs(t1 - t33) > 1.0) | (m1 < 1e-12)))
    beyond = np.asarray(a1["beyond_march"], bool)
    inside = cand[~beyond[cand]]
    ref = {}
    if inside.size:
        ages = np.asarray(coh.ages, float)[inside]
        d = float(coh.delta)
        sub = Cohort.from_arrays(ages, np.ones(inside.size, int), delta=d, time_unit=coh.time_unit)
        tau_end = float((ages + d).max() / g.L) + 4 * g.dt
        nodes = graded_start(1e-12, 1.02, g.dt / 64, tau_end)
        er = Evaluator(sub, numerics="v1", grid=GridPolicy("fixed", N=g.N, nodes=nodes, tau_max=tau_end, Gtol=g.Gtol,
                                                           L=g.L))
        tr, ar = er.terms(p)
        ref = {int(i): (float(tr[j]), float(ar["mass"][j])) for j, i in enumerate(inside)}
    deaths = []
    for i in cand:
        i = int(i)
        if beyond[i]:
            cat = "horizon"
        else:
            tref, mref = ref[i]
            if mref < 1e-12 or (m1[i] < 1e-12 and abs(tref - t1[i]) <= 1.0):
                cat = "impossible"
            elif abs(tref - t1[i]) > 1.0:
                cat = "grid"
            else:
                cat = "cancellation"
        deaths.append({"row": i + 1, "age": float(coh.ages[i]), "category": cat, "term_v033": float(t33[i]),
                       "term_v1": float(t1[i]), "term_graded": None if beyond[i] else ref[i][0]})
    return deaths, None


def phase_floors_expect(args):
    run = A.Run("floors-expect", [A.PANEL_V1], args.dev, f"~80 CPU-min (32 cohorts; the quick tier's theta-hats "
                f"where the stored records omit a point come from the port's fit, on {args.workers} worker(s))")
    A.preflight_or_exit(run)
    cases = _cases()
    quick = [name for name, _, th, _ in cases if th is None]
    thetas = _quick_thetas(quick, max(1, args.workers))
    out = {}
    for name, coh, th, tu in cases:
        th = thetas.get(name, th)
        deaths, why = predict(coh, th, tu)
        out[name] = {"theta": th, "deaths": deaths, "not_applicable": why}
        print(f"  {name:44s} " + (f"NOT APPLICABLE: {why}" if why else
                                  f"{len(deaths)} listed: {[(d['row'], d['category']) for d in deaths][:8]}"), flush=True)
    path = os.path.join(A.REPO, EXPECTED)
    with open(path, "w") as fh:
        json.dump({"written_at": A.now(), "repo_head": run.head0, "tree_dirty": run.dirty0,
                   "method": __doc__.split("`floors`")[0].strip(), "cases": out}, fh, indent=1)
        fh.write("\n")
    return run.write({"status": "measured", "expected_file": EXPECTED, "n_cases": len(out)})


def phase_floors(args):
    from sraging import floors
    from sraging import SRParams
    from sraging.result import FitResult
    run = A.Run("floors", [EXPECTED, A.PANEL_V1], args.dev, "~30 CPU-min (one CLI fit of the ITP example plus an "
                "age-0 death; floors.report at stored theta-hats on 31 cohorts)")
    A.preflight_or_exit(run)
    exp = json.load(open(os.path.join(A.REPO, EXPECTED)))["cases"]
    res = {}
    # (a) the real fit, through the CLI
    out_json = os.path.join(A.REPO, "var", "floors_itp_plus_age0.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    c0 = time.time()
    r = subprocess.run([sys.executable, "-m", "sraging.cli", "fit-cohort", "--csv", os.path.join(A.REPO, ITP_CSV),
                        "--delta", "1", "--time-unit", "days", "--calibrator", "profile_v033", "--json", out_json],
                       capture_output=True, text=True)
    fit = FitResult.from_json(out_json)
    fr = fit.diagnostics.get("floor_report") or {"deaths": []}
    got = [[d["row"], d["category"]] for d in fr["deaths"]]
    want = [[d["row"], d["category"]] for d in exp["itp_plus_age0"]["deaths"]]
    th_fit = {k: getattr(fit.theta_hat, k) for k in A.PARAMS}
    res["itp_plus_age0"] = {"got": got, "expected": want, "ok": got == want, "cli_exit": r.returncode,
                            "fired": any(c.code == "FLOORED_DEATHS" and c.fired for c in fit.checks),
                            "theta_hat_fit": th_fit, "theta_hat_expected_at": exp["itp_plus_age0"]["theta"],
                            "cli_stdout_tail": r.stdout[-1500:], "wall_s": time.time() - c0}
    print(f"  itp_plus_age0 (CLI fit): got {got} expected {want} -> {'ok' if res['itp_plus_age0']['ok'] else 'FAIL'}",
          flush=True)
    # (b) the stored theta-hats
    for name, coh, _, tu in _cases()[1:]:
        e = exp[name]
        if e["not_applicable"]:
            res[name] = {"ok": None, "not_applicable": e["not_applicable"]}
            continue
        rep = floors.report(coh, SRParams(**e["theta"]))              # the theta-hat committed with the prediction
        got = [[d["row"], d["category"]] for d in rep["deaths"]]
        want = [[d["row"], d["category"]] for d in e["deaths"]]
        res[name] = {"got": got, "expected": want, "ok": got == want, "cost_s": rep["cost_s"]}
        print(f"  {name:44s} {len(got)} listed -> {'ok' if got == want else 'FAIL ' + str(got[:5]) + ' vs ' + str(want[:5])}",
              flush=True)
    judged = [v for v in res.values() if v.get("ok") is not None]
    ok = bool(judged) and all(v["ok"] for v in judged) and res["itp_plus_age0"]["fired"]
    return run.write({"status": "pass" if ok else "fail", "results": res,
                      "n_not_applicable": sum(v.get("ok") is None for v in res.values())})
