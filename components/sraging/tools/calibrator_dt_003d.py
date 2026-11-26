"""Spec 003d: where does the witness forecast's offset come from, the generator's law or the calibrator's likelihood?

    .venv/bin/python tools/calibrator_dt_003d.py        # needs predict_003d.json, predict_witness_003d.json and their
                                                        # expected-count cohorts in .cache/out/predict_003d/

For each gen1 expected-count cohort of the forecast (the 28 censored configurations and the 12 witness cohorts), the
log-likelihood at the truth and at the calibrator's theta-hat under (a) the generator's own binned law (sraging-gen-1's
grid) and (b) profile_jax2's objective on its production grid, on the same grid with dt / 4 and dt / 16 (N x 1, x2), and on
the grid its own DT_RULE (dt <= CV/80, built and off in 1.0.0) would choose.
The generator's law must rate the truth highest (the cohort is its expected counts); if the calibrator's value at the
truth converges to the generator's as its dt shrinks, the offset is the calibrator's time step. Writes
tests/acceptance/results/calibrator_dt_003d.json. Fit-free: no search, a handful of likelihood evaluations each.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

REFINE = ((1, 1), (1, 4), (2, 4), (2, 16))          # (N factor, dt divisor) on the calibrator's production grid


def generator_ll(row, ages, rho):
    """sum log P(cell) of the recorded ages under the generator's binned law at (rho_eta, rho_beta, rho_eps, kappa),
    the censoring time imposed from the truth's law (the cohort's design)."""
    from sraging import generator as G
    from acceptance_003d import spec_from_library
    s0 = spec_from_library(row)
    import dataclasses
    s = dataclasses.replace(s0, rho_eta=rho[0], rho_beta=rho[1], rho_eps=rho[2], kappa=rho[3])
    C = G.binned_distribution(s0, G.GRID)["C"]
    law = G.binned_distribution(s, G.GRID, None, C)
    return law


def ll_under(law, ages, events, delta):
    k = np.clip(np.rint(ages / delta).astype(int), 0, int((law["events"] == 1).sum()) - 1)
    P = law["P"]
    idx = np.where(events == 1, k, P.size - 1) if law["C"] is not None else k
    with np.errstate(divide="ignore"):
        return float(np.sum(np.log(P[idx])))


def calibrator_lls(ages, events, delta, zs):
    """profile_jax2's objective (its observation model, its grid rule) at each z, on the production grid and REFINE."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import estimator as est
    from sraging.compat.jax2 import fp_jax as fj
    from sraging.compat.jax2 import observation_model as om
    kw = PJ.run_kwargs(PJ.CLASSES["profile_jax2"])
    prep = om.prepare(tuple(float(a) for a in ages), tuple(int(e) for e in events), delta, None, window_on=False,
                      relabel_on=kw["relabel"])
    a, e = np.asarray(prep.ages, float), np.asarray(prep.events, int)
    L, kw0, _ceil, _floor = est._grid(a, e, delta, floor_early=kw["floor_early"], n_rule=kw["n_rule"],
                                      dt_rule=kw["dt_rule"], **({"early_all": True} if kw["early_all"] else {}))
    if kw["exact_from_F"]:
        kw0 = dict(kw0, exact_from_F=True)
    out = []
    for nf, dd in REFINE:
        k = dict(kw0, N=int(kw0["N"] * nf), dt=float(kw0["dt"]) / dd)
        obj = est._Objective(fj, a, e, L, k, est.box_for(300.0))
        out.append({"N": k["N"], "dt": k["dt"], "ll": [-float(obj.nll(z)) for z in zs]})
    # jax2's own DT_RULE (dt <= CV/80; rc2 I2, built and OFF in 1.0.0): the grid its estimator would use with it on
    _L, kr, _c, _f = est._grid(a, e, delta, floor_early=kw["floor_early"], n_rule=kw["n_rule"], dt_rule=True,
                               **({"early_all": True} if kw["early_all"] else {}))
    if kw["exact_from_F"]:
        kr = dict(kr, exact_from_F=True)
    obj = est._Objective(fj, a, e, _L, kr, est.box_for(300.0))
    dt_rule = {"N": kr["N"], "dt": float(kr["dt"]), "ll": [-float(obj.nll(z)) for z in zs]}
    return {"L": float(L), "production": {"N": kw0["N"], "dt": float(kw0["dt"]), "tau_max": kw0["tau_max"]},
            "grids": out, "dt_rule": dt_rule}


def main(argv=None):
    import acceptance_003 as A
    import acceptance_003d as ACC
    import oracle_jax2 as OJ
    import retest_003d as RT
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.v033 import runtime
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args(argv)
    run = A.Run("calibrator-dt", manifests=[ACC.PANEL_PATH], dev=args.dev, suffix="003d",
                cost="40 cohorts x (2 generator laws + 8 calibrator likelihood evaluations); minutes")
    runtime.check()
    pred = json.load(open(os.path.join(ACC.RESULTS_DIR, "predict_003d.json")))
    wit = json.load(open(os.path.join(ACC.RESULTS_DIR, "predict_witness_003d.json")))
    v2 = {s["cohort_id"]: s for s in json.load(open(os.path.join(OJ.LIBRARY_V2, "manifest.json")))["specs"]}
    _m, rows = ACC.library_specs()
    v5 = {r["cohort_id"]: r for r in rows}
    cases = [(c["name"], v2[c["ids"][0]], c["laws"]["gen1"]["fit"]) for c in pred["configurations"]]
    cases += [(cid, v5[cid], w["fit"]) for cid, w in wit["witness_cohorts"].items()]
    out = []
    for name, row, fit in cases:
        z = np.load(os.path.join(ACC.PREDICT_DIR, f"{name}__gen1.npz"))
        ages, events = z["ages"].astype(float), z["events"].astype(int)
        L = float(row["L_phys"])
        truth = RT.physical_truth(row)
        th = fit["theta"]
        rho_t = (row["rho_eta"], row["rho_beta"], row["rho_eps"], row["kappa"])
        rho_h = (th["eta"] * L * L, th["beta"] * L, th["epsilon"] * L, th["kappa"])
        g_t = ll_under(generator_ll(row, ages, rho_t), ages, events, float(row["delta"]))
        g_h = ll_under(generator_ll(row, ages, rho_h), ages, events, float(row["delta"]))
        zs = [np.log([truth[p] for p in PJ.PARAMS]), np.log([th[p] for p in PJ.PARAMS])]
        cal = calibrator_lls(ages, events, float(row["delta"]), zs)
        n = int(ages.size)
        prod = cal["grids"][0]["ll"]
        rec = {"name": name, "n_expected_cohort": n, "generator": {"ll_truth": g_t, "ll_theta_hat": g_h},
               "calibrator": cal,
               "calibrator_deficit_at_truth_per_individual": (g_t - prod[0]) / n,
               "calibrator_deficit_at_theta_hat_per_individual": (g_h - prod[1]) / n,
               "refined_deficit_at_truth_per_individual": (g_t - cal["grids"][-1]["ll"][0]) / n,
               "dt_rule_deficit_at_truth_per_individual": (g_t - cal["dt_rule"]["ll"][0]) / n,
               "generator_prefers_truth": g_t >= g_h}
        out.append(rec)
        print(f"{name:28s} deficit at truth: production {rec['calibrator_deficit_at_truth_per_individual']:+.2e}, "
              f"refined {rec['refined_deficit_at_truth_per_individual']:+.2e} per individual; generator prefers "
              f"truth: {rec['generator_prefers_truth']}", flush=True)
    worst = max(out, key=lambda r: r["calibrator_deficit_at_truth_per_individual"])
    run.write({"status": "written", "cases": out, "refine": [list(r) for r in REFINE],
               "summary": {"n": len(out), "generator_prefers_truth": sum(r["generator_prefers_truth"] for r in out),
                           "worst": {k: worst[k] for k in ("name", "calibrator_deficit_at_truth_per_individual",
                                                           "refined_deficit_at_truth_per_individual")},
                           "max_abs_refined_deficit": max(abs(r["refined_deficit_at_truth_per_individual"])
                                                          for r in out),
                           "max_abs_dt_rule_deficit": max(abs(r["dt_rule_deficit_at_truth_per_individual"])
                                                          for r in out)},
               "read_as": "deficit = generator ll - calibrator ll, per individual of the expected-count cohort; a "
                          "production deficit that vanishes on the refined grid is the calibrator's time step"},
              name="calibrator_dt")


if __name__ == "__main__":
    main()
