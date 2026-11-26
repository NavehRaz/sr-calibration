"""Spec 003d's acceptance phases (specs/003d-cohort-generator/contracts/acceptance.md), fit-free up to the tag.

    .venv/bin/python tools/acceptance_003d.py sweep --pilot     # the 30 lowest-rho_eps points: prices the full sweep
    .venv/bin/python tools/acceptance_003d.py sweep             # SC-001: fixes the version's grid (R3, R6)

Every phase uses spec 003's `acceptance_003.Run` (suffix "003d"): it refuses a dirty tree or an uncommitted panel
unless --dev, prints its price first, and writes tests/acceptance/results/<phase>_003d.json. The tester's library v5
is read-only (manifest and arrays); its code is never imported.

The sweep (R3). For each library configuration (the generator's `configuration_key`: point, delta/L, censoring, h L,
heterogeneity) and grid rung (N, dt), lambda = 2 n_max KL(P_gen || P_ref) and dlnCV of the binned law against the
reference grid (2N, dt/2, 2K), n_max being the configuration's largest library n. Phase A searches the rungs in
ascending cost for every point's homogeneous configurations (a point stops at its first passing rung); the version's
(N, dt) is the highest of those minima. Phase B evaluates every configuration at that (N, dt), the heterogeneous ones
up the K ladder until all pass. If no rung up to the ladder's top passes, the sweep stops and reports (plan D2).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
import tomllib

import numpy as np
from scipy.stats import chi2, kstest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

from v5_reproduce_003d import V5, cohort_digest  # noqa: E402,F401  (the tester's checksum, shared with M4)

PANEL_PATH = "tests/acceptance/panel_003d.toml"
PANEL = tomllib.load(open(os.path.join(REPO, PANEL_PATH), "rb"))
TH = PANEL["thresholds"]
_DROP = ("cohort_id", "library_version", "group", "position", "arm", "stratum", "cluster_key", "real_name", "notes",
         "window")


# ----------------------------------------------------------------------------------------------- library driver
def library_specs():
    man = json.load(open(os.path.join(V5, "manifest.json")))
    return man, man["specs"]


def spec_from_library(row: dict, window: bool = False):
    """A generator spec from a v5 manifest row; the tester's organisational fields and the window are dropped (the
    window is passed as time_range only when asked, for phase `window`)."""
    from sraging.generator import CohortSpec
    kw = {"name": row["cohort_id"], **{k: row[k] for k in ("rho_eta", "rho_beta", "rho_eps", "kappa", "n", "L_phys",
                                                           "seed", "delta")},
          "censor_frac": row.get("censor_frac") or 0.0}
    if row.get("external_hazard") is not None:
        kw["external_hazard"] = row["external_hazard"]
    if row.get("het_param") and (row.get("het_var") or 0) > 0:
        kw["het_param"], kw["het_var"] = row["het_param"], row["het_var"]
    if window and row.get("window") is not None:
        kw["time_range"] = tuple(row["window"])
    return CohortSpec(**kw)


# ----------------------------------------------------------------------------------------------------- statistics
def dlncv_bound(n: int) -> float:
    return 0.5 * math.sqrt(TH["dlncv_chi"] / (2.0 * n))


def n_eval(n_max: int, heterogeneous: bool) -> int:
    """The n a configuration is judged at: its largest library n, capped for heterogeneous configurations at
    het_n_cap (Naveh, 2026-10-13; spec SC-001)."""
    return min(n_max, TH["het_n_cap"]) if heterogeneous else n_max


def passes_sc001(kl: float, dlncv: float, n: int) -> bool:
    """SC-001: lambda = 2 n KL <= lambda_max, and 4/3 of the refinement difference in lnCV within the bound."""
    return 2.0 * n * kl <= TH["lambda_max"] and TH["dlncv_refinement_factor"] * abs(dlncv) <= dlncv_bound(n)


def g_test(obs, exp, min_expected=None):
    """G statistic, df and p, cells merged left to right until each expected count is >= min_expected (the
    remainder joins the last merged cell)."""
    min_expected = TH["gtest_min_expected"] if min_expected is None else min_expected
    O, E, o, e = [], [], 0.0, 0.0
    for oi, ei in zip(np.asarray(obs, float), np.asarray(exp, float)):
        o, e = o + oi, e + ei
        if e >= min_expected:
            O.append(o), E.append(e)
            o = e = 0.0
    if e > 0 or o > 0:
        if E:
            O[-1] += o
            E[-1] += e
        else:
            O.append(o), E.append(e)
    O, E = np.array(O), np.array(E)
    m = O > 0
    g = float(2.0 * np.sum(O[m] * np.log(O[m] / E[m])))
    df = max(O.size - 1, 1)
    return g, df, float(chi2.sf(g, df))


def williams_p(g: float, df: int, n: float) -> float:
    """The G-test's p with Williams' correction for k = df + 1 cells: G / q, q = 1 + (k + 1) / (6 n)."""
    q = 1.0 + (df + 2) / (6.0 * n)
    return float(chi2.sf(g / q, df))


def adopted_statistic(raw: dict, null_raw: dict) -> str:
    """Naveh 2026-10-14 (SC-003): the plain G-test if every group passes; Williams' correction if every group the plain
    test fails also fails on exact draws from its own law (the null control: the test, not the generator); else a
    real failure."""
    failing = [k for k, v in raw.items() if not v["pass"]]
    if not failing:
        return "raw"
    return "williams" if all(not null_raw[k]["pass"] for k in failing) else "real-fail"


def fraction_bound(m: int) -> float:
    a = TH["gtest_alpha"]
    return a + 3.0 * math.sqrt(a * (1.0 - a) / m)


def group_passes(pvals) -> dict:
    p = np.asarray(pvals, float)
    ks = float(kstest(p, "uniform").pvalue)
    frac = float((p < TH["gtest_alpha"]).mean())
    return {"m": int(p.size), "ks_p": ks, "frac_below_alpha": frac, "frac_bound": fraction_bound(p.size),
            "pass": ks >= TH["gtest_ks_p_min"] and frac <= fraction_bound(p.size)}


# ---------------------------------------------------------------------------------------------------------- sweep
def configurations(rows):
    """{key: {"spec": a representative row's spec (n = the configuration's largest n), "ids": [...]}}."""
    from sraging import generator as G
    out = {}
    for r in rows:
        s = spec_from_library(r)
        k = json.dumps(G.configuration_key(s))
        c = out.setdefault(k, {"spec": s, "ids": [], "n_max": 0})
        c["ids"].append(r["cohort_id"])
        if s.n > c["n_max"]:
            c["n_max"], c["spec"] = s.n, s
    return out


def rungs():
    g = PANEL["grid"]
    rs = [(N, dt) for N in g["N_ladder"] for dt in sorted(g["dt_ladder"], reverse=True)]
    return sorted(rs, key=lambda r: r[0] / r[1])          # ascending cost: cells x steps


def _grid(N, dt, K):
    g = PANEL["grid"]
    return {"N": int(N), "dt": float(dt), "tau_max": float(g["tau_max"]), "Gtol": float(g["Gtol"]),
            "nstart": int(g["nstart"]), "K": int(K)}


def _init():
    from sraging.compat.v033 import runtime
    runtime.configure()


def _eval(specs, N, dt, K):
    """Each spec's KL and dlnCV at (N, dt, K) against the reference grid, models shared across specs of a point."""
    from sraging import generator as G
    g = _grid(N, dt, K)
    gr = G.reference_grid(g)
    c0 = time.process_time()
    cache = {}
    out = []
    for s in specs:
        mk = (s.heterogeneous, s.het_var)
        if mk not in cache:
            cache[mk] = (G._model(s, g), G._model(s, gr))
        mg, mr = cache[mk]
        bg = G.binned_distribution(s, g, mg)
        br = G.binned_distribution(s, gr, mr, bg["C"])          # the reference under the generator's own C
        n = min(bg["P"].size, br["P"].size)
        Pg = np.append(bg["P"][: n - 1], bg["P"][n - 1:].sum())
        Pr = np.append(br["P"][: n - 1], br["P"][n - 1:].sum())
        kl_, dl = G.kl_binned(Pg, Pr, s.n), math.log(G.binned_cv(bg) / G.binned_cv(br))   # s.n: the n_max
        out.append({"kl": kl_, "dlncv": dl, "max_abs_S_plus_F_minus_1": mg["max_abs_S_plus_F_minus_1"]})
    return out, time.process_time() - c0


def search_point(point_key, items, rung_list, K0):
    """Phase A for one point's homogeneous configurations: ascending rungs until all pass."""
    try:
        specs = [it["spec"] for it in items]
        trail = []
        for N, dt in rung_list:
            res, cpu = _eval(specs, N, dt, K0)
            ok = [passes_sc001(r["kl"], r["dlncv"], it["n_max"]) for r, it in zip(res, items)]
            trail.append({"N": N, "dt": dt, "cpu_s": cpu, "all_pass": all(ok),
                          "worst_lambda": max(2 * it["n_max"] * r["kl"] for r, it in zip(res, items)),
                          "worst_dlncv_ratio": max(TH["dlncv_refinement_factor"] * abs(r["dlncv"]) /
                                                   dlncv_bound(it["n_max"]) for r, it in zip(res, items))})
            if all(ok):
                return {"point": point_key, "min_rung": [N, dt], "trail": trail}
        return {"point": point_key, "min_rung": None, "trail": trail}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"point": point_key, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def verify_configs(keys, items, N, dt, K):
    """Phase B: every configuration at the version's (N, dt) and K, with its pass and numbers."""
    try:
        res, cpu = _eval([it["spec"] for it in items], N, dt, K)
        rows = []
        for k, r, it in zip(keys, res, items):
            ne = n_eval(it["n_max"], it["spec"].heterogeneous)
            rows.append({"key": json.loads(k), "n_max": it["n_max"], "n_eval": ne, "ids": it["ids"][:3],
                         "n_ids": len(it["ids"]), "kl": r["kl"], "dlncv": r["dlncv"], "lambda": 2 * ne * r["kl"],
                         "lambda_at_n_max": 2 * it["n_max"] * r["kl"], "pass": passes_sc001(r["kl"], r["dlncv"], ne),
                         "max_abs_S_plus_F_minus_1": r["max_abs_S_plus_F_minus_1"]})
        return {"cpu_s": cpu, "rows": rows}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:], "keys": keys}


def _pool(fn, jobs, workers):
    out = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for f in cf.as_completed(futs):
            out.append(f.result())
            print(f"  [sweep] {len(out)}/{len(jobs)}", flush=True)
    return out


NODE_PART_PATH = "tests/acceptance/results/het_nodes_all_003d.json"


def combine_het(node: dict, grid: dict) -> dict:
    """The split heterogeneous check (Naveh 2026-10-14): KL_total <= 2 (KL_node + KL_grid), a second-order bound
    for small errors; the lnCV differences add (|.| each)."""
    return {"kl": 2.0 * (node["kl"] + grid["kl"]), "dlncv": abs(node["dlncv"]) + abs(grid["dlncv"])}


def passes_het(kl: float, dlncv: float, n_eval: int, n_max: int) -> bool:
    """lambda = 2 n_eval KL <= het_lambda_max (n_eval = min(n, het_n_cap)), and the tester's own CV rule
    2 n dlnCV^2 <= dlncv_chi at the configuration's real (largest) n."""
    return 2.0 * n_eval * kl <= TH["het_lambda_max"] and 2.0 * n_max * dlncv * dlncv <= TH["dlncv_chi"]


def node_part(path: str = None) -> dict:
    """{(rho_eta, rho_beta, rho_eps, kappa, delta/L): {K: {kl, dlncv}}} from tools/het_nodes_003d.py --all."""
    d = json.load(open(path or os.path.join(REPO, NODE_PART_PATH)))
    out = {}
    for p in d["points"]:
        for c in p.get("configs", []):
            key = tuple(p["point"]) + (c["delta_over_L"],)
            out[key] = {r["K"]: {"kl": r["kl"], "dlncv": r["dlncv"]} for r in c["rows"]}
    return out


def het_grid_point(keys, items, N, dt):
    """The grid part for one heterogeneous point: a het_grid_nodes-node mixture's binned law at (N, dt) against
    (2N, dt/2), same nodes, the reference under the generator's own censoring time."""
    try:
        from sraging import generator as G
        K = int(TH["het_grid_nodes"])
        g, gr = _grid(N, dt, K), _grid(2 * N, dt / 2.0, K)
        c0 = time.process_time()
        s0 = items[0]["spec"]
        mg, mr = G._model(s0, g), G._model(s0, gr)
        rows = []
        for k, it in zip(keys, items):
            s = it["spec"]
            bg = G.binned_distribution(s, g, mg)
            br = G.binned_distribution(s, gr, mr, bg["C"])
            n = min(bg["P"].size, br["P"].size)
            Pg = np.append(bg["P"][: n - 1], bg["P"][n - 1:].sum())
            Pr = np.append(br["P"][: n - 1], br["P"][n - 1:].sum())
            rows.append({"key": json.loads(k), "ids": it["ids"][:3], "n_ids": len(it["ids"]), "n_max": it["n_max"],
                         "grid_kl": G.kl_binned(Pg, Pr, n_eval(it["n_max"], True)),
                         "grid_dlncv": math.log(G.binned_cv(bg) / G.binned_cv(br))})
        return {"cpu_s": time.process_time() - c0, "rows": rows}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:], "keys": keys}


def _save_partial(state):
    """Every phase and every finished heterogeneous point is saved at once (.cache/out, untracked): a stopped run
    keeps what it measured (the 2026-10-14 full sweep lost its K 161 round)."""
    path = os.path.join(REPO, ".cache", "out", f"sweep_003d_partial{'_pilot' if state.get('pilot') else ''}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(state, fh, default=float)
    os.replace(path + ".tmp", path)


def _pool_each(fn, jobs, workers, on_result):
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = [ex.submit(fn, *j) for j in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            on_result(f.result())
            print(f"  [sweep] {i}/{len(jobs)}", flush=True)


def sweep(args):
    import acceptance_003 as A
    man, rows = library_specs()
    confs = configurations(rows)
    by_point = {}
    for k, c in confs.items():
        s = c["spec"]
        by_point.setdefault((s.rho_eta, s.rho_beta, s.rho_eps, s.kappa), []).append((k, c))
    points = sorted(by_point, key=lambda p: p[2])                        # lowest rho_eps first
    het_points = [p for p in points if any(c["spec"].heterogeneous for _, c in by_point[p])]
    if args.pilot:
        points = points[: PANEL["grid"]["lowest_rho_eps_first"]]
        het_points = het_points[:: max(1, len(het_points) // TH["het_pilot_points"])][: TH["het_pilot_points"]]
    nodes = node_part()
    K_version = int(PANEL["grid"]["K_start"])
    K0 = PANEL["grid"]["K_ladder"][0]
    run = A.Run("sweep", manifests=[PANEL_PATH, NODE_PART_PATH], dev=args.dev, suffix="003d",
                cost=f"{'PILOT: ' if args.pilot else ''}{len(points)} points' homogeneous configurations up the rungs "
                     f"{rungs()} then at the chosen rung; {len(het_points)} heterogeneous points' grid part "
                     f"({TH['het_grid_nodes']}-node mixtures at (N, dt) vs (2N, dt/2)); node part from {NODE_PART_PATH} "
                     f"at K {K_version}; {args.workers} workers; saved after every phase and point")
    t0 = time.time()
    state = {"pilot": args.pilot, "phase_a": [], "phase_b": [], "het_grid": [], "started": time.time()}
    jobs = [(list(p), [c for k, c in by_point[p] if not c["spec"].heterogeneous], rungs(), K0)
            for p in points if any(not c["spec"].heterogeneous for _, c in by_point[p])]

    def keep(name):
        def f(r):
            state[name].append(r)
            _save_partial(state)
        return f
    _pool_each(search_point, jobs, args.workers, keep("phase_a"))
    errors = [r for r in state["phase_a"] if "error" in r]
    order = {tuple(r): i for i, r in enumerate(rungs())}
    mins = [r["min_rung"] for r in state["phase_a"] if "error" not in r]
    if errors or any(m is None for m in mins):
        run.write({"status": "no-rung" if not errors else "errors", **state}, name="sweep_pilot" if args.pilot else "sweep")
        return
    N, dt = max(mins, key=lambda r: order[tuple(r)])
    state["N"], state["dt"] = N, dt
    hom_jobs = []
    for p in points:
        hk = [(k, c) for k, c in by_point[p] if not c["spec"].heterogeneous]
        if hk:
            hom_jobs.append(([k for k, _ in hk], [c for _, c in hk], N, dt, K0))
    _pool_each(verify_configs, hom_jobs, args.workers, keep("phase_b"))
    het_jobs = [([k for k, c in by_point[p] if c["spec"].heterogeneous],
                 [c for k, c in by_point[p] if c["spec"].heterogeneous], N, dt) for p in het_points]
    _pool_each(het_grid_point, het_jobs, args.workers, keep("het_grid"))
    rows_b = [r for x in state["phase_b"] for r in x.get("rows", [])]
    het_rows, missing = [], []
    for x in state["het_grid"]:
        for r in x.get("rows", []):
            key = tuple(r["key"][:4]) + (r["key"][4],)
            nd = nodes.get(key, {}).get(K_version)
            if nd is None:
                missing.append(r["ids"][0])
                continue
            tot = combine_het(nd, {"kl": r["grid_kl"], "dlncv": r["grid_dlncv"]})
            ne = n_eval(r["n_max"], True)
            het_rows.append({**r, "node_kl": nd["kl"], "node_dlncv": nd["dlncv"], "kl": tot["kl"],
                             "dlncv": tot["dlncv"], "n_eval": ne, "lambda": 2 * ne * tot["kl"],
                             "cv_rule": 2 * r["n_max"] * tot["dlncv"] ** 2,
                             "pass": passes_het(tot["kl"], tot["dlncv"], ne, r["n_max"])})
    errs = [x for x in state["phase_b"] + state["het_grid"] if "error" in x]
    ok = not errs and not missing and all(r["pass"] for r in rows_b) and all(r["pass"] for r in het_rows)
    cpu = {"phase_a": sum(t["cpu_s"] for r in state["phase_a"] for t in r.get("trail", [])),
           "phase_b": sum(x.get("cpu_s", 0) for x in state["phase_b"]),
           "het_grid": sum(x.get("cpu_s", 0) for x in state["het_grid"])}
    out = {"status": "pass" if ok else "fail", "pilot": args.pilot, "choice": {"N": N, "dt": dt, "K": K_version} if ok
           else None, "n_points": len(points), "n_het_points": len(het_points), "phase_a": state["phase_a"],
           "rows": rows_b, "het_rows": het_rows, "missing_node_part": missing, "errors": errs, "cpu_s": cpu,
           "wall_s_total": time.time() - t0, "rungs": rungs(), "K_version": K_version,
           "rule": {"homogeneous": "lambda <= lambda_max at n_max and 4/3 |dlnCV| <= 1/2 sqrt(0.1/(2 n_max))",
                    "heterogeneous": "split: KL_total = 2 (KL_node + KL_grid), |dlnCV| adds; lambda = 2 min(n, "
                                     "het_n_cap) KL_total <= het_lambda_max; 2 n_max dlnCV^2 <= 0.1 (the tester's rule)"},
           "v5": {"manifest_sha256": hashlib.sha256(open(os.path.join(V5, "manifest.json"), "rb").read()).hexdigest()}}
    if args.pilot:
        n_het_all = len({p for p in by_point if any(c["spec"].heterogeneous for _, c in by_point[p])})
        out["price_full_cpu_h_inferred"] = ((cpu["phase_a"] + cpu["phase_b"]) / len(points) * len(by_point)
                                            + cpu["het_grid"] / max(len(het_points), 1) * n_het_all) / 3600
        out["price_basis"] = (f"the pilot's CPU per point: homogeneous phases over its {len(points)} lowest-rho_eps "
                              f"points, the heterogeneous grid part over {len(het_points)} points spread over the "
                              f"heterogeneous points' rho_eps order; times {len(by_point)} and {n_het_all} (INFERRED)")
    run.write(out, name="sweep_pilot" if args.pilot else "sweep")
    if not args.pilot and ok:
        table = {"generator_version_for": "sraging-gen-1", "grid": {**_grid(N, dt, K_version)},
                 "het_n_cap": TH["het_n_cap"],
                 "rows": [{"key": r["key"], "kl": r["kl"], "dlncv": r["dlncv"], "n_max": r["n_max"],
                           "lambda_at_n_max": 2 * r["n_max"] * r["kl"]} for r in rows_b + het_rows]}
        run.write({"status": "written", **table}, name="point_table")


# ------------------------------------------------------------------------------------------------------- fidelity
def ln_sample_cv(ages) -> float:
    a = np.asarray(ages, float)
    return math.log(float(a.std()) / float(a.mean()))


def fidelity_passes(lncvs, lncv_ref) -> bool:
    """SC-002: the replicates' mean ln CV of recorded ages within fidelity_max_abs_diff of the binned reference."""
    return abs(float(np.mean(lncvs)) - lncv_ref) <= TH["fidelity_max_abs_diff"]


def fidelity_point(name, row, seeds):
    """One named point: the generator's replicates and the backward-Euler control's (same seeds, SRK's grid), each
    against the reference grid's binned ln CV at the point's delta."""
    try:
        import dataclasses
        from sraging import generator as G
        import v5_reproduce_003d as R
        s0 = dataclasses.replace(spec_from_library(row), n=TH["fidelity_n"])
        lncv_ref = math.log(G.binned_cv(G.binned_distribution(s0, G.reference_grid(G.GRID))))
        c0 = time.process_time()
        gen = [ln_sample_cv(G.materialize(dataclasses.replace(s0, seed=sd)).death_times) for sd in seeds]
        tau, S = R.curve(s0.rho_eta, s0.rho_beta, s0.rho_eps, s0.kappa)
        base = {"n": s0.n, "L_phys": s0.L_phys, "delta": s0.delta, "censor_frac": 0.0, "external_hazard": None}
        ctrl = [ln_sample_cv(R.materialize_srk({**base, "seed": sd}, tau, S)[0]) for sd in seeds]
        return {"point": name, "source": row["cohort_id"], "lncv_ref": lncv_ref,
                "gen_mean": float(np.mean(gen)), "gen_sd": float(np.std(gen, ddof=1)),
                "gen_diff": float(np.mean(gen)) - lncv_ref, "gen_pass": fidelity_passes(gen, lncv_ref),
                "control_mean": float(np.mean(ctrl)), "control_diff": float(np.mean(ctrl)) - lncv_ref,
                "control_pass": fidelity_passes(ctrl, lncv_ref), "cpu_s": time.process_time() - c0}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"point": name, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def fidelity(args):
    import acceptance_003 as A
    man, rows = library_specs()
    by_id = {r["cohort_id"]: r for r in rows}
    f = PANEL["fidelity"]
    seeds = [f["seed0"] + i for i in range(1, TH["fidelity_replicates"] + 1)]
    run = A.Run("fidelity", manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"{len(f['points'])} points x {len(seeds)} replicates at n {TH['fidelity_n']}, generator and "
                     f"backward-Euler control; one reference-grid law per point; {args.workers} workers")
    res = []
    _pool_each(fidelity_point, [(nm, by_id[src], seeds) for nm, src in zip(f["points"], f["source"])],
               min(args.workers, len(f["points"])), res.append)
    errs = [r for r in res if "error" in r]
    sharp, broad = set(f["sharp"]), f["broad_control"]
    gen_ok = not errs and all(r["gen_pass"] for r in res)
    ctrl_ok = not errs and all((not r["control_pass"]) if r["point"] in sharp else r["control_pass"]
                               for r in res if r["point"] in sharp or r["point"] == broad)
    run.write({"status": "pass" if gen_ok and ctrl_ok else "fail", "generator_pass": gen_ok,
               "positive_control_as_expected": ctrl_ok, "points": sorted(res, key=lambda r: r["point"]),
               "rule": "|mean ln CV(recorded ages) - ln CV(binned reference)| <= fidelity_max_abs_diff; the "
                       "backward-Euler control must fail at the sharp points and pass at the broad control",
               "grid": __import__("sraging.generator", fromlist=["GRID"]).GRID}, name="fidelity")


# --------------------------------------------------------------------------------------- G-test and determinism
def cell_counts(ages, events, law, delta):
    """Counts of recorded (age, event) on the law's cells: deaths at k delta in cell k, censored rows in the last."""
    counts = np.zeros(law["ages"].size, float)
    a, e = np.asarray(ages, float), np.asarray(events, int)
    d = a[e == 1]
    k = np.rint(d / delta).astype(int)
    nd = int((law["events"] == 1).sum())
    np.add.at(counts, np.clip(k, 0, nd - 1), 1.0)
    if law["C"] is not None:
        counts[-1] += float((e == 0).sum())
    return counts


def recording_checks(ages, events, delta, C):
    a, e = np.asarray(ages, float), np.asarray(events, int)
    r = a / delta
    out = {"on_grid": bool(np.all(np.abs(r - np.rint(r)) <= 1e-9 * np.maximum(1.0, np.abs(r)))),
           "no_age_in_first_bin_but_zero": bool(not np.any((a > 0) & (a < delta)))}
    if C is not None:
        out["no_death_at_C"] = bool(not np.any((a == C) & (e == 1)))
        out["censored_at_C"] = bool(np.all(a[e == 0] == C))
    return out


def library_point(rows, mode):
    """Every library cohort at one point: generated, checksummed and (mode "gtest") G-tested against its law (the
    reference grid for homogeneous cohorts, the generator's own law for heterogeneous ones, Naveh 2026-10-14), with
    SC-004's recording checks."""
    try:
        from sraging import generator as G
        c0 = time.process_time()
        out, ref_models = [], {}
        for row in rows:
            s = spec_from_library(row)
            co = G.materialize(s)
            rec = {"cohort_id": row["cohort_id"], "arm": row["arm"], "group": row.get("group"), "n": s.n,
                   "heterogeneous": s.heterogeneous, "digest": cohort_digest(co.death_times, co.events)}
            if mode == "gtest":
                if s.heterogeneous:
                    law = G.binned_distribution(s, G.GRID, G._model(s, G.GRID))
                    law_src = "generator"
                else:
                    if "ref" not in ref_models:
                        ref_models["ref"] = G._model(s, G.reference_grid(G.GRID))
                    C = G.binned_distribution(s, G.GRID, G._model(s, G.GRID))["C"]
                    law = G.binned_distribution(s, G.reference_grid(G.GRID), ref_models["ref"], C)
                    law_src = "reference"
                obs = cell_counts(co.death_times, co.events, law, s.delta)
                expct = s.n * law["P"] / law["P"].sum()
                g, df, pv = g_test(obs, expct)
                # the null control: exact multinomial draws from the same law, the same test (Naveh 2026-10-14)
                null = np.random.default_rng([s.seed, 20261014]).multinomial(s.n, law["P"] / law["P"].sum())
                gn, dfn, pn = g_test(null, expct)
                rec.update({"G": g, "df": df, "p": pv, "p_williams": williams_p(g, df, s.n),
                            "null_p": pn, "null_p_williams": williams_p(gn, dfn, s.n), "law": law_src,
                            "checks": recording_checks(co.death_times, co.events, s.delta, co.censor_time),
                            "censor_frac": s.censor_frac, "realised_censored_fraction": co.realised_censored_fraction,
                            "n_counted": float(obs.sum())})
            out.append(rec)
        return {"cpu_s": time.process_time() - c0, "rows": out}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:],
                "ids": [r["cohort_id"] for r in rows]}


def v5_control_point(rows):
    """The positive control: v5's stored arrays (read-only) G-tested against the reference law."""
    try:
        from sraging import generator as G
        out, ref = [], None
        for row in rows:
            s = spec_from_library(row)
            if ref is None:
                ref = G._model(s, G.reference_grid(G.GRID))
            z = np.load(os.path.join(V5, "cohorts", f"{row['cohort_id']}.npz"))
            ages, ev = z["ages"].astype(float), z["events"].astype(int)
            C = float(ages[ev == 0][0]) if s.censor_frac > 0 and (ev == 0).any() else None
            if s.censor_frac > 0:            # v5 censors at the sample quantile, floored: compare its deaths' law only
                continue
            law = G.binned_distribution(s, G.reference_grid(G.GRID), ref, C)
            obs = cell_counts(ages, ev, law, s.delta)
            g, df, pv = g_test(obs, s.n * law["P"] / law["P"].sum())
            out.append({"cohort_id": row["cohort_id"], "arm": row["arm"], "group": row.get("group"), "p": pv})
        return {"rows": out}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def _by_point(rows):
    out = {}
    for r in rows:
        out.setdefault((r["rho_eta"], r["rho_beta"], r["rho_eps"], r["kappa"]), []).append(r)
    return out


def library_run(args, mode):
    import acceptance_003 as A
    from sraging import generator as G
    man, rows = library_specs()
    pts = _by_point(rows)
    name = {"gtest": "gtest", "determinism": "determinism"}[mode]
    run = A.Run(name, manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"the whole library ({len(rows)} cohorts at {len(pts)} points) generated at the frozen grid "
                     f"{G.GRID}{', each G-tested' if mode == 'gtest' else ''}; {args.workers} workers")
    state = {"pilot": False, "mode": mode, "points": []}
    t0 = time.time()

    def keep(r):
        state["points"].append(r)
        path = os.path.join(REPO, ".cache", "out", f"{name}_003d_partial.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(state, open(path + ".tmp", "w"), default=float)
        os.replace(path + ".tmp", path)
    _pool_each(library_point, [(rs, mode) for rs in pts.values()], args.workers, keep)
    errs = [x for x in state["points"] if "error" in x]
    recs = sorted([r for x in state["points"] for r in x.get("rows", [])], key=lambda r: r["cohort_id"])
    digests = {r["cohort_id"]: r["digest"] for r in recs}
    out = {"status": "written" if not errs else "errors", "n": len(recs), "errors": errs, "grid": G.GRID,
           "generator_version": G.GENERATOR_VERSION, "digests": digests,
           "digests_sha256": hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest(),
           "cpu_s": sum(x.get("cpu_s", 0) for x in state["points"]), "wall_s_total": time.time() - t0,
           "host_cpu": platform_cpu()}
    if mode == "determinism":
        g1 = json.load(open(os.path.join(RESULTS_DIR, "gtest_003d.json")))
        same = sum(1 for k, v in digests.items() if g1["digests"].get(k) == v)
        out.update({"run1": {"phase": "gtest", "repo_head": g1["repo_head"], "digests_sha256": g1["digests_sha256"]},
                    "n_match_run1": same, "n_run1": len(g1["digests"]),
                    "mismatches": sorted(k for k, v in digests.items() if g1["digests"].get(k) != v)[:50]})
        out["status"] = "pass" if not errs and same == len(digests) == len(g1["digests"]) else "fail"
    if mode == "gtest":
        def verdicts_of(field):
            groups = {}
            for r in recs:
                groups.setdefault(f"{r['arm']}/{r['group']}", []).append(r[field])
            return {k: group_passes(v) for k, v in sorted(groups.items())}
        v_raw, v_w = verdicts_of("p"), verdicts_of("p_williams")
        v_null, v_null_w = verdicts_of("null_p"), verdicts_of("null_p_williams")
        stat = adopted_statistic(v_raw, v_null)
        verdicts = v_w if stat == "williams" else v_raw
        checks = {k: sum(1 for r in recs if not r["checks"].get(k, True)) for k in
                  ("on_grid", "no_age_in_first_bin_but_zero", "no_death_at_C", "censored_at_C")}
        sharp_rows = [r for r in rows if r.get("group") == "sharp" and not (r.get("het_param") and r.get("het_var"))]
        ctrl = []
        _pool_each(v5_control_point, [(rs,) for rs in _by_point(sharp_rows).values()], args.workers,
                   lambda x: ctrl.append(x))
        cp = [r["p"] for x in ctrl for r in x.get("rows", [])]
        control = group_passes(cp) if cp else None
        cens = [r for r in recs if r["censor_frac"] > 0]
        out.update({"groups": verdicts, "adopted_statistic": stat, "groups_raw": v_raw, "groups_williams": v_w,
                    "null_groups_raw": v_null, "null_groups_williams": v_null_w,
                    "all_groups_pass": stat != "real-fail" and all(v["pass"] for v in verdicts.values()),
                    "recording_failures": checks, "positive_control": {"m": len(cp), **(control or {})},
                    "positive_control_fails_as_required": bool(control) and not control["pass"],
                    "realised_censoring": {"n": len(cens), "max_abs_diff": max((abs(r["realised_censored_fraction"]
                                           - r["censor_frac"]) for r in cens), default=None)},
                    "rows": [{k: r[k] for k in r if k != "digest"} for r in recs]})
        out["status"] = ("pass" if not errs and out["all_groups_pass"] and not any(checks.values())
                         and out["positive_control_fails_as_required"] else "fail")
    run.write(out, name=name)


def testers_apply_window():
    """The tester's apply_window, read with ast from its observation.py and compiled here (never imported)."""
    import ast
    from v5_reproduce_003d import TESTER
    path = os.path.join(TESTER, "srbench", "library", "observation.py")
    tree = ast.parse(open(path).read())
    ns = {"np": np}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "apply_window":
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns["apply_window"], hashlib.sha256(open(path, "rb").read()).hexdigest()


def window_point(rows):
    """SC-005 for one point's window specs: time_range equals apply_window on the same cohort generated without it,
    byte for byte; an off-grid window is refused by name."""
    try:
        import dataclasses
        from sraging import generator as G
        from sraging.errors import UnsupportedOptionError
        aw, _ = testers_apply_window()
        out = []
        for row in rows:
            s = spec_from_library(row)
            full = G.materialize(s)
            a0, e0 = aw(full.death_times, full.events, tuple(row["window"]))
            try:
                win = G.materialize(dataclasses.replace(s, time_range=tuple(row["window"])))
            except UnsupportedOptionError as ex:
                out.append({"cohort_id": row["cohort_id"], "refused": ex.option, "message": str(ex)[:160]})
                continue
            out.append({"cohort_id": row["cohort_id"], "refused": None,
                        "bytes_equal": bool(np.array_equal(win.death_times, np.asarray(a0, float))
                                            and np.array_equal(win.events, np.asarray(e0, int))),
                        "n_observed": win.n_observed, "deaths_at_hi": int(((win.death_times == row["window"][1])
                                                                           & (win.events == 1)).sum())})
        return {"rows": out}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def windows(args):
    import acceptance_003 as A
    from sraging import generator as G
    man, rows = library_specs()
    wrows = [r for r in rows if r.get("window") is not None]
    _, aw_sha = testers_apply_window()
    run = A.Run("window", manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"{len(wrows)} window specs, each generated twice (with and without time_range); "
                     f"{args.workers} workers")
    res = []
    _pool_each(window_point, [(rs,) for rs in _by_point(wrows).values()], args.workers, res.append)
    errs = [x for x in res if "error" in x]
    recs = sorted([r for x in res for r in x.get("rows", [])], key=lambda r: r["cohort_id"])
    refused = [r for r in recs if r["refused"]]
    compared = [r for r in recs if not r["refused"]]
    ok = (not errs and all(r["bytes_equal"] for r in compared) and all(r["refused"] == "time_range" for r in refused))
    run.write({"status": "pass" if ok else "fail", "n": len(recs), "n_compared": len(compared),
               "n_bytes_equal": sum(r["bytes_equal"] for r in compared), "n_refused": len(refused),
               "deaths_at_hi": sum(r["deaths_at_hi"] for r in compared), "errors": errs,
               "apply_window_source_sha256": aw_sha, "grid": G.GRID, "rows": recs}, name="window")


RESULTS_DIR = os.path.join(REPO, "tests", "acceptance", "results")


def shard_points(rows, index, of):
    """The points of shard `index` (1-based) of `of`: points sorted by their parameters, every of-th one."""
    pts = sorted(_by_point(rows).items())
    return [rs for i, (p, rs) in enumerate(pts) if i % of == index - 1]


def determinism_shard(args):
    """One WEXAC array element: generate its shard's cohorts and write their digests and the node's CPU."""
    from sraging.compat.v033 import runtime
    runtime.configure()
    from sraging import generator as G
    _, rows = library_specs()
    t0 = time.process_time()
    digests = {}
    for rs in shard_points(rows, args.index, args.of):
        for row in rs:
            co = G.materialize(spec_from_library(row))
            digests[row["cohort_id"]] = cohort_digest(co.death_times, co.events)
    json.dump({"index": args.index, "of": args.of, "digests": digests, "cpu": platform_cpu(),
               "cpu_s": time.process_time() - t0, "generator_version": G.GENERATOR_VERSION, "grid": G.GRID},
              open(args.out, "w"))


def compare_epyc(args):
    """Merge the EPYC shards and compare their digests with the Mac's (the G-test's run 1)."""
    import acceptance_003 as A
    import glob
    shards = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(args.compare_epyc, "shard_*.json")))]
    epyc = {k: v for sh in shards for k, v in sh["digests"].items()}
    g1 = json.load(open(os.path.join(RESULTS_DIR, "gtest_003d.json")))
    run = A.Run("determinism-epyc", manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"compare {len(shards)} EPYC shards ({len(epyc)} digests) with the Mac's {len(g1['digests'])}")
    same = sum(1 for k, v in epyc.items() if g1["digests"].get(k) == v)
    cpus = {}
    for sh in shards:
        cpus.setdefault(sh["cpu"].get("brand", "").strip(), []).append(sh["index"])
    run.write({"status": "pass" if same == len(epyc) == len(g1["digests"]) else "fail", "n_epyc": len(epyc),
               "n_mac": len(g1["digests"]), "n_match": same, "cpu_models": {k: len(v) for k, v in cpus.items()},
               "mismatches": sorted(k for k, v in epyc.items() if g1["digests"].get(k) != v)[:50],
               "missing": sorted(set(g1["digests"]) - set(epyc))[:50], "shards": len(shards)},
              name="determinism_epyc")


def cost(args):
    """SC-008, reported not gated: the library's generation CPU (Mac determinism run 2, generation only) against SRK
    v1's build (1,990 cohorts in build_seconds 835.9 wall, cores not recorded)."""
    import acceptance_003 as A
    d2 = json.load(open(os.path.join(RESULTS_DIR, "determinism_003d.json")))
    g1 = json.load(open(os.path.join(RESULTS_DIR, "gtest_003d.json")))
    v1 = json.load(open(os.path.join(os.path.dirname(V5), "v1", "manifest.json")))
    run = A.Run("cost", manifests=[PANEL_PATH], dev=args.dev, suffix="003d", cost="reads two result files")
    run.write({"status": "written",
               "library_generation": {"cohorts": d2["n"], "cpu_s": d2["cpu_s"], "wall_s": d2["wall_s_total"],
                                      "cpu_s_per_cohort": d2["cpu_s"] / max(d2["n"], 1), "workers": 10,
                                      "host_cpu": d2.get("host_cpu"), "source": "determinism_003d.json"},
               "with_g_tests": {"cpu_s": g1["cpu_s"], "wall_s": g1["wall_s_total"], "source": "gtest_003d.json"},
               "srk_v1": {"cohorts": v1.get("n_cohorts"), "build_seconds_wall": v1.get("build_seconds"),
                          "cores": "not recorded", "generator": v1.get("generator")},
               "note": "generation only: each process solves a point's curves once (the per-process curve cache); "
                       "heterogeneous points need 161 node curves at N 6000"}, name="cost")


def table_point(keys, items):
    """FR-015 for one point's configurations: the converged CV (homogeneous: the reference grid's continuous CV;
    heterogeneous: the generator's own 161-node mixture, labelled), the binned CV of the recorded outcome at the
    configuration's delta, the censoring time C and the model's censored fraction S_obs(C)."""
    try:
        from sraging import generator as G
        out = []
        cache = {}
        for k, it in zip(keys, items):
            s = it["spec"]
            mk = (s.heterogeneous, s.het_var)
            if mk not in cache:
                mg = G._model(s, G.GRID)
                if s.heterogeneous:
                    W, tau = mg["W"], mg["tau"]
                    Smix = 1.0 - sum(Wj * Fj for Wj, Fj in zip(W, mg["Fm"]))
                    mean = float(np.trapezoid(Smix, tau))
                    m2 = float(np.trapezoid(2.0 * tau * Smix, tau))
                    cv, src = math.sqrt(max(m2 - mean * mean, 0.0)) / mean, "generator's 161-node mixture"
                else:
                    cv = G.curve_moments(G._curve(s.point, G.reference_grid(G.GRID)))["cv"]
                    src = "reference grid (2N, dt/2)"
                cache[mk] = (mg, cv, src)
            mg, cv, src = cache[mk]
            b = G.binned_distribution(s, G.GRID, mg)
            C = b["C"]
            out.append({"key": json.loads(k), "ids": it["ids"], "n_max": it["n_max"],
                        "converged_cv": cv, "converged_cv_source": src, "binned_cv_at_delta": G.binned_cv(b),
                        "delta_over_L": s.delta / s.L_phys,
                        "censor_time": C, "censor_time_over_L": None if C is None else C / s.L_phys,
                        "model_censored_fraction": None if C is None else float(b["P"][-1] / b["P"].sum())})
        return {"rows": out}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def table(args):
    import acceptance_003 as A
    from sraging import generator as G
    man, rows = library_specs()
    confs = configurations(rows)
    by_point = {}
    for k, c in confs.items():
        s = c["spec"]
        by_point.setdefault((s.rho_eta, s.rho_beta, s.rho_eps, s.kappa), []).append((k, c))
    pt = {json.dumps(r["key"]): r for r in json.load(open(os.path.join(RESULTS_DIR, "point_table_003d.json")))["rows"]}
    run = A.Run("table", manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"FR-015 for {len(confs)} configurations at {len(by_point)} points; {args.workers} workers")
    res = []
    _pool_each(table_point, [([k for k, _ in v], [c for _, c in v]) for v in by_point.values()], args.workers,
               res.append)
    errs = [x for x in res if "error" in x]
    recs = []
    for x in res:
        for r in x.get("rows", []):
            q = pt.get(json.dumps(r["key"]), {})
            recs.append({**r, "kl": q.get("kl"), "dlncv": q.get("dlncv"),
                         "lambda_at_n_max": None if q.get("kl") is None else 2 * r["n_max"] * q["kl"]})
    recs.sort(key=lambda r: r["ids"][0])
    run.write({"status": "written" if not errs else "errors", "n": len(recs), "errors": errs, "grid": G.GRID,
               "generator_version": G.GENERATOR_VERSION,
               "fields": {"converged_cv": "continuous CV of the first-passage time (no hazard, no censoring)",
                          "binned_cv_at_delta": "CV of the recorded outcome (deaths at k delta, censored at C) under "
                                                "the generator's law at the configuration's delta, hazard, censoring",
                          "model_censored_fraction": "S_obs(C): the expected censored fraction at the chosen C",
                          "kl, dlncv": "the sweep's (run 3) grid error, lambda = 2 n kl"},
               "rows": recs}, name="point_table_full")


HANDOFF = "docs/planning/handoff_2026-10-14_generator_to_tester.md"


def handoff_claims() -> list:
    """Every number the hand-off message states, recomputed from the committed result files (constitution II)."""
    from scipy.stats import ks_2samp
    R = lambda f: json.load(open(os.path.join(RESULTS_DIR, f)))          # noqa: E731
    f, g, w, d = R("fidelity_003d.json"), R("gtest_003d.json"), R("window_003d.json"), R("determinism_003d.json")
    e, c, s = R("determinism_epyc_003d.json"), R("cost_003d.json"), R("sweep_003d.json")
    pc = s["choice"]
    sharp = sorted(p["control_diff"] for p in f["points"] if p["point"] != PANEL["fidelity"]["broad_control"])
    cens = [r for r in g["rows"] if r["censor_frac"] > 0]
    miss = [r["realised_censored_fraction"] - r["censor_frac"] for r in cens]
    unc = [r for r in cens if r["realised_censored_fraction"] < 0.005]
    het_hi = [r for r in s["het_rows"] if r["n_max"] > TH["het_n_cap"]]
    het_ids = [r for r in g["rows"] if r["heterogeneous"]]
    v5 = json.load(open(os.path.join(V5, "manifest.json")))["checksums"]
    same = sum(1 for k, v in g["digests"].items() if v5.get(k) == v)
    trip = sorted((r["ids"][0], 2e5 * r["dlncv"] ** 2) for r in s["rows"] if 2e5 * r["dlncv"] ** 2 > 0.1)
    models = sorted(m.split()[2] for m in e["cpu_models"])
    return [
        f"N {pc['N']}, dt {pc['dt']:.0e}".replace("e-04", "e-4"), f"{pc['K']} heterogeneity nodes",
        f"all {len(s['rows']) + len(s['het_rows'])} library configurations",
        f"{len(s['rows'])} homogeneous: worst lambda {max(r['lambda'] for r in s['rows']):.4f}",
        f"{len(s['het_rows'])} heterogeneous: worst lambda {max(r['lambda'] for r in s['het_rows']):.4f}",
        f"worst {max(2 * r['n_max'] * r['dlncv'] ** 2 for r in s['rows']):.4f} (homogeneous",
        f"and {max(2 * r['n_max'] * r['dlncv'] ** 2 for r in s['het_rows']):.4f}",
        f"at {len(f['points'])} homogeneous", f"largest {max(abs(p['gen_diff']) for p in f['points']):.4f}",
        f"all {len(sharp)} sharp points", f"(+{sharp[0]:.3f} to +{sharp[-1]:.3f})",
        f"{sum(not r['checks']['no_age_in_first_bin_but_zero'] for r in g['rows'])} violations in {g['n']:,} cohorts",
        f"({sum(not r['checks'].get('no_death_at_C', True) for r in cens)} in {len(cens)} censored cohorts)",
        f"{w['n_bytes_equal']}/{w['n_compared']} on-grid windows", f"{w['n_refused']}/{w['n_refused']} off-grid windows",
        f"{d['n_match_run1']:,}/{d['n']:,} checksums across two Mac runs",
        f"{e['n_match']:,}/{e['n_epyc']:,} Mac against WEXAC AMD EPYC",
        f"{e['shards']} jobs on EPYC {models[0]}, {models[1]} and {models[2]}",
        f"({len(g['rows']) - len(het_ids):,} homogeneous)", f"own law ({len(het_ids)}",
        f"{sum(v['pass'] for v in g['groups'].values())}/{len(g['groups'])} of your arm x group cells pass",
        f"lowest KS p {min(v['ks_p'] for v in g['groups'].values()):.3f}",
        f"fails pure/main (KS p {g['groups_raw']['pure/main']['ks_p']:.0e}",
        f"exact draws from the law (KS p {g['null_groups_raw']['pure/main']['ks_p']:.4f})",
        f"exact draws fail {sum(not v['pass'] for v in g['null_groups_williams'].values())}/{len(g['groups'])} cells",
        f"(two-sample KS p {ks_2samp([r['p_williams'] for r in g['rows']], [r['null_p_williams'] for r in g['rows']]).pvalue:.2f})",
        f"({round(100 * g['positive_control']['frac_below_alpha'])}% below 0.01)",
        f"{c['library_generation']['cpu_s_per_cohort']:.2f} CPU-s per cohort",
        f"{c['library_generation']['cpu_s'] / 3600:.2f} CPU-h",
        f"{same} of your {len(g['digests']):,} cohorts are byte-identical to v5; {len(g['digests']) - same:,} change",
        f"{len(set(r['cohort_id'] for r in het_ids if r['n'] > TH['het_n_cap']))} of yours lie above",
        f"worst lambda is {max(2 * r['n_max'] * r['kl'] for r in het_hi):.2f}",
        f"{len(unc)} censored specs", f"({sum(r['realised_censored_fraction'] == 0 for r in unc)} exactly 0)",
        f"Of the {len(cens)} censored cohorts, {sum(m < -0.1 for m in miss)} fall short by more than 0.1",
        f"censoring miss is {float(np.median(np.abs(miss))):.3f}",
        f"{sum(2e5 * r['kl'] > 0.05 for r in s['rows'])}/{len(s['rows'])} homogeneous",
        f"{len(trip)} would trip your rule",
        f"({', '.join(f'{k.split(chr(45))[0]}-{k.split(chr(45))[1]}-{k.split(chr(45))[2]} {v:.2f}' for k, v in trip)})",
    ]


def verify_handoff(args):
    """T038: every claim string must appear in the hand-off's message; the tag line must name the tag's commit."""
    import subprocess
    text = open(os.path.join(REPO, HANDOFF)).read()
    msg = text.split("```", 2)[1]
    bad = [c for c in handoff_claims() if c not in msg]
    tag = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "sraging-gen-1^{commit}"],
                         capture_output=True, text=True).stdout.strip()
    if not tag or f"sraging-gen-1 at {tag}" not in msg:
        bad.append(f"the tag line: sraging-gen-1 is {'not tagged yet' if not tag else tag}")
    for b in bad:
        print("MISMATCH:", b)
    print(f"verify-handoff: {len(handoff_claims()) - len([b for b in bad if not b.startswith('the tag')])} of "
          f"{len(handoff_claims())} numbers match; tag {'ok' if not any(b.startswith('the tag') for b in bad) else 'NOT ok'}")
    sys.exit(1 if bad else 0)


# ------------------------------------------------------------------------------------ the fit-free prediction (R12)
PREDICT_DIR = os.path.join(REPO, ".cache", "out", "predict_003d")
DIST_FIELDS = ("rho_eta", "rho_beta", "rho_eps", "kappa", "n", "L_phys", "delta", "censor_frac", "external_hazard",
               "het_var")
MLE_FLAGS = ("shadow", "exact_from_F", "self_consistent", "floor_early", "n_rule", "dt_rule", "early_all")


def be_law(row, tau, S):
    """The backward-Euler control's recorded outcome as n -> infinity: SRK cb191f5's materialize
    (`v5_reproduce_003d.materialize_srk`, which rebuilds v2's 55 censored cohorts byte for byte). T/L has survival S
    interpolated linearly in tau (its inverse-CDF draw); C is the population (1 - f) quantile (the sample quantile's
    limit); every age is floored to the delta grid, so the deaths in [floor(C), C] share the censored records' age.
    Cells, in `generator.binned_distribution`'s format: deaths at k delta for k = 0..floor(C/delta), then the censored
    cell (at floor(C/delta) delta)."""
    if row.get("external_hazard") is not None:
        raise ValueError("be_law: external_hazard is not modelled (the hazard points wait for _h 1.0.1)")
    f = float(row.get("censor_frac") or 0.0)
    if not f > 0:
        raise ValueError("be_law: censor_frac must be > 0 (the prediction covers the censored clusters only)")
    L, d = float(row["L_phys"]), float(row["delta"])
    C = float(np.interp(f, S[::-1], tau[::-1])) * L
    kc = int(math.floor(C / d))
    Se = np.interp(np.append(np.arange(kc + 1) * d, C) / L, tau, S)
    Se[0] = 1.0                                      # a draw above S[0] lands at tau[0]: the first cell
    P = np.append(Se[:-1] - Se[1:], Se[-1])
    return {"ages": np.append(np.arange(kc + 1) * d, kc * d), "events": np.append(np.ones(kc + 1, int), 0),
            "P": np.clip(P, 0.0, None), "C": C}


def expected_cohort(law, n, scale):
    """R12's expected-count cohort: each cell's age repeated round(scale n P(cell)) times, sorted by age."""
    m = np.rint(scale * n * np.asarray(law["P"], float)).astype(int)
    ages = np.repeat(np.asarray(law["ages"], float), m)
    events = np.repeat(np.asarray(law["events"], int), m)
    o = np.argsort(ages, kind="stable")
    return ages[o], events[o], m


def half_width(claim, side):
    """log10 distance from a closed claim's point to its edge on `side` ("upper" or "lower"); None otherwise."""
    if not isinstance(claim, dict) or claim.get("shape") != "closed":
        return None
    p, e = claim.get("point"), claim.get(side)
    if p is None or e is None or not (p > 0 and e > 0):
        return None
    return math.log10(e / p) if side == "upper" else math.log10(p / e)


def predict_rule(off_dex, claims):
    """R12's rule at one configuration. `off_dex` = log10(pseudo-true) - log10(truth). The truth lies beyond the
    claims' edge on the side facing it (pseudo-true below the truth: the upper edge) when |off_dex| exceeds the
    median half-width there of the v2 rows' closed claims. None when no closed claim is there."""
    side = "upper" if off_dex < 0 else "lower"
    hws = [h for h in (half_width(c, side) for c in claims) if h is not None]
    med = float(np.median(hws)) if hws else None
    return {"side": side, "half_widths": hws, "median_half_width": med,
            "predicted_miss": (abs(off_dex) > med) if med is not None else None}


def predict_configurations(rows, specs):
    """{name: {name, ids, spec, n}}: the panel rows grouped by cohort id less the seed suffix ("-s<k>"); every
    cohort of a group must share the distribution's fields (DIST_FIELDS)."""
    import re
    out = {}
    for r in rows:
        cid = r["cohort_id"]
        sp = specs[cid]
        name = re.sub(r"-s\d+$", "", cid)
        g = out.setdefault(name, {"name": name, "ids": [], "spec": sp, "n": int(sp["n"])})
        for k in DIST_FIELDS:
            if sp.get(k) != g["spec"].get(k):
                raise ValueError(f"predict: {cid} differs from {g['ids'][0]} in {k}")
        g["ids"].append(cid)
    for g in out.values():
        g["ids"].sort()
    return dict(sorted(out.items()))


def theta_hat_only(ages, events, delta, name, truth=None, dt_rule=False):
    """The calibrator's own fit stopped after its MLE stage: `run_fit`'s call (the observation model, then
    `estimator.fit` with the registered name's class constants) with stages=("mle",) added -- no profile, no claim.
    With `truth`, also the log-likelihood there on the same objective (the search's witness). `dt_rule=True` adds
    jax2's own DT_RULE (dt <= CV/80; built and off in 1.0.0) and changes nothing else: an analysis, never a
    calibrator version."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.jax2 import claims as jc
    from sraging.compat.jax2 import estimator as est
    from sraging.compat.jax2 import fp_jax as fj
    from sraging.compat.jax2 import observation_model as om
    from sraging.compat.v033 import runtime
    runtime.check()
    kw = PJ.run_kwargs(PJ.CLASSES[name])
    obs = PJ.Observation(name, tuple(float(a) for a in ages), tuple(int(e) for e in events), delta, None)
    prep = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=kw["window"], relabel_on=kw["relabel"])
    flags = {k: True for k in MLE_FLAGS if kw[k]}
    if dt_rule:
        flags["dt_rule"] = True
    c0 = time.process_time()
    with PJ.capture_objectives() as objs:
        res = est.fit(fj, prep.ages, prep.events, delta=obs.delta, levels=(jc.LEVEL,), kappa_max=jc.KAPPA_MAX,
                      hazard=kw["hazard"], window=prep.window_eff, stages=("mle",), **flags)
    cpu = time.process_time() - c0
    dg = res.get("diagnostics") or {}
    out = {"theta": dict(zip(PJ.PARAMS, (float(v) for v in res["theta"][:4]))), "loglik": res.get("loglik"),
           "cpu_s": cpu, "censor_tie": prep.censor_tie,
           "diagnostics": {k: dg.get(k) for k in ("solver", "dt_floor", "n_eval", "n_grad", "mle_s", "compile_s",
                                                  "exc_counts", "error")},
           "dt_ceiling": res.get("dt_ceiling")}
    obj = next((o for o in objs if o.NP == len(PJ.PARAMS)), None)
    if truth is not None and obj is not None:
        out["loglik_truth"] = -float(obj.nll(np.log([float(truth[p]) for p in PJ.PARAMS])))
    return out


def _fit_init():
    import acceptance_003_coverage as COV
    for k, v in COV.THREAD_ENV.items():
        os.environ[k] = v
    from sraging.compat.v033 import runtime
    runtime.configure()


def predict_fit(name, law, path, truth, delta, calibrator):
    """One expected-count cohort's theta-hat (a spawned, single-threaded worker)."""
    try:
        z = np.load(path)
        return {"name": name, "law": law, **theta_hat_only(z["ages"], z["events"], delta, calibrator, truth=truth)}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"name": name, "law": law, "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def v2_rows(rel):
    """Spec 003's coverage rows at d903979 (the WEXAC shards, read-only) and the sha256 of their bytes in order."""
    import glob
    paths = sorted(glob.glob(os.path.join(REPO, rel, "shard_*.jsonl")))
    h = hashlib.sha256()
    rows = []
    for p in paths:
        raw = open(p, "rb").read()
        h.update(raw)
        rows += [json.loads(ln) for ln in raw.decode().splitlines() if ln.strip()]
    return rows, {"path": rel, "n_shards": len(paths), "sha256": h.hexdigest()}


def predict(args):
    """R12 (SC-010): for each configuration of spec 003's censored panel, the calibrator's theta-hat on the
    expected-count cohort of each law ("be": the backward-Euler control, v2's law; "gen1": sraging-gen-1's), and the
    misses R12's rule predicts. Fit-free for the panel itself; committed before the re-test's first fit."""
    import acceptance_003 as A
    import acceptance_003_coverage as COV
    import oracle_jax2 as OJ
    import v5_reproduce_003d as R5
    P = PANEL["predict"]
    laws = P["laws_pilot"] if args.pilot else P["laws"]
    rows, rows_rec = v2_rows(P["rows"])
    rows = [r for r in rows if r.get("arm") == P["arm"] and r["calibrator"] == P["calibrator"]]
    man_path = os.path.join(OJ.LIBRARY_V2, "manifest.json")
    raw = open(man_path, "rb").read()
    specs = {s["cohort_id"]: s for s in json.loads(raw)["specs"]}
    cfgs = predict_configurations(rows, specs)
    if args.pilot:
        cfgs = {k: cfgs[k] for k in P["pilot"]}
    run = A.Run("predict-pilot" if args.pilot else "predict", manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"{len(cfgs) * len(laws)} theta-hat-only fits ({len(cfgs)} configurations x laws {laws}) of "
                     f"expected-count cohorts at {P['scale']} x n, {args.workers} workers"
                     + ("" if args.pilot else "; the pilot's price is in predict_pilot_003d.json"))
    by_id = {r["cohort_id"]: r for r in rows}
    os.makedirs(PREDICT_DIR, exist_ok=True)
    curves, jobs, built = {}, [], {}
    for name, g in cfgs.items():
        sp = g["spec"]
        truths = {json.dumps(by_id[i]["truth"], sort_keys=True) for i in g["ids"]}
        if len(truths) != 1:
            raise ValueError(f"predict: {name}'s cohorts carry different truths")
        truth = by_id[g["ids"][0]]["truth"]
        for law_name in laws:
            if law_name == "be":
                p = (sp["rho_eta"], sp["rho_beta"], sp["rho_eps"], sp["kappa"])
                if p not in curves:
                    curves[p] = R5.curve(*p)
                law = be_law(sp, *curves[p])
            elif law_name == "gen1":
                from sraging import generator as G
                s = spec_from_library(sp)
                law = G.binned_distribution(s, G.GRID)
            else:
                raise ValueError(f"predict: unknown law {law_name!r}")
            ages, events, m = expected_cohort(law, g["n"], P["scale"])
            path = os.path.join(PREDICT_DIR, f"{name}__{law_name}.npz")
            np.savez(path, ages=ages, events=events)
            built[(name, law_name)] = {"n": int(ages.size), "n_censored": int((events == 0).sum()),
                                       "C": law["C"], "censored_mass": float(law["P"][-1]),
                                       "tied_death_mass": (float(law["P"][-2]) if law_name == "be" else 0.0),
                                       "n_cells": int(law["P"].size), "first_cell_deaths": int(m[0]),
                                       "digest": cohort_digest(ages, events), "path": os.path.relpath(path, REPO)}
            jobs.append((name, law_name, path, truth, float(sp["delta"]), P["calibrator"]))
    fits = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                initializer=_fit_init) as ex:
        futs = [ex.submit(predict_fit, *j) for j in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            fits[(r["name"], r["law"])] = r
            print(f"  [predict] {i}/{len(jobs)} {r['name']} {r['law']} "
                  f"{'ERROR ' + r['error'] if 'error' in r else 'cpu %.0f s' % r['cpu_s']}", flush=True)
    q = P["quantity"]
    out_cfgs, summary = [], {law: {"predicted": [], "not_predicted": [], "undetermined": [], "failed": []}
                            for law in laws}
    for name, g in cfgs.items():
        crow = [by_id[i] for i in g["ids"]]
        truth = crow[0]["truth"]
        t_q = COV.truth_value(q, truth)
        claims = [COV.claim_of(r, q) for r in crow]
        hits = COV.hits_of(crow, q, "closed")
        rec = {"name": name, "ids": g["ids"], "n": g["n"], "truth": truth, "truth_" + q: t_q,
               "v2": {i: {"hit": hits[i], "claim": c} for i, c in zip(g["ids"], claims)}, "laws": {}}
        for law_name in laws:
            fr = fits[(name, law_name)]
            lr = {"cohort": built[(name, law_name)], "fit": fr}
            if "error" in fr or fr.get("loglik") is None or not math.isfinite(fr["loglik"]):
                summary[law_name]["failed"].append(name)
            else:
                pq = COV.truth_value(q, fr["theta"])
                off = math.log10(pq) - math.log10(t_q)
                lr.update({"pseudo_true_" + q: pq, "off_dex": off, **predict_rule(off, claims)})
                llt = fr.get("loglik_truth")
                if llt is not None and math.isfinite(llt):
                    lr["dll_hat_minus_truth"] = fr["loglik"] - llt
                    lr["dll_per_individual"] = lr["dll_hat_minus_truth"] / built[(name, law_name)]["n"]
                    lr["search_short"] = lr["dll_hat_minus_truth"] < 0
                pm = lr["predicted_miss"]
                summary[law_name]["undetermined" if pm is None else "predicted" if pm else "not_predicted"].append(name)
            rec["laws"][law_name] = lr
        out_cfgs.append(rec)
    for law_name, sm in summary.items():
        def misses(names):
            return sum(rec["v2"][i]["hit"] is False for rec in out_cfgs if rec["name"] in names for i in rec["ids"])

        def hit_n(names):
            return sum(rec["v2"][i]["hit"] is True for rec in out_cfgs if rec["name"] in names for i in rec["ids"])
        sm.update({"v2_misses_at_predicted": misses(sm["predicted"]), "v2_hits_at_predicted": hit_n(sm["predicted"]),
                   "v2_misses_elsewhere": misses(sm["not_predicted"] + sm["undetermined"] + sm["failed"]),
                   "predicted_cohorts": [i for rec in out_cfgs if rec["name"] in sm["predicted"] for i in rec["ids"]],
                   "search_short": [rec["name"] for rec in out_cfgs
                                    if rec["laws"][law_name].get("search_short")]})
    cpus = [f["cpu_s"] for f in fits.values() if "cpu_s" in f]
    run.write({"status": "written", "rule": "R12: a miss is predicted where |log10(pseudo-true) - log10(truth)| of "
                                            f"{q} exceeds the median half-width, on the side facing the truth, of the "
                                            "v2 rows' closed claims at that configuration",
               "laws": {"be": "SRK cb191f5's materialize on SRK's backward-Euler generation grid (v2's law: "
                              "55/55 v2 censored cohorts rebuilt byte for byte), large-n limit",
                        "gen1": "sraging-gen-1's binned_distribution on its grid (the re-test's law)"},
               "read_as": {"be": "a retrodiction of v2's misses: which ones the backward-Euler law explains",
                           "gen1": "the prediction for the re-test: the misses the calibrator's own large-n fit "
                                   "leaves on the clean cohorts"},
               "scale": P["scale"], "calibrator": P["calibrator"], "quantity": q, "stage": "mle (no profile, no claim)",
               "v2_rows": rows_rec, "library_v2": {"manifest": man_path,
                                                   "manifest_sha256": hashlib.sha256(raw).hexdigest()},
               "pilot": bool(args.pilot), "summary": summary, "configurations": out_cfgs,
               "fit_cpu_s": {"total": sum(cpus), "max": max(cpus, default=0.0), "median": float(np.median(cpus))
                             if cpus else None, "n": len(cpus)}, "host_cpu": platform_cpu()},
              name="predict_pilot" if args.pilot else "predict")


def witness_forecast(lams, df=4) -> dict:
    """SC-011's witness forecast from the fit-free offsets: cohort i's D taken as noncentral chi-squared with `df`
    degrees of freedom and noncentrality lam_i = 2 n_i x (ll(theta-hat) - ll(truth)) per individual on its law's
    expected-count cohort. Returns the expected count above witness_D_cut and P(the fraction rule fails), the count
    being a Poisson-binomial (exact convolution)."""
    from scipy.stats import chi2, ncx2
    cut = TH["witness_D_cut"]
    ps = [float(chi2.sf(cut, df) if lam <= 0 else ncx2.sf(cut, df, lam)) for lam in lams]
    dist = np.array([1.0])
    for p in ps:
        dist = np.convolve(dist, [1.0 - p, p])
    k_fail = math.floor(TH["witness_frac_max"] * len(ps)) + 1
    return {"n": len(ps), "df": df, "p_above_cut": ps, "expected_above_cut": float(sum(ps)), "k_fail": k_fail,
            "p_fraction_fails": float(dist[k_fail:].sum()), "noncentrality_median": float(np.median(lams)) if lams
            else None}


def predict_witness(args):
    """The witness forecast (Naveh 2026-10-14, "Yes, forecast all 12 first"): R11's 12 witness cohorts, each its own
    configuration, fitted theta-hat-only on its gen1 expected-count cohort; with the 55 re-test cohorts' gen1 offsets
    (predict_003d.json), the forecast of SC-011's rule over all 67. Committed before the WEXAC fits."""
    import acceptance_003 as A
    import retest_003d as RT
    from sraging import generator as G
    P = PANEL["predict"]
    ids = [p["cohort_id"] for p in RT.panel_ids() if p["kind"] == "witness"]
    _man, rows = library_specs()
    v5 = {r["cohort_id"]: r for r in rows}
    run = A.Run("predict-witness", manifests=[PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"{len(ids)} theta-hat-only fits of gen1 expected-count cohorts at {P['scale']} x n, "
                     f"{args.workers} workers (~30 CPU-min INFERRED from predict_003d.json)")
    os.makedirs(PREDICT_DIR, exist_ok=True)
    jobs, built = [], {}
    for cid in ids:
        row = v5[cid]
        law = G.binned_distribution(spec_from_library(row), G.GRID)
        ages, events, m = expected_cohort(law, int(row["n"]), P["scale"])
        path = os.path.join(PREDICT_DIR, f"{cid}__gen1.npz")
        np.savez(path, ages=ages, events=events)
        built[cid] = {"n": int(row["n"]), "n_expected_cohort": int(ages.size), "n_cells": int(law["P"].size),
                      "first_cell_deaths": int(m[0]), "digest": cohort_digest(ages, events),
                      "path": os.path.relpath(path, REPO)}
        jobs.append((cid, "gen1", path, RT.physical_truth(row), float(row["delta"]), P["calibrator"]))
    fits = {}
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                initializer=_fit_init) as ex:
        for i, f in enumerate(cf.as_completed([ex.submit(predict_fit, *j) for j in jobs]), 1):
            r = f.result()
            fits[r["name"]] = r
            print(f"  [predict-witness] {i}/{len(jobs)} {r['name']} "
                  f"{'ERROR ' + r['error'] if 'error' in r else 'cpu %.0f s' % r['cpu_s']}", flush=True)
    out, lams = {}, {}
    for cid in ids:
        fr = fits[cid]
        rec = {"cohort": built[cid], "fit": fr}
        if "error" not in fr and fr.get("loglik_truth") is not None:
            dll = (fr["loglik"] - fr["loglik_truth"]) / built[cid]["n_expected_cohort"]
            rec.update(dll_per_individual=dll, forecast_noncentrality=max(0.0, 2.0 * built[cid]["n"] * dll),
                       search_short=dll < 0)
            lams[cid] = rec["forecast_noncentrality"]
        out[cid] = rec
    pred = json.load(open(os.path.join(RESULTS_DIR, "predict_003d.json")))
    man = json.load(open(RT.MANIFEST))
    n_of = {r["cohort_id"]: r["n"] for r in man["cohorts"]}
    for c in pred["configurations"]:
        g = c["laws"]["gen1"]
        for i in c["ids"]:
            lams[i] = max(0.0, 2.0 * n_of[i] * g["dll_per_individual"])
    order = [p["cohort_id"] for p in RT.panel_ids()]
    missing = [c for c in order if c not in lams]
    run.write({"status": "written" if not missing else "incomplete", "missing": missing,
               "rule": "D_i ~ noncentral chi-squared(df, lam_i), lam_i = 2 n_i dll_i, dll_i = (ll(theta-hat) - "
                       "ll(truth)) per individual on the gen1 expected-count cohort (the calibrator's own large-n "
                       "offset from the truth on the clean law); df 4 = the fitted parameters, 3 as a sensitivity",
               "witness_cohorts": out, "noncentrality": {c: lams[c] for c in order if c in lams},
               "forecast": {"df4": witness_forecast([lams[c] for c in order if c in lams], 4),
                            "df3": witness_forecast([lams[c] for c in order if c in lams], 3)},
               "scale": P["scale"], "calibrator": P["calibrator"], "host_cpu": platform_cpu()},
              name="predict_witness")


# ----------------------------------------------------------------------- the re-test's scoring (T048, SC-011)
def witness_rule(Ds, ids=None) -> dict:
    """SC-011's truth witness on D = 2 (ll_fit - ll_truth): pass if the fraction with D > witness_D_cut is at most
    witness_frac_max, the median is at most witness_median_max, and no D is below witness_D_min (each such D is a
    calibrator CODE finding, reported apart)."""
    D = [float(d) for d in Ds]
    ids = list(ids) if ids is not None else [str(i) for i in range(len(D))]
    above = sum(d > TH["witness_D_cut"] for d in D)
    med = float(np.median(D)) if D else None
    code = [{"cohort_id": i, "D": d} for i, d in zip(ids, D) if d < TH["witness_D_min"]]
    fails = []
    if D and above / len(D) > TH["witness_frac_max"]:
        fails.append(f"fraction with D > {TH['witness_D_cut']}: {above}/{len(D)} > {TH['witness_frac_max']:.4f}")
    if med is not None and med > TH["witness_median_max"]:
        fails.append(f"median D {med:.3f} > {TH['witness_median_max']}")
    if code:
        fails.append(f"{len(code)} D below {TH['witness_D_min']} (calibrator code findings)")
    return {"n": len(D), "n_above_cut": above, "fraction_above_cut": (above / len(D)) if D else None, "median": med,
            "code_findings": code, "fails": fails, "pass": bool(D) and not fails}


def v2_pairs(new_rows, old_rows, q) -> dict:
    """Per cohort, [v2 hit, re-test hit] of q's closed claims (spec 003's `hits_of`); reported, never gated. The
    sqrt(eta)/eps pairs are not like for like: v2 widened it after six walks, the re-test after three (D1)."""
    import acceptance_003_coverage as COV
    hn, ho = COV.hits_of(new_rows, q, "closed"), COV.hits_of(old_rows, q, "closed")
    pairs = {c: [ho[c], hn[c]] for c in sorted(hn) if c in ho}
    count = lambda a, b: sum(x is a and y is b for x, y in pairs.values())  # noqa: E731
    return {"pairs": pairs, "v2_miss_new_hit": count(False, True), "v2_hit_new_miss": count(True, False),
            "both_hit": count(True, True), "both_miss": count(False, False), "like_for_like": q != "sqrt_eta_over_eps"}


def retest_rows(rows_dir, kinds=("retest", "witness")):
    """The re-test's rows (every shard under `rows_dir`), each checked against the arrays' manifest: the panel's
    cohorts of `kinds` complete, one row per cohort, the digest the fit read equal to the manifest's."""
    import glob
    import retest_003d as RT
    man = json.load(open(RT.MANIFEST))
    want = {r["cohort_id"]: r for r in man["cohorts"] if r["kind"] in kinds}
    rows = [json.loads(ln) for p in sorted(glob.glob(os.path.join(rows_dir, "shard_*.jsonl"))) for ln in open(p)
            if ln.strip()]
    got = {}
    problems = []
    for r in rows:
        if r["cohort_id"] in got:
            problems.append(f"{r['cohort_id']}: more than one row")
        if r["cohort_id"] not in want:
            if r["cohort_id"] not in {c["cohort_id"] for c in man["cohorts"]}:
                problems.append(f"{r['cohort_id']}: not in the manifest")
            continue
        if want[r["cohort_id"]]["digest"] != r.get("digest"):
            problems.append(f"{r['cohort_id']}: not the manifest's array")
        got[r["cohort_id"]] = r
    problems += [f"{c}: no row" for c in want if c not in got]
    return man, got, problems


def retest_score(args):
    """T048: spec 003's SC-005 functions, unchanged, on the 55 re-test fits (gated in spec 003: profile_jax2 on the
    censor arm), on the three combinations against eta and epsilon; the v2 pairs beside them, and the prediction
    (predict_003d.json) against the observed misses. A finding about the calibrator, not a gate on the generator."""
    import acceptance_003 as A
    import acceptance_003_coverage as COV
    from sraging.compat.jax2 import estimator as JE
    P = PANEL["retest"]
    man, got, problems = retest_rows(args.rows, kinds=("retest",))        # the coverage needs the 55 only
    run = A.Run("retest-score", manifests=[PANEL_PATH], dev=args.dev or bool(problems), suffix="003d",
                cost="scores stored rows; no fit")
    new = [r for r in got.values() if r.get("kind") == "retest"]
    sc = COV.score(new, JE.box_for(300.0))
    cell = sc.get("profile_jax2|censor", {})
    qs = list(P["combinations"])
    g = COV.gate_outcome(sc)
    losses = [x for x in g["losses"] if x[1] in qs]
    under = [x for x in g["underpowered"] if x[1] in qs]
    v2, rows_rec = v2_rows(PANEL["predict"]["rows"])
    old = [r for r in v2 if r.get("arm") == "censor" and r["calibrator"] == "profile_jax2"]
    pred = json.load(open(os.path.join(RESULTS_DIR, "predict_003d.json")))
    q = pred["quantity"]
    hits = COV.hits_of(new, q, "closed")
    vs = []
    for c in pred["configurations"]:
        obs = [hits.get(i) for i in c["ids"]]
        vs.append({"name": c["name"], "ids": c["ids"], "observed": obs,
                   **{f"predicted_{law}": c["laws"][law].get("predicted_miss") for law in c["laws"]}})
    run.write({"status": ("loss" if losses else "underpowered" if under else "no loss") if not problems
               else "incomplete", "problems": problems, "read_as": "a finding about the calibrator (SC-011), not a "
               "gate on the generator", "losses": losses, "underpowered": under, "quantities": qs,
               "cell": {k: cell.get(k) for k in ("n", "n_informative", "n_failed")},
               "scores": {k: cell.get("quantities", {}).get(k) for k in qs + ["eta", "epsilon"]},
               "v2_pairs": {k: v2_pairs(new, old, k) for k in qs + ["eta", "epsilon"]}, "v2_rows": rows_rec,
               "prediction_vs_observed": vs, "arrays": {"repo_head": man["repo_head"], "tag": man["tag"]},
               "fit_heads": sorted({r.get("head") for r in new}), "fit_hosts": COV.fit_hosts(new),
               "cost_cpu_s": sum(r.get("cpu_s") or 0.0 for r in new)}, name="retest")


def pending_filter(problems, pending):
    """(problems without the named rows' "no row", the pending ids): a row still computing is named, never hidden; a
    pending id that is not missing is refused."""
    pending = sorted(pending or [])
    absent = {p.split(":")[0] for p in problems if p.endswith(": no row")}
    wrong = [c for c in pending if c not in absent]
    if wrong:
        raise ValueError(f"pending rows that are not missing: {', '.join(wrong)}")
    return [p for p in problems if not (p.endswith(": no row") and p.split(":")[0] in pending)], pending


def witness(args):
    """T048: SC-011's truth witness over every re-test and witness fit (67). `--pending ID` names a row still computing
    (Naveh 2026-10-15: score on 66, add the 67th later); it is recorded, and the rule runs on the rows in hand."""
    import acceptance_003 as A
    man, got, problems = retest_rows(args.rows)
    problems, pending = pending_filter(problems, args.pending)
    run = A.Run("witness", manifests=[PANEL_PATH], dev=args.dev or bool(problems), suffix="003d",
                cost="reads stored rows; no fit")
    ids = sorted(c for c, r in got.items() if "D" in (r.get("witness") or {}))
    rule = witness_rule([got[c]["witness"]["D"] for c in ids], ids)
    missing = sorted(c for c in got if c not in ids)
    rule["pass"] = rule["pass"] and not missing and not problems
    status = "incomplete" if problems else ("pass" if rule["pass"] else "fail")
    run.write({"status": status, "problems": problems, "no_witness": missing, "pending": pending,
               "n_expected": len(man["cohorts"]), **rule,
               "by_kind": {k: witness_rule([got[c]["witness"]["D"] for c in ids if got[c]["kind"] == k],
                                           [c for c in ids if got[c]["kind"] == k]) for k in ("retest", "witness")},
               "D": {c: got[c]["witness"]["D"] for c in ids},
               "thresholds": {k: TH[k] for k in ("witness_D_cut", "witness_frac_max", "witness_median_max",
                                                  "witness_D_min")},
               "read_as": "a witness failure opens a generator finding; a D below the minimum is a calibrator code "
                          "finding"}, name="witness")


def platform_cpu():
    import platform
    import subprocess
    brand = ""
    try:
        if os.path.exists("/proc/cpuinfo"):
            brand = next((ln.split(":", 1)[1] for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
        else:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    except OSError:
        pass
    return {"machine": platform.machine(), "processor": platform.processor(), "brand": brand.strip(),
            "node": platform.node()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["sweep", "fidelity", "gtest", "determinism", "window", "determinism-shard", "cost",
                                      "table", "verify-handoff", "predict", "retest-score", "witness"])
    ap.add_argument("--pilot", action="store_true", help="sweep: the 30 lowest-rho_eps points only; predict: the "
                                                         "panel's [predict] pilot configurations. Each prices the rest")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--dev", action="store_true", help="a development run on a dirty tree (never counts)")
    ap.add_argument("--index", type=int, help="determinism-shard: this element (1-based)")
    ap.add_argument("--of", type=int, help="determinism-shard: the number of elements")
    ap.add_argument("--out", help="determinism-shard: the output JSON")
    ap.add_argument("--compare-epyc", help="determinism: a directory of EPYC shard_*.json to compare with the Mac's")
    ap.add_argument("--rows", help="retest-score, witness: the directory of the WEXAC re-test rows (shard_*.jsonl)")
    ap.add_argument("--witness", action="store_true", help="predict: the 12 witness cohorts' forecast (gen1 law)")
    ap.add_argument("--pending", nargs="*", default=[], help="witness: cohort ids still computing (named, not hidden)")
    args = ap.parse_args(argv)
    if args.phase == "determinism-shard":
        return determinism_shard(args)
    if args.phase == "determinism" and args.compare_epyc:
        return compare_epyc(args)
    if args.phase == "predict" and args.witness:
        return predict_witness(args)
    {"sweep": sweep, "fidelity": fidelity, "gtest": lambda a: library_run(a, "gtest"),
     "determinism": lambda a: library_run(a, "determinism"), "window": windows, "cost": cost, "table": table,
     "verify-handoff": verify_handoff, "predict": predict, "retest-score": retest_score,
     "witness": witness}[args.phase](args)


if __name__ == "__main__":
    main()
