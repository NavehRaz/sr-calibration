"""Spec 003d: how many heterogeneity nodes K the generator needs, and with which interpolation (Naveh's decision of
2026-10-13 rests on this; research R6).

Node count only: every curve on one grid (N 1500, dt 2e-4), so the grid error is held fixed. For each named point,
641 node curves (uniform in z on the generator's node range) are solved once; K = 21, 41, 81, 161 are every
(640/(K-1))-th of them. The reference is the hat-weighted CDF mixture at K = 641. Two schemes:
- CDF interpolation (the generator's, R6 as amended): the exact hat-weighted mixture of node CDFs;
- quantile interpolation (the plan's original R6): an individual's death time interpolates the two bracketing nodes'
  quantiles; its law by quadrature over z (4,001 points) and a 6,001-point u grid.
Per K and scheme: lambda = 2 n KL(P_K || P_ref) and 4/3 |dlnCV| / (1/2 sqrt(0.1/(2n))), at n = the configuration's
largest library n and at min(n, 20,000) (the cap Naveh set for heterogeneous configurations).

    .venv/bin/python tools/het_nodes_003d.py            # ~10 Mac CPU-min; needs a clean committed tree
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.special import ndtr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

GRID = {"N": 1500, "dt": 2e-4, "tau_max": 30.0, "Gtol": 1e-12, "nstart": 20}
KS, KREF, SIGMA, CAP = (21, 41, 81, 161), 641, 0.2, 20000
NAMED = ("k0.01-p05", "k0.0316-p05", "k0.1-p05", "k0.1-p00")       # the pilot's K = 81 failures


def _bound(n):
    return 0.5 * math.sqrt(0.1 / (2.0 * n))


def measure(point, rows):
    from sraging import generator as G
    P4 = dict(zip(("rho_eta", "rho_beta", "rho_eps", "kappa"), point))
    dL = min(r["delta"] / r["L_phys"] for r in rows)
    nmax = max(r["n"] for r in rows)
    nodes = G._het_nodes(SIGMA, KREF)
    curves = [G._curve(G._node_point(P4, s), {**GRID, "K": KREF}) for s in nodes["s"]]
    L = max(c["F"].size for c in curves)
    tau = np.arange(L) * GRID["dt"]
    Fm = np.vstack([np.concatenate([c["F"], np.full(L - c["F"].size, c["F"][-1])]) for c in curves])
    edges = np.arange(0.0, tau[-1] + dL, dL)

    def binP(c):
        P = np.diff(c)
        P[-1] += 1.0 - c[-1]
        return np.clip(P, 0.0, None)

    def bcv(P):
        a, P = edges[:-1], P / P.sum()
        mu = float(P @ a)
        return math.sqrt(max(float(P @ (a * a)) - mu * mu, 0.0)) / mu

    def cdf_scheme(K):
        idx = np.arange(0, KREF, (KREF - 1) // (K - 1))
        return binP(sum(Wj * np.interp(edges, tau, Fm[i]) for Wj, i in zip(G._het_nodes(SIGMA, K)["W"], idx)))

    U = ndtr(np.linspace(-8.5, 8.5, 6001))
    Q = np.vstack([G._inverse(U, tau, Fm[i][None, :], np.zeros(U.size, int), np.zeros(U.size))[0]
                   for i in range(KREF)])
    zq = np.linspace(nodes["z"][0], nodes["z"][-1], 4001)
    Z = 1.0 - ndtr(-1.0 / SIGMA)
    wq = np.exp(-0.5 * zq ** 2) / math.sqrt(2 * math.pi) / Z * (zq[1] - zq[0])
    wq[[0, -1]] *= 0.5
    wq[0] += (ndtr(zq[0]) - ndtr(-1.0 / SIGMA)) / Z
    wq[-1] += (1.0 - ndtr(zq[-1])) / Z

    def q_scheme(K):
        idx = np.arange(0, KREF, (KREF - 1) // (K - 1))
        j, w, _ = G._bracket(zq, nodes["z"][idx])
        c = np.zeros(edges.size)
        for i in range(zq.size):
            c += wq[i] * np.interp(edges, (1 - w[i]) * Q[idx[j[i]]] + w[i] * Q[idx[min(j[i] + 1, K - 1)]], U)
        return binP(c)

    Pref = cdf_scheme(KREF)
    cvr = bcv(Pref)
    out = {"point": list(point), "ids": sorted(r["cohort_id"] for r in rows)[:4], "delta_over_L": dL, "n_max": nmax,
           "base_cv": G.curve_moments(G._curve(P4, {**GRID, "K": 1}))["cv"],
           "reference_check_lambda_quantile_vs_cdf_at_641": 2 * nmax * G.kl(q_scheme(KREF), Pref), "rows": []}
    for K in KS:
        for scheme, P in (("cdf", cdf_scheme(K)), ("quantile", q_scheme(K))):
            kl, dl = G.kl(P, Pref), math.log(bcv(P) / cvr)
            for n in sorted({nmax, min(nmax, CAP)}):
                out["rows"].append({"K": K, "scheme": scheme, "n": n, "lambda": 2 * n * kl,
                                    "cv_ratio": 4 / 3 * abs(dl) / _bound(n),
                                    "pass": 2 * n * kl <= 0.05 and 4 / 3 * abs(dl) <= _bound(n)})
    for scheme in ("cdf", "quantile"):
        for n in sorted({nmax, min(nmax, CAP)}):
            out[f"K_needed_{scheme}_n{n}"] = next((r["K"] for r in out["rows"] if r["scheme"] == scheme
                                                   and r["n"] == n and r["pass"]), None)
    return out


TOLS = (0.05, 0.2, 0.5, 1.0)                      # lambda tolerances reported in --all mode


def _init():
    from sraging.compat.v033 import runtime
    runtime.configure()


def measure_all_point(point, configs):
    """--all: one point's node curves once; per heterogeneous configuration (its own delta/L and n), the CDF scheme's
    lambda and CV ratio at each K against K_ref, at n = min(n_max, CAP) and at n_max."""
    try:
        from sraging import generator as G
        P4 = dict(zip(("rho_eta", "rho_beta", "rho_eps", "kappa"), point))
        nodes = G._het_nodes(SIGMA, KREF)
        curves = [G._curve(G._node_point(P4, s), {**GRID, "K": KREF}) for s in nodes["s"]]
        L = max(c["F"].size for c in curves)
        tau = np.arange(L) * GRID["dt"]
        Fm = np.vstack([np.concatenate([c["F"], np.full(L - c["F"].size, c["F"][-1])]) for c in curves])
        base_cv = G.curve_moments(G._curve(P4, {**GRID, "K": 1}))["cv"]
        out = []
        for cf in configs:
            edges = np.arange(0.0, tau[-1] + cf["dL"], cf["dL"])

            def binP(c):
                P = np.diff(c)
                P[-1] += 1.0 - c[-1]
                return np.clip(P, 0.0, None)

            def bcv(P):
                a, P = edges[:-1], P / P.sum()
                mu = float(P @ a)
                return math.sqrt(max(float(P @ (a * a)) - mu * mu, 0.0)) / mu

            def cdf_scheme(K):
                idx = np.arange(0, KREF, (KREF - 1) // (K - 1))
                return binP(sum(Wj * np.interp(edges, tau, Fm[i])
                                for Wj, i in zip(G._het_nodes(SIGMA, K)["W"], idx)))
            Pref = cdf_scheme(KREF)
            cvr = bcv(Pref)
            rows = []
            ne = min(cf["n_max"], CAP)
            for K in KS:
                P = cdf_scheme(K)
                # lambda's KL merges cells to >= 5 expected individuals at the n it is judged at (Naveh 2026-10-14)
                kl, dl = G.kl_binned(P, Pref, ne), abs(math.log(bcv(P) / cvr))
                rows.append({"K": K, "kl": kl, "kl_at_n_max": G.kl_binned(P, Pref, cf["n_max"]), "dlncv": dl})
            rec = {"point": list(point), "base_cv": base_cv, "delta_over_L": cf["dL"], "n_max": cf["n_max"],
                   "n_eval": ne, "ids": cf["ids"][:3], "n_ids": len(cf["ids"]), "rows": rows}
            for n_label, n in (("n_eval", ne), ("n_max", cf["n_max"])):
                for tol in TOLS:
                    kk = "kl" if n_label == "n_eval" else "kl_at_n_max"
                    rec[f"K_lambda_{tol}_{n_label}"] = next((r["K"] for r in rows if 2 * n * r[kk] <= tol), None)
                rec[f"K_cv_{n_label}"] = next((r["K"] for r in rows if 4 / 3 * r["dlncv"] <= _bound(n)), None)
            out.append(rec)
        return {"point": list(point), "configs": out}
    except Exception as ex:  # noqa: BLE001
        import traceback
        return {"point": list(point), "error": f"{type(ex).__name__}: {ex}", "trace": traceback.format_exc()[-1500:]}


def main_all(args):
    import concurrent.futures as cf_
    import multiprocessing as mp
    import acceptance_003 as A
    import acceptance_003d as ACC
    _, rows = ACC.library_specs()
    het = [r for r in rows if r.get("het_param")]
    by = {}
    for r in het:
        p = (r["rho_eta"], r["rho_beta"], r["rho_eps"], r["kappa"])
        k = r["delta"] / r["L_phys"]
        c = by.setdefault(p, {}).setdefault(k, {"dL": k, "n_max": 0, "ids": []})
        c["n_max"] = max(c["n_max"], r["n"])
        c["ids"].append(r["cohort_id"])
    run = A.Run("het-nodes-all", manifests=[ACC.PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"all {len(by)} heterogeneous points x {KREF} node curves at N {GRID['N']}, dt {GRID['dt']}; "
                     f"{sum(len(v) for v in by.values())} configurations (~2-3 Mac CPU-h, {args.workers} workers)")
    res = []
    with cf_.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=_init) as ex:
        futs = [ex.submit(measure_all_point, list(p), list(v.values())) for p, v in by.items()]
        for f in cf_.as_completed(futs):
            res.append(f.result())
            print(f"  [het-nodes-all] {len(res)}/{len(futs)}", flush=True)
    run.write({"status": "written" if not any("error" in r for r in res) else "errors", "grid": GRID, "K": list(KS),
               "K_ref": KREF, "het_var": SIGMA, "cap": CAP, "tolerances": list(TOLS), "points": res},
              name="het_nodes_all")


def main(argv=None):
    import acceptance_003 as A
    import acceptance_003d as ACC
    from sraging import generator as G
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--all", action="store_true", help="every heterogeneous point, per configuration (CDF scheme)")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args(argv)
    if args.all:
        return main_all(args)
    _, rows = ACC.library_specs()
    het = [r for r in rows if r.get("het_param")]
    pts = {}
    for r in het:
        pts.setdefault((r["rho_eta"], r["rho_beta"], r["rho_eps"], r["kappa"]), []).append(r)
    cvs = sorted((G.curve_moments(G._curve(dict(zip(("rho_eta", "rho_beta", "rho_eps", "kappa"), p)),
                                           {**GRID, "K": 1}))["cv"], p) for p in pts)
    chosen = {"sharpest": cvs[0][1], "median": cvs[len(cvs) // 2][1], "broadest": cvs[-1][1]}
    for tag in NAMED:
        chosen[tag] = next(p for p, rs in pts.items() if any(r["cohort_id"].startswith(f"het-{tag}-") for r in rs))
    run = A.Run("het-nodes", manifests=[ACC.PANEL_PATH], dev=args.dev, suffix="003d",
                cost=f"{len(chosen)} heterogeneous points x {KREF} node curves at N {GRID['N']}, dt {GRID['dt']} "
                     f"(~10 Mac CPU-min)")
    res = {name: measure(p, pts[p]) for name, p in chosen.items()}
    run.write({"status": "written", "grid": GRID, "K": list(KS), "K_ref": KREF, "het_var": SIGMA, "cap": CAP,
               "n_het_points": len(pts), "points": res}, name="het_nodes")


if __name__ == "__main__":
    main()
