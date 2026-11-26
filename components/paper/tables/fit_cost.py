"""CPU cost of a full fit (fit + profile intervals) from the tester's stored fit records, by run, sharpness stratum
(DT_RULE in scope: CV_c < 0.05) and cohort size n. Also CPU per counted likelihood evaluation by grid size N.
Reads sr_calibration_tester/runs/<run>.json and data/store/<k[:2]>/<k[2:4]>/<key>.json. Writes fit_cost.csv,
fit_cost_per_eval.csv and fit_cost_cpus.csv next to this script.
Run: python -B fit_cost.py"""
import csv, json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)


HERE = os.path.dirname(os.path.abspath(__file__))
T = "aging:code_5/sr_calibration_tester"
RUNS = ["jax2-v6-full-bign", "jax2h-v6-full-bign", "jax2-v6-all-groups"]
BUCKETS = [("<1e3", 0, 1e3), ("1e3-5e3", 1e3, 5e3), ("5e3-2e4", 5e3, 2e4), ("2e4-5e4", 2e4, 5e4), (">=5e4", 5e4, 1e99)]
NS = [900, 1800, 3600]


def load(run):
    out = []
    for cid, d in PP.records(T, run):
        dg = d["claim_set"]["diagnostics"]
        df = dg["dt_floor"]; gr = df["grid_rule"]
        assert d["status"] == "ok"
        out.append(dict(cid=cid, cpu=d["cpu_seconds"], wall=d["wall_clock_seconds"], n=gr["n"], cv_c=gr["cv_c"],
                        scope=bool(gr["dt_in_scope"]), N=df["N"], nev=dg["n_eval"], ng=dg["n_grad"],
                        cpumodel=d["code_provenance"]["environment"]["cpu"]))
    return out


def rq(x):
    return round(x, 1) if x != "" else ""


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else ""


rows, per_eval, cpus = [], [], []
for run in RUNS:
    R = load(run)
    assert all(r["scope"] == (r["cv_c"] < 0.05) for r in R), "scope != CV_c<0.05"
    for stratum, f in [("all", lambda r: True), ("in_scope", lambda r: r["scope"]), ("out_of_scope", lambda r: not r["scope"])]:
        for bname, lo, hi in [("all", 0, 1e99)] + BUCKETS:
            S = [r for r in R if f(r) and lo <= r["n"] < hi]
            c = [r["cpu"] for r in S]
            row = dict(run=run, stratum=stratum, n_bucket=bname, count=len(S),
                       cpu_median=rq(q(c, 50)), cpu_q90=rq(q(c, 90)), cpu_max=round(max(c), 1) if c else "")
            for N in NS:
                row[f"N{N}"] = sum(1 for r in S if r["N"] == N)
            rows.append(row)
    for N in ["all"] + NS:
        S = [r for r in R if N == "all" or r["N"] == N]
        e = [r["cpu"] / (r["nev"] + r["ng"]) * 1e3 for r in S]
        per_eval.append(dict(run=run, N=N, count=len(S), cpu_ms_per_eval_median=round(q(e, 50), 1),
                             cpu_ms_per_eval_q90=round(q(e, 90), 1), cpu_median_s=round(q([r["cpu"] for r in S], 50), 1)))
    C = collections.Counter(r["cpumodel"] for r in R)
    for m, k in sorted(C.items(), key=lambda x: -x[1]):
        cpus.append(dict(run=run, cpu_model=m, count=k, cpu_median_s=round(q([r["cpu"] for r in R if r["cpumodel"] == m], 50), 1)))
for name, data in [("fit_cost.csv", rows), ("fit_cost_per_eval.csv", per_eval), ("fit_cost_cpus.csv", cpus)]:
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
print("saved", len(rows), len(per_eval), len(cpus))
