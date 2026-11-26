"""Grid-convergence tables for the current engine's N rule, from the stored I2 study (no new runs).
Inputs: reports/rc2/I2/data/{analysis_rows,census,grid2d_summary,rule_eval}.json.
Outputs: tables/convergence_cohorts.csv, convergence_grid2d.csv, convergence_summary.csv.
Run: python -B convergence.py"""
import csv, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)


HERE = os.path.dirname(os.path.abspath(__file__))
I2 = "aging:code_5/sr_calibration_tester_jax2/reports/rc2/I2/data/"
A = json.load(open(PP.path(I2 + "analysis_rows.json")))
C = {r["cid"]: r for r in json.load(open(PP.path(I2 + "census.json")))}
G2 = json.load(open(PP.path(I2 + "grid2d_summary.json")))
NS = (900, 1800, 3600, 7200, 14400)
COEF = (-3.034, 0.428, -0.900)   # the rounded constants of the implemented rule
Q, TARGET, NMAX = 0.6, 0.2, 3600


def g(d, N):
    return d.get(str(N), d.get(N))


def regime(r):
    nm, nb = g(r["noise_mle"], 7200), g(r["noise_best"], 7200)
    nb = 0.0 if nb is None or (isinstance(nb, float) and math.isnan(nb)) else nb
    return "A" if (nm is not None and nm < 0.01 and nb < 0.05) else "B"


def e900_pred(n, cv):
    return 10 ** (COEF[0] + COEF[1] * math.log10(n) + COEF[2] * math.log10(cv))


def rule_N(n, cv):
    e, k, N = e900_pred(n, cv), 0, 900
    while e * Q ** k > TARGET and N < NMAX:
        k += 1; N *= 2
    return N


rows = []
for r in A:
    if r["cid"] not in C:
        continue
    cv = C[r["cid"]]["cv"]            # the rule's own Kaplan-Meier CV
    E = {N: g(r["E_edge"], N) for N in NS}
    ok = r["n_edge"] > 0 and E[900] is not None and np.isfinite(E[900]) and E[900] > 0
    inreg = ok and regime(r) == "A" and "p15" not in r["cid"]
    Nr = rule_N(r["n"], cv)
    rows.append(dict(cid=r["cid"], group=r["group"], human=int(r["group"].startswith("H")), n=r["n"], cv=cv,
                     regime=regime(r), n_edge=r["n_edge"], in_regression=int(inreg),
                     **{f"E{N}": (E[N] if ok else float("nan")) for N in NS},
                     **{f"ratio_{N}_{2*N}": (E[2 * N] / E[N] if ok and E[N] > 0.05 else float("nan")) for N in NS[:3]},
                     e900_pred=e900_pred(r["n"], cv), rule_N=Nr,
                     E_rule=(E[Nr] if ok else float("nan")),
                     dt_shift_7200=r["dt_shift_7200"]))
cols = list(rows[0])
with open(os.path.join(HERE, "convergence_cohorts.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, cols); w.writeheader(); w.writerows(rows)

# N x dt study (12 cohorts): max edge error against the dt/8, N->inf reference
with open(os.path.join(HERE, "convergence_grid2d.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["cid", "n", "cv", "N", "dt_divisor", "E_edge", "E_best"])
    for cid, s in G2.items():
        for N, t in s["tab"].items():
            for i, d in enumerate((1, 2, 4, 8)):
                w.writerow([cid, s["n"], C[cid]["cv"], N, d, t["edge"][i], t["best"][i]])

RA = [r for r in rows if r["in_regression"]]
X = np.array([[1, math.log10(r["n"]), math.log10(r["cv"])] for r in RA])
y = np.array([math.log10(r["E900"]) for r in RA])
coef = np.linalg.lstsq(X, y, rcond=None)[0]
res = y - X @ np.array(COEF)
out = []


def add(k, v, note=""):
    out.append(dict(key=k, value=v, note=note))


add("n_measured", len(rows)); add("n_regime_A", sum(r["regime"] == "A" for r in rows))
add("n_regression", len(RA), "regime A without the 3 p15 cohorts and cohorts without an edge point")
add("n_regression_human", sum(r["human"] for r in RA)); add("n_regression_nonhuman", sum(1 - r["human"] for r in RA))
for i, c in enumerate(coef):
    add(f"refit_coef_{i}", c, "refit of the regression on the cohorts and CV used here")
add("resid_sd_dex_rounded_coef", res.std(), "residual sd of log10 E900 about the rounded rule")
add("resid_q90_dex", np.quantile(res, 0.9)); add("resid_max_dex", res.max())
add("resid_mean_human", np.mean([x for x, r in zip(res, RA) if r["human"]]))
add("resid_mean_nonhuman", np.mean([x for x, r in zip(res, RA) if not r["human"]]))
add("share_within_factor10_of_pred", np.mean(np.abs(res) <= 1.0), "share of regression cohorts with |residual| <= 1 dex")
add("share_within_1sd_of_pred", np.mean(np.abs(res) <= 0.56))
add("n_range_regression", f"{min(r['n'] for r in RA)}-{max(r['n'] for r in RA)}")
add("cv_range_regression", f"{min(r['cv'] for r in RA):.3f}-{max(r['cv'] for r in RA):.3f}")


def ratios(sel, Ns):
    return [r[f"ratio_{N}_{2*N}"] for r in sel for N in Ns if np.isfinite(r[f"ratio_{N}_{2*N}"])]


for lab, sel in (("all", RA), ("human", [r for r in RA if r["human"]]), ("nonhuman", [r for r in RA if not r["human"]]),
                 ("cv_lt_0.05", [r for r in RA if r["cv"] < 0.05])):
    v = ratios(sel, NS[:3])
    add(f"ratio_n_{lab}", len(v), "ratios E(2N)/E(N) with E(N)>0.05 nats, N=900,1800,3600 pooled")
    add(f"ratio_median_{lab}", np.median(v)); add(f"ratio_q75_{lab}", np.quantile(v, 0.75))
    add(f"ratio_q90_{lab}", np.quantile(v, 0.9))
v = ratios(RA, (900,)); add("ratio_900_1800_median_all", np.median(v))
v = ratios(RA, (3600,)); add("ratio_3600_7200_median_all", np.median(v))
for lab, sel in (("all", RA), ("human", [r for r in RA if r["human"]]), ("nonhuman", [r for r in RA if not r["human"]])):
    add(f"E900_median_{lab}", np.median([r["E900"] for r in sel])); add(f"E900_q90_{lab}", np.quantile([r["E900"] for r in sel], 0.9))
    add(f"E900_max_{lab}", max(r["E900"] for r in sel))
    for t in (0.2, 0.5):
        add(f"n_E900_le_{t}_{lab}", sum(r["E900"] <= t for r in sel))
        add(f"n_Erule_le_{t}_{lab}", sum(r["E_rule"] <= t for r in sel))
    add(f"Erule_median_{lab}", np.median([r["E_rule"] for r in sel])); add(f"Erule_max_{lab}", max(r["E_rule"] for r in sel))
    add(f"n_{lab}", len(sel))
for t in (0.2, 0.5):
    add(f"share_E900_le_{t}", np.mean([r["E900"] <= t for r in RA])); add(f"share_Erule_le_{t}", np.mean([r["E_rule"] <= t for r in RA]))
add("n_rule_N900", sum(r["rule_N"] == 900 for r in RA)); add("n_rule_N1800", sum(r["rule_N"] == 1800 for r in RA))
add("n_rule_N3600", sum(r["rule_N"] == 3600 for r in RA))
add("n_E14400_gt_0.2_human_n1e5", sum(1 for r in RA if r["human"] and r["n"] >= 1e5 and r["E14400"] > 0.2))
add("n_human_n1e5", sum(1 for r in RA if r["human"] and r["n"] >= 1e5))
for N in NS:
    add(f"human_E_median_N{N}", np.median([r[f"E{N}"] for r in RA if r["human"]]))
    add(f"nonhuman_E_median_N{N}", np.median([r[f"E{N}"] for r in RA if not r["human"]]))
# dt: humans, regime A, dt vs dt/2 at N=7200
hs = [r["dt_shift_7200"] for r in RA if r["human"]]
add("dt_shift_7200_median_human", np.median(hs)); add("dt_shift_7200_max_human", max(hs))
ss = [r["dt_shift_7200"] for r in RA if r["cv"] < 0.05]
add("dt_shift_7200_median_cvlt005", np.median(ss)); add("n_cvlt005", len(ss))
# grid2d: at N=3600, error with dt vs dt/8
def g2(cid, N, i):
    return G2[cid]["tab"][str(N)]["edge"][i]
sharp = [c for c in G2 if "real" not in c and "p15" not in c]; hum = [c for c in G2 if "real" in c]
add("grid2d_n", len(G2), "6 human + 6 sharp synthetic; the p15 cohort is left out of the sharp summaries as in the report"); add("grid2d_n_human", len(hum)); add("grid2d_n_sharp", len(sharp))
add("grid2d_sharp_N3600_dt1_median", np.median([g2(c, 3600, 0) for c in sharp]))
add("grid2d_sharp_N3600_dt8_median", np.median([g2(c, 3600, 3) for c in sharp]))
add("grid2d_sharp_N3600_dt1_max", max(g2(c, 3600, 0) for c in sharp)); add("grid2d_sharp_N3600_dt8_max", max(g2(c, 3600, 3) for c in sharp))
add("grid2d_human_N3600_dt1_median", np.median([g2(c, 3600, 0) for c in hum]))
add("grid2d_human_N3600_dt8_median", np.median([g2(c, 3600, 3) for c in hum]))
add("grid2d_human_maxabs_dt_effect", max(abs(g2(c, N, 0) - g2(c, N, 3)) for c in hum for N in ("900", "1800", "3600", "7200")),
    "max over human cohorts and N<=7200 of |E(dt)-E(dt/8)|")
add("n_min_regression", min(r["n"] for r in RA)); add("n_max_regression", max(r["n"] for r in RA))
add("cv_min_regression", min(r["cv"] for r in RA)); add("cv_max_regression", max(r["cv"] for r in RA))
add("share_resid_positive", np.mean(res > 0), "share of regression cohorts whose measured E900 exceeds the rule's prediction")
add("n_cv_lt_0.05_regression", sum(r["cv"] < 0.05 for r in RA))
h1 = [r for r in RA if r["human"] and r["n"] >= 1e5]
for N in (900, 3600, 14400):
    add(f"human_n1e5_E_median_N{N}", np.median([r[f"E{N}"] for r in h1]))
for lab, n_, cv_ in (("a", 1000, 0.2), ("b", 10000, 0.1), ("c", 100000, 0.15)):
    add(f"example_{lab}_n", n_); add(f"example_{lab}_cv", cv_)
    add(f"example_{lab}_E900_pred", e900_pred(n_, cv_)); add(f"example_{lab}_rule_N", rule_N(n_, cv_))
    add(f"example_{lab}_E_at_ruleN_pred", e900_pred(n_, cv_) * Q ** int(math.log2(rule_N(n_, cv_) / 900)))
with open(os.path.join(HERE, "convergence_summary.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, ["key", "value", "note"]); w.writeheader()
    for o in out:
        w.writerow({**o, "value": (f"{o['value']:.6g}" if isinstance(o["value"], (float, np.floating)) else o["value"])})
print(len(rows), len(RA))
