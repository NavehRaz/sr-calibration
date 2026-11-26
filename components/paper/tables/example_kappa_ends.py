"""Model curves of the worked example at the two ends of kappa's range: the exact FP survival and hazard at
diagnostics["kappa_ends"]["theta_lo"] and ["theta_hi"] of tables/example_refit_v010.json (the example refit with
sraging tag v0.1.0; the ends are points within 1.92 nats of the best fit, as sraging's plots.overlay draws them).
Same ages as tables/example_curve.csv; no fit, two engine evaluations.

Needs the tagged package (sraging 0.1.0), so it runs in a venv of the tag:
  <venv of sraging v0.1.0>/bin/python -B tables/example_kappa_ends.py
    ->  tables/example_kappa_ends.csv (curves) and tables/example_kappa_ends_summary.csv (key,value)
The header line records the package version and the sha256 of the refit.
"""
import hashlib, json, os

import numpy as np

import sraging
from sraging.params import SRParams
from sraging.population import population

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.join(HERE, "example_refit_v010.json")

raw = open(FIT, "rb").read()
fit = json.loads(raw)
ends = fit["diagnostics"]["kappa_ends"]
unit = fit["theta_hat"]["time_unit"]
cur = np.genfromtxt(os.path.join(HERE, "example_curve.csv"), delimiter=",", names=True, skip_header=1)
ages = cur["age_days"]

cols = {}
for end in ("lo", "hi"):
    eta, beta, eps, kappa = (float(v) for v in ends[f"theta_{end}"])
    pop = population(SRParams(eta=eta, beta=beta, epsilon=eps, kappa=kappa, h_ext=0.0, time_unit=unit), ages)
    cols[f"S_k{end}"] = np.asarray(pop.S, float)
    cols[f"hazard_k{end}"] = np.asarray(pop.hazard, float)

out = os.path.join(HERE, "example_kappa_ends.csv")
with open(out, "w") as f:
    f.write(f"# sraging {sraging.__version__}; refit sha256 {hashlib.sha256(raw).hexdigest()}; "
            f"kappa_lo {ends['kappa_lo']:.6g}, kappa_hi {ends['kappa_hi']:.6g}; {ends['definition']}\n")
    f.write("age_days," + ",".join(cols) + "\n")
    for i, a in enumerate(ages):
        f.write(f"{a:.6g}," + ",".join(f"{cols[k][i]:.10g}" for k in cols) + "\n")

# how far apart the two ends are, over the ages of the data (to the last recorded age)
obs = ages <= ages.max() / 1.05                  # example_curve.py runs its ages to 1.05 x the last recorded age
dS = np.abs(cols["S_khi"] - cols["S_klo"])[obs]
hz = obs & (cols["hazard_klo"] > 1e-6) & (cols["hazard_khi"] > 1e-6)
ratio = cols["hazard_khi"][hz] / cols["hazard_klo"][hz]
mid = hz & (cols["S_klo"] <= 0.99) & (cols["S_klo"] >= 0.01)    # the ages of the central 98 % of the model's deaths
rmid = cols["hazard_khi"][mid] / cols["hazard_klo"][mid]
summ = {"kappa_lo": ends["kappa_lo"], "kappa_hi": ends["kappa_hi"],
        "box_limited_lo": int(ends["box_limited_lo"]), "box_limited_hi": int(ends["box_limited_hi"]),
        "max_abs_dS": float(dS.max()), "hazard_ratio_min": float(ratio.min()), "hazard_ratio_max": float(ratio.max()),
        "hazard_from_day": float(ages[hz].min()), "n_ages": int(obs.sum()),
        "mid_from_day": float(ages[mid].min()), "mid_to_day": float(ages[mid].max()),
        "hazard_ratio_mid_min": float(rmid.min()), "hazard_ratio_mid_max": float(rmid.max())}
out2 = os.path.join(HERE, "example_kappa_ends_summary.csv")
with open(out2, "w") as f:
    f.write("key,value\n")
    for k, v in summ.items():
        f.write(f"{k},{v:.6g}\n")
print("wrote", out, out2)
for k, v in summ.items():
    print(f"  {k} = {v:.6g}")
