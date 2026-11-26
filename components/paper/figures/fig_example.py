"""Worked-example figure (NIA ITP UM-HET3 control females, cohort C2016): A, Kaplan-Meier survival of the data with
the model survival at the stored theta-hat and at the two ends of kappa's range; B, the same model hazards against the
data's hazard in 60-day bins.
Reads the stored example CSV, ../tables/example_curve.csv (one engine evaluation at the stored fit, made by
tables/example_curve.py) and ../tables/example_kappa_ends.csv (the kappa ends of the v0.1.0 refit, made by
tables/example_kappa_ends.py); no fit.
Run: python -B fig_example.py   ->  out/fig_example.pdf
"""
import csv, hashlib, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)

CSV = PP.path("~/code/sraging/examples/data/itp_c2016_het3_control_female.csv")
SHA = "d2c1066a8907c6475bbff153bd557e21171ce7fc3eb23077e7debaff9b365aa3"
assert hashlib.sha256(open(CSV, "rb").read()).hexdigest() == SHA, "example CSV changed"

rows = list(csv.DictReader(open(CSV)))
age = np.array([float(r["death times"]) for r in rows])
ev = np.array([int(r["events"]) for r in rows])

# Kaplan-Meier (ties handled by grouping equal ages)
order = np.argsort(age, kind="stable")
age, ev = age[order], ev[order]
ts, ss = [0.0], [1.0]
S, at_risk = 1.0, len(age)
i = 0
while i < len(age):
    j = i
    while j < len(age) and age[j] == age[i]:
        j += 1
    d = int(ev[i:j].sum())
    if d:
        S *= 1 - d / at_risk
        ts.append(age[i]); ss.append(S)
    at_risk -= j - i
    i = j

plt.rcParams.update({"font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 18, "ytick.labelsize": 18,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
mod = np.genfromtxt(os.path.join(HERE, "..", "tables", "example_curve.csv"), delimiter=",", names=True, skip_header=1)
ke = np.genfromtxt(os.path.join(HERE, "..", "tables", "example_kappa_ends.csv"), delimiter=",", names=True, skip_header=1)
assert np.array_equal(ke["age_days"], mod["age_days"]), "kappa-end curves on other ages"
# the kappa = 0.001 end lies under theta-hat (kappa-hat = 0.001), so it is drawn wide and pale to show beneath it
ENDS = (("klo", r"$\kappa=0.001$ end", "C0", "-", 5.0, 0.4), ("khi", r"$\kappa=300$ end", "C2", ":", 2.6, 1.0))

# data hazard in 60-day bins: deaths / person-days at risk
BIN = 60.0
edges = np.arange(0.0, age.max() + BIN, BIN)
mid, haz = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    exposure = np.clip(np.minimum(age, hi) - lo, 0, None).sum()
    d = int(((age >= lo) & (age < hi) & (ev == 1)).sum())
    if d > 0 and exposure > 0:
        mid.append(0.5 * (lo + hi)); haz.append(d / exposure)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
a1.step(ts, ss, where="post", color="k", lw=2.2, label="data (KM)")
a1.plot(mod["age_days"], mod["S"], color="C3", lw=2.2, ls="--", label=r"model at $\hat\theta$", zorder=3)
for k, lab, col, ls, lw, al in ENDS:
    a1.plot(ke["age_days"], ke[f"S_{k}"], color=col, lw=lw, ls=ls, alpha=al, label=lab)
a1.set_xlabel("Age (days)"); a1.set_ylabel("Survival"); a1.set_ylim(0, 1.02); a1.set_xlim(0, None)
a1.legend(frameon=False, fontsize=15, loc="lower left")
a2.semilogy(mid, haz, "o", color="k", ms=6, label=f"data ({int(BIN)}-day bins)")
keep = (mod["hazard_per_day"] > 1e-6) & (mod["age_days"] <= age.max())
a2.semilogy(mod["age_days"][keep], mod["hazard_per_day"][keep], color="C3", lw=2.2, ls="--", label=r"model at $\hat\theta$", zorder=3)
for k, lab, col, ls, lw, al in ENDS:
    kk = (ke[f"hazard_{k}"] > 1e-6) & (ke["age_days"] <= age.max())
    a2.semilogy(ke["age_days"][kk], ke[f"hazard_{k}"][kk], color=col, lw=lw, ls=ls, alpha=al, label=lab)
a2.set_xlabel("Age (days)"); a2.set_ylabel(r"Hazard (day$^{-1}$)"); a2.set_xlim(0, None)
a2.legend(frameon=False, fontsize=15, loc="upper left")
for a, L in ((a1, "A"), (a2, "B")):
    a.text(-0.16, 1.02, L, transform=a.transAxes, fontsize=22, fontweight="bold", va="bottom")
fig.tight_layout()
out = os.path.join(HERE, "out", "fig_example.pdf")
fig.savefig(out)
print("wrote", out, "KM points", len(ts), "hazard bins", len(mid))
