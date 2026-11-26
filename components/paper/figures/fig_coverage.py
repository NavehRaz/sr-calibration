"""Closed-interval coverage (Wilson 95% CI) and median closed width per arm, J and H, from tables/coverage_by_arm.csv.
Run: python -B fig_coverage.py"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "tables", "coverage_by_arm.csv"))))
R = {(r["arm"], r["method"], r["param"]): r for r in rows}
ARMS = ["pure", "censor", "hazard", "window", "het", "realfull"]
PARS = [("eta", r"$\eta$", "#1f77b4", "o"), ("epsilon", r"$\varepsilon$", "#d95f02", "s"), ("beta", r"$\beta$", "#2ca02c", "^")]
plt.rcParams.update({"font.size": 16, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(15, 5.6))
x0 = np.arange(len(ARMS))
slots = [(m, p) for p in PARS for m in "JH"]
off = np.linspace(-0.32, 0.32, len(slots))
for k, (m, (p, lab, col, mk)) in enumerate(slots):
    cov, lo, hi, wd, xs = [], [], [], [], []
    for i, a in enumerate(ARMS):
        r = R[(a, m, p)]
        if int(r["n_closed"]) == 0:
            continue
        c = float(r["coverage"]); xs.append(x0[i] + off[k]); cov.append(c)
        lo.append(c - float(r["wilson_lo"])); hi.append(float(r["wilson_hi"]) - c); wd.append(float(r["median_width_dex"]))
    kw = dict(marker=mk, ms=8, color=col, mfc=col if m == "J" else "white", mew=1.8, ls="none", capsize=3, elinewidth=1.5)
    axs[0].errorbar(xs, cov, yerr=[lo, hi], **kw)
    axs[1].plot(xs, wd, **{k_: v for k_, v in kw.items() if k_ not in ("capsize", "elinewidth")})
axs[0].axhline(0.95, color="k", lw=1.2, ls="--")
axs[0].set_ylim(-0.02, 1.04); axs[0].set_ylabel("closed-interval coverage")
axs[1].set_yscale("log"); axs[1].set_ylabel("median closed width (dex)")
for ax in axs:
    ax.set_xticks(x0); ax.set_xticklabels(ARMS, rotation=20)
    for xx in x0[:-1] + 0.5: ax.axvline(xx, color="0.92", lw=1, zorder=0)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker=mk, color=col, ls="none", ms=8, label=lab) for _, lab, col, mk in PARS]
h += [Line2D([], [], marker="o", color="k", ls="none", ms=8, label="J"),
      Line2D([], [], marker="o", color="k", mfc="white", mew=1.8, ls="none", ms=8, label="H")]
axs[1].legend(handles=h, loc="upper left", ncol=1, frameon=False, bbox_to_anchor=(1.0, 1.0))
axs[0].text(-0.14, 1.05, "A", transform=axs[0].transAxes, fontsize=20, fontweight="bold")
axs[1].text(-0.14, 1.05, "B", transform=axs[1].transAxes, fontsize=20, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "out", "fig_coverage.pdf"))
print("saved")
