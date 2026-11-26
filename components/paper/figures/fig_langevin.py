"""KS distance between 25,000 Langevin death times and the FP CDF, per cell, grouped by preset; from tables/langevin_ks.csv.
Run: python -B fig_langevin.py"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "tables", "langevin_ks.csv"))))
PRE = ["mice_F", "Jack_Russell", "ecoli", "human"]
LAB = ["mouse", "dog", r"$E.\ coli$", "human"]
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7.2, 4.8))
rng = np.random.default_rng(1)
crit = {w: float(next(r["crit"] for r in rows if r["wall"] == w)) for w in ("clamp", "reflect")}
for i, p in enumerate(PRE):
    for r in rows:
        if r["preset"] != p: continue
        d = float(r["ks_D"])
        if r["row"] == "cell":
            x = i + (-0.12 if r["wall"] == "clamp" else 0.2) + rng.uniform(-0.1, 0.1)
            if r["wall"] == "clamp":
                ax.plot(x, d, "o", color="#1f77b4", ms=5.5, alpha=0.8, mew=0)
            else:
                ax.plot(x, d, "s", color="#d95f02", ms=6, alpha=0.9, mew=0)
        else:
            ax.plot(i - 0.12, d, "*", color="k", ms=15, mfc="#d62728", mew=1)
ax.axhline(crit["clamp"], color="#1f77b4", ls="--", lw=1.3)
ax.axhline(crit["reflect"], color="#d95f02", ls=":", lw=1.5)
ax.set_xticks(range(4)); ax.set_xticklabels(LAB)
ax.set_ylabel("KS distance $D$ (Langevin vs FP)")
ax.set_ylim(0, 0.042); ax.set_xlim(-0.6, 3.6)
h = [Line2D([], [], marker="o", color="#1f77b4", ls="none", ms=6, label="clamp cell"),
     Line2D([], [], marker="s", color="#d95f02", ls="none", ms=6, label="reflect cell"),
     Line2D([], [], marker="*", color="k", mfc="#d62728", ls="none", ms=12, label=r"plant, $\eta(1+\delta)$"),
     Line2D([], [], color="#1f77b4", ls="--", label="crit, clamp"), Line2D([], [], color="#d95f02", ls=":", label="crit, reflect")]
ax.legend(handles=h, loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.02))
fig.tight_layout()
fig.savefig(os.path.join(HERE, "out", "fig_langevin.pdf"))
print("saved")
