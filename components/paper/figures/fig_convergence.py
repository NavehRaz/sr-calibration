"""Grid convergence of the current engine's deviance (stored I2 study): error vs N, rule prediction, error vs time step.
Reads tables/convergence_cohorts.csv and convergence_grid2d.csv (written by tables/convergence.py).
Run: python -B fig_convergence.py"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "..", "tables")
rows = [r for r in csv.DictReader(open(os.path.join(T, "convergence_cohorts.csv"))) if r["in_regression"] == "1"]
g2 = list(csv.DictReader(open(os.path.join(T, "convergence_grid2d.csv"))))
NS = [900, 1800, 3600, 7200, 14400]
plt.rcParams.update({"font.size": 16, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 13, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 3, figsize=(17, 5.6))
cmap = plt.get_cmap("viridis")

# A: error vs N, coloured by log10 CV
ax = axs[0]
lcv = np.array([np.log10(float(r["cv"])) for r in rows])
norm = matplotlib.colors.Normalize(lcv.min(), lcv.max())
for r, c in zip(rows, lcv):
    ax.plot(NS, [float(r[f"E{N}"]) for N in NS], color=cmap(norm(c)), lw=0.9, alpha=0.7)
ax.axhline(0.2, color="k", ls="--", lw=1.3); ax.text(14400, 0.22, "0.2 nats", ha="right", va="bottom", fontsize=13)
x = np.array([3600, 14400.0])
ax.plot(x, 6 * (x / 3600) ** -1.0, color="0.2", lw=2, ls=":"); ax.plot(x, 6 * (x / 3600) ** -0.737, color="0.2", lw=2, ls="-.")
ax.text(3600, 6.6, r"slope $-1$ ($q=0.5$)", fontsize=12, color="0.2", ha="left", va="bottom")
ax.text(14400, 4.9, r"$q=0.6$", fontsize=12, color="0.2", ha="right", va="bottom")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS], rotation=20)
ax.minorticks_off(); ax.set_ylim(3e-3, 14); ax.set_xlabel("space cells $N$"); ax.set_ylabel("deviance error (nats)")
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
cb.set_label(r"$\log_{10}$ CV", fontsize=14); cb.ax.tick_params(labelsize=12)

# B: measured E900 vs prediction
ax = axs[1]
pred = np.array([float(r["e900_pred"]) for r in rows]); meas = np.array([float(r["E900"]) for r in rows])
hum = np.array([r["human"] == "1" for r in rows])
xs = np.array([1e-3, 30.0]); sd = 0.557
ax.fill_between(xs, xs * 10 ** -sd, xs * 10 ** sd, color="0.88", lw=0, zorder=0)
ax.plot(xs, xs, color="k", lw=1.3)
ax.scatter(pred[~hum], meas[~hum], s=38, color="#1f77b4", alpha=0.8, lw=0, label="non-human")
ax.scatter(pred[hum], meas[hum], s=46, color="#d95f02", alpha=0.9, lw=0, marker="s", label="human")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2e-3, 20); ax.set_ylim(2e-3, 20)
ax.set_xlabel(r"predicted $E_{900}$ (nats)"); ax.set_ylabel(r"measured $E_{900}$ (nats)")
ax.legend(frameon=False, loc="upper left")

# C: error vs time step at N=3600
ax = axs[2]
cids = list(dict.fromkeys(r["cid"] for r in g2))
for cid in cids:
    if "p15" in cid:
        continue
    v = [(int(r["dt_divisor"]), float(r["E_edge"])) for r in g2 if r["cid"] == cid and r["N"] == "3600"]
    v.sort(); real = "real" in cid
    ax.plot([a for a, _ in v], [b for _, b in v], marker="s" if real else "o", ms=7, lw=1.6,
            color="#d95f02" if real else "#1f77b4", alpha=0.85)
ax.axhline(0.2, color="k", ls="--", lw=1.3)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
ax.minorticks_off(); ax.set_xlabel(r"time-step divisor ($\Delta t/\Delta t_0$)"); ax.set_ylabel("deviance error at $N=3600$ (nats)")
ax.set_ylim(0.03, 6)
ax.legend(handles=[Line2D([], [], marker="s", color="#d95f02", ls="-", ms=7, label="human (6)"),
                   Line2D([], [], marker="o", color="#1f77b4", ls="-", ms=7, label="sharp synthetic (5)")],
          frameon=False, loc="lower left")
for ax, l in zip(axs, "ABC"):
    ax.text(-0.17 if l != "A" else -0.17, 1.05, l, transform=ax.transAxes, fontsize=20, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "out", "fig_convergence.pdf"))
print("saved")
