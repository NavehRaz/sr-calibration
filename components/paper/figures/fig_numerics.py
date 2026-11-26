"""Figure for Section 7a (numerical accuracy). Reads stored acceptance results only; no evaluations are run.
Run: python -B fig_numerics.py   ->  out/fig_numerics.pdf
Sources: ~/code/sraging/tests/acceptance/results/{mpref_002,score_002,score_window_002,score_het_002}.json
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)

R = "~/code/sraging/tests/acceptance/results/"
plt.rcParams.update({"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16, "legend.fontsize": 12,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
OLD, NEW = "#d62728", "#1f77b4"
FLOOR = 1e-16

mp = json.load(open(PP.path(R + "mpref_002.json")))
fig, ax = plt.subplots(1, 3, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
fig.subplots_adjust(wspace=0.42, left=0.075, right=0.99, top=0.9, bottom=0.2)

# A: low-mass rows, 108 mpmath-referenced evaluations
a = ax[0]
rows = sorted(mp["routed"], key=lambda r: r["sum_abs_dl_v033"])
x = np.arange(len(rows))
a.plot(x, [max(r["sum_abs_dl_v033"], FLOOR) for r in rows], "o", color=OLD, ms=4, label="earlier subtraction form")
a.plot(x, [max(r["sum_abs_dl_v1"], FLOOR) for r in rows], "o", color=NEW, ms=4, label="absorbed-flux form")
a.axhline(mp["thresholds"]["per_evaluation"], color="k", ls="--", lw=1.3)
a.text(2, 1.6e-5, r"$10^{-5}$ nats", fontsize=12)
a.set_yscale("log"); a.set_ylim(1e-17, 1e4)
a.set_xlabel("evaluation (sorted by earlier error)")
a.set_ylabel(r"$\sum|\Delta\ell|$ vs mpmath (nats)")
a.set_title("A  Low-probability rows", loc="left")
a.legend(frameon=False, loc="upper left")

# B: round-off panel (17 evaluations)
a = ax[1]
rows = sorted(mp["roundoff"], key=lambda r: abs(r["dl_v033"]))
x = np.arange(len(rows))
a.plot(x, [max(abs(r["dl_v033"]), FLOOR) for r in rows], "o", color=OLD, ms=7, label="earlier subtraction form")
a.plot(x, [max(abs(r["dl_v1"]), FLOOR) for r in rows], "o", color=NEW, ms=7, label="absorbed-flux form")
bad = [i for i, r in enumerate(rows) if not r["ok"]]
a.plot(bad, [max(abs(rows[i]["dl_v1"]), FLOOR) for i in bad], "o", mfc="none", mec="k", ms=13, mew=1.5,
       label="fails the acceptance test")
a.axhline(mp["thresholds"]["per_evaluation"], color="k", ls="--", lw=1.3)
a.set_yscale("log"); a.set_ylim(1e-17, 1e9)
a.set_xlabel("evaluation (sorted by earlier error)")
a.set_ylabel(r"$|\Delta\ell_{\rm total}|$ vs mpmath (nats)")
a.set_title("B  Round-off panel", loc="left")
a.legend(frameon=False, loc="upper left", fontsize=11)

# C: score tests
a = ax[2]
labels, zs, cols = [], [], []
def add(path, feat, name, key=None):
    f = json.load(open(PP.path(R + path)))["features"][feat]
    for c, z in zip(f["components"], f["z"]):
        labels.append(f"{name} {c[:3]}"); zs.append(z)
    return json.load(open(PP.path(R + path)))["zcrit"]
zcrit = add("score_002.json", "pure", "pure")
add("score_002.json", "hazard", "hazard")
add("score_002.json", "entry", "entry")
add("score_window_002.json", "window", "window")
add("score_het_002.json", "heterogeneity", "het")
feat_of = [l.split()[0] for l in labels]
order = ["pure", "hazard", "entry", "window", "het"]
palette = {"pure": "#1f77b4", "hazard": "#ff7f0e", "entry": "#2ca02c", "window": "#9467bd", "het": "#8c564b"}
for f in order:
    idx = [i for i, ff in enumerate(feat_of) if ff == f]
    a.plot(idx, [zs[i] for i in idx], "o", color=palette[f], ms=8, label=f)
a.axhline(zcrit, color="k", ls="--", lw=1.3); a.axhline(-zcrit, color="k", ls="--", lw=1.3)
a.axhline(0, color="#888888", lw=0.8)
a.set_ylim(-5.4, 4.3)
a.set_xlabel("feature and parameter"); a.set_ylabel(r"$z$ of the mean score at the truth")
a.set_xticks([])
a.set_title("C  Score tests", loc="left")
a.legend(frameon=False, ncol=3, loc="lower center", fontsize=11, columnspacing=0.8, handletextpad=0.2)
a.text(len(zs) - 0.5, zcrit + 0.12, f"$|z|={zcrit:.2f}$", ha="right", fontsize=12)

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
fig.savefig(os.path.join(HERE, "out", "fig_numerics.pdf"))
print("ok", zcrit, len(zs))
