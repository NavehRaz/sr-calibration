"""Figure for Section 7c (identifiability in the benchmark). Reads stored kfix and v6 results only (through the
derived tables of tables/d2b_ident.py); no fits. Run tables/d2b_ident.py first.
Run: python -B fig_ident.py   ->  out/fig_ident.pdf
"""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "..", "tables")
plt.rcParams.update({"font.size": 16, "axes.labelsize": 17, "axes.titlesize": 17, "legend.fontsize": 13,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
COL = {"eta": "#1f77b4", "epsilon": "#ff7f0e", "sqrt_eta_over_eps": "#2ca02c", "beta_rbar": "#d62728",
       "beta_over_kp1": "#9467bd", "beta": "#7f7f7f"}
LAB = {"eta": r"$\eta$", "epsilon": r"$\varepsilon$", "sqrt_eta_over_eps": r"$\sqrt{\eta}/\varepsilon$",
       "beta_rbar": r"$\beta\,\bar r$", "beta_over_kp1": r"$\beta/(1+\kappa)$", "beta": r"$\beta$ (at $\kappa_0$)"}


def rd(name):
    with open(os.path.join(T, name)) as f:
        return list(csv.DictReader(f))


ridge = rd("d2b_ridge.csv")
coh = {}
for r in ridge:
    coh.setdefault(r["cohort_id"], []).append(r)
for v in coh.values():
    v.sort(key=lambda r: float(r["kappa0"]))
valid = {c: v for c, v in coh.items() if v[0]["valid"] == "1"}
K0 = [float(r["kappa0"]) for r in next(iter(coh.values()))]

fig, ax = plt.subplots(2, 3, figsize=(14.5, 9.5))
fig.subplots_adjust(hspace=0.5, wspace=0.42, left=0.08, right=0.99, top=0.95, bottom=0.1)

# A: one ridge
ex = "pure-k0.1-p12-c0-s0"
a = ax[0, 0]
v = coh[ex]
for q in ("beta", "beta_over_kp1", "beta_rbar"):
    a.plot(K0, [float(r[q]) for r in v], "o-", color=COL[q], lw=2, ms=6, label=LAB[q])
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"pinned $\kappa_0$"); a.set_ylabel("fitted value")
a.set_title("A  One cohort", loc="left")
a.legend(frameon=False, loc="upper left")
a.text(0.03, 0.40, "true $\\kappa=0.1$", transform=a.transAxes, ha="left", fontsize=12)

# B: loglik cost of pinning
a = ax[0, 1]
for c, v in coh.items():
    ok = c in valid
    a.plot(K0, [-float(r["dll"]) for r in v], "-", color="#444444" if ok else "#bbbbbb", lw=1.2 if ok else 0.8,
           alpha=0.9 if ok else 0.7)
a.axhline(1.92, color="#d62728", ls="--", lw=1.8)
a.text(0.001 * 1.2, 2.05, "1.92 nats", color="#d62728", fontsize=12, va="bottom")
a.set_xscale("log"); a.set_xlabel(r"pinned $\kappa_0$")
a.set_ylabel(r"$\ell_{\max}-\ell(\kappa_0)$ (nats)")
a.set_ylim(-0.1, 7)
a.set_title("B  Cost of pinning", loc="left")
a.text(0.03, 0.95, "dark: valid (19)\nlight: other (17)", transform=a.transAxes, ha="left", va="top", fontsize=12)

# C: drift of each readout across kappa0
a = ax[0, 2]
sel = {}
for c, v in valid.items():
    br = np.array([float(r["beta_rbar"]) for r in v])
    if np.median(br) >= 1e-3:
        sel[c] = v
for q in ("beta", "beta_over_kp1", "beta_rbar"):
    Y = []
    for c, v in sel.items():
        y = np.array([float(r[q]) for r in v])
        Y.append(np.log10(y / np.median(y)))
    Y = np.array(Y)
    for y in Y:
        a.plot(K0, y, "-", color=COL[q], lw=0.7, alpha=0.25)
    a.plot(K0, np.median(Y, axis=0), "o-", color=COL[q], lw=3, ms=7, label=LAB[q])
a.set_xscale("log"); a.set_xlabel(r"pinned $\kappa_0$")
a.set_ylabel("log$_{10}$(value / cohort median)")
a.set_title("C  Readout drift", loc="left")
a.legend(frameon=False, loc="upper left")

# D: coverage lost on the panel
a = ax[1, 0]
P = {}
for r in rd("d2b_kfix_panel.csv"):
    P[(r["kappa0"], r["quantity"])] = int(r["hits"])
ks = ["0.001", "0.1", "0.5", "1", "2", "300"]
xs = np.arange(len(ks))
for q in ("beta_over_kp1", "beta_rbar", "eta", "epsilon"):
    lost = [P[("free", q)] - P[(k, q)] for k in ks]
    off = {"beta_over_kp1": -0.09, "beta_rbar": -0.03, "eta": 0.03, "epsilon": 0.09}[q]
    a.plot(xs + off, lost, "o-", color=COL[q], lw=2, ms=7, label=LAB[q])
a.set_xticks(xs); a.set_xticklabels(ks)
a.set_xlabel(r"pinned $\kappa_0$"); a.set_ylabel("valid cohorts lost (of 19)")
a.set_title("D  Calibration lost", loc="left")
a.legend(frameon=False, loc="upper left")

# E: net coverage loss at the three released names
a = ax[1, 1]
N = rd("d2b_kfix_net.csv")
groups = [("J", 0.001), ("J", 0.5), ("J", 1.0), ("H", 0.001), ("H", 0.5), ("H", 1.0)]
qs = ["eta", "epsilon", "sqrt_eta_over_eps", "beta_rbar"]
w = 0.19
for i, q in enumerate(qs):
    vals = []
    for var, k in groups:
        r = [x for x in N if x["variant"] == var and float(x["kappa0"]) == k and x["quantity"] == q][0]
        vals.append(int(r["net"]))
    a.bar(np.arange(len(groups)) + (i - 1.5) * w, vals, w, color=COL[q], label=LAB[q])
a.set_xticks(range(len(groups)))
a.set_xticklabels([f"{v}\n{k:g}" for v, k in groups])
a.set_xlabel(r"variant and $\kappa_0$"); a.set_ylabel("net cohorts lost")
a.set_title("E  Net loss (v5)", loc="left")
a.legend(frameon=False, ncol=2, loc="upper left")
a.set_ylim(0, 17)

# F: kappa declined by arm
a = ax[1, 2]
D = rd("d2b_kappa_declined.csv")
arms = ["pure", "censor", "window", "het", "realfull", "hazard", "all"]
w = 0.38
for i, (var, c) in enumerate((("J", "#444444"), ("H", "#9ecae1"))):
    vals = [float([x for x in D if x["variant"] == var and x["arm"] == arm][0]["frac"]) for arm in arms]
    a.bar(np.arange(len(arms)) + (i - 0.5) * w, vals, w, color=c, label=var)
a.set_xticks(range(len(arms))); a.set_xticklabels(arms, rotation=35, ha="right")
a.set_ylabel(r"fraction with $\kappa$ declined"); a.set_ylim(0, 1.0)
a.set_title("F  $\\kappa$ declined (v6)", loc="left")
a.legend(frameon=False, loc="upper right", ncol=2)

os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
fig.savefig(os.path.join(HERE, "out", "fig_ident.pdf"))
print("ok", len(sel), "cohorts in C")
