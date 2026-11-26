"""Explanatory figure for the profile walk, its three passes, the interval and the two safeguards.
A: walk replay (the real step rules) on an illustrative profile. B: schematic passes. C: real stored walk
(sqrt(eta)/eps of one benchmark cohort; a one-sided case) + interpolation sketch. D: schematic safeguards.
Run: python -B fig_profile.py"""
import json, os, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)

WALK = PP.path("~/code/sraging/tests/acceptance/results/walker_step1_003.json")
BLUE, ORG, GRN, RED, GRY = "#1f77b4", "#d95f02", "#2ca02c", "#c0392b", "0.45"
plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False})
ANN = dict(fontsize=12)

# constants of the text
STEP0, STEPMAX, JUMP, STOP, SHRINK, CUT = 0.05, 2.0, 4.0, 10.0, 3.0, 1.92

def replay(Dfun, sgn, box):
    """The walk's step rule (first step 0.05 dex; x3 if D<1, else x2 if D rose by <0.3; cap 2; a step that
    raises D by more than 4 is rejected and retried at step/3; stop at D>=10 or the box)."""
    x, step, dprev, nodes, rej, steps = 0.0, STEP0, 0.0, [], [], []
    while len(nodes) < 24:
        nxt = x + sgn * step
        at_box = abs(nxt) >= box
        if at_box: nxt = sgn * box
        d = Dfun(nxt)
        if d - dprev > JUMP:
            rej.append((nxt, d, x)); step /= SHRINK; continue
        nodes.append((nxt, d)); steps.append(abs(nxt - x))
        if d >= STOP or at_box: break
        if d < 1: step = min(step * 3, STEPMAX)
        elif d - dprev < 0.3: step = min(step * 2, STEPMAX)
        dprev, x = d, nxt
    return nodes, rej, steps, ("box" if at_box else "D>=10")

def Dill(x):  # illustrative profile: steep, cliff-like rise on the + side; shallow plateau on the - side
    if x >= 0:
        return (x / 0.8) ** 2 + 20 * max(0, x - 1.3) ** 2
    return 2.6 * (1 - math.exp(-(x / 1.1) ** 2)) + 0.03 * abs(x)

fig = plt.figure(figsize=(15, 16.5))
gs = GridSpec(4, 12, figure=fig, height_ratios=[1.55, 1.15, 1.15, 1.2], hspace=0.62, wspace=2.0,
              left=0.065, right=0.985, top=0.975, bottom=0.05)

# ---------------- A ----------------
gA = gs[0, :].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08)
a1 = fig.add_subplot(gA[0]); a2 = fig.add_subplot(gA[1], sharex=a1)
BOXL, BOXR = 6.0, 3.0
up, upr, ups, upst = replay(Dill, +1, BOXR); upr = upr[:1]
lo, lor, los, lost = replay(Dill, -1, BOXL)
xx = np.linspace(-BOXL, BOXR, 600)
a1.plot(xx, [Dill(x) for x in xx], color="0.75", lw=2.5, zorder=1)
a1.plot([0] + [n[0] for n in up], [0] + [n[1] for n in up], color=BLUE, lw=1.2, zorder=2)
a1.plot([0] + [n[0] for n in lo], [0] + [n[1] for n in lo], color=BLUE, lw=1.2, zorder=2)
a1.scatter([n[0] for n in up + lo], [n[1] for n in up + lo], s=46, color=BLUE, zorder=4)
a1.scatter([0], [0], s=90, marker="*", color="k", zorder=5)
a1.scatter([r[0] for r in upr], [r[1] for r in upr], s=70, marker="X", color=RED, zorder=5)
a1.axhline(STOP, color=ORG, ls="--", lw=1.4); a1.axhline(JUMP, color="none")
a1.text(-3.9, STOP + 0.35, r"stop when $D\geq10$", color=ORG, va="bottom", **ANN)
a1.axvline(-BOXL, color=GRY, ls=":", lw=2); a1.axvline(BOXR, color=GRY, ls=":", lw=2)
a1.text(-BOXL + 0.1, 14.0, "search box", color=GRY, rotation=90, va="center", ha="left", **ANN)
a1.text(BOXR - 0.1, 14.0, "search box", color=GRY, rotation=90, va="center", ha="right", **ANN)
a1.annotate(r"start at $\hat\theta_p$", xy=(0, 0.3), xytext=(-0.55, 3.6), ha="center", va="center", fontsize=12, arrowprops=dict(arrowstyle="->", color="k"))
a1.set_ylim(-0.9, 17.8); a1.set_xlim(-BOXL - 0.15, BOXR + 0.15)
a1.set_ylabel(r"deviance $D$"); plt.setp(a1.get_xticklabels(), visible=False)
r0 = upr[0][:2]
a1.annotate("rejected: $D$ jumps by\nmore than 4; retry with\nstep/3", xy=r0, xytext=(r0[0] - 1.25, 15.6),
            arrowprops=dict(arrowstyle="->", color=RED), color=RED, ha="center", va="center", **ANN)
a1.annotate("first step 0.05 decades;\nsteps then grow while $D$\nchanges slowly", xy=(lo[2][0], lo[2][1]),
            xytext=(-2.9, 6.6), arrowprops=dict(arrowstyle="->", color=BLUE), color=BLUE, ha="center", va="center", **ANN)
a1.annotate("stops at $D\\geq10$", xy=up[-1], xytext=(2.45, 7.6), color=ORG, ha="center", va="center",
            arrowprops=dict(arrowstyle="->", color=ORG), **ANN)
a1.annotate("this side reaches\nthe box: stops", xy=lo[-1], xytext=(lo[-1][0] + 1.6, 6.0), color=GRY, ha="center",
            arrowprops=dict(arrowstyle="->", color=GRY), va="center", **ANN)
# step-size strip
for (x, d), s in zip(up, ups): a2.bar(x, s, width=0.09, color=BLUE)
for (x, d), s in zip(lo, los): a2.bar(x, s, width=0.09, color=BLUE)
for r in upr:
    a2.bar(r[0], abs(r[0] - r[2]), width=0.09, color="none", edgecolor=RED, hatch="////", lw=1.2)
a2.axhline(STEPMAX, color=GRY, ls="--", lw=1.2)
a2.text(-BOXL + 0.1, STEPMAX + 0.12, "cap: 2 decades", color=GRY, va="bottom", **ANN)
a2.set_ylim(0, 3.3); a2.set_ylabel("step\n(decades)")
a2.set_xlabel(r"parameter value, $\log_{10}v$ relative to $\hat\theta_p$ (decades)")
a2.text(-4.2, 2.85, "bars: accepted steps (hatched: the rejected one)", fontsize=11, color=BLUE, va="center")
a1.text(-0.065, 1.03, "A", transform=a1.transAxes, fontsize=22, fontweight="bold")
a1.set_title("The walk on both sides of the maximum (walk rules replayed on an illustrative profile)",
             fontsize=14, loc="left", pad=8)

# ---------------- B ----------------
def bax(i, title):
    ax = fig.add_subplot(gs[1, 4 * i:4 * i + 4]); ax.set_title(title, fontsize=14, loc="left", pad=6)
    ax.set_xticks([]); ax.set_ylabel(r"$\hat\ell-\ell_p(v)$ (nats)"); ax.set_xlabel(r"outward in $\log v$ $\rightarrow$")
    return ax
xs = np.arange(1, 9)
true = lambda x: 0.09 * x ** 2
xf = np.linspace(0.8, 8.4, 200)
# B1 repair
b1 = bax(0, "Repair")
b1.plot(xf, true(xf), color="0.8", lw=2.5, zorder=1)
stall = np.array([true(1), true(2), true(3), 1.9, 3.4, 5.3, true(7), true(8)])
b1.plot(xs, stall, color=BLUE, lw=1.2, zorder=2)
b1.scatter(xs, stall, s=50, color=BLUE, zorder=3)
b1.scatter([6], [5.3], s=130, facecolor="none", edgecolor=RED, lw=2, zorder=4)
b1.scatter([6], [true(6)], s=70, marker="o", facecolor="white", edgecolor=GRN, lw=2, zorder=5)
b1.annotate("", xy=(6.15, true(6) + 0.1), xytext=(6.9, true(7) - 0.05), arrowprops=dict(arrowstyle="->", color=GRN, lw=1.8,
            connectionstyle="arc3,rad=-0.3"))
b1.annotate("stalled chain: this node's\nvalue is more than 0.2 nats\nbelow that of a node\nfurther out", xy=(5.85, 5.4), xytext=(2.3, 7.2),
            color=RED, arrowprops=dict(arrowstyle="->", color=RED), ha="left", va="center", **ANN)
b1.annotate("re-solved from\nthe further node:\nrepaired value", xy=(6.1, true(6) - 0.12), xytext=(6.9, 1.2), color=GRN,
            arrowprops=dict(arrowstyle="->", color=GRN), ha="left", va="center", **ANN)
b1.set_ylim(-0.3, 8.6); b1.set_xlim(0.5, 9.8)
b1.text(-0.2, 1.08, "B", transform=b1.transAxes, fontsize=22, fontweight="bold")
# B2 polish
b2 = bax(1, "Polish")
b2.plot(xf, true(xf) * 0.62, color="0.8", lw=2.5, zorder=1)
pr = true(xs) * 0.62 ; pr = pr.copy()
raw = pr.copy(); raw[4] = pr[4] + 0.9; raw[3] = pr[3] + 0.35   # nodes 5 and 4 too high (stalled starts)
b2.plot(xs, raw, color=BLUE, lw=1.2, zorder=2); b2.scatter(xs, raw, s=50, color=BLUE, zorder=3)
b2.axhline(CUT, color=ORG, ls="--", lw=1.4); b2.text(0.6, CUT + 0.12, "1.92 nats", color=ORG, va="bottom", **ANN)
b2.scatter([5], [pr[4]], s=70, facecolor="white", edgecolor=GRN, lw=2, zorder=5)
b2.annotate("", xy=(5, pr[4] + 0.07), xytext=(5, raw[4] - 0.07), arrowprops=dict(arrowstyle="->", color=GRN, lw=1.8))
b2.annotate("edge node re-solved\nfrom several starts:\nnow inside the cut", xy=(5.0, pr[4]), xytext=(0.7, 4.2), color=GRN,
            arrowprops=dict(arrowstyle="->", color=GRN), ha="left", va="center", **ANN)
b2.annotate("", xy=(6, CUT + 0.0), xytext=(5, CUT), arrowprops=dict(arrowstyle="->", color=RED, lw=2.2))
b2.plot([5, 5], [CUT - 0.0, 0.0], color="none")
b2.text(5.6, -0.15, "edge moves outward to\nthe next node (at most\n12 times per side)", color=RED, va="center", **ANN)
b2.set_ylim(-0.9, 5.6); b2.set_xlim(0.5, 9.8)
# B3 bisection
b3 = bax(2, "Bisection")
f = lambda x: 0.55 * x ** 2
x1, x2 = 1.0, 3.45
xf3 = np.linspace(0.4, 4.2, 200)
b3.plot(xf3, f(xf3), color="0.8", lw=2.5, zorder=1)
b3.axhline(CUT, color=ORG, ls="--", lw=1.4); b3.text(5.2, CUT + 0.15, "1.92 nats", color=ORG, va="bottom", **ANN)
x3 = math.sqrt(CUT / 0.55)
L0, R0 = 1.0, 3.0
mid1 = (L0 + R0) / 2          # 2.0 above cut?
pts = [L0, R0]; lab = []
L, R = L0, R0
widths = []
for k in range(3):
    m = (L + R) / 2; lab.append(m)
    if f(m) < CUT: L = m
    else: R = m
    widths.append(R - L)
b3.scatter([L0, R0], [f(L0), f(R0)], s=50, color=BLUE, zorder=3)
b3.scatter(lab, [f(m) for m in lab], s=60, facecolor="white", edgecolor=GRN, lw=2, zorder=4)
for k, m in enumerate(lab):
    b3.text(m + (-0.28 if f(m) > CUT else 0.0), f(m) + (0.15 if f(m) > CUT else -0.62), str(k + 1), color=GRN, ha="center", fontsize=12, fontweight="bold")
yb = [4.3, 3.7, 3.1]
b3.annotate("", xy=(R0, 5.35), xytext=(L0, 5.35), arrowprops=dict(arrowstyle="<->", color=RED))
b3.text(L0, 5.55, "bracket wider than 0.3 decades:\nthe edge would rest on a coarse node", color=RED, ha="left", va="bottom", **ANN)
b3.text(3.35, 0.3, "midpoint solved, bracket\nhalved (2, 1, 0.5, 0.25\ndecades) until below 0.3;\nat most 6 times per side", color=GRN,
        ha="left", va="center", **ANN)
b3.set_ylim(-1.1, 7.0); b3.set_xlim(0.3, 7.2)

# ---------------- C ----------------
W = json.load(open(WALK))
def real(row, name):
    w = W["rows"][row]["walks"][name]
    z = np.array([n[0] for n in w["nodes"]]) / math.log(10)
    nl = w["ll_ref"] - np.array([n[1] for n in w["nodes"]])
    o = np.argsort(z)
    return z[o], nl[o], w["ci"], w["box_image"]
cx = lambda ax, t: ax.set_title(t, fontsize=14, loc="left", pad=6)
# C1 closed
z, nl, ci, box = real(18, "sqrt_eta_over_eps")
c1 = fig.add_subplot(gs[2, 0:4]); cx(c1, "Closed interval: a value")
c1.plot(z, nl, color=BLUE, lw=1.2); c1.scatter(z, nl, s=26, color=BLUE, zorder=3)
c1.axhline(CUT, color=ORG, ls="--", lw=1.4)
elo, ehi = math.log10(ci["ci_lo"]), math.log10(ci["ci_hi"])
c1.axvspan(elo, ehi, color=ORG, alpha=0.12, lw=0)
c1.plot([elo, ehi], [CUT, CUT], "v", color=ORG, ms=9)
c1.text(1.40, CUT + 0.12, "1.92 nats", color=ORG, va="bottom", **ANN)
c1.annotate("interval", xy=((elo + ehi) / 2, 0.15), xytext=((elo + ehi) / 2, 3.6), color=ORG, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=ORG), **ANN)
c1.set_ylim(-0.3, 5.9); c1.set_xlabel(r"$\log_{10}$ of $\sqrt{\eta}/\varepsilon$"); c1.set_ylabel(r"$\hat\ell-\ell_p(v)$ (nats)")
sec = c1.secondary_yaxis("right", functions=(lambda y: 2 * y, lambda y: y / 2)); sec.set_yticks([0, 4, 8]); sec.set_ylabel(r"$D$", fontsize=14)
sec.spines["right"].set_visible(True)
c1.text(-0.2, 1.08, "C", transform=c1.transAxes, fontsize=22, fontweight="bold")
# C2 one-sided
z2, nl2, ci2, box2 = real(1, "rbar")
c2 = fig.add_subplot(gs[2, 4:8], sharey=c1); cx(c2, "One side at the box: a bound"); plt.setp(c2.get_yticklabels(), visible=False)
c2.plot(z2, nl2, color=BLUE, lw=1.2); c2.scatter(z2, nl2, s=26, color=BLUE, zorder=3)
c2.axhline(CUT, color=ORG, ls="--", lw=1.4)
e2 = math.log10(ci2["ci_hi"])
c2.axvspan(math.log10(box2[0]), e2, color=ORG, alpha=0.12, lw=0)
c2.plot([e2], [CUT], "v", color=ORG, ms=9)
c2.axvline(math.log10(box2[0]), color=GRY, ls=":", lw=2)
c2.text(z2.min() + 0.25, 4.7, "box\nedge", color=GRY, va="center", **ANN)
c2.text(-14.2, CUT + 0.12, "1.92 nats", color=ORG, va="bottom", **ANN)
c2.annotate("never crosses on\nthis side: reported\nas an upper bound", xy=(e2 - 0.1, CUT), xytext=(-9.5, 3.7), color=ORG, ha="center",
            arrowprops=dict(arrowstyle="->", color=ORG), va="center", **ANN)
c2.set_ylim(-0.3, 5.9); c2.set_xlabel(r"$\log_{10}$ of mean removal $\beta\bar r(\kappa)$")
# C3 interpolation sketch
c3 = fig.add_subplot(gs[2, 8:12]); cx(c3, "Edge by interpolation")
xq = np.linspace(0.3, 2.0, 100); c3.plot(xq, xq ** 2, color="0.8", lw=2.5)
n1, n2 = 0.8, 1.8
c3.scatter([n1, n2], [n1 ** 2, n2 ** 2], s=60, color=BLUE, zorder=3)
c3.plot([n1, n2], [n1 ** 2, n2 ** 2], color=GRY, ls=":", lw=1.6)
c3.axhline(CUT, color=ORG, ls="--", lw=1.4)
edge = math.sqrt(CUT)
c3.plot([edge], [CUT], "v", color=ORG, ms=10, zorder=5)
chord = n1 ** 2 + (CUT - n1 ** 2 + 0) / 1 * 0
xc = n1 + (CUT - n1 ** 2) / (n2 ** 2 - n1 ** 2) * (n2 - n1)
c3.plot([xc], [CUT], "o", mfc="white", mec=GRY, ms=8, zorder=5)
c3.annotate(r"linear in $\sqrt{D}$:" "\nedge lands on the\ncurve", xy=(edge, CUT), xytext=(1.5, 0.45), color=ORG,
            arrowprops=dict(arrowstyle="->", color=ORG), ha="left", va="center", **ANN)
c3.annotate("linear in $D$:\nedge too narrow", xy=(xc, CUT), xytext=(0.3, 3.3), color=GRY,
            arrowprops=dict(arrowstyle="->", color=GRY), ha="left", va="center", **ANN)
c3.text(0.3, 4.55, "bracket: the two nodes\nthat straddle the cut", color=BLUE, ha="left", va="center", **ANN)
c3.set_ylim(-0.3, 5.5); c3.set_xlim(0.2, 2.9); c3.set_xticks([]); c3.set_xlabel(r"$\log v$ (quadratic profile, schematic)")
c3.set_ylabel(r"$\hat\ell-\ell_p(v)$ (nats)")

# ---------------- D ----------------
d1 = fig.add_subplot(gs[3, 0:6]); cx(d1, "Self-consistency widening")
q = 0.38
xf = np.linspace(-3.2, 3.4, 300)
d1.plot(xf, q * xf ** 2, color="0.8", lw=2.5, zorder=1)
nxr = np.array([-1.0, -0.4, 0.0, 0.5, 1.1, 1.8, 2.4]); nyr = q * nxr ** 2
nxl = np.array([-1.0, -1.7]); nyl = np.array([0.55, 2.6])          # left chain stalled: nodes too high
d1.plot(np.r_[nxl[::-1], nxr], np.r_[nyl[::-1], nyr], color=BLUE, lw=1.2, zorder=2)
d1.scatter(np.r_[nxl, nxr], np.r_[nyl, nyr], s=40, color=BLUE, zorder=3)
d1.axhline(CUT, color=ORG, ls="--", lw=1.4)
f_ = (math.sqrt(CUT) - math.sqrt(nyl[0])) / (math.sqrt(nyl[1]) - math.sqrt(nyl[0]))
el = -(1.0 + f_ * 0.7); er = math.sqrt(CUT / q)
d1.axvspan(el, er, color=ORG, alpha=0.10, lw=0)
d1.text(0.0, CUT + 0.12, "1.92 nats", color=ORG, va="bottom", ha="center", **ANN)
pt = (-2.15, 1.3)
d1.scatter([pt[0]], [pt[1]], s=110, marker="D", color=RED, zorder=5)
d1.annotate("a point evaluated elsewhere in the fit (any\nmaximisation, walk, repair or polish solve):\nwithin 1.92 nats of the best value,\nbut outside the current interval",
            xy=pt, xytext=(-3.15, 4.6), color=RED, arrowprops=dict(arrowstyle="->", color=RED), ha="left", va="center", **ANN)
d1.annotate("", xy=(pt[0], -0.45), xytext=(el, -0.45), arrowprops=dict(arrowstyle="->", color=GRN, lw=2.2))
d1.text(el + 0.15, -0.45, "interval widened to cover it", color=GRN, va="center", **ANN)
d1.plot([el, el], [-0.65, -0.25], color=ORG, lw=2); d1.plot([er, er], [-0.65, -0.25], color=ORG, lw=2)
d1.set_ylim(-0.9, 5.9); d1.set_xlim(-3.3, 3.6); d1.set_xticks([]); d1.set_xlabel(r"$\log v$")
d1.set_ylabel("nats below the best\nvalue found")
d1.text(-0.09, 1.08, "D", transform=d1.transAxes, fontsize=22, fontweight="bold")
# D2 basin refit
d2 = fig.add_subplot(gs[3, 6:12]); cx(d2, "Basin re-fit")
xb = np.linspace(-3, 3.6, 400)
old = -0.42 * (xb + 1.0) ** 2
new = -0.55 * (xb - 2.2) ** 2 + 3.2
d2.plot(xb, old, color="0.7", lw=2.5, ls="--", label="first profile (maximum at the star)")
d2.plot(xb, np.maximum(new, old), color=BLUE, lw=2.2, label="profile after the re-fit")
d2.scatter([-1.0], [0], s=110, marker="*", color="k", zorder=5)
d2.scatter([2.2], [3.2], s=70, color=RED, zorder=5)
d2.annotate("", xy=(-1.0, 3.2), xytext=(-1.0, 0.0), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
d2.plot([-1.0, 2.2], [3.2, 3.2], color=RED, lw=1, ls=":")
d2.text(-1.15, 1.6, "a node beats the\nmaximum by more\nthan 1.92 nats", color=RED, va="center", ha="right", **ANN)
d2.annotate("the maximisation had\nstopped in the wrong basin", xy=(-1.0, -0.12), xytext=(-4.9, -3.5), color="k", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="k"), **ANN)
d2.annotate("restart from the best nodes,\nrecompute all profiles\n(at most 3 times)", xy=(2.2, 3.2), xytext=(3.1, 1.0), color=BLUE,
            arrowprops=dict(arrowstyle="->", color=BLUE), ha="left", va="center", **ANN)
d2.legend(loc="lower right", frameon=False, fontsize=11)
d2.set_ylim(-4.8, 4.6); d2.set_xlim(-5.0, 9.2); d2.set_xticks([]); d2.set_xlabel(r"$\log v$")
d2.set_ylabel(r"$\ell_p(v)-\ell(\hat\theta)$ (nats)")
fig.savefig(os.path.join(HERE, "out", "fig_profile.pdf"))
print("saved", len(up), len(lo), upr, up[-1], lo[-1], upst, lost)
