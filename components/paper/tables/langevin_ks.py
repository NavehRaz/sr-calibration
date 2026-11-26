"""Langevin vs Fokker-Planck: KS distance per cell, from sraging's stored S4 result (rule HR-6, spec 004).
Source: git object a2ead86:tests/acceptance/results/s4_004.json in ~/code/sraging (the commit that stored the file).
Each cell compares n = 25,000 Langevin death times with the FP CDF; crit = kstwobign.isf(0.05/G)/sqrt(n), G = rows per wall.
Plants are the same comparison with eta -> eta (1 + delta); they must be REJECTED (D > crit).
Run: python -B langevin_ks.py"""
import csv, json, os, sys, math
from scipy.stats import kstwobign
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)


HERE = os.path.dirname(os.path.abspath(__file__))
REPO, REV, PATH = "~/code/sraging", "a2ead86", "tests/acceptance/results/s4_004.json"
s = json.loads(PP.git_show(REPO, REV, PATH))
crit = {w: float(kstwobign.isf(0.05 / s["G"][w]) / math.sqrt(25000)) for w in s["G"]}
rows = []
for c in s["cells"]:
    pre, par, *rest = c["id"].split("/")
    cv = 0.0 if par == "v0" else float(rest[0][2:] if rest[0].startswith("cv") else 0)
    if par == "v0" and rest[0].startswith("cv"): cv = float(rest[0][2:])
    cr = crit[c["wall"]]
    assert abs(cr - [v for v in s["verdicts"] if v["wall"] == c["wall"]][0]["crit"]) < 1e-9
    rows.append(dict(row="cell", preset=pre, parameter=par, cv=cv, wall=c["wall"], n=c["n"], m=c["m"], ks_D=round(c["ks_D"], 5),
                     crit=round(cr, 5), ratio=round(c["ks_D"] / cr, 3), delta="", **{"pass": c["ks_D"] <= cr}))
for p in s["plants"]:
    pre, par, *rest = p["id"].split("/")
    wall = "clamp"
    cv = float(rest[0][2:]) if rest[0].startswith("cv") else 0.0
    rows.append(dict(row="plant", preset=pre, parameter=par, cv=cv, wall=wall, n=p["n"], m="", ks_D=round(p["ks_D"], 5),
                     crit=round(crit[wall], 5), ratio=round(p["ks_D"] / crit[wall], 3), delta=round(p["delta"], 5),
                     **{"pass": p["ks_D"] > crit[wall]}))   # a plant passes when rejected
with open(os.path.join(HERE, "langevin_ks.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("saved", len(rows), "all cells pass:", all(r["pass"] for r in rows))
