"""Per-arm x method x parameter coverage tables from the stored explorer payloads (read-only; no refits).
Run: python -B coverage_tables.py
Outputs: coverage_by_arm.csv, coverage_main.tex, coverage_full.tex, library_design.tex, onesided_null.tex
"""
import json, csv, math, hashlib, os, sys, statistics as st
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)


R = "aging:code_5/sr_calibration_tester/reports/"
FILES = {"J": R + "jax2-v6-full-bign/explorer_data.json", "H": R + "jax2h-v6-full-bign/explorer_data.json"}
MANIFEST = R.replace("reports/","") + "data/library/v6/manifest.json"
CMP = R + "jax2h-v6-full-bign/comparison.json"
HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["pure", "censor", "hazard", "window", "het", "realfull"]
PARAMS = [("eta", r"$\eta$"), ("beta", r"$\beta$"), ("epsilon", r"$\varepsilon$")]
PARAMS_ALL = PARAMS + [("kappa", r"$\kappa$")]
# stored REPORT_H_V6 counts (hits, closed n) for the reproduction check
REPORT = {("J", "pure", "eta"): (617, 644), ("J", "censor", "eta"): (194, 199), ("J", "hazard", "eta"): (18, 237),
          ("J", "window", "eta"): (130, 235), ("J", "het", "eta"): (37, 233),
          ("H", "pure", "eta"): (594, 623), ("H", "hazard", "eta"): (178, 178), ("H", "het", "eta"): (36, 211),
          ("J", "pure", "beta"): (272, 281), ("J", "pure", "epsilon"): (627, 654), ("H", "realfull", "beta"): (26, 26)}


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sha(p):
    return hashlib.sha256(open(PP.path(p), "rb").read()).hexdigest()


def rows_for(meth, data):
    q = data["quantities"]
    c = data["cohort"]
    res = data["result"]
    out = []
    for arm in ARMS:
        idx = [i for i, a in enumerate(c["arm"]) if a == arm]
        for pname, _ in PARAMS_ALL:
            j = q.index(pname)
            shp = [res["shape"][j][i] for i in idx]
            closed = [i for i in idx if res["shape"][j][i] == 3]
            hits = sum(1 for i in closed if res["hit"][j][i] == 1)
            errs = [res["err"][j][i] for i in closed if res["err"][j][i] is not None]
            wid = [res["width"][j][i] for i in closed if res["width"][j][i] is not None]
            n = len(closed)
            lo, hi = wilson(hits, n)
            out.append(dict(arm=arm, method=meth, param=pname, n_cohorts=len(idx), n_closed=n, hits=hits,
                            coverage=(hits / n if n else float("nan")), wilson_lo=lo, wilson_hi=hi,
                            bias_log10=(st.median(errs) if errs else float("nan")),
                            median_width_dex=(st.median(wid) if wid else float("nan")),
                            decline_rate=sum(1 for s in shp if s == 0) / len(idx),
                            n_declined=sum(1 for s in shp if s == 0),
                            n_onesided=sum(1 for s in shp if s in (1, 2))))
    return out


def main():
    allrows, designs = [], {}
    for m, p in FILES.items():
        d = json.load(open(PP.path(p)))
        assert d["n_cohorts"] == 1776
        allrows += rows_for(m, d)
        if m == "J":
            c = d["cohort"]
            designs = Counter(zip(c["arm"], c["group"]))
            valid = Counter((a, v) for a, v in zip(c["arm"], c["valid"]))
    # reproduction check against REPORT_H_V6 (stop if it fails)
    idx = {(r["method"], r["arm"], r["param"]): r for r in allrows}
    bad = [(k, (idx[k]["hits"], idx[k]["n_closed"]), v) for k, v in REPORT.items() if (idx[k]["hits"], idx[k]["n_closed"]) != v]
    print("reproduction mismatches:", bad)
    if bad:
        raise SystemExit("explorer counts do not reproduce REPORT_H_V6; stopping")

    fields = list(allrows[0].keys())
    with open(os.path.join(HERE, "coverage_by_arm.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in allrows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    def f3(x): return "--" if x != x else f"{x:.3f}"
    def f2(x): return "--" if x != x else f"{x:.2f}"
    def f2s(x): return "--" if x != x else f"{x:+.2f}"
    armtex = {"pure": "pure", "censor": "censor", "hazard": "hazard", "window": "window", "het": "het", "realfull": "realfull"}

    # compact main table: coverage (hits/n) per arm, eta/beta/eps, J and H
    L = [r"\begin{tabular}{l" + "ccc" * 2 + "}", r"\toprule",
         r" & \multicolumn{3}{c}{J} & \multicolumn{3}{c}{H} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Arm & $\eta$ & $\beta$ & $\varepsilon$ & $\eta$ & $\beta$ & $\varepsilon$ \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for m in "JH":
            for p, _ in PARAMS:
                r = idx[(m, arm, p)]
                cells.append(f"{f3(r['coverage'])} ({r['hits']}/{r['n_closed']})")
        L.append(f"{armtex[arm]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "coverage_main.tex"), "w").write("\n".join(L) + "\n")

    # full longtable
    L = [r"\begin{longtable}{llrrrcrrr}", r"\toprule",
         r"Arm & Meth. & Par. & $n_{\mathrm{closed}}$ & hits & coverage [Wilson 95\%] & bias & width & decl. \\",
         r"\midrule", r"\endfirsthead", r"\toprule",
         r"Arm & Meth. & Par. & $n_{\mathrm{closed}}$ & hits & coverage [Wilson 95\%] & bias & width & decl. \\",
         r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    sym = dict(PARAMS_ALL)
    for arm in ARMS:
        for p, _ in PARAMS_ALL:
            for m in "JH":
                r = idx[(m, arm, p)]
                cov = "--" if r["n_closed"] == 0 else f"{f3(r['coverage'])} [{f3(r['wilson_lo'])}, {f3(r['wilson_hi'])}]"
                L.append(f"{armtex[arm]} & {m} & {sym[p]} & {r['n_closed']} & {r['hits']} & {cov} & {f2s(r['bias_log10'])} & "
                         f"{f2(r['median_width_dex'])} & {f3(r['decline_rate'])} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule" if False else L[-1]
    L.pop()
    L.append(r"\end{longtable}")
    open(os.path.join(HERE, "coverage_full.tex"), "w").write("\n".join(L) + "\n")

    # library design table: arm x group counts from the v6 manifest (2,408) and reference-run population (1,776)
    man = json.load(open(PP.path(MANIFEST)))["specs"]
    cnt = Counter((x["arm"], x["group"]) for x in man)
    GROUPS = ["main", "bign", "sharp", "nladder"]
    L = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Arm & main & bign & sharp & n-ladder & library & reference run & valid in reference run \\", r"\midrule"]
    with open(os.path.join(HERE, "library_design.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "main", "bign", "sharp", "nladder", "library", "reference_run", "valid_reference_run", "invalid_reference_run"])
        for arm in ARMS:
            row = [cnt[(arm, g)] for g in GROUPS]
            ref = designs[(arm, "main")] + designs[(arm, "bign")]
            v = valid[(arm, 1)]
            w.writerow([arm] + row + [sum(row), ref, v, ref - v])
            L.append(f"{arm} & " + " & ".join(str(x) if x else "--" for x in row) + f" & {sum(row)} & {ref} & {v} \\\\")
        tot = [sum(cnt[(a, g)] for a in ARMS) for g in GROUPS]
        ref = sum(designs[(a, "main")] + designs[(a, "bign")] for a in ARMS)
        v = sum(valid[(a, 1)] for a in ARMS)
        w.writerow(["all"] + tot + [sum(tot), ref, v, ref - v])
        L += [r"\midrule", "all & " + " & ".join(map(str, tot)) + f" & {sum(tot)} & {ref} & {v} \\\\", r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "library_design.tex"), "w").write("\n".join(L) + "\n")

    # one-sided and null summary (comparison.json)
    cm = json.load(open(PP.path(CMP)))
    L = [r"\begin{tabular}{lcccc}", r"\toprule",
         r" & $\eta$ & $\beta$ & $\varepsilon$ & $\kappa$ \\", r"\midrule"]
    def cell(p, key, sub):
        return cm["by_parameter"][p]["coverage"][key][sub]
    for lab, key in (("J one-sided", "reference_one_sided"), ("H one-sided", "method_one_sided")):
        L.append(lab + " & " + " & ".join(f3(cell(p, key, "coverage")) for p, _ in PARAMS_ALL) + r" \\")
    L.append("null closed & " + " & ".join(f3(cm["by_parameter"][p]["null"]["coverage"]) for p, _ in PARAMS_ALL) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "onesided_null.tex"), "w").write("\n".join(L) + "\n")
    print("ok; sha J", sha(FILES["J"]), "H", sha(FILES["H"]))


if __name__ == "__main__":
    main()
