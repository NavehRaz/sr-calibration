"""Table of reportable quantities: what survival data identify, what to report, and the v6 benchmark scores
(calibrator J 1.0.1, pure arm, reference run). Read-only; no fits.
Run: python -B ident_table.py
Outputs (beside this file): ident_table.csv, ident_table.tex (tabular, booktabs).
Sources: explorer_data.json of the J v6 reference run (per-cohort shape/hit/err/width; err = |log10 estimate/truth| at the
claimed point; err_theta = same for combinations at the fitted maximum-likelihood point, kappa filled from the fit when
declined), and KAPPA0_TABLE.txt (free-kappa calibrator, version 1.0.0, library v5 medium tier) for the truth-scored
combinations that 1.0.1 does not store as intervals.
"""
import csv, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)


HERE = os.path.dirname(os.path.abspath(__file__))
AG = "aging:code_5/"
EXPL = AG + "sr_calibration_tester/reports/jax2-v6-full-bign/explorer_data.json"
K0T = AG + "sr_calibration_tester_jax2_rel/reports/rel101/G3_KFIX/BOTH/KAPPA0_TABLE.txt"

# (key, tex label, explorer quantity, which error, identified, report)
ROWS = [
    ("eta", r"$\eta$", "eta", "err", "pinned", "value"),
    ("epsilon", r"$\varepsilon$", "epsilon", "err", "pinned", "value"),
    ("sqrt_eta_over_eps", r"$\sqrt{\eta}/\varepsilon$", "sqrt_eta_over_eps", "err", "pinned", "value"),
    ("eta_over_eps", r"$\eta/\varepsilon^{b}$", "eta_over_eps", "err", "pinned", "value"),
    ("beta_rbar", r"$\beta\,\bar r(\kappa)$", "beta_rbar", "err_theta", "pinned", "value or bound"),
    ("beta_rbar_over_eta", r"$\beta\,\bar r(\kappa)/\eta$", "beta_rbar_over_eta", "err_theta", "pinned", "value or bound"),
    ("beta", r"$\beta$", "beta", "err", "not pinned", "bound"),
    ("kappa", r"$\kappa$", "kappa", "err", "not pinned", "no claim"),
    ("beta_over_kp1", r"$\beta/(1+\kappa)$", "beta_over_kp1", "err_theta", "not pinned", "no claim"),
    ("xc", r"$X_c$", None, None, "fixed by convention", "none ($X_c=1$)"),
]


def num(x, f):
    return f.format(x)


def main():
    d = json.load(open(PP.path(EXPL)))
    assert d["provenance"]["method_version"] == "1.0.1" and d["provenance"]["library_version"] == "v6"
    c, r, qs = d["cohort"], d["result"], d["quantities"]
    idx = [i for i, a in enumerate(c["arm"]) if a == "pure"]
    n = len(idx)
    # version 1.0.0, library v5 medium tier, free-kappa calibrator: closed coverage of the two combinations
    txt = open(PP.path(K0T)).read()
    blk = txt.split("--- J kfix kappa0 = 0.001")[1].split("--- J kfix kappa0 = 0.5")[0]
    v100 = {}
    for key, pat in (("sqrt_eta_over_eps", r"sqrt\(eta\)/eps .*?comp 0\.\d+ \((\d+)/(\d+)\)"),
                     ("beta_rbar", r"beta\*rbar .*?comp 0\.\d+ \((\d+)/(\d+)\)")):
        m = re.search(pat, blk)
        v100[key] = (int(m.group(1)), int(m.group(2)))

    out, tex = [], []
    for key, lab, q, errk, ident, rep in ROWS:
        row = dict(quantity=key, identified=ident, report=rep, n_pure=n, closed_n="", closed_share="", hits="", coverage="",
                   err_source="", err_n="", err_median="", err_q25="", err_q75="", width_median="", v100_hits="", v100_n="",
                   one_sided_n="", declined_n="")
        if q is not None:
            j = qs.index(q)
            sh = [r["shape"][j][i] for i in idx]
            closed = [i for i in idx if r["shape"][j][i] == 3]
            if q in ("eta", "epsilon", "beta", "kappa"):
                row["declined_n"] = sh.count(0)
                row["one_sided_n"] = sh.count(1) + sh.count(2)
                row["closed_n"] = len(closed)
                row["closed_share"] = round(len(closed) / n, 4)
                row["hits"] = sum(1 for i in closed if r["hit"][j][i] == 1)
                if closed:
                    row["coverage"] = round(row["hits"] / len(closed), 4)
                    w = [r["width"][j][i] for i in closed if r["width"][j][i] is not None]
                    row["width_median"] = round(float(np.median(w)), 3)
            e = [r[errk][j][i] for i in idx if r[errk][j][i] is not None]
            row["err_source"] = "claimed point" if errk == "err" else "fitted point"
            row["err_n"] = len(e)
            if e:
                q1, q2, q3 = np.percentile(e, [25, 50, 75])
                row["err_median"], row["err_q25"], row["err_q75"] = (round(float(v), 3) for v in (q2, q1, q3))
        if key in v100:
            row["v100_hits"], row["v100_n"] = v100[key]
        out.append(row)

        # tex cells
        sup = {"sqrt_eta_over_eps": "b", "eta_over_eps": "b", "beta_rbar": "b", "beta_rbar_over_eta": "b"}.get(key, "")
        if q is None:
            cells = ["--", "--", "--", "--"]
        else:
            if row["closed_n"] != "":
                cs = f"{100 * row['closed_share']:.1f}\\,\\% ({row['closed_n']}/{n})"
                cov = f"{row['coverage']:.3f} ({row['hits']}/{row['closed_n']})" if row["closed_n"] else "--"
                wd = f"{row['width_median']:.2f}" if row["closed_n"] else "--"
            else:
                cs, cov, wd = "no interval stored$^{a}$", "not measured$^{a}$", "not measured$^{a}$"
                if key in v100:
                    h, nn = v100[key]
                    cov = f"{h / nn:.2f} ({h}/{nn})$^{{c}}$"
            if key == "kappa":
                cov, wd = "--", "--"
            er = "--"
            if row["err_median"] != "":
                er = f"{row['err_median']:.2f} [{row['err_q25']:.2f}, {row['err_q75']:.2f}]"
                if row["err_n"] != n:
                    er += f" ($n={row['err_n']}$)"
                if errk == "err_theta":
                    er += "$^{d}$"
            cells = [cs, cov, er, wd]
        tex.append(f"{lab} & {ident} & {rep} & " + " & ".join(cells) + r" \\")

    with open(os.path.join(HERE, "ident_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    L = [r"\begin{tabular}{llllllc}", r"\toprule",
         r"Quantity & Identified? & Report & Closed claims & Coverage (hits/closed) & Median error [IQR] & Median width \\", r"\midrule"]
    L += tex[:4] + [r"\addlinespace"] + tex[4:6] + [r"\addlinespace"] + tex[6:]
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(HERE, "ident_table.tex"), "w").write("\n".join(L) + "\n")
    print("ok", n)


if __name__ == "__main__":
    main()
