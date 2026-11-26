"""Write the README's example cohort from the ITP lifespan file on the Mouse Phenome Database (T055; R13).

Source: MPD project ITP1 (https://phenome.jax.org/projects/ITP1), file ITP_C2016_Lifespan.xlsx, which MPD
serves to scripts (2 s between requests, per its terms of use):

    curl -sSLJ -o .cache/itp/ITP_C2016_Lifespan.xlsx "https://phenome.jax.org/getcurationfile?id=754&mode=download"

Selection, fixed before any fit (specs/001-reference-fit/review/itp_selection.md): UM-HET3 controls,
cohort C2016, females, all three sites (TJL, UM, UT) pooled, because no single site x sex x year control
group reaches n = 200 and the three sites' survival does not differ (log-rank p = 0.50).

Transformation (README, "Fit a cohort"): mice were checked daily and `age` is the day a mouse was found
dead or euthanized when moribund, so a death is moved to the previous day, age - 1, the start of its
1-day recording interval (delta = 1). A mouse removed from the study alive (`dead` = 0) was known alive at
`age`, which is kept.

    .venv/bin/python tools/make_example_cohort.py
"""
import csv
import hashlib
import os
import re
import sys
import zipfile
import xml.etree.ElementTree as ET

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, ".cache", "itp", "ITP_C2016_Lifespan.xlsx")
SHA256 = "8c23eb208a1c4dd46b6ce9cee788796a25a2844cbee62e72a7693198cda330df"   # MPD file id 754, 2026-10-03
DELTA = 1
OUT = os.path.join(REPO, "examples", "data", "itp_c2016_het3_control_female.csv")
_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_xlsx(path):
    """Rows of the first sheet (shared strings and plain numbers), with the standard library only."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    rows = []
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(f"{{{_NS['m']}}}row"):
        cells = {}
        for c in row.findall("m:c", _NS):
            col = 0
            for ch in re.match(r"[A-Z]+", c.get("r")).group():
                col = col * 26 + (ord(ch) - 64)
            v = c.find("m:v", _NS)
            if v is None:
                continue
            cells[col - 1] = shared[int(v.text)] if c.get("t") == "s" else (
                v.text if c.get("t") in ("str", "inlineStr") else float(v.text))
        if cells:
            rows.append([cells.get(i) for i in range(max(cells) + 1)])
    return rows


def selection_check(data, path):
    """The pre-registered test behind pooling the three sites (review/itp_selection.md): log-rank across sites
    on the selected mice (raw ages), per-site sizes and medians, and the pooled hazard in 100-day windows."""
    import json
    import numpy as np
    from lifelines.statistics import multivariate_logrank_test       # dev group (oracle pins)
    t = np.array([float(d["age"]) for d in data])
    e = np.array([int(d["dead"]) for d in data])
    g = np.array([d["site"] for d in data])
    lr = multivariate_logrank_test(t, g, e)
    win = []
    for lo in range(500, 1300, 100):
        ex = float(np.sum(np.clip(np.minimum(t, lo + 100) - lo, 0, None)))
        dd = int(np.sum((e == 1) & (t >= lo) & (t < lo + 100)))
        win.append({"from": lo, "to": lo + 100, "deaths": dd, "hazard_per_1000_days": 1000 * dd / ex if ex else None})
    out = {"logrank_chi2": float(lr.test_statistic), "logrank_p": float(lr.p_value),
           "sites": {s: {"n": int(np.sum(g == s)), "median_death_age": float(np.median(t[(g == s) & (e == 1)]))}
                     for s in sorted(set(g))},
           "pooled_hazard": win,
           "hazard_monotone": all(a["hazard_per_1000_days"] <= b["hazard_per_1000_days"] for a, b in zip(win, win[1:]))}
    json.dump(out, open(path, "w"), indent=1)
    return out


def main():
    if not os.path.exists(SRC):
        sys.exit(f"download ITP_C2016_Lifespan.xlsx into .cache/itp/ first (see this file's docstring)")
    if hashlib.sha256(open(SRC, "rb").read()).hexdigest() != SHA256:
        sys.exit(f"{SRC}: sha256 differs from the pinned {SHA256} (MPD may have revised the file)")
    rows = read_xlsx(SRC)
    head = [str(h).strip() for h in rows[0]]
    data = [dict(zip(head, r)) for r in rows[1:]]
    sel = [d for d in data if d["cohort"] == "C2016" and d["group"] == "Control" and d["sex"] == "f"]
    if any(float(d["age"]) != int(d["age"]) or int(d["dead"]) not in (0, 1) for d in sel):
        sys.exit("unexpected non-integer age or dead flag in the selected rows")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["death times", "events"])
        for d in sel:                                         # file order kept
            age, dead = int(d["age"]), int(d["dead"])
            w.writerow([repr(float(age - DELTA if dead else age)), dead])
    chk = selection_check(sel, os.path.join(REPO, "specs", "001-reference-fit", "review", "itp_selection.json"))
    print(f"site check: log-rank chi2 {chk['logrank_chi2']:.2f}, p {chk['logrank_p']:.3f}; pooled hazard monotone "
          f"{chk['hazard_monotone']}")
    deaths = sum(int(d["dead"]) for d in sel)
    sites = {s: sum(d["site"] == s for d in sel) for s in sorted({d["site"] for d in sel})}
    print(f"{os.path.relpath(OUT, REPO)}: {len(sel)} mice, {deaths} deaths, {len(sel) - deaths} censored, "
          f"sites {sites}; sha256 {hashlib.sha256(open(OUT, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
