"""Independently recomputing every number in a generated document.

THIS MODULE SHARES NO CODE WITH `comparison.py`, AND THAT IS THE POINT. A verifier built on the
generator's helpers re-tests the generator against itself: if the helper is wrong, both agree and
the check passes. So this one parses the finished document, recomputes each claim from the stored
results by its own route, and compares.

WHY IT EXISTS. Documents in this project drift from their own data within weeks. Writing exactly
this for a sibling analysis caught three already-published errors: a wrong superlative, two
incompatible definitions of "median" in simultaneous circulation, and a stall that had been claimed
as fixed.

WHAT IT CHECKS. Not only decimals. Counts and superlatives too -- those are the ones that rot
silently, because a number that is merely stale still looks like a number, while "the worst arm is
X" quietly becomes false when the data moves.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Mismatch:
    location: str
    claim: str
    in_document: str
    recomputed: str

    def format(self) -> str:
        return (f"  {self.location}\n    claim      : {self.claim}\n"
                f"    document   : {self.in_document}\n    recomputed : {self.recomputed}")


def _recompute_coverage(store_results, truths, parameter: str) -> tuple[float | None, int, int]:
    """Coverage, claimed count and decline count -- computed HERE, from first principles.

    Deliberately re-implemented rather than imported: two-sided claims only, a decline counted but
    excluded from the denominator, a failure in neither.
    """
    hits = claimed = declined = 0
    for cs in store_results:
        t = truths.get(cs.cohort_id, {}).get(parameter)
        if t is None or not cs.ok:
            continue
        c = cs.claim_for(parameter)
        if c is None or c.shape == "none":
            declined += 1
            continue
        if c.shape != "closed":
            continue
        claimed += 1
        lo = -math.inf if c.lower is None else c.lower
        hi = math.inf if c.upper is None else c.upper
        if lo <= t <= hi:
            hits += 1
    return (hits / claimed if claimed else None), claimed, declined


def _recompute_coverage_one_sided(store_results, truths, parameter: str) -> tuple[float | None, int]:
    """ONE-SIDED coverage and claimed count, from first principles (re-implemented, not imported):
    upper_bound hits when truth <= upper, lower_bound when truth >= lower; closed claims and
    declines are not in this figure."""
    hits = claimed = 0
    for cs in store_results:
        t = truths.get(cs.cohort_id, {}).get(parameter)
        if t is None or not cs.ok:
            continue
        c = cs.claim_for(parameter)
        if c is None or c.shape not in ("upper_bound", "lower_bound"):
            continue
        claimed += 1
        if (c.shape == "upper_bound" and c.upper is not None and t <= c.upper) or \
           (c.shape == "lower_bound" and c.lower is not None and t >= c.lower):
            hits += 1
    return (hits / claimed if claimed else None), claimed


def verify_markdown(md_path: str, comparison_json_path: str, library, store) -> list[Mismatch]:
    """Recompute every figure in the rendered document. Returns the mismatches."""
    from ..run.execute import load_results

    with open(md_path, encoding="utf-8") as fh:
        text = fh.read()
    with open(comparison_json_path, encoding="utf-8") as fh:
        cmp_ = json.load(fh)

    run_path = os.path.join("runs", f"{cmp_['method']['run_id']}.json")
    ref_path = os.path.join("runs", f"{cmp_['reference']['run_id']}.json")
    problems: list[Mismatch] = []
    if not (os.path.exists(run_path) and os.path.exists(ref_path)):
        return [Mismatch("inputs", "run manifests present", "expected", "missing")]

    with open(run_path, encoding="utf-8") as fh:
        a = load_results(json.load(fh), store)
    with open(ref_path, encoding="utf-8") as fh:
        b = load_results(json.load(fh), store)
    ids = sorted({cs.cohort_id for cs in a} & {cs.cohort_id for cs in b})
    truths = library.truths(ids)
    a_all = list(a)          # the nuisance-h section is over the method's WHOLE run, unpaired
    a = [cs for cs in a if cs.cohort_id in set(ids)]
    b = [cs for cs in b if cs.cohort_id in set(ids)]

    # ---- the paired-cohort count, which is the denominator for everything below
    m = re.search(r"\*\*Paired cohorts\*\*\s+(\d+)", text)
    if m and int(m.group(1)) != len(ids):
        problems.append(Mismatch("Paired cohorts", "paired cohort count",
                                 m.group(1), str(len(ids))))

    # ---- the coverage table, recomputed from first principles
    for line in text.split("\n"):
        mm = re.match(r"\|\s*`(\w+)`\s*\|\s*([\d.]+|n/a)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
                      r"\s*([\d.]+|n/a)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|", line)
        if not mm:
            continue
        param, cov_s, n_s, dec_s, rcov_s, rn_s, rdec_s = mm.groups()
        for label, sets, cov_str, n_str, dec_str in (
                ("method", a, cov_s, n_s, dec_s), ("reference", b, rcov_s, rn_s, rdec_s)):
            cov, claimed, declined = _recompute_coverage(sets, truths, param)
            want_cov = "n/a" if cov is None else f"{cov:.3f}"
            if cov_str != want_cov:
                problems.append(Mismatch(f"coverage table / {param} / {label}",
                                         "coverage", cov_str, want_cov))
            if int(n_str) != claimed:
                problems.append(Mismatch(f"coverage table / {param} / {label}",
                                         "denominator (claims)", n_str, str(claimed)))
            total = sum(1 for cs in sets if truths.get(cs.cohort_id, {}).get(param) is not None)
            want_dec = f"{declined / total:.3f}" if total else "n/a"
            if dec_str != want_dec:
                problems.append(Mismatch(f"coverage table / {param} / {label}",
                                         "decline rate", dec_str, want_dec))

    # ---- the one-sided table, recomputed from first principles (added with the table, 2026-09-25)
    for line in text.split("\n"):
        mm = re.match(r"\|\s*`(\w+)` \(one-sided\)\s*\|\s*([\d.]+|n/a)\s*\|\s*(\d+)\s*\|"
                      r"\s*([\d.]+|n/a)\s*\|\s*(\d+)\s*\|", line)
        if not mm:
            continue
        param, cov_s, n_s, rcov_s, rn_s = mm.groups()
        for label, sets, cov_str, n_str in (("method", a, cov_s, n_s), ("reference", b, rcov_s, rn_s)):
            cov, claimed = _recompute_coverage_one_sided(sets, truths, param)
            want_cov = "n/a" if cov is None else f"{cov:.3f}"
            if cov_str != want_cov:
                problems.append(Mismatch(f"one-sided table / {param} / {label}",
                                         "coverage", cov_str, want_cov))
            if int(n_str) != claimed:
                problems.append(Mismatch(f"one-sided table / {param} / {label}",
                                         "denominator (claims)", n_str, str(claimed)))

    # ---- the verdicts
    for param, entry in cmp_["by_parameter"].items():
        pat = rf"\|\s*`{param}`\s*\|\s*\*\*(.+?)\*\*\s*\|"
        m = re.search(pat, text)
        if m and m.group(1) != entry["overall"]:
            problems.append(Mismatch(f"verdict table / {param}", "verdict",
                                     m.group(1), entry["overall"]))

    # ---- counts and superlatives, which rot silently
    m = re.search(r"UNTESTED REGIMES in this run: ([^.]+)\.", text)
    if m:
        stated = sorted(x.strip() for x in m.group(1).split(","))
        actual = sorted(cmp_["claim_strength"]["arms_absent"])
        if stated != actual:
            problems.append(Mismatch("untested regimes", "list of absent arms",
                                     ", ".join(stated), ", ".join(actual)))

    m = re.search(r"Median cost ratio \*\*([\d.]+)x\*\*", text)
    if m:
        w_a = sorted(cs.wall_clock_seconds for cs in a if cs.wall_clock_seconds)
        w_b = sorted(cs.wall_clock_seconds for cs in b if cs.wall_clock_seconds)
        if w_a and w_b:
            med = lambda x: (x[len(x) // 2] if len(x) % 2
                             else 0.5 * (x[len(x) // 2 - 1] + x[len(x) // 2]))
            want = f"{med(w_a) / med(w_b):.3g}"
            if m.group(1) != want:
                problems.append(Mismatch("run time", "median cost ratio", m.group(1), want))

    # ---- the nuisance external-hazard section (added 2026-10-29), by its own route
    if _H_HEADING in text:
        problems += verify_nuisance_hazard(text, a_all, library)

    return problems


_H_HEADING = "## External hazard h"


def _h_group(spec, planted: bool) -> str:
    """The section's group labels, written out again here ON PURPOSE (see the module docstring)."""
    het = (getattr(spec, "het_var", 0.0) or 0.0) > 0
    if planted:
        if het:
            return f"{spec.arm} + heterogeneity, h planted (misspecified: H has no heterogeneity term)"
        return "hazard arm (h planted)" if spec.arm == "hazard" else f"{spec.arm} (h planted)"
    if het:
        return f"{spec.arm}, heterogeneous, no h (misspecified: H has no heterogeneity term)"
    return f"{spec.arm} (no h)"


def _h_covers(c: dict, t: float) -> bool | None:
    sh = c.get("shape")
    if sh == "closed":
        return c["lower"] <= t <= c["upper"]
    if sh == "upper_bound":
        return t <= c["upper"]
    if sh == "lower_bound":
        return t >= c["lower"]
    return None


def _h_band(x: float) -> str:
    return ("< 0.1" if x < 0.1 else "0.1-0.3" if x < 0.3 else "0.3-1" if x < 1.0 else ">= 1")


def _n_band(n: int) -> str:
    return ("n < 1000" if n < 1000 else "n 1000-10000" if n < 10000
            else "n 10000-50000" if n < 50000 else "n >= 50000")


def _h_quantile(v, p):
    s = sorted(v)
    pos = p * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _h_wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"({c - h:.3f}-{c + h:.3f})"


def verify_nuisance_hazard(text: str, sets, library) -> list[Mismatch]:
    """Recompute EVERY figure of the nuisance-h section from each record's raw `diagnostics.hazard` --
    the would-be claim read as a plain dict; grouping, HB4, Wilson, quantiles, widths, parked MLEs and
    caps all re-derived here. A coverage row printed WITHOUT the box null beside it is itself a
    mismatch: an h coverage is only readable against the box (sr-manager gate, 2026-10-29)."""
    import statistics
    sec = text.split(_H_HEADING, 1)[1].split("\n## ", 1)[0]
    neg = 0.01
    m = re.search(r"of which lower edge hL > ([\d.e-]+)", sec)
    if m:
        neg = float(m.group(1))
    caps = [float(c) for c in re.findall(r"upper edge hL <= ([\d.e-]+)", sec)]
    ids = [cs.cohort_id for cs in sets]
    truths = library.truths(ids)
    groups: dict[str, list] = {}
    failed: dict[str, int] = {}
    for cs in sets:
        sp = library.spec(cs.cohort_id)
        t = (truths.get(cs.cohort_id) or {}).get("external_hazard")
        lab = _h_group(sp, t is not None)
        if not cs.ok:
            failed[lab] = failed.get(lab, 0) + 1
            continue
        hz = (cs.diagnostics or {}).get("hazard")
        if not isinstance(hz, dict) or not isinstance(hz.get("h_claim"), dict):
            continue
        h, hl = hz.get("h"), hz.get("hL")
        L = float(hz["L"]) if hz.get("L") else ((hl / h) if (h and hl is not None) else None)
        bh = hz.get("box_hL")
        floor = hz.get("at_floor") in (True, "true") or bool(bh and hl is not None
                                                             and hl <= float(bh[0]) * (1 + 1e-6))
        w = str((hz.get("mle") or {}).get("winner") or "")
        start = False
        if w.startswith("ladder:hL=") and hl:
            try:
                xv = float(w.split("=", 1)[1])
                start = xv > 0 and abs(math.log10(hl / xv)) < 1e-6
            except ValueError:
                pass
        groups.setdefault(lab, []).append({"c": hz["h_claim"], "t": t, "h": h, "L": L, "box": hz.get("box"),
                                           "n": sp.n, "ck": sp.cluster_key, "floor": floor, "start": start})
    out: list[Mismatch] = []

    def same(loc, what, got, want):
        if got.strip() != want:
            out.append(Mismatch(loc, what, got.strip() or "(missing)", want))

    def f3(v):
        return "n/a" if v is None else f"{v:.3f}"

    mode = None
    for line in sec.split("\n"):
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        head = cells[0]
        if head.startswith("group (h planted)"):
            mode = "planted"; continue
        if head.startswith("group (no h planted)"):
            mode = "zero"; continue
        if head.startswith("hazard arm by "):
            mode = "by_hL" if "true hL" in head else "by_n"; continue
        if mode in ("planted", "zero"):
            recs = groups.get(head)
            loc = f"h section / {head}"
            if recs is None:
                out.append(Mismatch(loc, "group exists in the run", head, "absent"))
                continue
            same(loc, "cohorts (clusters)", cells[1], f"{len(recs)} ({len({r['ck'] for r in recs})})")
            same(loc, "failed", cells[-1], str(failed.get(head, 0)))
        if mode == "planted":
            if len(cells) < 12:
                out.append(Mismatch(loc, "row has every column, the box null included", f"{len(cells)} cells", "12"))
                continue
            cl = [r for r in recs if r["c"].get("shape") == "closed"]
            k = sum(1 for r in cl if _h_covers(r["c"], r["t"]))
            same(loc, "closed coverage (Wilson)", cells[2], f"{k}/{len(cl)}" + (f" {_h_wilson(k, len(cl))}" if cl else ""))
            miss = {r["ck"] for r in cl if not _h_covers(r["c"], r["t"])}
            same(loc, "clusters with a closed miss", cells[3], f"{len(miss)} of {len({r['ck'] for r in cl})}")
            one = [r for r in recs if r["c"].get("shape") in ("upper_bound", "lower_bound")]
            same(loc, "one-sided coverage", cells[4], f"{sum(1 for r in one if _h_covers(r['c'], r['t']))}/{len(one)}")
            same(loc, "declined", cells[5], str(sum(1 for r in recs if r["c"].get("shape") == "none")))
            wd = [math.log10(r["c"]["upper"] / r["c"]["lower"]) for r in cl
                  if r["c"].get("lower") and r["c"].get("upper") and r["c"]["lower"] > 0]
            same(loc, "claim width", cells[6], ("n/a" if not wd else f"{statistics.median(wd):.2f}") + f" ({len(wd)})")
            bn = sum(1 for r in recs if r["box"] and r["box"][0] <= r["t"] <= r["box"][1])
            same(loc, "box null printed beside the coverage", cells[7], f"{bn}/{len(recs)}")
            ec = [abs(math.log10(r["h"] / r["t"])) for r in cl if r["h"] and r["h"] > 0]
            same(loc, "error on closed claims", cells[8],
                 f"{f3(statistics.median(ec) if ec else None)} · {f3(_h_quantile(ec, 0.9) if ec else None)}")
            ea = [math.log10(r["h"] / r["t"]) for r in recs if r["h"] and r["h"] > 0]
            aa = [abs(e) for e in ea]
            same(loc, "error, all cohorts (HB8)", cells[9],
                 f"{f3(statistics.median(aa) if aa else None)} · {f3(_h_quantile(aa, 0.9) if aa else None)} · "
                 + ("n/a" if not ea else f"{statistics.median(ea):+.3f}"))
            same(loc, "MLE at floor / start (closed)", cells[10],
                 f"{sum(1 for r in recs if r['floor'])} / {sum(1 for r in recs if r['start'])} "
                 f"({sum(1 for r in cl if r['start'])})")
        elif mode == "zero":
            det = [r for r in recs if r["c"].get("shape") in ("closed", "lower_bound")
                   and r["c"].get("lower") is not None and r["box"]
                   and r["c"]["lower"] > 10.0 * float(r["box"][0])]
            same(loc, "false detection (HB4, Wilson)", cells[2], f"{len(det)}/{len(recs)} {_h_wilson(len(det), len(recs))}")
            same(loc, "detections above negligible h", cells[3],
                 str(sum(1 for r in det if r["L"] and r["c"]["lower"] * r["L"] > neg)))
            up = [r["c"]["upper"] * r["L"] for r in recs if r["c"].get("shape") in ("closed", "upper_bound")
                  and r["c"].get("upper") is not None and r["L"]]
            for j, cv in enumerate(caps):
                same(loc, f"upper edge hL <= {cv:g}", cells[5 + j], f"{sum(1 for u in up if u <= cv)}/{len(recs)}")
        elif mode in ("by_n", "by_hL"):
            hz_recs = groups.get("hazard arm (h planted)", [])
            if mode == "by_n":
                sel = [r for r in hz_recs if _n_band(r["n"]) == head]
            else:
                sel = [r for r in hz_recs if r["L"] and f"true hL {_h_band(r['t'] * r['L'])}" == head]
            loc = f"h strata / {head}"
            same(loc, "cohorts", cells[1], str(len(sel)))
            cl = [r for r in sel if r["c"].get("shape") == "closed"]
            same(loc, "closed coverage", cells[2], f"{sum(1 for r in cl if _h_covers(r['c'], r['t']))}/{len(cl)}")
    return out


def format_report(problems: list[Mismatch], md_path: str) -> str:
    if not problems:
        return (f"VERIFIED  {md_path}\n"
                "Every quantitative claim was recomputed from the stored results and agrees.\n"
                "0 calibrations performed.")
    lines = [f"FAILED  {md_path}", f"{len(problems)} claim(s) do not match the stored results:", ""]
    lines += [p.format() for p in problems]
    lines.append("")
    lines.append("A report is not allowed to disagree with its own data. Regenerate it, or find "
                 "out why the data moved.")
    return "\n".join(lines)
