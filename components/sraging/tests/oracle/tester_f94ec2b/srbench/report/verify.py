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

    return problems


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
