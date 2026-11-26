"""Re-scoring the predecessor benchmark's stored results with THIS scorer, and diffing.

THIS IS THE REASON THIS REPOSITORY SITS BESIDE THE OLD ONE RATHER THAN FORKING IT. Two independent
implementations that agree constitute evidence; shared code cannot disagree and therefore tests
nothing. It consumes stored results and performs no calibration, so it is the cheapest validation
of the riskiest thing here — a scorer re-implemented from prose that has itself been wrong.

IT MEASURES TWO DIFFERENT THINGS, AND THE SECOND IS THE POINT.

  1. `any` shape class — the predecessor pools one-sided and two-sided claims into one coverage.
     Scored the same way, this scorer should REPRODUCE its published numbers. A mismatch here is a
     bug in one of the two implementations and must be chased.

  2. `closed` shape class — this scorer's default, which refuses to pool. This will DIFFER, and the
     difference is not an error: it is the size of the shape-pooling effect on the published
     figures. The predecessor's own documentation records the phenomenon (`kappa` pooled reads
     0.676 while its two-sided intervals cover 0.250); this puts a number on it for the matched arm.

A difference is a FINDING, not a failure. It means one of two independent implementations is wrong,
or that a documented methodological choice has a measurable size. Either is worth knowing, and
neither is resolved by silently reconciling them.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass

PARAMS = ("eta", "beta", "epsilon", "kappa")


@dataclass(frozen=True, slots=True)
class Row:
    quantity: str
    shape_class: str
    ours: float | None
    theirs: float | None
    our_n: int
    their_n: int

    @property
    def delta(self) -> float | None:
        if self.ours is None or self.theirs is None:
            return None
        return self.ours - self.theirs

    def format(self) -> str:
        o = "n/a" if self.ours is None else f"{self.ours:.4f}"
        t = "n/a" if self.theirs is None else f"{self.theirs:.4f}"
        d = "" if self.delta is None else f"{self.delta:+.4f}"
        flag = ""
        if self.delta is not None and abs(self.delta) > 5e-4:
            flag = "  <-- DIFFERS"
        return (f"  {self.quantity:<10} {self.shape_class:<10} ours {o:>8} (n={self.our_n:<4}) "
                f"theirs {t:>8} (n={self.their_n:<4}) {d:>9}{flag}")


def _shape_of(lo, hi) -> str:
    """Infer a claim shape from the predecessor's stored bounds.

    INFERENCE IS EXACTLY WHAT THIS PROJECT'S CONTRACT FORBIDS A METHOD FROM RELYING ON, and this
    function is why: the predecessor stores only numbers, so its shapes must be reconstructed, and
    a reconstruction is a guess about what the estimator meant. The contract makes a method STATE
    its shape precisely so that this step never has to happen for a new method.
    """
    lo_ok = lo is not None and isinstance(lo, (int, float)) and math.isfinite(lo)
    hi_ok = hi is not None and isinstance(hi, (int, float)) and math.isfinite(hi)
    if lo_ok and hi_ok:
        return "closed"
    if hi_ok:
        return "upper_bound"
    if lo_ok:
        return "lower_bound"
    return "none"


def rescore(results_path: str, method: str = "mle_profile", level: float = 0.95) -> dict:
    """Coverage by our rules, from their stored per-cohort output."""
    with open(results_path, encoding="utf-8") as fh:
        rows = json.load(fh)

    out: dict[str, dict[str, dict]] = {}
    for q in PARAMS:
        counts = {"closed": [0, 0], "one_sided": [0, 0], "any": [0, 0]}  # [hits, claimed]
        declined = total = 0
        for r in rows:
            m = (r.get("methods") or {}).get(method)
            if not m:
                continue
            truth = (r.get("truth") or {}).get(q)
            if truth is None:
                continue
            total += 1
            ci = (m.get("ci") or {}).get(f"{q}@{level}")
            if ci is None:
                declined += 1
                continue
            lo, hi = ci
            shape = _shape_of(lo, hi)
            if shape == "none":
                declined += 1
                continue
            lo_v = -math.inf if not (isinstance(lo, (int, float)) and math.isfinite(lo)) else lo
            hi_v = math.inf if not (isinstance(hi, (int, float)) and math.isfinite(hi)) else hi
            hit = 1 if lo_v <= truth <= hi_v else 0
            counts["any"][0] += hit
            counts["any"][1] += 1
            k = "closed" if shape == "closed" else "one_sided"
            counts[k][0] += hit
            counts[k][1] += 1
        out[q] = {
            k: {"coverage": (v[0] / v[1]) if v[1] else None, "hits": v[0], "claimed": v[1]}
            for k, v in counts.items()
        }
        out[q]["declined"] = declined
        out[q]["total"] = total
    return out


def published(tables_dir: str, method: str = "mle_profile", level: float = 0.95) -> dict:
    """Their published figures, read from the shipped table."""
    p = os.path.join(tables_dir, "coverage_all.csv")
    out = {}
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] != method or float(row["level"]) != level:
                continue
            if row["quantity"] not in PARAMS:
                continue
            try:
                cov = float(row["coverage"])
            except (TypeError, ValueError):
                cov = None
            out[row["quantity"]] = {"coverage": cov, "n": int(float(row["n"])),
                                    "unidentified": int(float(row["unidentified"]))}
    return out


def compare(results_path: str, tables_dir: str, method: str = "mle_profile",
            level: float = 0.95) -> list[Row]:
    ours = rescore(results_path, method, level)
    theirs = published(tables_dir, method, level)
    rows = []
    for q in PARAMS:
        t = theirs.get(q, {})
        # `any` is the like-for-like comparison: it is how they score.
        rows.append(Row(q, "any", ours[q]["any"]["coverage"], t.get("coverage"),
                        ours[q]["any"]["claimed"], t.get("n", 0)))
        # `closed` is our rule. It SHOULD differ; the difference is the shape-pooling effect.
        rows.append(Row(q, "closed", ours[q]["closed"]["coverage"], t.get("coverage"),
                        ours[q]["closed"]["claimed"], t.get("n", 0)))
    return rows


def format_report(rows: list[Row], results_path: str) -> str:
    like = [r for r in rows if r.shape_class == "any"]
    agree = [r for r in like if r.delta is not None and abs(r.delta) <= 5e-4]
    L = ["=" * 92,
         "CROSS-CHECK -- this scorer against the predecessor's published figures",
         "=" * 92,
         f"source: {results_path}",
         "",
         "LIKE-FOR-LIKE (`any`): scored the way they score, pooling claim shapes.",
         "  This SHOULD reproduce their numbers. A difference is a bug in one of two independent",
         "  implementations and must be chased, not reconciled.",
         ""]
    L += [r.format() for r in like]
    L += ["",
          f"  -> {len(agree)} of {len(like)} agree to within 5e-4.",
          "",
          "OUR RULE (`closed`): two-sided claims only, because a one-sided claim contains the",
          "  truth almost for free. This SHOULD differ; the difference is the size of the",
          "  shape-pooling effect on the published figures.",
          ""]
    L += [r.format() for r in rows if r.shape_class == "closed"]
    L += ["", "=" * 92,
          "A difference in the first block is a FINDING: one of two independent implementations",
          "is wrong. A difference in the second is the measured cost of a methodological choice.",
          "Neither is resolved by silently reconciling them.",
          "", "0 calibrations performed."]
    return "\n".join(L)
