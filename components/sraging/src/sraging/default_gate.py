"""The gate a candidate default calibrator must pass (constitution V and XI; spec 003g "Pre-registered rules", Naveh's
OK 2026-10-22). Pure functions: each takes plain rows and returns a `Verdict`; none reads a file or a record.

- PR-1 `witness_verdict`: the truth witness W = ll(truth) - ll(theta-hat) on valid informative cohorts; a stratum fails
  if any has W > 1.92, and is incomplete if any lacks a fresh witness.
- PR-2 `coverage_inside`: inside the reference set, the tester's own paired comparison of the candidate against its
  predecessor; a stratum fails if any quantity's calibration outcome reads "loses" (z < -2.0 in srbench's verdict).
  When every axis reads "not measured" it measured nothing and is "reported", not passed (DV-2, Naveh 2026-10-28);
  likewise when no measured axis has the n at which "loses" can occur (DV-5, Naveh 2026-10-29).
- PR-3 `coverage_outside`: outside it (no predecessor record), the candidate's coverage against its own coverage on the
  same arm and quantity inside, a one-sided 95 % Newcombe lower bound above -0.10.
- PR-4 `null_verdict`: the candidate's intervals narrower than the data-ignoring null's on eta and epsilon (XI).
- PR-5 `predecessor_verdict`: the predecessor is shown failing on its defect's stratum when at least one cohort shows
  the defect; a cohort whose point could not be recovered does not.
- PR-6 `gate_summary`: blocked if any PR-1..PR-4 verdict is FAIL or incomplete, any PR-5 is "not shown", or a
  required verdict is missing. An empty stratum never passes; it is listed as not covered.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

W_THR = 1.92
PR3_MARGIN = 0.10            # Naveh 2026-10-22, correcting 0.05 (which failed equal coverage at the panel's N)
ALPHA = 0.05
PR4_QUANTITIES = ("eta", "epsilon")
Z_ONE_SIDED_95 = 1.6448536269514722            # scipy.stats.norm.ppf(0.95)
#: DV-5 (Naveh 2026-10-29): the fewest paired cells at which srbench's calibration axis can read "loses" (z < -2.0).
#: Enumerated with the tester's own cluster_robust_mean over every cell difference in {-1, 0, 1} and every partition into
#: clusters: the most negative finite z is -1.0 at 2 cells and -3.0 at 3 (tests/unit/test_default_gate.py).
PR2_N_MIN_LOSES = 3

BLOCKING = {"PR-1": ("FAIL", "incomplete"), "PR-2": ("FAIL", "incomplete"), "PR-3": ("FAIL", "incomplete"),
            "PR-4": ("FAIL", "incomplete"), "PR-5": ("not shown",)}
NOT_COVERED = ("empty", "reported")


@dataclass
class Verdict:
    stratum: str
    rule: str
    verdict: str
    arm: str | None = None
    quantity: str | None = None
    n_valid: int = 0
    n_all: int = 0
    statistic: dict = field(default_factory=dict)
    interval: tuple | None = None
    reasons: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------------------------------- intervals
def clopper_pearson_upper(k: int, n: int, alpha: float = ALPHA) -> float:
    """The exact one-sided (1 - alpha) upper bound on a binomial rate, k failures in n."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def wilson(k: int, n: int, z: float) -> tuple[float, float]:
    """Wilson's score interval for k/n at normal quantile z."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float) -> tuple[float, float]:
    """Newcombe's hybrid score interval (1998, method 10) for p1 - p2, two independent proportions."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1, z)
    l2, u2 = wilson(k2, n2, z)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


# --------------------------------------------------------------------------------------------------------- rules
def witness_verdict(stratum: str, rows: list, *, w_thr: float = W_THR, arm: str | None = None) -> Verdict:
    """PR-1. `rows`: {cohort_id, valid, informative, witness: None | {W, fresh}}. W is undefined on a non-informative
    (heterogeneous) cohort, where the candidate's likelihood cannot score the truth."""
    if not rows:
        return Verdict(stratum, "PR-1", "empty", arm=arm)
    pop = [r for r in rows if r.get("valid") and r.get("informative")]
    if not pop:
        return Verdict(stratum, "PR-1", "reported", arm=arm, n_all=len(rows),
                       reasons=["no valid informative cohort"])
    missing = [r["cohort_id"] for r in pop if not r.get("witness") or not r["witness"].get("fresh", True)
               or r["witness"].get("W") is None]
    fails = [r["cohort_id"] for r in pop if r.get("witness") and r["witness"].get("W") is not None
             and r["witness"]["W"] > w_thr]
    n, k = len(pop), len(fails)
    n_w = n - len(missing)                       # N10 (Naveh 2026-10-29): the bound counts witnessed cohorts only
    stat = {"n": n, "n_witnessed": n_w, "k": k, "upper95": clopper_pearson_upper(k, n_w), "w_thr": w_thr,
            "failing": sorted(fails),
            "missing": sorted(missing),
            "w_max": max((r["witness"]["W"] for r in pop if r.get("witness") and r["witness"].get("W") is not None),
                         default=None)}
    verdict = "FAIL" if k else "incomplete" if missing else "PASS"
    reasons = ([f"{k} of {n} valid informative cohorts have W > {w_thr}"] if k else []) + \
              ([f"{len(missing)} without a fresh witness"] if missing else [])
    return Verdict(stratum, "PR-1", verdict, arm=arm, n_valid=n, n_all=len(rows), statistic=stat, reasons=reasons)


def coverage_inside(stratum: str, arm: str, axes: dict) -> Verdict:
    """PR-2. `axes`: {quantity: {outcome, n, ...}} from the tester's own paired comparison (candidate vs predecessor on
    the stratum's valid ids of this arm)."""
    if not axes:
        return Verdict(stratum, "PR-2", "reported", arm=arm, reasons=["no valid cohort in this arm"])
    loses = [f"{q}: {a['outcome']}" for q, a in sorted(axes.items()) if a.get("outcome") == "loses"]
    n = max((int(a.get("n") or 0) for a in axes.values()), default=0)
    stat = {q: dict(a) for q, a in axes.items()}
    if all(a.get("outcome") == "not measured" for a in axes.values()):     # DV-2, Naveh 2026-10-28: WARN is not PASS
        return Verdict(stratum, "PR-2", "reported", arm=arm, n_valid=n, n_all=n, statistic=stat,
                       reasons=["no axis measured"])
    if not loses and not any(a.get("outcome") != "not measured" and int(a.get("n") or 0) >= PR2_N_MIN_LOSES
                             for a in axes.values()):                   # DV-5, Naveh 2026-10-29: a PASS that cannot fail
        return Verdict(stratum, "PR-2", "reported", arm=arm, n_valid=n, n_all=n, statistic=stat,
                       reasons=[f"no measured axis has the {PR2_N_MIN_LOSES} cells at which 'loses' can occur"])
    return Verdict(stratum, "PR-2", "FAIL" if loses else "PASS", arm=arm, n_valid=n, n_all=n, statistic=stat,
                   reasons=loses)


def coverage_outside(stratum: str, arm: str, quantity: str, *, k_out: int, n_out: int, k_in: int, n_in: int,
                     margin: float = PR3_MARGIN, z: float = Z_ONE_SIDED_95, n_valid_out: int | None = None) -> Verdict:
    """PR-3. Coverage hits outside the reference set against the candidate's own hits on the same arm inside. `n_out`
    counts closed claims; `n_valid_out`, when given, the valid cohorts with a truth for the quantity (reasons only)."""
    if n_out <= 0:
        among = "" if n_valid_out is None else f" among {n_valid_out} valid"
        return Verdict(stratum, "PR-3", "reported", arm=arm, quantity=quantity,
                       reasons=[f"no closed claim{among} outside"])
    if n_in <= 0:
        return Verdict(stratum, "PR-3", "incomplete", arm=arm, quantity=quantity, n_valid=n_out, n_all=n_out,
                       reasons=["no comparator inside the reference set"])
    lo, hi = newcombe_diff(k_out, n_out, k_in, n_in, z)
    stat = {"k_out": k_out, "n_out": n_out, "k_in": k_in, "n_in": n_in, "p_out": k_out / n_out, "p_in": k_in / n_in,
            "margin": margin, "wilson95_out": wilson(k_out, n_out, 1.959963984540054)}
    ok = lo > -margin
    return Verdict(stratum, "PR-3", "PASS" if ok else "FAIL", arm=arm, quantity=quantity, n_valid=n_out,
                   n_all=n_out, statistic=stat, interval=(lo, hi),
                   reasons=[] if ok else [f"lower bound {lo:.3f} <= -{margin}"])


def null_verdict(stratum: str, arm: str, widths: dict, *, quantities=PR4_QUANTITIES) -> Verdict:
    """PR-4. `widths`: {quantity: {j, null, outcome?}}: the candidate's and the null's median closed-interval width
    (dex) on the stratum, and, when given, the tester's own information outcome with the null as the reference side
    (srbench verdict.information_axis: "wins" = narrower and the closed share not collapsed)."""
    missing = [q for q in quantities if q not in widths or widths[q].get("outcome") == "not measured"]
    if missing:
        return Verdict(stratum, "PR-4", "incomplete", arm=arm, statistic=dict(widths),
                       reasons=[f"no width for {', '.join(missing)}"])
    wide = []
    for q in quantities:
        w = widths[q]
        if "outcome" in w:
            if w["outcome"] != "wins":
                wide.append(f"{q}: the tester's information axis reads {w['outcome']}")
        elif not w["j"] < w["null"]:
            wide.append(f"{q}: J {w['j']:g} >= null {w['null']:g}")
    return Verdict(stratum, "PR-4", "FAIL" if wide else "PASS", arm=arm, statistic=dict(widths), reasons=wide)


def predecessor_verdict(stratum: str, rows: list, *, measure: str = "W33 > 1.92") -> Verdict:
    """PR-5. `rows`: {cohort_id, shown: True | False | None}; None = no point recovered (not shown)."""
    if not rows:
        return Verdict(stratum, "PR-5", "empty", statistic={"measure": measure})
    k = sum(r.get("shown") is True for r in rows)
    stat = {"measure": measure, "n": len(rows), "k_shown": k, "n_null": sum(r.get("shown") is None for r in rows),
            "shown_ids": sorted(r["cohort_id"] for r in rows if r.get("shown") is True)}
    return Verdict(stratum, "PR-5", "shown" if k else "not shown", n_valid=len(rows), n_all=len(rows), statistic=stat,
                   reasons=[] if k else ["no cohort shows the predecessor's defect"])


def gate_summary(verdicts: list, *, required=()) -> dict:
    """PR-6. The blocking list (each with its stratum, rule and verdict) and what is not covered."""
    def key(v):
        return {"stratum": v.stratum, "rule": v.rule, "verdict": v.verdict, "arm": v.arm, "quantity": v.quantity,
                "reasons": list(v.reasons)}
    blocking = [key(v) for v in verdicts if v.verdict in BLOCKING.get(v.rule, ())]
    have = {(v.stratum, v.rule) for v in verdicts}
    blocking += [{"stratum": s, "rule": r, "verdict": "missing"} for s, r in required if (s, r) not in have]
    return {"status": "blocked" if blocking else "passed", "blocking": blocking,
            "not_covered": [key(v) for v in verdicts if v.verdict in NOT_COVERED],
            "n_verdicts": len(verdicts)}
