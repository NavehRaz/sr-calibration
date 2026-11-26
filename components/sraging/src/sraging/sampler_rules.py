"""Spec 004's pre-registered rules HR-1 to HR-11 as pure functions (contracts/register-and-panel.md §3): each takes
plain rows and returns a `Verdict`; none reads a file, a clock or a record, so tests/acceptance/test_results_004.py can
recompute every stored verdict from its stored rows.

The skeleton (tasks T007): the verdict vocabulary, `require_assessable` and `summarise`. The rules join with their
stories: HR-1 and HR-2 (T022), HR-3 to HR-5 (T035), HR-8 and HR-9 (T069), HR-6 (T076), HR-7 (T083), HR-10 (T097),
HR-11 (T109). From freeze-H (HR-1 to HR-5) and freeze-L (HR-6 to HR-11) on, a change to a rule's code is named in the
results test's RULE_CHANGES. Each rule's constants are arguments whose defaults are the values
`panel_004.toml` / `langevin_004.toml` freeze; an acceptance phase passes the frozen ones.

- **No vacuous pass** (gate C2, Naveh 2026-10-29): an empty set, a pre-registered cell unassessable at m <= m_max, or a
  plant that is not informative reads `incomplete`, never `pass`. The incident is constitution V's: the tester's 19
  regression fingerprints hashed identically because the fixture produced zero deaths, and would have passed whatever
  broke.
- **A run at its CPU cap** reads `capped` (gate C4), never `pass`.
- **WARN is not PASS** (constitution V): `summarise` counts PASS, WARN (incomplete, capped) and FAIL separately, and
  its green line cannot be printed while any verdict is incomplete, capped or failed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

#: register-and-panel.md "Statuses"; `reported` is a measurement only
STATUSES = ("pass", "fail", "incomplete", "capped", "reported")
WARN_STATUSES = ("incomplete", "capped")
WALLS = ("clamp", "reflect")                    # OD-12: a per-wall verdict names its wall; None for the FP rules
GREEN = "ALL PASS"


@dataclass(frozen=True)
class Verdict:
    """One rule's verdict, per wall where the rule has one (HR-6, HR-10). Every verdict carries its denominators
    (constitution III) and, for a capped phase, the CPU spent against its cap."""

    rule: str
    wall: str | None
    status: str
    n_valid: int = 0
    n_unassessable: int = 0
    statistic: dict = field(default_factory=dict)
    crit: float | dict | None = None
    incomplete_reasons: tuple = ()
    cpu_spent: float | None = None          # CPU-h, parent plus workers
    cap: float | None = None                # CPU-h

    def __post_init__(self):
        if self.status not in STATUSES:
            raise ValueError(f"{self.rule}: status must be one of {STATUSES}, got {self.status!r}")
        if self.wall is not None and self.wall not in WALLS:
            raise ValueError(f"{self.rule}: wall must be one of {WALLS} or None, got {self.wall!r}")
        object.__setattr__(self, "incomplete_reasons", tuple(self.incomplete_reasons))
        if self.status == "incomplete" and not self.incomplete_reasons:
            raise ValueError(f"{self.rule}: an incomplete verdict needs its reasons")
        if self.status == "pass" and self.incomplete_reasons:
            raise ValueError(f"{self.rule}: a pass with something unassessed is not a pass (gate C2): "
                             f"{list(self.incomplete_reasons)}")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["incomplete_reasons"] = list(self.incomplete_reasons)
        return d


def require_assessable(cells) -> tuple:
    """Gate C2: the reasons a rule's input makes it `incomplete`, each naming its cell; empty when every cell is
    assessable and every plant informative. A cell row is flagged by `assessable: False` and a plant row by
    `informative: False`, each with an optional `reason`; an unflagged row counts as assessable."""
    cells = list(cells)
    if not cells:
        return ("nothing to assess: the set is empty (gate C2)",)
    out = []
    for i, c in enumerate(cells):
        label, why = c.get("id", f"cell {i}"), f" ({c['reason']})" if c.get("reason") else ""
        if c.get("assessable", True) is False:
            out.append(f"{label}: unassessable{why}")
        if c.get("informative", True) is False:
            out.append(f"{label}: plant not informative{why}")
    return tuple(out)


def raised(rows) -> tuple:
    """(the rows that are not error rows, the reasons naming each error row): a cell whose work raised is kept by its
    phase as a row with `error` {type, message, ...} (Naveh 2026-10-30, after hr3's crash); a rule reads it as
    unassessable, never as a value, so its verdict is incomplete (gate C2), never pass."""
    rows = list(rows)
    ok = [r for r in rows if not r.get("error")]
    why = [f"{r.get('id', '?')}: the cell raised ({r['error'].get('type')}: {r['error'].get('message')}) (gate C2)"
           for r in rows if r.get("error")]
    return ok, tuple(why)


def summarise(verdicts) -> dict:
    """PASS, WARN (incomplete, capped) and FAIL counted separately, and `reported` apart (a measurement, not a
    verdict). Green only when something passed and nothing warned or failed; the line starts with GREEN only then."""
    verdicts = list(verdicts)
    by = {s: sum(v.status == s for v in verdicts) for s in STATUSES}
    n_pass, n_warn, n_fail = by["pass"], by["incomplete"] + by["capped"], by["fail"]
    green = n_pass > 0 and n_warn == 0 and n_fail == 0
    counts = (f"PASS {n_pass}, WARN {n_warn} (incomplete {by['incomplete']}, capped {by['capped']}), FAIL {n_fail}, "
              f"reported {by['reported']}")
    return {"PASS": n_pass, "WARN": n_warn, "FAIL": n_fail, "REPORTED": by["reported"], "by_status": by,
            "green": green, "line": f"{GREEN}: {counts}" if green else f"NOT A PASS: {counts}",
            "not_passed": [(v.rule, v.wall, v.status) for v in verdicts if v.status in WARN_STATUSES + ("fail",)]}


# ------------------------------------------------------------------------------------------------------------ HR-1
#: the panel's strata (research R9); HR-1 needs a valid cell at sd <= HR1_SD_STRATA in each
HR1_STRATA = ("syn-broad", "syn-mid", "syn-sharp", "real-lo-r", "real-hi-r", "human-10k", "human-25k", "human-100k",
              "realfull")
HR1_TARGET = 0.1                # nats: a passing check's error against the reference must be below it
HR1_THRESHOLD = 0.05            # the check's threshold (sraging.engine.mixture.THRESHOLD): the plant's check fires at it
HR1_SD_STRATA = 0.3


def _cp_upper(k: int, n: int, level: float = 0.95) -> float:
    """The one-sided Clopper-Pearson upper bound on a rate with k events in n (1 - (1 - level)^(1/n) at k = 0)."""
    if n == 0:
        return 1.0
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(level, k + 1, n - k))


def hr1_verdict(cells, *, strata=HR1_STRATA, target=HR1_TARGET, threshold=HR1_THRESHOLD,
                sd_strata=HR1_SD_STRATA) -> Verdict:
    """HR-1 (contracts/register-and-panel.md §3): automatic M's accuracy, and the M = 11 plant.

    A cell row: `id, stratum, sd, ll` (automatic M's value), `verdict` (its check's), `M, abs_dll` (the last check),
    `ll_ref` (M = 192), `ref_valid` (|l(128) - l(192)| < 0.01), `ll_11` (the plant: M = 11 unchecked) and `check_11`
    (its check value |l(11) - l(17)|). Only valid cells are judged; invalid ones are counted as reference-unconverged.
    PASS iff no false pass (a converged check with |ll - ll_ref| >= target) and the plant's check (check_11 >= the
    threshold) fires on every cell where |ll_11 - ll_ref| > target. Incomplete (gate C2) if a stratum has no valid cell
    at sd <= sd_strata, or no cell has M = 11 more than `target` off (the plant is not informative). A failure is a FAIL
    even with incomplete reasons, which it keeps."""
    cells = list(cells)
    valid = [c for c in cells if c["ref_valid"]]
    checked = [c for c in valid if c["verdict"] == "converged"]
    false_passes = [c["id"] for c in checked if abs(c["ll"] - c["ll_ref"]) >= target]
    informative = [c for c in valid if abs(c["ll_11"] - c["ll_ref"]) > target]
    misses = [c["id"] for c in informative if not c["check_11"] >= threshold]
    reasons = [] if cells else ["nothing to assess: no cell (gate C2)"]
    for s in strata:
        if not any(c["stratum"] == s and c["sd"] <= sd_strata for c in valid):
            reasons.append(f"stratum {s}: no valid cell at sd <= {sd_strata:g} (gate C2)")
    if cells and not informative:
        reasons.append(f"plant M = 11: not informative, no valid cell has M = 11 more than {target:g} nats off (gate "
                       f"C2)")
    status = "fail" if (false_passes or misses) else ("incomplete" if reasons else "pass")
    stat = {"false_passes": false_passes, "n_false_pass": len(false_passes), "plant_misses": misses,
            "n_plant_informative": len(informative), "n_checked_valid": len(checked),
            "cp_upper_95": _cp_upper(len(false_passes), len(checked)),
            "n_reference_unconverged": len(cells) - len(valid),
            "unconverged": [{"id": c["id"], "M": c["M"], "abs_dll": c["abs_dll"]} for c in valid
                            if c["verdict"] != "converged"],
            "max_abs_err_checked": max((abs(c["ll"] - c["ll_ref"]) for c in checked), default=None)}
    return Verdict("HR-1", None, status, n_valid=len(valid), n_unassessable=len(cells) - len(valid), statistic=stat,
                   crit=target, incomplete_reasons=tuple(reasons))


# ------------------------------------------------------------------------------------------------------------ HR-2
HR2_PLANT_FACTOR = 10.0         # a plant is informative when its jump exceeds 10 x the tolerance


def hr2_verdict(cells, plants, *, tol, plant_factor=HR2_PLANT_FACTOR) -> Verdict:
    """HR-2 (contracts/register-and-panel.md §3; research R11): no jump in sd.

    A cell row: `id, sd, cd` (the central difference of l in sd) and `grad` (the engine's dl/dsd); `tol` comes from the
    noise pilot (hr2-noise). A plant row: `M` (5, or the fallback 3), `jump` (|l(0.35+) - l(0.35-)| of normal_frac_sd,
    or across its M = 3 crossing) and the same `cd`, `grad` across the crossing. The M = 5 plant is used when its jump
    exceeds plant_factor x tol, else the M = 3 fallback when its does; the chosen plant is detected when |cd - grad| >
    tol. PASS iff every cell has |cd - grad| <= tol and the chosen plant is detected. Incomplete (gate C2) if no cell,
    or neither plant is informative."""
    if tol is None or not (isinstance(tol, (int, float)) and math.isfinite(tol) and tol > 0):
        raise ValueError(f"HR-2: tol must be a finite number > 0, from the noise pilot, got {tol!r}")
    cells = list(cells)
    over = [c["id"] for c in cells if abs(c["cd"] - c["grad"]) > tol]
    by_m = {p["M"]: p for p in plants}
    chosen = next((by_m[m] for m in (5, 3) if m in by_m and by_m[m]["jump"] > plant_factor * tol), None)
    reasons = [] if cells else ["nothing to assess: no (cohort, sd) cell (gate C2)"]
    plant = None
    if chosen is None:
        jumps = ", ".join(f"M = {m}: jump {by_m[m]['jump']:.3g}" if m in by_m else f"M = {m}: not run" for m in (5, 3))
        reasons.append(f"plant: neither M = 5 nor M = 3 is informative ({jumps}; needs > {plant_factor:g} x tol = "
                       f"{plant_factor * tol:.3g}) (gate C2)")
    else:
        plant = {"M": chosen["M"], "jump": chosen["jump"], "abs_diff": abs(chosen["cd"] - chosen["grad"]),
                 "detected": abs(chosen["cd"] - chosen["grad"]) > tol}
    status = "fail" if (over or (plant is not None and not plant["detected"])) else ("incomplete" if reasons
                                                                                    else "pass")
    stat = {"over_tol": over, "max_abs_diff": max((abs(c["cd"] - c["grad"]) for c in cells), default=None),
            "plant": plant, "plants": [dict(p) for p in plants]}
    return Verdict("HR-2", None, status, n_valid=len(cells), statistic=stat, crit=tol, incomplete_reasons=tuple(reasons))


# ------------------------------------------------------------------------------------------------------------ HR-3
HR3_THRESHOLD = 0.05            # nats: |l_adaptive - l_full| at a common M below it, for every pair
HR3_CAP = 10.0                  # CPU-h (gate C4)


def _upper_95_mean(x) -> float:
    """The one-sided 95 % upper bound of a mean (Student t; +inf with fewer than two values)."""
    x = [float(v) for v in x]
    if len(x) < 2:
        return math.inf
    from scipy.stats import t
    m = sum(x) / len(x)
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))
    return m + float(t.ppf(0.95, len(x) - 1)) * sd / math.sqrt(len(x))


def hr3_verdict(pairs, plants, *, cpu_spent, cap=HR3_CAP, threshold=HR3_THRESHOLD) -> Verdict:
    """HR-3 (contracts/register-and-panel.md §3; research R7): adaptive M against full M.

    A pair row: `id, sharp, ll_adaptive, ll_full` (the two maxima's final l at a common M), `M_common`, `cpu_adaptive,
    cpu_full` and `checked` (both arms carry a checked final value). A plant row: `id, sharp, ll_plant, ll_full` (the
    never-raise run). FAIL if any |dl| >= threshold. `capped` at the cap. Incomplete (gate C2) if a pair lacks a
    checked final value, or the plant does not exceed the threshold on every sharp cell (or there is none). The CPU
    saving is claimed iff the upper 95 % bound of the mean log(cpu_adaptive / cpu_full) is < 0 (a statistic: it does
    not set the status)."""
    pairs, plants = list(pairs), list(plants)
    over = [p["id"] for p in pairs if not abs(p["ll_adaptive"] - p["ll_full"]) < threshold]
    reasons = [] if pairs else ["nothing to assess: no pair (gate C2)"]
    reasons += [f"{p['id']}: no checked final value (gate C2)" for p in pairs if not p["checked"]]
    sharp = [p for p in plants if p["sharp"]]
    weak = [p["id"] for p in sharp if not abs(p["ll_plant"] - p["ll_full"]) > threshold]
    if not sharp or weak:
        reasons.append(f"the never-raise plant: not informative, it does not exceed {threshold:g} nats on the sharp "
                       f"cells {weak if sharp else '(none run)'} (gate C2)")
    logs = [math.log(p["cpu_adaptive"] / p["cpu_full"]) for p in pairs if p["cpu_adaptive"] > 0 and p["cpu_full"] > 0]
    up = _upper_95_mean(logs)
    saving = {"n": len(logs), "mean_log_ratio": sum(logs) / len(logs) if logs else None, "upper_95_log_ratio": up,
              "claimed": up < 0}
    status = "fail" if over else "capped" if cpu_spent >= cap else "incomplete" if reasons else "pass"
    stat = {"over": over, "max_abs_dll": max((abs(p["ll_adaptive"] - p["ll_full"]) for p in pairs), default=None),
            "plant": {"n_sharp": len(sharp), "not_exceeding": weak,
                      "min_abs_dll_sharp": min((abs(p["ll_plant"] - p["ll_full"]) for p in sharp), default=None)},
            "saving": saving}
    return Verdict("HR-3", None, status, n_valid=sum(p["checked"] for p in pairs),
                   n_unassessable=sum(not p["checked"] for p in pairs), statistic=stat, crit=threshold,
                   incomplete_reasons=tuple(reasons), cpu_spent=float(cpu_spent), cap=float(cap))


# ------------------------------------------------------------------------------------------------------------ HR-4
def hr4_verdict(rows, plants, *, walls=None, stated_W=None) -> Verdict:
    """HR-4 (FR-007): a byte row is `W, M, kind` (value or grad) and `equal` (the pool's bytes are the in-process ones);
    a plant row is `W, M, dropped` (M mod W nodes) and `differs`. PASS iff every byte row is equal and the plant differs
    on every cell where it drops a node (a cell that drops none has no size and is not judged). Incomplete (gate C2) if
    no plant drops a node. A wall-time speedup at `stated_W` is claimed iff the upper 95 % bound of the mean paired
    log(wall_pool / wall_serial) is < 0 (a statistic)."""
    rows, plants = list(rows), list(plants)
    unequal = [{"W": r["W"], "M": r["M"], "kind": r["kind"]} for r in rows if not r["equal"]]
    sized = [p for p in plants if p["dropped"] > 0]
    same = [{"W": p["W"], "M": p["M"]} for p in sized if not p["differs"]]
    reasons = [] if rows else ["nothing to assess: no byte comparison (gate C2)"]
    if not sized:
        reasons.append("the remainder plant: not informative, no cell drops a node (gate C2)")
    speedup = None
    if walls is not None and stated_W is not None:
        logs = [math.log(w["wall_pool"] / w["wall_serial"]) for w in walls if w["W"] == stated_W]
        up = _upper_95_mean(logs)
        speedup = {"W": stated_W, "n": len(logs), "upper_95_ratio": math.exp(up) if math.isfinite(up) else math.inf,
                   "claimed": up < 0}
    status = "fail" if (unequal or same) else "incomplete" if reasons else "pass"
    return Verdict("HR-4", None, status, n_valid=len(rows), statistic={"unequal": unequal, "plant_same": same,
                                                                       "n_plant_sized": len(sized),
                                                                       "speedup": speedup},
                   incomplete_reasons=tuple(reasons))


# ------------------------------------------------------------------------------------------------------------ HR-5
HR5_BAND = (0.5, 2.0)           # measured / stated per cell
HR5_SHARE = 0.95                # of the cells inside the band (OD-5 (a))
HR5_MEDIAN = (0.8, 1.25)        # the median ratio's band


def _hr5_core(ratios, band, share, median_band):
    import statistics
    inside = sum(band[0] <= r <= band[1] for r in ratios)
    med = statistics.median(ratios) if ratios else None
    ok = bool(ratios) and inside >= share * len(ratios) and median_band[0] <= med <= median_band[1]
    return ok, {"n_outside": len(ratios) - inside, "share_inside": inside / len(ratios) if ratios else None,
                "median_ratio": med}


def hr5_verdict(cells, *, band=HR5_BAND, share=HR5_SHARE, median_band=HR5_MEDIAN) -> Verdict:
    """HR-5 (FR-008; OD-5 (a)): a cell row is `id, stated_cpu_s, measured_cpu_s` (compilation apart) and
    `blind_stated_cpu_s` (the M-blind predictor's statement, cost.blind). PASS iff >= share of the cells have
    measured / stated in the band and the median ratio is in median_band; the M-blind predictor must FAIL the same
    rule, else the plant is not informative and the verdict is incomplete (gate C2)."""
    cells = list(cells)
    ok, stat = _hr5_core([c["measured_cpu_s"] / c["stated_cpu_s"] for c in cells], band, share, median_band)
    pok, pstat = _hr5_core([c["measured_cpu_s"] / c["blind_stated_cpu_s"] for c in cells], band, share, median_band)
    reasons = [] if cells else ["nothing to assess: no cell (gate C2)"]
    if cells and pok:
        reasons.append("the M-blind predictor passes the rule too: not informative (gate C2)")
    stat["plant"] = {**pstat, "status": "pass" if pok else "fail"}
    status = "fail" if (cells and not ok) else "incomplete" if reasons else "pass"
    return Verdict("HR-5", None, status, n_valid=len(cells), statistic=stat, crit={"band": list(band), "share": share,
                                                                                  "median_band": list(median_band)},
                   incomplete_reasons=tuple(reasons))


# ------------------------------------------------------------------------------------------------------------ HR-8
def hr8_verdict(rows, plants) -> Verdict:
    """HR-8 (S7; research R17): a row is `W, n` and `equal` (the sample's bytes at W equal W = 1's); a plant row is `W,
    dropped` (n mod W) and `differs` (B4's dropped-and-padded sample against the real one). PASS iff every row is equal
    and the plant differs wherever it drops someone; incomplete (gate C2) if no plant drops anyone, or no row."""
    rows, plants = list(rows), list(plants)
    unequal = [r["W"] for r in rows if not r["equal"]]
    sized = [p for p in plants if p["dropped"] > 0]
    same = [p["W"] for p in sized if not p["differs"]]
    reasons = [] if rows else ["nothing to assess: no byte comparison (gate C2)"]
    if not sized:
        reasons.append("the n mod W plant: not informative, it drops no one (gate C2)")
    status = "fail" if (unequal or same) else "incomplete" if reasons else "pass"
    return Verdict("HR-8", None, status, n_valid=len(rows), statistic={"unequal": unequal, "plant_same": same,
                                                                       "n_plant_sized": len(sized)},
                   incomplete_reasons=tuple(reasons))


# ------------------------------------------------------------------------------------------------------------ HR-9
HR9_ALPHA = {"plant": 0.05, "twin": 0.025, "fix": 0.025}


def hr9_verdict(arms, *, alpha=None) -> Verdict:
    """HR-9 (B1, like for like; gate C3; research R14): `arms` maps plant, twin and fix to their KS {D, p_value, n}.
    The plant, SRK(dt, m = 16) against SRK(dt/16, m = 1), must reject at alpha 0.05; the twin, sraging(dt/16) against
    SRK(dt/16, m = 1), and the fix, sraging(dt/16) against FP through the same dt recording, must not reject at alpha
    0.025 each. FAIL if the plant does not reject or a must-pass arm does; incomplete (gate C2) if an arm was not run."""
    alpha = dict(HR9_ALPHA if alpha is None else alpha)
    missing = [a for a in alpha if a not in arms]
    failed = []
    if "plant" in arms and not arms["plant"]["p_value"] < alpha["plant"]:
        failed.append("plant")
    failed += [a for a in ("twin", "fix") if a in arms and arms[a]["p_value"] < alpha[a]]
    reasons = [f"arm {a!r} not run (gate C2)" for a in missing]
    status = "fail" if failed else "incomplete" if reasons else "pass"
    return Verdict("HR-9", None, status, n_valid=len(arms), statistic={"failed": failed, "arms": dict(arms)},
                   crit=alpha, incomplete_reasons=tuple(reasons))


# ------------------------------------------------------------------------------------------------------------ HR-6
HR6_CRIT = {"clamp": 1.973, "reflect": 1.698}   # x / sqrt(n): Bonferroni over the wall's family
HR6_G = {"clamp": 60, "reflect": 8}
HR6_BAR = 10.0                                   # a cell is assessable iff its judged bias <= crit / HR6_BAR
HR6_CAP = 35.0


#: R18's branch (Naveh 2026-11-03; contracts/register-and-panel.md §3, amendment A1): a borrowed row (its step is
#: another cell's) is unassessable when its own coupled D exceeds GUARD_X times its cell's mean per-key D
GUARD_X = 3.0


def family_crit(G: int, alpha: float = 0.05) -> float:
    """x with crit = x / sqrt(n): the one-sample KS bound at Bonferroni alpha / G (the Kolmogorov distribution)."""
    from scipy.stats import kstwobign
    return float(kstwobign.isf(alpha / int(G)))


def frozen_bias(row, ladder_bias) -> tuple:
    """(the judged bias, the reasons it is unassessable beyond the bar): at a frozen step (`frozen_b`, the frozen
    cell's pooled bias; amendment A1) the row is not re-measured, and a `borrowed` row is guarded by its own D against
    GUARD_X x `D_key_mean`; otherwise the row's own ladder bias, `ladder_bias(row)`."""
    if row.get("frozen_b") is None:
        return ladder_bias(row), []
    why = []
    if row.get("borrowed") and row.get("D_own") is not None and row.get("D_key_mean"):
        if float(row["D_own"]) > GUARD_X * float(row["D_key_mean"]):
            why.append(f"borrowed step: its own D {float(row['D_own']):.3g} > {GUARD_X:g} x its cell's mean per-key D "
                       f"{float(row['D_key_mean']):.3g} (the guard, amendment A1)")
    return float(row["frozen_b"]), why


def hr6_judged_bias(cell) -> float:
    """b(dt/2) = D / (2^p - 1), the bias of the judged dt/2 sample, with p clipped to [1/2, 1] (gate C1)."""
    p = min(max(float(cell["p_hat"]), 0.5), 1.0)
    return float(cell["D"]) / (2.0 ** p - 1.0)


def hr6_verdict(cells, plants, *, wall, cpu_spent, cap=HR6_CAP, G=None) -> Verdict:
    """HR-6 (S4; contracts/register-and-panel.md §3; research R18): one verdict per wall. A cell row is `id, n, D` (the
    coupled pair's sup|S_dt - S_dt/2|), `ks_D` (the dt/2 rung's KS against FP), `p_hat` (P1's), `m, m_max`. A cell is
    assessable iff m <= m_max and b(dt/2) <= crit/10, with crit = x / sqrt(n): x = family_crit(G) for the family's
    actual row count `G` (amendment A3, Naveh 2026-11-03), or HR6_CRIT[wall] without it. At a frozen step (amendment
    A1) b is the frozen cell's pooled bias (`frozen_b`) and a borrowed row is guarded (frozen_bias). A plant row is
    `id, ks_D, n` (the eta(1 + delta) plant against FP at eta; the clamp's 2 cells), judged at the clamp family's crit.
    FAIL if an assessable cell's ks_D > crit or a plant does not reject; `capped` past the cap; incomplete (gate C2) if
    a cell is unassessable, there is no cell, or the clamp has no plant."""
    if wall not in HR6_CRIT:
        raise ValueError(f"HR-6: wall is one of {tuple(HR6_CRIT)}, got {wall!r}")
    x = family_crit(G) if G is not None else HR6_CRIT[wall]
    cells, plants = list(cells), list(plants)
    n_cells, err_ids = len(cells), [c.get("id") for c in cells if c.get("error")]
    cells, cell_errors = raised(cells)
    plants, plant_errors = raised(plants)
    crits = {c["id"]: x / math.sqrt(c["n"]) for c in cells}
    rows, unassessable, failing, guarded = [], [], [], []
    for c in cells:
        cr = crits[c["id"]]
        b, why = frozen_bias(c, hr6_judged_bias)
        ok = c["m"] <= c["m_max"] and b <= cr / HR6_BAR and not why
        guarded += [f"{c['id']}: {w}" for w in why]
        rows.append({"id": c["id"], "crit": cr, "judged_bias": b, "bar": cr / HR6_BAR, "assessable": ok,
                     "ks_D": c["ks_D"], "frozen": c.get("frozen_b") is not None})
        if not ok:
            unassessable.append(c["id"])
        elif c["ks_D"] > cr:
            failing.append(c["id"])
    px = x if wall == "clamp" else HR6_CRIT["clamp"]
    pcrit = {p["id"]: px / math.sqrt(p["n"]) for p in plants}
    weak = [p["id"] for p in plants if not p["ks_D"] > pcrit[p["id"]]]
    reasons = [] if n_cells else ["nothing to assess: no cell (gate C2)"]
    reasons += list(cell_errors) + list(plant_errors) + guarded
    reasons += [f"{cid}: unassessable (m > m_max, or b(dt/2) > crit/10) (gate C1, C2)" for cid in unassessable]
    reasons += [f"{c['id']}: m {c['m']} > m_max {c['m_max']}" for c in cells if c["m"] > c["m_max"]]
    if wall == "clamp" and not plants:
        reasons.append("the eta(1 + delta) plant was not run (gate C2)")
    status = "fail" if (failing or weak) else "capped" if cpu_spent >= cap else "incomplete" if reasons else "pass"
    crit = x / math.sqrt(cells[0]["n"]) if cells else None
    unassessable += err_ids
    return Verdict("HR-6", wall, status, n_valid=n_cells - len(unassessable), n_unassessable=len(unassessable),
                   statistic={"failing": failing, "unassessable": unassessable, "plant_not_rejecting": weak,
                              "cells": rows, "G": G if G is not None else HR6_G[wall], "crit_x": x},
                   crit=crit, incomplete_reasons=tuple(reasons), cpu_spent=float(cpu_spent), cap=float(cap))


# ------------------------------------------------------------------------------------------------------------ HR-7
HR7_ALPHA = 0.05


def hr7_zcrit(k: int, alpha: float = HR7_ALPHA) -> float:
    """The two-sided Bonferroni critical |z| over k score components."""
    from statistics import NormalDist
    return NormalDist().inv_cdf(1.0 - alpha / (2 * k))


def hr7_detected(arm, zcrit) -> bool:
    return any(abs(float(z)) > zcrit for z in arm["z"].values())


def hr7_verdict(arms, *, alpha=HR7_ALPHA, branch=1, smallest_detectable=None, n_errors=0,
                recheck_error=None) -> Verdict:
    """HR-7 (S5; research R20; gate C5): `arms` maps plant and twin to {"z": {component: mean score / SE}} (and KS,
    reported). A detection is any component past the Bonferroni bound over the components. PASS iff the plant is
    detected and the twin is not; FAIL if the twin is detected; `capped` on gate C5's third branch (clamp at the cap),
    with the smallest detectable planted shift reported; incomplete (gate C2) if the plant is not detected, if
    `n_errors` replicates raised (their cohorts are missing from the arms), or if `recheck_error` names why the re-check
    of S5's step (amendment A6: P2's coupled halving at S5's own step, run first) gave no bias to cite."""
    missing = [a for a in ("plant", "twin") if a not in arms]
    k = max((len(a["z"]) for a in arms.values()), default=1)
    zc = hr7_zcrit(k, alpha)
    plant = "plant" in arms and hr7_detected(arms["plant"], zc)
    twin = "twin" in arms and hr7_detected(arms["twin"], zc)
    reasons = [f"arm {a!r} not run (gate C2)" for a in missing]
    if "plant" in arms and not plant:
        reasons.append("the plant is not detected at this R (gate C2)")
    if n_errors:
        reasons.append(f"{n_errors} replicate(s) raised: the arms stand on fewer than R cohorts (gate C2)")
    if recheck_error:
        reasons.append(f"the re-check of S5's step gave no bias to cite ({recheck_error}) (A6, gate C2)")
    status = "fail" if twin else "capped" if branch == 3 else "incomplete" if reasons else "pass"
    return Verdict("HR-7", None, status, n_valid=len(arms), statistic={
        "plant_detected": plant, "twin_detected": twin, "zcrit": zc, "branch": branch,
        "smallest_detectable_nats_per_1e5": smallest_detectable, "arms": dict(arms)},
        crit=zc, incomplete_reasons=tuple(reasons))


def s5_wall_branch(proj_clamp, proj_reflect, reflect_converged, cap=86.0, *, smallest_detectable, plant_size) -> dict:
    """Gate C5, Naveh's three branches for S5's wall: 1, clamp if its projection fits the cap; 2, else reflect if P2
    measured it converged at S5's resolution and it fits; 3, else clamp at the cap, where a smallest detectable shift
    above the plant's size means asking Naveh before the run."""
    if proj_clamp <= cap:
        return {"branch": 1, "wall": "clamp", "action": "run", "reason": f"clamp projected at {proj_clamp:g} <= {cap:g}"}
    if reflect_converged and proj_reflect <= cap:
        return {"branch": 2, "wall": "reflect", "action": "run",
                "reason": f"clamp at {proj_clamp:g} > {cap:g}; reflect converged and projected at {proj_reflect:g}"}
    ask = smallest_detectable > plant_size
    return {"branch": 3, "wall": "clamp", "action": "ask_naveh" if ask else "run",
            "reason": f"clamp at the cap {cap:g}: the smallest detectable shift {smallest_detectable:g} nats per 1e5 "
                      f"is {'above' if ask else 'within'} the plant's {plant_size:g}"}


# ----------------------------------------------------------------------------------------------------------- HR-10
HR10_CRIT = 1.797                   # x / sqrt(n_alive): the KS bound at Bonferroni alpha 0.05 / 16 (kstwobign)
HR10_G = 16                         # 8 configurations x 2 ages, per wall
HR10_BAR = 10.0                     # a test is assessable iff its path bias <= crit / HR10_BAR
HR10_CAP = 6.0                      # CPU-h
HR10_CHECKS = ("nan_iff_dead", "times_equal", "bytes_by_W")
HR10_U_MIN, HR10_U_LUMP, HR10_X_STEPS = 0.01, 3.0, 5.0


def hr10_censoring(p0, *, eps_max, dt, quantile) -> tuple:
    """The clamp's censoring point (research R25): u_c = max(0.01, 3 p0) and x_c = max(the FP quantile at u_c,
    5 sqrt(2 eps_max dt)); x_c is None when the quantile is (u_c >= 1: the lump covers the law)."""
    u_c = max(HR10_U_MIN, HR10_U_LUMP * float(p0))
    q = quantile(u_c) if u_c < 1.0 else None
    if q is None:
        return u_c, None
    return u_c, max(float(q), HR10_X_STEPS * math.sqrt(2.0 * float(eps_max) * float(dt)))


def hr10_nan_check(times, ages, X) -> dict:
    """The deterministic NaN check (research R24): X[i, j] is NaN iff times[i] <= ages[j]. Counts the dead with a
    value (zeros apart: SRK's getISR writes 0 for the dead, D-411) and the living with NaN."""
    import numpy as np
    times, ages, X = np.asarray(times, float), np.asarray(ages, float), np.asarray(X, float)
    dead = times[:, None] <= ages[None, :]
    nan = np.isnan(X)
    out = {"zeros_among_dead": int(np.sum(dead & (X == 0.0))), "values_among_dead": int(np.sum(dead & ~nan)),
           "nan_among_alive": int(np.sum(~dead & nan)), "n_dead": int(dead.sum()), "n_alive": int((~dead).sum())}
    out["holds"] = out["values_among_dead"] == 0 and out["nan_among_alive"] == 0
    return out


def hr10_judged_bias(test) -> float:
    """The path bias of the judged dt/2 sample, D / (2^p - 1) over the judged range, p clipped to [1/2, 1]."""
    p = min(max(float(test["p_hat"]), 0.5), 1.0)
    return float(test["D_bias"]) / (2.0 ** p - 1.0)


def hr10_verdict(tests, checks, plants, *, wall, cpu_spent, cap=HR10_CAP, crit_x=HR10_CRIT) -> Verdict:
    """HR-10 (paths; research R24, R25; register §3): one verdict per wall. A test row (a configuration's age, the
    dt/2 rung) is `id, n_alive, p0, u_c, x_c, D_c, D, D_bias` (the coupled pair's sup over the judged range), `p_hat,
    m, m_max, reference_grid_error`; crit = crit_x / sqrt(n_alive). A test is assessable iff m <= m_max, its path bias
    <= crit/10 and the reference's grid error <= crit/10. The judged statistic is D_c under clamp (D reported) and D
    under reflect. `checks` maps HR10_CHECKS to {"holds": bool}. `plants`: kind "plant" (the eta(1 + delta) draws;
    `statistic` the wall's judged one, `n_alive`) must reject, kind "twin" (delta = 0) must not, and kind
    "srk_recorder" (`nan_check`) must fail the NaN check. FAIL if a judged statistic of an assessable test exceeds
    crit, a check fails, the plant does not reject or the twin does; `capped` past the cap; incomplete (gate C2) if a
    test is unassessable, there is none, a check or a plant was not run, or SRK's recorder passes the NaN check."""
    if wall not in WALLS:
        raise ValueError(f"HR-10: wall is one of {WALLS}, got {wall!r}")
    tests, plants, checks = list(tests), list(plants), dict(checks)
    rows, unassessable, failing, reasons = [], [], [], [] if tests else ["nothing to assess: no test (gate C2)"]
    unassessable += [t.get("id") for t in tests if t.get("error")]
    tests, why = raised(tests)
    plants, pwhy = raised(plants)
    reasons += list(why) + list(pwhy)
    n_tests = len(tests) + len(unassessable)
    for t in tests:
        cr = crit_x / math.sqrt(t["n_alive"]) if t.get("n_alive") else None
        if cr is None:
            unassessable.append(t["id"])
            reasons.append(f"{t['id']}: no one alive (gate C2)")
            continue
        b, gwhy = frozen_bias(t, hr10_judged_bias)          # amendment A2: the death-time frozen step gates
        g = float(t.get("reference_grid_error") or 0.0)
        why = list(gwhy) + ([f"m {t['m']} > m_max {t['m_max']}"] if t["m"] > t["m_max"] else []) + \
              ([f"path bias not measured ({b!r})"] if not math.isfinite(b) else []) + \
              ([f"path bias {b:.3g} > crit/10 = {cr / HR10_BAR:.3g} (gate C1)"] if b > cr / HR10_BAR else []) + \
              ([f"the reference's grid error {g:.3g} > crit/10 (grid unconverged)"] if g > cr / HR10_BAR else [])
        if wall == "clamp" and t.get("D_c") is None:
            why.append("no censored statistic (x_c undefined: the lump covers the law)")
        judged = t.get("D_c") if wall == "clamp" else t["D"]
        rows.append({"id": t["id"], "crit": cr, "judged": judged, "judged_bias": b, "assessable": not why,
                     "D_reported": t["D"], "path_bias_reported": t.get("path_bias")})
        if why:
            unassessable.append(t["id"])
            reasons += [f"{t['id']}: unassessable: {w}" for w in why]
        elif judged > cr:
            failing.append(t["id"])
    bad_checks = [c for c in HR10_CHECKS if c in checks and not checks[c]["holds"]]
    reasons += [f"deterministic check {c!r} not run (gate C2)" for c in HR10_CHECKS if c not in checks]
    by_kind = {}
    for p in plants:
        by_kind.setdefault(p["kind"], []).append(p)
    weak = [p["id"] for p in by_kind.get("plant", []) if not p["statistic"] > crit_x / math.sqrt(p["n_alive"])]
    twin_rejects = [p["id"] for p in by_kind.get("twin", []) if p["statistic"] > crit_x / math.sqrt(p["n_alive"])]
    srk_uninformative = [p["id"] for p in by_kind.get("srk_recorder", []) if p["nan_check"]["holds"]]
    reasons += [f"the {k} was not run (gate C2)" for k in ("plant", "twin", "srk_recorder") if k not in by_kind]
    reasons += [f"{pid}: SRK's zero-for-dead recorder passes the NaN check: the check is not informative (gate C2)"
                for pid in srk_uninformative]
    status = ("fail" if (failing or bad_checks or weak or twin_rejects) else "capped" if cpu_spent >= cap
              else "incomplete" if reasons else "pass")
    return Verdict("HR-10", wall, status, n_valid=n_tests - len(unassessable), n_unassessable=len(unassessable),
                   statistic={"failing": failing, "unassessable": unassessable, "checks_failing": bad_checks,
                              "plant_not_rejecting": weak, "twin_rejecting": twin_rejects, "tests": rows,
                              "judged": "D_c (censored)" if wall == "clamp" else "D (full)", "G": HR10_G},
                   crit={"x": crit_x, "G": HR10_G}, incomplete_reasons=tuple(reasons), cpu_spent=float(cpu_spent),
                   cap=float(cap))


# ----------------------------------------------------------------------------------------------------------- HR-11
HR11_CRIT = 1.655                   # x / sqrt(n): the KS bound at Bonferroni alpha 0.05 / 6 (kstwobign)
HR11_G = 6                          # the 6 multi-parameter cells
HR11_BAR = 10.0                     # a cell is assessable iff its judged bias <= crit / HR11_BAR
HR11_DETECT = 1.86                  # the 90 %-detection shift x sqrt(n) at G = 6 (research R25's c_G)
HR11_INFORMATIVE = 2.0              # a plant is informative iff its predicted sup|dS| >= 2 x the detection shift
HR11_COMPOSITE = 1e-4               # the composite cell's agreement in sup|dS|
HR11_CAP = 4.0                      # CPU-h


def hr11_verdict(reductions, cells, plants, composite, above_budget, *, cpu_spent, cap=HR11_CAP,
                 crit_x=HR11_CRIT) -> Verdict:
    """HR-11 (multi-parameter heterogeneity; research R27-R29; register §3), its (ii) on the default clamp.
    (i) `reductions`: kind "case" rows (`equal`: a tuple with every other sd at 0 against its single-parameter draw,
    bytes, under each wall) and the kind "plant" row (`breaks`: the shared-stream draw breaks the identity).
    (ii) `cells`: `id, n, D` (the dt/2 rung's KS against the tensor reference), `D_bias, p_hat, m, m_max,
    reference_kind, reference_verdict`; crit = crit_x / sqrt(n); assessable iff m <= m_max, D_bias / (2^p - 1) <=
    crit/10 and the tensor reference converged. `plants`: the dropped-spread draws, `id, n, D, predicted_sup_dS`;
    informative iff predicted_sup_dS >= 2 x 1.86 / sqrt(n), and an informative plant must reject. `composite`:
    {"sup_dS"} of the tensor against the composite rule, <= 1e-4. (iii) `above_budget`: the statement of a
    configuration beyond the tensor reference (amendment A7, Naveh 2026-11-04: a 4-parameter cell, past its d <= 3
    limit; the node-budget path rests on a unit test), {"kind", "clean"}: it must read "none" and not be clean. FAIL
    on any violation; `capped` past the cap; incomplete (gate C2) on a missing part, an unassessable cell, no
    informative plant, or a shared-stream plant that does not break the identity."""
    reductions, cells, plants = list(reductions), list(cells), list(plants)
    reasons, failing, rows = [], [], []
    unassessable = [c.get("id") for c in cells if c.get("error")]
    n_cells = len(cells)
    reductions, rwhy = raised(reductions)
    cells, cwhy = raised(cells)
    plants, pwhy = raised(plants)
    reasons += list(rwhy) + list(cwhy) + list(pwhy)
    if composite is not None and composite.get("error"):
        reasons.append(f"(ii) the composite cross-check raised ({composite['error'].get('type')}) (gate C2)")
        composite = None
    if above_budget is not None and above_budget.get("error"):
        reasons.append(f"(iii) the above-budget statement raised ({above_budget['error'].get('type')}) (gate C2)")
        above_budget = None
    cases = [r for r in reductions if r.get("kind") == "case"]
    stream_plants = [r for r in reductions if r.get("kind") == "plant"]
    unequal = [r["id"] for r in cases if not r["equal"]]
    if not cases:
        reasons.append("(i) no reduction case was run (gate C2)")
    if not stream_plants:
        reasons.append("(i) the shared-stream plant was not run (gate C2)")
    reasons += [f"(i) {r['id']}: the shared-stream plant does not break the identity: not informative (gate C2)"
                for r in stream_plants if not r["breaks"]]
    if not n_cells:
        reasons.append("(ii) nothing to assess: no cell (gate C2)")
    for c in cells:
        cr = crit_x / math.sqrt(c["n"])
        b, gwhy = frozen_bias(c, lambda r: float(r["D_bias"]) / (2.0 ** min(max(float(r["p_hat"]), 0.5), 1.0)
                                                                - 1.0))
        why = list(gwhy) + ([f"m {c['m']} > m_max {c['m_max']}"] if c["m"] > c["m_max"] else []) + \
              ([f"bias {b:.3g} > crit/10 = {cr / HR11_BAR:.3g} (gate C1)"] if not b <= cr / HR11_BAR else []) + \
              ([f"reference {c['reference_kind']} ({c['reference_verdict']})"]
               if c["reference_kind"] != "tensor" or c["reference_verdict"] != "converged" else [])
        rows.append({"id": c["id"], "crit": cr, "judged_bias": b, "assessable": not why, "D": c["D"]})
        if why:
            unassessable.append(c["id"])
            reasons += [f"(ii) {c['id']}: unassessable: {w}" for w in why]
        elif c["D"] is None or c["D"] > cr:
            failing.append(c["id"])
    informative = [p for p in plants if p["predicted_sup_dS"] >= HR11_INFORMATIVE * HR11_DETECT / math.sqrt(p["n"])]
    weak = [p["id"] for p in informative if not p["D"] > crit_x / math.sqrt(p["n"])]
    if not informative:
        reasons.append("(ii) no informative dropped-spread plant (predicted sup|dS| >= 2 x the detection shift) "
                       "(gate C2)")
    comp_bad = composite is not None and not composite["sup_dS"] <= HR11_COMPOSITE
    if composite is None:
        reasons.append("(ii) the composite cross-check was not run (gate C2)")
    above_bad = above_budget is not None and (above_budget["kind"] != "none" or above_budget["clean"])
    if above_budget is None:
        reasons.append("(iii) no above-budget statement (gate C2)")
    fail = bool(unequal or failing or weak or comp_bad or above_bad)
    status = "fail" if fail else "capped" if cpu_spent >= cap else "incomplete" if reasons else "pass"
    return Verdict("HR-11", "clamp", status, n_valid=n_cells - len(unassessable),
                   n_unassessable=len(unassessable),
                   statistic={"reductions_unequal": unequal, "failing": failing, "unassessable": unassessable,
                              "cells": rows, "plant_not_rejecting": weak,
                              "plants_informative": [p["id"] for p in informative],
                              "composite": composite, "composite_disagrees": comp_bad,
                              "above_budget": above_budget, "above_budget_wrong": above_bad, "G": HR11_G},
                   crit={"x": crit_x, "G": HR11_G}, incomplete_reasons=tuple(reasons), cpu_spent=float(cpu_spent),
                   cap=float(cap))
