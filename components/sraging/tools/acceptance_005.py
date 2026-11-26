"""Spec 005's acceptance phases (specs/005-data-plots-cli/contracts/acceptance.md; research R10, R11, R14).

    .venv/bin/python tools/acceptance_005.py <phase> [--dev]

Phases: exit (HR-1), km (HR-3), noise (HR-4), workers (HR-6, WEXAC), presets (HR-7), units (HR-8), unchanged (SC-008).

Every phase runs through spec 003's `acceptance_003.Run` (suffix "005"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_005.json with the repo head, the host and its cost. A non-dev run
needs tests/acceptance/rules_005.toml committed with `frozen = true` (Naveh's OK, T047); `--dev` runs a pilot, labelled
"development", which never counts. A phase that reaches its cap (CAPS, the same as rules_005.toml's [caps]) stops,
keeps what finished and reads `capped`, never PASS. A phase that reads the cleaning records (exit, noise) checks each
file against tests/acceptance/cleaning_records_005.toml first: a missing file or a sha256 mismatch makes it
not_applicable, naming the file, never PASS (R10; HR-2).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import resource
import sys
import time
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

import numpy as np  # noqa: E402

import acceptance_003 as A3  # noqa: E402

RECORDS_REL = "tests/acceptance/cleaning_records_005.toml"
RULES_REL = "tests/acceptance/rules_005.toml"
RECORDS = os.path.join(REPO, RECORDS_REL)
RULES = os.path.join(REPO, RULES_REL)

PHASES = ("exit", "km", "noise", "workers", "presets", "units", "unchanged")
#: R14 and tasks T003, in CPU-h; rules_005.toml's [caps] repeats them (tests/unit/test_acceptance_005.py checks it)
CAPS = {"exit": 0.1, "km": 0.1, "noise": 0.5, "workers": 8.0, "presets": 0.1, "units": 0.1, "unchanged": 1.0}
#: the phases that read the cleaning records
READS_RECORDS = ("exit", "noise")
PRICE = {"exit": "seconds, Mac, one core [I]", "km": "seconds, Mac [I]", "noise": "minutes, Mac [I]",
         "workers": "5.1-6.4 CPU-h, WEXAC, one node [M P1]", "presets": "seconds, Mac [I]", "units": "minutes, Mac [I]",
         "unchanged": "about 0.2-0.5 CPU-h, Mac (004's took 747 s wall)"}


def read_rules() -> dict:
    return tomllib.load(open(RULES, "rb"))


# ------------------------------------------------------------------------------------------ the records (R10)
def record_files(path=None) -> list:
    """[(record, absolute path, pinned sha256)] for every file the manifest pins: cleaning.csv, its stamp, the fly's
    floors and nulls ([<record>.sha256]) and each per-cage cohort file ([<record>.cohorts_allcages])."""
    recs = tomllib.load(open(path or RECORDS, "rb"))
    out = []
    for name, r in recs.items():
        out += [(name, os.path.join(r["dir"], f), h) for f, h in r["sha256"].items()]
        out += [(name, os.path.join(r["dir"], "cohorts_allcages", f), h) for f, h in r["cohorts_allcages"].items()]
    return out


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def records_problems(path=None) -> list[str]:
    """Each pinned file that is missing or whose sha256 differs from its pin, named; empty when every file is there
    and pinned right (R10)."""
    out = []
    for name, p, pin in record_files(path):
        if not os.path.isfile(p):
            out.append(f"{name}: missing {p}")
        elif file_sha256(p) != pin:
            out.append(f"{name}: sha256 mismatch {p} (pinned {pin[:12]}..., found {file_sha256(p)[:12]}...)")
    return out


def not_applicable(problems) -> dict:
    """The payload of a phase whose inputs are absent or changed: never PASS (HR-2)."""
    return {"status": "not_applicable", "reasons": list(problems)}


# ------------------------------------------------------------------------------------------------ caps (R14)
def _children_cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


class Budget:
    """One phase's CPU against its cap: this process's CPU since the start plus, with `children`, its finished child
    processes' (a subprocess's CPU is not in the parent's process time), plus `remote_cpu_s`, a remote job's CPU (the
    workers phase's LSF job; P1 found it left out, 2026-11-08)."""

    def __init__(self, phase, cap_cpu_h, *, clock=time.process_time, children=True):
        self.phase, self.cap_cpu_h, self.clock, self.children = phase, float(cap_cpu_h), clock, children
        self.c0, self.k0, self.capped = clock(), _children_cpu_s() if children else 0.0, False
        self.remote_cpu_s = 0.0

    @property
    def spent_cpu_h(self) -> float:
        kids = _children_cpu_s() - self.k0 if self.children else 0.0
        return (self.clock() - self.c0 + kids + self.remote_cpu_s) / 3600.0

    def reached(self) -> bool:
        return self.spent_cpu_h >= self.cap_cpu_h

    def record(self) -> dict:
        return {"cap_cpu_h": self.cap_cpu_h, "cpu_spent_h": self.spent_cpu_h, "capped": self.capped}


def run_capped(budget: Budget, cells, work) -> list:
    """`work(cell) -> row` for each cell in order; once the cap is reached no new cell starts, the finished rows are
    kept and the budget reads capped."""
    rows = []
    for cell in cells:
        if budget.reached():
            budget.capped = True
            break
        rows.append(work(cell))
    return rows


def phase_status(status, budget: Budget | None) -> str:
    """A capped run's status is `capped`, whatever its finished cells read (never PASS)."""
    return "capped" if budget is not None and budget.capped else status


def counts_as_pass(status) -> bool:
    """Constitution V: only pass is PASS; capped, not_applicable and the rest never are."""
    return str(status).lower() == "pass"


# --------------------------------------------------------------------------------------------- the envelope
def manifests(phase) -> list:
    return [RULES_REL] + ([RECORDS_REL] if phase in READS_RECORDS else [])


def start(phase, *, dev) -> "A3.Run":
    """Spec 003's Run (suffix 005): refuses a dirty tree or an uncommitted manifest unless dev; a non-dev run also
    needs rules_005.toml frozen (T047)."""
    if not dev and not read_rules().get("frozen"):
        sys.exit(f"[{phase}] REFUSED: {RULES_REL} is not frozen (frozen = false): no non-dev run before the freeze "
                 f"with Naveh's OK (tasks T047); pass --dev for a pilot that never counts")
    return A3.Run(phase, manifests=manifests(phase), dev=dev, cost=PRICE[phase], suffix="005")


def finish(run, payload: dict, budget: Budget | None = None):
    out = dict(payload)
    out["manifests_sha256"] = {m: file_sha256(os.path.join(REPO, m)) for m in run.manifests
                               if os.path.exists(os.path.join(REPO, m))}
    if budget is not None:
        out["cap"] = budget.record()
        out["status"] = phase_status(out.get("status"), budget)
    return run.write(out)


def require_records(run, budget=None):
    """not_applicable (exit code 3) when a record file is missing or changed; else nothing."""
    problems = records_problems(RECORDS)
    if problems:
        finish(run, not_applicable(problems), budget)
        sys.exit(3)


# ------------------------------------------------------------------------------------------- exit (HR-1, T023)
EXIT_SEED = 20261108
EXIT_NOISE_REPS = 200            # the noise baseline is not compared by HR-1; a small R keeps the phase in seconds
K_SWAP = {2.5: 3.0, 3.0: 2.5}
PLANTED = ("no leave-one-out", "no floors", "k swapped", "first-crossing quartiles")


def exit_key(cohort, cage) -> tuple:
    """(cohort, cage) as strings, a whole float written as an int, so that the record's parsed cage_id and the reader's
    typed one compare equal (3, 3.0 and "3" alike)."""
    def norm(v):
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v.item() if isinstance(v, np.generic) else v)
    return norm(cohort), norm(cage)


def _same(a, b) -> bool:
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


def compare_record(ours: dict, record: list, *, kept_col, distance_col, tol) -> dict:
    """HR-1 on one record at one k: per (cohort, cage) of the record, the verdict identical, |Δ distance| <=
    tol["distance_abs"], the median exact, |Δ steepness| <= tol["steepness_abs"], n equal; a record cage we lack, or
    a cage of the record's cohorts the record lacks, is a mismatch. `ours`: exit_key -> {verdict, distance, median,
    steepness, n}; `record`: rows parsed round-trip."""
    mism, seen, dmax, smax = [], set(), 0.0, 0.0
    cohorts = {exit_key(r["cohort"], "")[0] for r in record}
    for r in record:
        key = exit_key(r["cohort"], r["cage_id"])
        seen.add(key)
        o = ours.get(key)
        if o is None:
            mism.append({"key": list(key), "what": "missing"})
            continue
        want = "kept" if bool(r[kept_col]) else "flagged"
        if o["verdict"] != want:
            mism.append({"key": list(key), "what": "verdict", "ours": o["verdict"], "record": want})
        dd = abs(float(o["distance"]) - float(r[distance_col]))
        dmax = max(dmax, dd) if math.isfinite(dd) else math.inf
        if not dd <= tol["distance_abs"] and not _same(float(o["distance"]), float(r[distance_col])):
            mism.append({"key": list(key), "what": "distance", "ours": o["distance"], "record": r[distance_col]})
        if not _same(float(o["median"]), float(r["median"])):
            mism.append({"key": list(key), "what": "median", "ours": o["median"], "record": r["median"]})
        ds = abs(float(o["steepness"]) - float(r["steepness_iqr"]))
        smax = max(smax, ds) if math.isfinite(ds) else math.inf
        if not ds <= tol["steepness_abs"] and not _same(float(o["steepness"]), float(r["steepness_iqr"])):
            mism.append({"key": list(key), "what": "steepness", "ours": o["steepness"], "record": r["steepness_iqr"]})
        if int(o["n"]) != int(r["n"]):
            mism.append({"key": list(key), "what": "n", "ours": int(o["n"]), "record": int(r["n"])})
    for key in ours:
        if key[0] in cohorts and key not in seen:
            mism.append({"key": list(key), "what": "extra"})
    return {"n": len(record), "mismatches": mism, "flips": sum(m["what"] == "verdict" for m in mism),
            "max_abs_distance": dmax, "max_abs_steepness": smax,
            "n_flagged_ours": sum(o["verdict"] == "flagged" for k, o in ours.items() if k in seen),
            "n_flagged_record": sum(not bool(r[kept_col]) for r in record)}


def compare_floors(ours, record, tol_rel) -> dict:
    rel = [abs(a - b) / abs(b) for a, b in zip(ours, record)]
    return {"ours": list(ours), "record": list(record), "rel": rel, "ok": all(x <= tol_rel for x in rel)}


def read_record_csv(path) -> list:
    """The record's rows, its floats parsed round-trip (R10: pandas' default parser leaves a 4.4e-16 residual)."""
    import pandas as pd
    return pd.read_csv(path, float_precision="round_trip").to_dict("records")


def _find_floors(obj):
    """(φ_ml, φ_st) from a stamp or floors JSON: the first mapping holding rel_ml with rel_steepness / rel_steep, or
    floor_rel_ml with floor_rel_steep, searched depth first."""
    pairs = (("rel_ml", "rel_steepness"), ("rel_ml", "rel_steep"), ("floor_rel_ml", "floor_rel_steep"),
             ("floor_ml", "floor_steep"), ("floor_ml", "floor_steepness"))
    if isinstance(obj, dict):
        for a, b in pairs:
            if isinstance(obj.get(a), (int, float)) and isinstance(obj.get(b), (int, float)):
                return float(obj[a]), float(obj[b])
        for v in obj.values():
            got = _find_floors(v)
            if got:
                return got
    if isinstance(obj, list):
        for v in obj:
            got = _find_floors(v)
            if got:
                return got
    return None


def record_floors(name, rec) -> tuple:
    """(φ_ml, φ_st, source file) the record states: the fly's cleaning_floors.json, the mosquito's stamp."""
    import json
    for f in ("cleaning_floors.json", "cleaning.csv.stamp.json"):
        p = os.path.join(rec["dir"], f)
        if os.path.isfile(p):
            got = _find_floors(json.load(open(p)))
            if got:
                return got[0], got[1], f
    raise RuntimeError(f"{name}: no floors found in cleaning_floors.json or cleaning.csv.stamp.json")


def read_record_cages(rec, cohorts):
    """(CohortSet of the record's cohorts' cages, refused rows) from cohorts_allcages/<cohort>.csv, through
    read_cohorts (one cohort per file, cages by cage_id)."""
    from sraging.cohorts import CohortSet, read_cohorts
    members, refused = [], []
    for co in cohorts:
        s, bad = read_cohorts(os.path.join(rec["dir"], "cohorts_allcages", f"{co}.csv"), delta=1.0,
                              time_unit="days", cage_col="cage_id")
        members += [(k, s[k], dict(s.design[k])) for k in s]
        refused += [r._asdict() for r in bad]
    return CohortSet.from_members(members, time_unit="days"), refused


def _ours(report) -> dict:
    return {exit_key(r.cohort, r.cage): {"verdict": r.verdict, "distance": r.distance, "median": r.median,
                                         "steepness": r.steepness, "n": r.n} for r in report.cages}


def _first_crossing_by_cohort(cages, k, tol_cols, record, tol):
    """The first-crossing defect cohort by cohort with floors pooled from first-crossing scores: a cohort with a
    non-finite steepness is refused (D-503) and counted; the others are compared."""
    from sraging import cleaning as CL
    from sraging.errors import CohortDataError
    scores = CL.cage_scores(cages, quartiles="first-crossing")
    groups = cages.by("cohort")
    phi = CL.pooled_floors({co: ([scores[x][0] for x in sub], [scores[x][1] for x in sub])
                            for co, sub in groups.items()})
    ours, refused = {}, []
    for co, sub in groups.items():
        try:
            rep = CL.clean(sub, by="cohort", k=k, seed=EXIT_SEED, noise_reps=EXIT_NOISE_REPS,
                           quartiles="first-crossing", floors=phi[:2])
        except CohortDataError as ex:
            refused.append({"cohort": co, "reason": str(ex).splitlines()[0]})
            continue
        ours.update(_ours(rep))
    kept_rows = [r for r in record if exit_key(r["cohort"], "")[0] not in {str(x["cohort"]) for x in refused}]
    cmp = compare_record(ours, kept_rows, kept_col=tol_cols[0], distance_col=tol_cols[1], tol=tol)
    return {**cmp, "refused_cohorts": refused, "floors": list(phi[:2])}


def exit_record(name, rec, tol) -> dict:
    """HR-1 on one record: the four planted defects first (each must fail), then every recorded k."""
    from sraging.cleaning import clean
    record = read_record_csv(os.path.join(rec["dir"], "cleaning.csv"))
    cohorts = sorted({str(r["cohort"]) for r in record})
    cages, refused_rows = read_record_cages(rec, cohorts)
    ks = [float(k) for k in rec["k"]]
    cols = {2.5: (rec["kept_column"], "distance"), 3.0: (rec.get("kept_k3_column", "kept_k3"), "distance_k3")}
    if len(ks) == 1:
        cols = {ks[0]: (rec["kept_column"], "distance")}
    phi_rec = record_floors(name, rec)

    def run(k, **kw):
        return clean(cages, by="cohort", k=k, seed=EXIT_SEED, noise_reps=EXIT_NOISE_REPS, **kw)

    planted = []
    k0 = ks[0]
    for defect in PLANTED:
        row = {"defect": defect}
        if defect == "first-crossing quartiles":
            cmp = _first_crossing_by_cohort(cages, k0, cols[k0], record, tol)
            row.update(k=k0, **cmp)
        else:
            kw, k_run = {}, k0
            if defect == "no leave-one-out":
                kw["leave_one_out"] = False
            elif defect == "no floors":
                kw["floors"] = (0.0, 0.0)
            elif defect == "k swapped":
                k_run = K_SWAP[k0]
            cmp = compare_record(_ours(run(k_run, **kw)), record, kept_col=cols[k0][0], distance_col=cols[k0][1],
                                 tol=tol)
            row.update(k=k_run, compared_with=cols[k0][0], **cmp)
        row["fails"] = bool(row["mismatches"] or row.get("refused_cohorts"))
        row["mismatches"] = row["mismatches"][:20]
        planted.append(row)
    per_k = []
    floors_cmp = None
    for k in ks:
        rep = run(k)
        cmp = compare_record(_ours(rep), record, kept_col=cols[k][0], distance_col=cols[k][1], tol=tol)
        floors_cmp = compare_floors(rep.floors, phi_rec[:2], tol["floors_rel"])
        per_k.append({"k": k, "column": cols[k][0], "floors": list(rep.floors),
                      "floor_cohorts": list(rep.floor_cohorts),
                      "report_sha256": rep.sha256, **cmp, "mismatches_total": len(cmp["mismatches"]),
                      "mismatches": cmp["mismatches"][:50]})
    ok = (all(r["fails"] for r in planted) and all(r["mismatches_total"] == 0 for r in per_k)
          and floors_cmp["ok"] and not refused_rows)
    return {"record": name, "cohorts": len(cohorts), "cages": len(cages), "refused_rows": refused_rows[:20],
            "n_refused_rows": len(refused_rows), "floors_record": {"phi": list(phi_rec[:2]), "source": phi_rec[2]},
            "floors_compare": floors_cmp, "planted": planted, "per_k": per_k, "ok": ok}


# ------------------------------------------------------------------------------------------------ the phases
def phase_exit(args):
    """HR-1 (T023): for each record, the planted defects (each must fail), then the cleaning at every recorded k
    against cleaning.csv (verdicts identical, distances, floors, medians, steepness within HR-1's tolerances)."""
    run = start("exit", dev=args.dev)
    budget = Budget("exit", CAPS["exit"])
    require_records(run, budget)
    h1 = read_rules()["rules"]["HR-1"]
    tol = {"distance_abs": float(h1["distance_abs"]), "steepness_abs": float(h1["steepness_abs"]),
           "floors_rel": float(h1["floors_rel"])}
    recs = tomllib.load(open(RECORDS, "rb"))
    out = run_capped(budget, list(recs), lambda name: exit_record(name, recs[name], tol))
    status = "pass" if len(out) == len(recs) and all(r["ok"] for r in out) else "fail"
    for r in out:
        print(f"[exit] {r['record']}: {r['cages']} cages, {r['cohorts']} cohorts; planted defects fail: "
              f"{[(p['defect'], p['fails'], p['n_flagged_ours'], p['flips']) for p in r['planted']]}; "
              f"per k (k, flagged ours, flagged record, mismatches, max |dd|): "
              f"{[(x['k'], x['n_flagged_ours'], x['n_flagged_record'], x['mismatches_total'], x['max_abs_distance'])
                  for x in r['per_k']]}; "
              f"floors rel {r['floors_compare']['rel']}; refused rows {r['n_refused_rows']}", flush=True)
    finish(run, {"status": status, "tolerances": tol, "seed": EXIT_SEED, "noise_reps": EXIT_NOISE_REPS,
                 "definitions": {"median": "first-crossing", "quartiles": "nearest"}, "records": out}, budget)


# ------------------------------------------------------------------------------------------ units (HR-8, T026)
UNITS_CHAIN = ("days", "weeks", "years")
#: one delta cohort (synthetic, deterministic: Weibull quantiles on a 1-day census, every 7th censored) and a theta in
#: days near spec 004's mice_F preset (tools/presets_004.py, weeks -> days); the invariance holds at any theta
UNITS_N, UNITS_SCALE, UNITS_SHAPE = 200, 800.0, 4.0
UNITS_THETA_DAYS = {"eta": 0.0011978543153447314 / 49, "beta": 0.14843519768275326 / 7,
                    "epsilon": 0.011737266323603139 / 7, "kappa": 0.016835016835016835, "xc": 1.0, "h_ext": 0.0}


def units_cohort():
    from sraging.cohort import Cohort
    q = (np.arange(UNITS_N) + 0.5) / UNITS_N
    ages = np.maximum(np.round(UNITS_SCALE * (-np.log1p(-q)) ** (1 / UNITS_SHAPE)), 1.0)
    events = np.where(np.arange(UNITS_N) % 7 == 6, 0, 1)
    return Cohort.from_arrays(ages, events, delta=1.0, name="units_005", time_unit="days")


def units_round_trips() -> dict:
    """The largest relative round-trip error over every ordered pair of units, for the parameters and the cohort."""
    from sraging.params import SRParams
    from sraging.units import UNITS, convert
    p, c = SRParams(**UNITS_THETA_DAYS, time_unit="days"), units_cohort()
    worst_p = worst_c = 0.0
    for a in UNITS:
        for b in UNITS:
            if a == b:
                continue
            pa, ca = convert(p, to=a), convert(c, to=a)
            pb, cb = convert(convert(pa, to=b), to=a), convert(convert(ca, to=b), to=a)
            for k in ("eta", "beta", "epsilon", "kappa", "xc"):
                worst_p = max(worst_p, abs(getattr(pb, k) - getattr(pa, k)) / abs(getattr(pa, k)))
            worst_c = max(worst_c, float(np.max(np.abs(cb.ages - ca.ages) / ca.ages)),
                          abs(cb.delta - ca.delta) / ca.delta)
    return {"params_rel": worst_p, "cohort_rel": worst_c}


def proposed_tolerance(measured) -> float:
    """HR-8's log-likelihood tolerance proposed from pilot P3 (tasks T045): about 10x the measured |dloglik|, with a
    floor of 1e-12."""
    return max(10.0 * float(measured), 1e-12)


def presets_hr7(*, types=None, ext=(False, True), units=None, planted=None) -> dict:
    """HR-7 and HR-8's factor table: every preset x type x ExtH of sraging.presets.get against SRK's getTheta from the
    frozen snapshot (tools/oracle_presets_005.py), at each unit: in days eta, beta, epsilon, xc and kappa exact; in
    hours, weeks, years and generations, units.convert(get(...), to=u) within 1e-15 relative for eta, beta, epsilon
    and exact for xc and kappa; the raw ExtH exact (getTheta returns it unconverted at every unit, D-502). SRK's
    rescaleTheta (SRK/utils.py) is in no frozen snapshot; getTheta applies the same unit table.
    With ExtH, one h_ext cell per preset x type (sr-naveh_005 B1): convert(get(n, type=t, ExtH=True),
    to=info.ext_h_unit).h_ext against exp(-ExtH) of getTheta(n, type=t, ExtH=True, time_unit="auto") within HR-7's
    convert_rel (the table's ExtH is in the auto unit); ext_h_unit is SRK's auto unit when getTheta at "auto" equals
    getTheta at ext_h_unit; a NaN ExtH gives h_ext = 0 exactly. `planted`: "eta_s" (eta scaled by s, not s^2),
    "ratio" (eta = (xc/eta) / xc) or "h_ext_per_day" (exp(-ExtH) read per day, the reading B1 found) must fail."""
    import oracle_presets_005 as OP
    from sraging import presets as PR
    from sraging.params import SRParams
    from sraging.units import convert
    rel_tol = float(read_rules()["rules"]["HR-7"]["convert_rel"])
    types = tuple(types or OP.TYPES)
    units = tuple(units or OP.UNITS)
    cells = [(n, t, e, u) for t in types for n in PR.names(type=t) for e in ext for u in units]
    hcells = ([(n, t, PR.get(n, type=t, ExtH=True)[1].ext_h_unit) for t in types for n in PR.names(type=t)]
              if True in ext else [])
    hreq = ([{"name": n, "type": t, "ExtH": True, "time_unit": "auto"} for n, t, _ in hcells]
            + [{"name": n, "type": t, "ExtH": True, "time_unit": u} for n, t, u in hcells if u is not None])
    ref = OP.get_theta([{"name": n, "type": t, "ExtH": e, "time_unit": u} for n, t, e, u in cells] + hreq)
    ref, href_auto, href_unit = ref[:len(cells)], ref[len(cells):len(cells) + len(hcells)], \
        iter(ref[len(cells) + len(hcells):])
    mism, worst = [], 0.0
    for (n, t, e, u), r in zip(cells, ref):
        if r["theta"] is None:
            mism.append({"cell": [n, t, e, u], "what": "getTheta raised", "error": r["error"]})
            continue
        p, info = PR.get(n, type=t, ExtH=e)
        if planted == "ratio":
            tab = PR._table(t)
            xc = tab["xc"][n]
            p = SRParams(**{**p.as_dict(), "eta": (tab["xc/eta"][n] / xc) / xc})
        c = convert(p, to=u)
        if planted == "eta_s":
            s = _unit_factor(u)
            c = SRParams(**{**c.as_dict(), "eta": p.eta * s})
        th = r["theta"]
        for k, want in zip(("eta", "beta", "epsilon"), th[:3]):
            got = getattr(c, k)
            rel = abs(got - want) / abs(want)
            worst = max(worst, rel)
            if (got != want) if u == "days" else not rel <= 1e-15:
                mism.append({"cell": [n, t, e, u], "what": k, "ours": got, "srk": want, "rel": rel})
        for k, want in zip(("xc", "kappa"), th[3:5]):
            if getattr(c, k) != want:
                mism.append({"cell": [n, t, e, u], "what": k, "ours": getattr(c, k), "srk": want})
        if e:
            want = th[5]
            if not ((info.ext_h == want) if info.ext_h is not None else math.isnan(want)):
                mism.append({"cell": [n, t, e, u], "what": "ExtH", "ours": info.ext_h, "srk": want})
    worst_h, n_finite = 0.0, 0
    for (n, t, unit), ra in zip(hcells, href_auto):
        ru = next(href_unit) if unit is not None else None
        cell = [n, t, True, "auto"]
        if ra["theta"] is None or (ru is not None and ru["theta"] is None):
            mism.append({"cell": cell, "what": "getTheta raised", "error": ra["error"] or ru["error"]})
            continue
        p, info = PR.get(n, type=t, ExtH=True)
        want = ra["theta"][5]
        if info.ext_h is None:                          # no ExtH in the table: no external hazard at any unit
            if not (math.isnan(want) and p.h_ext == 0.0):
                mism.append({"cell": cell, "what": "h_ext", "ours": p.h_ext, "srk_ExtH": want})
            continue
        n_finite += 1
        if ru["theta"] != ra["theta"]:                  # SRK's auto unit is not ours
            mism.append({"cell": cell, "what": "auto unit", "ours": unit})
        if planted == "h_ext_per_day":
            p = SRParams(**{**p.as_dict(), "h_ext": math.exp(-info.ext_h)})
        got, w = convert(p, to=unit).h_ext, math.exp(-want)
        rel = abs(got - w) / w
        worst_h = max(worst_h, rel)
        if not rel <= rel_tol:
            mism.append({"cell": cell, "what": "h_ext", "unit": unit, "ours": got, "srk": w, "rel": rel})
    return {"cells": len(cells), "mismatches": mism, "max_rel": worst, "types": list(types), "units": list(units),
            "ext": list(ext), "h_ext_cells": len(hcells), "h_ext_finite": n_finite, "h_ext_max_rel": worst_h}


def _unit_factor(u):
    from sraging.units import UNITS
    return UNITS[u] / UNITS["days"]


def phase_presets(args):
    """HR-7 (T042): every preset x type x ExtH against getTheta from the frozen snapshot, in days and converted to each
    listed unit, and each preset x type's h_ext in its auto unit; the planted wrong unit factor, wrong ratio rebuild
    and per-day ExtH must fail first."""
    run = start("presets", dev=args.dev)
    budget = Budget("presets", CAPS["presets"])
    planted = []
    for defect, kw in (("eta_s", {"ext": (False,), "units": ("days", "weeks")}),
                       ("ratio", {"ext": (False,), "units": ("days", "weeks")}),
                       ("h_ext_per_day", {"ext": (True,), "units": ("days",)})):
        r = presets_hr7(types=("mode_overall",), planted=defect, **kw)
        planted.append({"defect": defect, "mismatches": len(r["mismatches"]), "fails": bool(r["mismatches"])})
    out = run_capped(budget, ["all"], lambda _: presets_hr7())
    res = out[0] if out else {"cells": 0, "mismatches": ["capped"], "max_rel": math.nan, "h_ext_cells": 0,
                              "h_ext_finite": 0, "h_ext_max_rel": math.nan}
    ok = bool(out) and not res["mismatches"] and all(p["fails"] for p in planted)
    print(f"[presets] {res['cells']} cells and {res['h_ext_cells']} h_ext cells ({res['h_ext_finite']} with an ExtH), "
          f"{len(res['mismatches'])} mismatches, max rel {res['max_rel']:.3g}, h_ext max rel "
          f"{res['h_ext_max_rel']:.3g}; planted {planted}", flush=True)
    finish(run, {"status": "pass" if ok else "fail", "planted": planted,
                 **{k: v for k, v in res.items() if k != "mismatches"}, "mismatches": res["mismatches"][:50],
                 "n_mismatches": len(res["mismatches"])}, budget)


def phase_units(args):
    """HR-8 (T026): one delta cohort's engine log-likelihood (numerics v1, its default grid) under joint conversion of
    the cohort and theta, days -> weeks -> years; round trips over every pair of units; and the factor table, as
    convert(get(name), to=u) against getTheta(name, time_unit=u) from the frozen snapshot for every preset (type
    mode_overall) and listed unit (SRK's rescaleTheta, in SRK/utils.py, is in no snapshot; getTheta applies the same
    table)."""
    from sraging.engine.evaluator import Evaluator
    from sraging.params import SRParams
    from sraging.units import convert
    run = start("units", dev=args.dev)
    budget = Budget("units", CAPS["units"])
    p, c = SRParams(**UNITS_THETA_DAYS, time_unit="days"), units_cohort()

    def loglik(unit):
        pu, cu = convert(p, to=unit), convert(c, to=unit)
        return {"unit": unit, "loglik": Evaluator(cu, numerics="v1").value(pu)}
    rows = run_capped(budget, UNITS_CHAIN, loglik)
    ref = rows[0]["loglik"] if rows else math.nan
    for r in rows:
        r["dloglik"] = r["loglik"] - ref
    worst = max((abs(r["dloglik"]) for r in rows), default=math.nan)
    trips = units_round_trips()
    factors = presets_hr7(types=("mode_overall",), ext=(False,))
    planted = presets_hr7(types=("mode_overall",), ext=(False,), units=("weeks",), planted="eta_s")
    tol_rule = read_rules()["rules"]["HR-8"]
    tol = tol_rule.get("loglik_tolerance")
    trips_ok = trips["params_rel"] <= float(tol_rule["round_trip_rel"]) and \
        trips["cohort_rel"] <= float(tol_rule["round_trip_rel"])
    factors_ok = not factors["mismatches"] and bool(planted["mismatches"])
    if isinstance(tol, (int, float)):
        status = "pass" if len(rows) == len(UNITS_CHAIN) and worst <= float(tol) and trips_ok and factors_ok \
            else "fail"
    else:
        status = "written"                       # pilot P3: no frozen tolerance yet
    print(f"[units] loglik {[(r['unit'], r['loglik'], r['dloglik']) for r in rows]}; worst |dloglik| {worst:.3g}; "
          f"round trips {trips}; factors {factors['cells']} cells, {len(factors['mismatches'])} mismatches (planted "
          f"eta*s: {len(planted['mismatches'])}); proposed tolerance {proposed_tolerance(worst):.3g}", flush=True)
    finish(run, {"status": status, "cohort": {"n": c.n, "n_deaths": c.n_deaths, "delta": c.delta, "unit": "days"},
                 "theta_days": UNITS_THETA_DAYS, "rows": rows, "worst_abs_dloglik": worst,
                 "loglik_tolerance": tol, "proposed_tolerance": proposed_tolerance(worst), "round_trips": trips,
                 "factors": {**{k: v for k, v in factors.items() if k != "mismatches"},
                             "mismatches": factors["mismatches"][:50], "planted_eta_s": len(planted["mismatches"]),
                             "source": "getTheta(name, time_unit=u) from tests/oracle/fp_piecewise_cb191f5 "
                                       "(tools/oracle_presets_005.py); SRK/utils.py's rescaleTheta is in no snapshot"}},
           budget)


# --------------------------------------------------------------------------------- workers (HR-6, T037; WEXAC)
#: WEXAC (research R8; plan "Target Platform"): ~/.ssh/config's alias, the pinned venv, the run's directories (relative
#: to the remote home). No GitHub there: the committed head goes over by rsync of `git archive`.
WEXAC = {"host": "wexac", "python": "~/sraging_cov003/venv/bin/python", "root": "sraging_005",
         "repo": "sraging_005/repo", "out": "sraging_005/out"}
WORKERS_JOB = "sraging005_workers"
WORKERS_WALL = "8:00"           # P1 measured ~2,540 CPU-s (~32 min) per fit: the full job, 9 fits, ~4.2 h
WORKERS_POLL_S = 60


def _remote(cmd, *, capture=True):
    """Run one ssh / rsync command (the only way the phase reaches WEXAC; the dry run never calls it)."""
    import subprocess
    r = subprocess.run(cmd, capture_output=capture, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited {r.returncode}: {(r.stderr or '')[-500:]}")
    return r.stdout


def workers_commands(head, export_dir, job_out=None, *, pilot=False) -> dict:
    """The exact commands of the WEXAC run, by step (the dry run prints them); `pilot` (a dev run) runs P1's job."""
    import shlex
    host, py, repo, out = WEXAC["host"], WEXAC["python"], WEXAC["repo"], WEXAC["out"]
    job_out = job_out or f"~/{out}/workers_job.json"
    job = (f"cd ~/{repo} && PYTHONPATH=$HOME/{repo}/src:$HOME/{repo}/tools {py} tools/acceptance_005.py "
           f"workers-job --out {job_out}" + (" --pilot" if pilot else ""))
    return {
        "export": [["git", "-C", REPO, "archive", "--format=tar", "-o", os.path.join(export_dir, "head.tar"), head],
                   ["tar", "-x", "-C", export_dir, "-f", os.path.join(export_dir, "head.tar")]],
        "mkdir": [["ssh", host, f"mkdir -p ~/{repo} ~/{out}"]],
        "rsync": [["rsync", "-az", "--delete", "--exclude", "head.tar", f"{export_dir}/", f"{host}:{repo}/"]],
        "python": [["ssh", host, f"{py} -c {shlex.quote('import platform; print(platform.python_version())')}"]],
        # the venv has no pip (built with uv; P1, 2026-11-08): the installed distributions through importlib.metadata
        "freeze": [["ssh", host, f"{py} -c {shlex.quote(FREEZE_PY)}"]],
        "submit": [["ssh", host, f"bsub -J {WORKERS_JOB} -n 3 -R {shlex.quote('span[hosts=1]')} -W {WORKERS_WALL} "
                                 f"-o ~/{out}/%J.out -e ~/{out}/%J.err {shlex.quote(job)}"]],
        "poll": [["ssh", host, f"bjobs -noheader -o stat <JOBID>"]],
        "fetch": [["rsync", "-az", f"{host}:{out}/workers_job.json", os.path.join(export_dir, "workers_job.json")]],
    }


#: `pip freeze`-style lines (name==version) from importlib.metadata, sorted, for an interpreter without pip
FREEZE_PY = ("import importlib.metadata as m; "
             "print(chr(10).join(sorted(d.metadata['Name'] + '==' + d.version for d in m.distributions())))")


def _freeze_map(text) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "-e")) or " @ " in line or "==" not in line:
            continue
        name, ver = line.split("==", 1)
        name = name.strip().lower().replace("_", "-")
        if name != "sraging":
            out[name] = ver.strip()
    return out


def runtime_pins() -> dict:
    """What the WEXAC venv must match exactly (the coordinator, 2026-11-08; R8): uv.lock's requires-python and this
    interpreter's exact version, and sraging's numerical runtime stack, its [project] dependencies (numpy, scipy, jax,
    jaxlib, ml_dtypes, opt_einsum) at the versions uv.lock resolves."""
    import platform
    lock = tomllib.load(open(os.path.join(REPO, "uv.lock"), "rb"))
    proj = tomllib.load(open(os.path.join(REPO, "pyproject.toml"), "rb"))["project"]["dependencies"]
    names = [d.split("==")[0].split(">")[0].split("<")[0].strip().lower().replace("_", "-") for d in proj]
    resolved = {p["name"]: p.get("version") for p in lock["package"]}
    return {"requires_python": lock["requires-python"], "python": platform.python_version(),
            "packages": {n: resolved[n] for n in names}}


def runtime_problems(remote_freeze, remote_python, pins) -> list[str]:
    """Each difference from `pins`, named: the Python version (uv.lock's requires-python and the exact version) and
    every runtime package (a version differs, or it is missing). The rest of the freeze is recorded, not compared:
    HR-6 compares the node against itself, so a host difference cannot make it pass falsely."""
    got = _freeze_map(remote_freeze)
    out = []
    major_minor = pins["requires_python"].lstrip("=").rstrip(".*")
    if not remote_python.startswith(major_minor + ".") or remote_python != pins["python"]:
        out.append(f"python: {remote_python} on WEXAC, pinned {pins['python']} (uv.lock {pins['requires_python']})")
    for n, want in sorted(pins["packages"].items()):
        if n not in got:
            out.append(f"{n}: missing on WEXAC (uv.lock {want})")
        elif got[n] != want:
            out.append(f"{n}: {got[n]} on WEXAC, uv.lock {want}")
    return out


def workers_cohorts() -> dict:
    """Three distinct small synthetic delta cohorts (an odd count; HR-6), deterministic: Weibull quantiles on a 1-day
    census at three scales and shapes, every 7th censored."""
    from sraging.cohort import Cohort
    out = {}
    for name, (n, scale, shape) in {"w005_a": (120, 60.0, 3.0), "w005_b": (150, 90.0, 4.0),
                                    "w005_c": (180, 40.0, 2.5)}.items():
        q = (np.arange(n) + 0.5) / n
        ages = np.maximum(np.round(scale * (-np.log1p(-q)) ** (1 / shape)), 1.0)
        out[name] = Cohort.from_arrays(ages, np.where(np.arange(n) % 7 == 6, 0, 1), delta=1.0, name=name,
                                       time_unit="days")
    return out


def _identity(d) -> dict:
    """What HR-6 compares of one result JSON: theta-hat's bytes (float.hex), the loglik's, and the claims."""
    th = d.get("theta_hat") or {}
    return {"theta_hex": {k: float(v).hex() for k, v in sorted(th.items()) if isinstance(v, (int, float))
                          and not isinstance(v, bool)},
            "loglik_hex": float(d["loglik"]).hex() if d.get("loglik") is not None else None,
            "claims": d.get("claims")}


def compare_workers(job) -> list[str]:
    """HR-6: every cohort's identity equal in-process, at W = 1 and at W = 2 (P1's job: W = 1 only); each difference
    named (a result attached to the wrong cohort differs, the cohorts being distinct)."""
    out = []
    ref = job["inproc"]
    for run in job.get("runs", ("w1", "w2")):
        for name in ref:
            got = job[run].get(name)
            if got is None:
                out.append(f"{name}: missing from {run}")
            elif got != ref[name]:
                diff = sorted(k for k in ref[name] if got.get(k) != ref[name][k])
                out.append(f"{name}: {run} differs from the in-process fit in {diff}")
    return out


def workers_job(out_path, *, pilot=False):
    """On the WEXAC node (one LSF job, -n 3, span[hosts=1]): the three cohorts fitted in-process with the default
    calibrator, then by fit_set at W = 1 and W = 2; writes the identities, each worker error, the comparison and the
    cost. `pilot` is P1 (rules_005.toml HR-6): one cohort, in-process and spawned at W = 1."""
    import json
    import socket
    import tempfile
    import sraging
    from sraging.cohorts import CohortSet
    from sraging.fitset import fit_set
    t0, c0 = time.time(), time.process_time()
    cs = workers_cohorts()
    if pilot:
        cs = {"w005_a": cs["w005_a"]}
    ws = (1,) if pilot else (1, 2)
    rec = {"pilot": pilot, "runs": [f"w{w}" for w in ws],
           "inproc": {n: _identity(json.loads(sraging.fit_cohort(c).to_json())) for n, c in cs.items()}}
    s = CohortSet.from_members([(n, c, {"cohort": n}) for n, c in cs.items()], time_unit="days")
    for w in ws:
        with tempfile.TemporaryDirectory() as d:
            run = fit_set(s, workers=w, out_dir=d)
            rec[f"w{w}"] = {f.name: _identity(json.load(open(f.path))) for f in run.fits if f.path}
            rec[f"w{w}_status"] = {f.name: f.status for f in run.fits}
            rec[f"w{w}_errors"] = {f.name: f.error for f in run.fits if f.error}
    cpuinfo = ""
    if os.path.exists("/proc/cpuinfo"):
        cpuinfo = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), "")
    rec.update(problems=compare_workers(rec), host=socket.gethostname(), cpu=cpuinfo,
               job_id=os.environ.get("LSB_JOBID"), wall_s=time.time() - t0,
               cpu_s=time.process_time() - c0 + _children_cpu_s())
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")
    print(f"[workers-job] {len(rec['problems'])} problem(s); cpu {rec['cpu_s']:.0f} s; -> {out_path}", flush=True)


def phase_workers(args):
    """HR-6 (T037; R8): rsync the committed head to WEXAC, check the venv's Python and runtime stack against uv.lock
    (a mismatch refuses, naming the package; the full freeze is recorded), submit one
    LSF job on one node that fits three distinct small cohorts in-process and at W = 1 and 2, wait for it, fetch its
    record and compare. `--dry-run` prints the exact commands and contacts nothing. `--resume JOBID --job-head SHA`
    collects a job already submitted from SHA (the poller died, 2026-11-09): it refuses unless WEXAC's copy is
    byte-identical to SHA's tree and HEAD differs from SHA only in RESUME_MAY_DIFFER(_DIRS)."""
    import json
    import re
    import subprocess
    import tempfile
    head = A3.git("rev-parse", "HEAD")
    if args.dry_run:
        for step, cmds in workers_commands(head, "<export-dir>", pilot=args.dev).items():
            for c in cmds:
                print(f"[workers] {step}: {' '.join(_q(x) for x in c)}")
        sys.exit(0)
    if bool(args.resume) != bool(args.job_head):
        sys.exit("[workers] REFUSED: --resume JOBID and --job-head SHA go together")
    run = start("workers", dev=args.dev)
    budget = Budget("workers", CAPS["workers"])
    job_head = A3.git("rev-parse", args.job_head) if args.resume else head
    with tempfile.TemporaryDirectory(prefix="sraging005_workers_") as tmp:
        cmds = workers_commands(job_head, tmp, pilot=args.dev)
        for c in cmds["export"]:
            subprocess.run(c, check=True)
        if args.resume:
            why = resume_problems(job_head) + remote_tree_problems(tmp)
            if why:
                sys.exit("[workers] REFUSED: " + "; ".join(why[:10]))
        else:
            for step in ("mkdir", "rsync"):
                for c in cmds[step]:
                    _remote(c)
        freeze, remote_python = _remote(cmds["freeze"][0]), _remote(cmds["python"][0]).strip()
        pins = runtime_pins()
        problems = runtime_problems(freeze, remote_python, pins)
        env = {"python": remote_python, "pins": pins, "pip_freeze": freeze.splitlines()}
        if problems:
            finish(run, {**not_applicable([f"runtime mismatch with uv.lock (R8): {p}" for p in problems]),
                         "head": job_head, "environment": env}, budget)
            sys.exit(3)
        if args.resume:
            job_id = str(args.resume)
            print(f"[workers] resuming LSF job {job_id}, submitted from {job_head[:7]}", flush=True)
        else:
            sub = _remote(cmds["submit"][0])
            m = re.search(r"Job <(\d+)>", sub)
            if not m:
                raise RuntimeError(f"bsub gave no job id: {sub!r}")
            job_id = m.group(1)
            print(f"[workers] submitted LSF job {job_id}", flush=True)
        stat = poll_job(cmds["poll"][0], job_id)
        if stat == "EXIT":
            finish(run, {"status": "fail", "job_id": job_id, "reason": "the LSF job exited abnormally"}, budget)
            sys.exit(1)
        _remote(cmds["fetch"][0])
        job = json.load(open(os.path.join(tmp, "workers_job.json")))
    if str(job.get("job_id")) != job_id:
        sys.exit(f"[workers] REFUSED: the fetched record is job {job.get('job_id')}'s, not {job_id}'s")
    problems = compare_workers(job)
    budget.remote_cpu_s = float(job.get("cpu_s", 0.0))
    budget.capped = budget.reached()
    status = "pass" if not problems else "fail"
    for p in problems + [f"{r}: {n}: {e}" for r in job.get("runs", ()) for n, e in job.get(f"{r}_errors", {}).items()]:
        print(f"[workers] {p}", flush=True)
    finish(run, {"status": status, "pilot": bool(job.get("pilot")), "head": job_head, "resumed": bool(args.resume),
                 "job_id": job_id, "environment": env, "host": job.get("host"),
                 "cpu": job.get("cpu"),
                 "job_cpu_s": job.get("cpu_s"), "job_wall_s": job.get("wall_s"), "problems": problems,
                 "record": job}, budget)


#: what HEAD may change, against the head a resumed job ran, without touching the code under test: this driver (its
#: poller and resume), its unit test, stored results and documentation (specs/, docs/)
RESUME_MAY_DIFFER = ("tools/acceptance_005.py", "tests/unit/test_acceptance_005.py")
RESUME_MAY_DIFFER_DIRS = ("tests/acceptance/results/", "specs/", "docs/")
#: consecutive failed polls tolerated (one ssh failure killed the first HR-6 poller, 2026-11-09)
WORKERS_POLL_RETRIES = 30


def resume_problems(job_head, diff=None) -> list[str]:
    """Each file HEAD changed since `job_head` outside RESUME_MAY_DIFFER (a resumed result would not describe HEAD's
    code); `diff` (the changed paths) is for tests."""
    changed = diff if diff is not None else [f for f in A3.git("diff", "--name-only", job_head, "HEAD").split() if f]
    return [f"{f} changed since the job's head {job_head[:7]}" for f in changed
            if f not in RESUME_MAY_DIFFER and not f.startswith(RESUME_MAY_DIFFER_DIRS)]


def _skipped(rel) -> bool:
    """Not the code under test: the export's tarball and the bytecode the job's Python wrote (__pycache__/, and .cache/,
    where the repo's gitignored pycache prefix puts it)."""
    return rel == "head.tar" or rel.startswith(".cache/") or "__pycache__/" in rel


def tree_sha256(root) -> dict:
    """{relative path: sha256} of every file under `root` but the `_skipped` ones."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            rel = os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/")
            if not _skipped(rel):
                out[rel] = file_sha256(os.path.join(d, f))
    return out


def remote_tree_command() -> list:
    """sha256sum of every file of WEXAC's copy but the `_skipped` ones. macOS's openrsync itemizes files of identical
    content in a checksum dry run (2026-11-09), so the copies are compared by explicit hashes, not by rsync."""
    host, repo = WEXAC["host"], WEXAC["repo"]
    return ["ssh", host, f"cd ~/{repo} && find . -type f -not -path './.cache/*' -not -path '*/__pycache__/*' "
                         f"-not -name head.tar -print0 | xargs -0 sha256sum"]


def tree_problems(local: dict, remote: dict) -> list[str]:
    """Each file missing on WEXAC, extra there, or of different content."""
    out = [f"{f}: missing on WEXAC" for f in sorted(set(local) - set(remote))]
    out += [f"{f}: on WEXAC only" for f in sorted(set(remote) - set(local))]
    out += [f"{f}: content differs" for f in sorted(set(local) & set(remote)) if local[f] != remote[f]]
    return out


def remote_tree_problems(export_dir) -> list[str]:
    remote = {}
    for ln in _remote(remote_tree_command()).splitlines():
        if ln.strip():
            sha, path = ln.split(None, 1)
            remote[path.strip().lstrip("*").removeprefix("./")] = sha
    return [f"WEXAC's copy differs from the job's head: {p}" for p in tree_problems(tree_sha256(export_dir), remote)]


def poll_job(poll_cmd, job_id, *, retries=WORKERS_POLL_RETRIES, wait_s=None, remote=None) -> str:
    """bjobs until DONE or EXIT; up to `retries` consecutive failed polls are waited out, not fatal."""
    remote = remote or _remote
    wait_s = WORKERS_POLL_S if wait_s is None else wait_s
    failed = 0
    while True:
        try:
            stat = remote([x.replace("<JOBID>", job_id) for x in poll_cmd]).strip()
            failed = 0
        except RuntimeError as err:
            failed += 1
            print(f"[workers] poll failed ({failed}/{retries}): {str(err)[:200]}", flush=True)
            if failed > retries:
                raise
            stat = ""
        if stat in ("DONE", "EXIT"):
            return stat
        time.sleep(wait_s)


def _q(x):
    import shlex
    return shlex.quote(x)


# ---------------------------------------------------------------------------------- km (HR-3; T048's phase code)
KM_SEED = 20261108
KM_TIES = ((1.0, 2.0, 2.0, 2.0, 3.0), (1, 1, 0, 0, 1))      # one death and two censorings at 2 (brief §1e)


def km_vs_lifelines(ages, events) -> dict:
    """sraging.survival.km against lifelines 0.30.3 (oracle group) on one cohort: max |dS| at the vertices, max |d|
    of the band, and the medians."""
    from lifelines import KaplanMeierFitter
    from sraging.survival import km, summary
    ages, events = np.asarray(ages, float), np.asarray(events, int)
    k = km((ages, events))
    ref = KaplanMeierFitter().fit(ages, events)
    sf, ci = ref.survival_function_.iloc[:, 0], ref.confidence_interval_
    if k.times.size:
        dS = float(np.max(np.abs(k.S - sf.loc[k.times].to_numpy())))
        dband = float(max(np.max(np.abs(k.S_lo - ci.iloc[:, 0].loc[k.times].to_numpy())),
                          np.max(np.abs(k.S_hi - ci.iloc[:, 1].loc[k.times].to_numpy()))))
    else:
        dS = dband = 0.0
    med, med_ref = summary(k).median.value, float(ref.median_survival_time_)
    return {"dS": dS, "dband": dband, "median": med, "median_ref": med_ref,
            "median_equal": med == med_ref or (math.isinf(med) and math.isinf(med_ref))}


def _km_worst(rows) -> dict:
    return {"cohorts": len(rows), "max_dS": max((r["dS"] for r in rows), default=0.0),
            "max_dband": max((r["dband"] for r in rows), default=0.0),
            "median_mismatches": sum(not r["median_equal"] for r in rows)}


def km_synthetic(*, seed=KM_SEED, reps=200) -> dict:
    """HR-3's synthetic ties: `reps` random cohorts on a coarse grid (heavy ties, censoring) and the tie example."""
    rng = np.random.default_rng(seed)
    rows = [km_vs_lifelines(*KM_TIES)]
    for _ in range(reps):
        n = int(rng.integers(3, 400))
        ages = rng.integers(1, 30, n).astype(float) * float(rng.choice([1.0, 0.5, 2.0]))
        events = (rng.random(n) < rng.uniform(0.4, 1.0)).astype(int)
        events[rng.integers(n)] = 1
        rows.append(km_vs_lifelines(ages, events))
    return _km_worst(rows)


def km_planted_tie() -> dict:
    """The planted tie order: SRK's row walk with a censoring before the death at t = 2 gives S(2) = 8/15, against
    the deaths-first 0.600; it must fail HR-3's 1e-12."""
    from sraging.survival import km
    ours = float(km(KM_TIES).S_at([2.0])[0])
    S, r = 1.0, len(KM_TIES[0])
    for t, d in [(1.0, 1), (2.0, 0), (2.0, 1)]:
        S, r = (S * (1 - 1 / r) if d else S), r - 1
    return {"ours": ours, "planted": S, "fails": abs(S - ours) > 1e-12}


def phase_km(args):
    """HR-3 (T048): the KM against lifelines 0.30.3 side by side on the 539 record cages (read through the manifest)
    and the synthetic ties: S at the vertices, the band, the medians; the planted tie order fails first."""
    run = start("km", dev=args.dev)
    budget = Budget("km", CAPS["km"])
    require_records(run, budget)
    h3 = read_rules()["rules"]["HR-3"]
    planted = km_planted_tie()
    recs = tomllib.load(open(RECORDS, "rb"))

    def cell(name):
        if name == "synthetic":
            return {"cell": name, **km_synthetic()}
        rec = recs[name]
        record = read_record_csv(os.path.join(rec["dir"], "cleaning.csv"))
        cages, refused = read_record_cages(rec, sorted({str(r["cohort"]) for r in record}))
        return {"cell": name, "refused_rows": len(refused),
                **_km_worst([km_vs_lifelines(cages[k].ages, cages[k].events) for k in cages])}
    rows = run_capped(budget, ["synthetic", *recs], cell)
    ok = (planted["fails"] and len(rows) == 1 + len(recs)
          and all(r["max_dS"] <= float(h3["S_abs"]) and r["max_dband"] <= float(h3["band_abs"])
                  and r["median_mismatches"] == 0 and not r.get("refused_rows") for r in rows))
    print(f"[km] planted {planted}; {rows}", flush=True)
    finish(run, {"status": "pass" if ok else "fail", "planted": planted, "cells": rows,
                 "tolerances": {"S_abs": h3["S_abs"], "band_abs": h3["band_abs"]}}, budget)


# ------------------------------------------------------------------------------- noise (HR-4; T051's phase code)
NOISE_SEED = 20261108
NOISE_SE = 4.0


def rate_cell(ours, theirs, n_ours, n_theirs, se_multiple=NOISE_SE) -> dict:
    """Two flag rates, each a binomial estimate over its own cage count: |difference| <= se_multiple x the SE of the
    difference, sqrt(p (1 - p) (1/n1 + 1/n2)) with p pooled."""
    p = (ours * n_ours + theirs * n_theirs) / (n_ours + n_theirs)
    se = math.sqrt(p * (1 - p) * (1 / n_ours + 1 / n_theirs))
    return {"ours": ours, "theirs": theirs, "diff": ours - theirs, "se": se,
            "ok": abs(ours - theirs) <= se_multiple * se}


def group_spread(cohorts: dict, floors, *, seed=NOISE_SEED) -> tuple:
    """(mult_ml, mult_st): a group's own relative spread as a multiple of the floor, the fly redo's group_spreads
    (recal_2026-10/src/clean.py:150-164): each cohort's leave-one-out relative deviations divided by r(m), pooled over
    the group's cohorts, robust sd. The library's own construction (cleaning.pooled_spread, which clean's own-spread
    null uses: sr-naveh_005 B2), so HR-4's group cells test the spread the report computes."""
    from sraging.cleaning import pooled_spread
    s_ml, s_st = pooled_spread(cohorts, seed=seed)
    return s_ml / floors[0], s_st / floors[1]


def group_expected(sizes: dict, k, floors, spread, *, reps, seed=NOISE_SEED) -> dict:
    """The own-spread null's expected flags over a group's cohorts ({m: number of cohorts of m cages}), and the
    Monte Carlo variance of that estimate (sum of n_m^2 m p_m (1 - p_m) / R)."""
    from sraging.cleaning import noise_baseline
    e = v = 0.0
    for m, n_m in sizes.items():
        row = noise_baseline(int(m), k, floors=floors, spread=spread, reps=reps, seed=seed)["own_spread"]
        e += n_m * row["expected_flags"]
        v += n_m ** 2 * m * row["rate"] * (1 - row["rate"]) / reps
    return {"expected": e, "var": v}


def noise_plants(phi, ref, *, seed=NOISE_SEED) -> list:
    """HR-4's planted nulls, each judged as a floor cell against `ref` (the fly's null_drops row at m = 3, k = 2.5) and
    required to fail. Both draw scatter of size phi (the rule's floors):
    - "no leave-one-out": judged by the rule without leave-one-out (the floor null's entry);
    - "no floors": judged by a rule WITHOUT floors, floors = (0, 0), so the scatter phi goes in as the own-spread
      null's spread and the rate is that entry's. The floor entry of the same call draws scatter of size floors =
      (0, 0): no noise at all (rate 0.000 before; now not measurable), a null without scatter rather than a rule
      without floors (sr-naveh_005 N1)."""
    from sraging.cleaning import noise_baseline
    m, k, reps = int(ref["m"]), float(ref["k"]), int(ref["reps"])
    rates = {"no leave-one-out": noise_baseline(m, k, floors=phi, spread=phi, reps=reps, seed=seed,
                                                leave_one_out=False)["floor"]["rate"],
             "no floors": noise_baseline(m, k, floors=(0.0, 0.0), spread=phi, reps=reps,
                                         seed=seed)["own_spread"]["rate"]}
    out = []
    for defect, rate in rates.items():
        c = rate_cell(rate, float(ref["rate"]), reps * m, int(ref["cages"]))
        out.append({"defect": defect, **c, "fails": not c["ok"]})
    return out


def phase_noise(args):
    """HR-4 (T051): the floor null's flag rate per (m, k) against the fly's null_drops.csv (its sigma_mult = 1 cells),
    and the own-spread null's expected drops per group and k against null_by_group.csv (the groups' spreads rebuilt from
    the fly's cleaning.csv), each within 4 binomial SE; a null without leave-one-out, and one without floors, must fail
    first."""
    from sraging.cleaning import noise_baseline
    run = start("noise", dev=args.dev)
    budget = Budget("noise", CAPS["noise"])
    require_records(run, budget)
    fly = tomllib.load(open(RECORDS, "rb"))["fly"]
    phi = record_floors("fly", fly)[:2]
    drops = [r for r in read_record_csv(os.path.join(fly["dir"], "null_drops.csv")) if float(r["sigma_mult"]) == 1.0]
    by_group = read_record_csv(os.path.join(fly["dir"], "null_by_group.csv"))
    cl = read_record_csv(os.path.join(fly["dir"], "cleaning.csv"))
    ref3 = next(r for r in drops if int(r["m"]) == 3 and float(r["k"]) == 2.5)
    planted = noise_plants(phi, ref3)

    def floor_cell(r):
        m, k, reps = int(r["m"]), float(r["k"]), int(r["reps"])
        o = noise_baseline(m, k, floors=phi, spread=phi, reps=reps, seed=NOISE_SEED)["floor"]
        return {"m": m, "k": k, **rate_cell(o["rate"], float(r["rate"]), reps * m, int(r["cages"]))}
    floors_rows = run_capped(budget, drops, floor_cell)

    groups = {}
    for r in cl:
        g = groups.setdefault(str(r["group"]), {})
        g.setdefault(str(r["cohort"]), ([], []))
        g[str(r["cohort"])][0].append(float(r["median"]))
        g[str(r["cohort"])][1].append(float(r["steepness_iqr"]))

    def group_cell(r):
        cohorts = groups[str(r["group"])]
        mult = group_spread(cohorts, phi)
        sizes = {}
        for ml, _ in cohorts.values():
            sizes[len(ml)] = sizes.get(len(ml), 0) + 1
        out = {"group": str(r["group"]), "mult": list(mult), "mult_file": [float(r["mult_ml"]), float(r["mult_st"])],
               "sizes": sizes}
        for k, col in ((2.5, "expected_k25"), (3.0, "expected_k3")):
            e = group_expected(sizes, k, phi, (mult[0] * phi[0], mult[1] * phi[1]), reps=4000)
            se = math.sqrt(2 * e["var"])
            out[col] = {"ours": e["expected"], "theirs": float(r[col]), "se": se,
                        "ok": abs(e["expected"] - float(r[col])) <= NOISE_SE * se}
        return out
    group_rows = run_capped(budget, by_group, group_cell)
    ok = (all(p["fails"] for p in planted) and len(floors_rows) == len(drops) and len(group_rows) == len(by_group)
          and all(c["ok"] for c in floors_rows)
          and all(g[c]["ok"] for g in group_rows for c in ("expected_k25", "expected_k3")))
    print(f"[noise] {len(drops)} floor cells, {sum(c['ok'] for c in floors_rows)} within {NOISE_SE:g} SE; groups "
          f"{[(g['group'], g['expected_k25']['ok'], g['expected_k3']['ok']) for g in group_rows]}; planted "
          f"{[(p['defect'], p['fails']) for p in planted]}", flush=True)
    finish(run, {"status": "pass" if ok else "fail", "floors": list(phi), "seed": NOISE_SEED, "se_multiple": NOISE_SE,
                 "planted": planted, "n_floor_cells": len(floors_rows), "n_group_cells": len(group_rows),
                 "floor_cells": floors_rows, "group_cells": group_rows}, budget)


# --------------------------------------------------------------------------- unchanged (SC-008; T053's phase code)
def unchanged_groups() -> dict:
    """The suites `unchanged` runs, each by pytest in its own process: the unit and parity suites, every earlier
    spec's results test (004's bounded at 759c003), and the closure check."""
    import glob
    results = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "tests", "acceptance",
                                                                               "test_results*.py"))
                     if not p.endswith("test_results_005.py"))
    return {"unit": ["tests/unit"], "parity": ["tests/parity"], "results": results,
            "closure": ["tests/acceptance/test_results_005.py::test_claims_closure_untouched"]}


def spec003_reading() -> dict:
    """What spec 003's `unchanged` reads on this branch, from its own functions, without running it (its run writes
    results/claims-001_003.json): the closure files changed against main's merge base, the untouched-path
    violations, the spec-exempt files, and whether spec 001's claims re-run would be required."""
    base = A3.merge_base()
    closure = A3.import_closure()
    changed = A3.git("diff", "--name-only", base, "HEAD", "--", *closure).split()
    doc_only, exempt = [], []
    violations = A3.untouched_problems(doc_only, exempt, base)
    return {"base": base, "closure_files": len(closure), "changed": changed, "violations": violations,
            "spec_exempt": exempt, "docstring_only": doc_only, "rerun_required": bool(changed) or bool(exempt)}


def spec003_ok(r) -> bool:
    """SC-008: no closure change, no violation, no claims re-run."""
    return not r["changed"] and not r["violations"] and not r["rerun_required"]


def phase_unchanged(args):
    """SC-008 (T053), in the MAIN checkout: the unit suite, the parity suite, every earlier results test and the
    closure check, each by pytest in its own process (003e's rule: 0 failures and 0 errors, every skip on its lists),
    and spec 003's `unchanged` reading no closure change and no claims re-run."""
    import tempfile
    import acceptance_003e as E
    import acceptance_004 as A4
    run = start("unchanged", dev=args.dev)
    budget = Budget("unchanged", CAPS["unchanged"])
    reading = spec003_reading()
    groups_def = unchanged_groups()
    with tempfile.TemporaryDirectory(prefix="sraging005_unchanged_") as tmp:
        rows = run_capped(budget, list(groups_def),
                          lambda g: {"group": g, **A4.unchanged_group(g, groups_def[g], tmp)})
    groups = {r.pop("group"): r for r in rows}
    groups_ok, unlisted = E.unchanged_verdict(groups) if len(groups) == len(groups_def) else (False, [])
    for g in groups.values():
        g.pop("cases", None)
    status = "pass" if groups_ok and spec003_ok(reading) else "fail"
    print(f"[unchanged] groups ok {groups_ok}; spec 003 reading {reading['changed']} / rerun "
          f"{reading['rerun_required']}", flush=True)
    finish(run, {"status": status, "groups": groups, "groups_ok": groups_ok, "skips_unlisted": unlisted,
                 "spec003": reading}, budget)


def _todo(name, task):
    def phase(args):
        sys.exit(f"{name}: not implemented yet (specs/005-data-plots-cli/tasks.md {task})")
    return phase


PHASE_FUNCS = {"exit": phase_exit, "km": phase_km, "noise": phase_noise,
               "workers": phase_workers, "presets": phase_presets,
               "units": phase_units, "unchanged": phase_unchanged}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=PHASES + ("workers-job",))
    ap.add_argument("--dev", action="store_true", help="development run (a pilot): never counts")
    ap.add_argument("--dry-run", action="store_true", help="workers: print the exact commands; contact nothing")
    ap.add_argument("--out", help="workers-job (on the WEXAC node): where to write its record")
    ap.add_argument("--pilot", action="store_true", help="workers-job: P1's job, one cohort in-process and at W = 1")
    ap.add_argument("--resume", metavar="JOBID", help="workers: collect this already-submitted LSF job")
    ap.add_argument("--job-head", metavar="SHA", help="workers --resume: the commit the job was submitted from")
    args = ap.parse_args(argv)
    if args.dry_run and args.phase != "workers":
        ap.error("--dry-run is the workers phase's")
    if args.pilot and args.phase != "workers-job":
        ap.error("--pilot is workers-job's (the workers phase passes it for --dev)")
    if args.phase == "workers-job":
        if not args.out:
            ap.error("workers-job needs --out")
        workers_job(args.out, pilot=args.pilot)
        return
    PHASE_FUNCS[args.phase](args)


if __name__ == "__main__":
    main()
