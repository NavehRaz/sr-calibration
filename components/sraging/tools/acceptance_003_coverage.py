"""Spec 003 `coverage` phase (SC-005, SC-006; research R14; task T044; sr-manager gate 5; sr-naveh review 4).

Fits: profile_jax2 on panel_003.toml [coverage].profile_jax2 (the medium J set less windowed cohorts and the panel's
cuts: het, the hazard arm, the realfull C. elegans cohorts; [coverage.cuts] gives each reason) and profile_jax2_h on
[coverage].profile_jax2_h (the medium hazard arm, where it is the right model), combinations on. Each fit's row
(every claim, the truth, arm, cluster, informativeness, costs) is appended to .cache/out/coverage_003_rows.jsonl as it
finishes, so a crashed run resumes where it stopped (rows of another HEAD are discarded). A fit that RAISES IN THE
FROZEN CODE (the innermost sraging frame of the exception or its cause lies in compat/jax2) is a row too: a decline
of every quantity, never `missing`. Any other error is sraging's own failure: `missing`, re-run on resume, a fail.

Before any fit, free: the gate's power on the panel's cluster structure (`paired.gate_power`), then the run-time
refusals (`prerequisites`): the walker result committed and passed with its cost decided (above cost_limit, Naveh's
`cost_decision` recorded in it), the order and floor results committed, and the frozen claims/order.json and
claims/floors.json equal to them. `--dev` reports a refusal and runs anyway (a development result never counts).

Scoring, per calibrator and arm:
  GATE (Naveh 2026-10-05; sr-manager 5; sr-naveh 4) on (profile_jax2, pure), (profile_jax2, censor) and
  (profile_jax2_h, hazard) only; profile_jax2 on hazard, if fitted, is reported and never gated. For every
  combination, paired against the same fits' eta claims and, apart, their epsilon claims (`paired.gate`): a loss is
  z < -2 (Naveh 2026-10-05 dropped the -0.05 non-inferiority margin: decisions_003.toml [coverage_gate]; the CI is
  reported beside each verdict); fewer than 30 paired cells or 10 clusters, or fewer than 50 informative cohorts on
  the arm, is "underpowered", and the phase is then not_applicable, never a pass. The gate counts every panel cohort,
  valid or not (Naveh 2026-10-08, [coverage_validity]: it is paired, so an invalid truth affects both sides alike).
  REPORTED beside it: absolute coverage with its Wilson 95% interval, and again on the cohorts the tester's
  reports/validity/validity.csv marks valid (its path and sha256 recorded); the decline rate (failed fits included); the
  tester's availability and information axes against eta and epsilon (verdict.py's tolerances; not gating); one-sided
  claims apart, with their distance inside the data-ignoring null's box image; box-limited claims apart (a cap set by
  the box scores coverage for free); the widened subset scored against its own claims_profile (coverage, width,
  shape); the closed width against the null's box image; the labels' firing rates; for "negligible", the costly-error
  rate (labelled, truth above 3x the floor) and SC-006's rate of truths above 3x the floor that carry the label.
"""
from __future__ import annotations

import concurrent.futures as cf
import itertools
import json
import math
import multiprocessing as mp
import os
import statistics as st
import subprocess
import sys
import time
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = os.path.join(REPO, ".cache", "out", "coverage_003_rows.jsonl")
XHOST_ROWS = os.path.join(REPO, ".cache", "out", "coverage_003_xhost_rows.jsonl")   # the Mac subset, never in ROWS
XHOST_N, XHOST_SEED = 5, 20261008                     # per gated cell (Naveh 2026-10-08, [coverage_host])
PANEL_REL = os.path.join("tests", "acceptance", "panel_003.toml")
FROZEN = os.path.join(REPO, "src", "sraging", "compat", "jax2") + os.sep
SRAGING = os.path.join(REPO, "src", "sraging") + os.sep
GATED = (("profile_jax2", "pure"), ("profile_jax2", "censor"), ("profile_jax2_h", "hazard"))
MIN_INFORMATIVE = 50
VALIDITY = os.path.join("reports", "validity", "validity.csv")    # under the tester root (acceptance_003.T)
COSTLY_X = 3.0                  # "negligible"'s costly error: a truth above 3x the floor labelled negligible
PARAMS = ("eta", "beta", "epsilon", "kappa")
POWER_GRID = {"delta": (0.0, -0.05, -0.10), "p_ref": (0.95, 0.80), "icc": (0.0, 1.0), "concordance": (0.0, 0.5, 0.9)}
POWER_REPS = 400
LABEL_MARKERS = {"box_limited": "box-limited", "negligible": "negligible:", "saturated": "saturated removal",
                 "widened": "edge not found"}
THREAD_ENV = {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
              "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def _init():
    for k, v in THREAD_ENV.items():
        os.environ[k] = v
    sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
    for p in (os.path.join(REPO, "tools"), os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


# ------------------------------------------------------------------------------------------------ the fits
_LIB = None


def library():
    """J2's library v2 through the v033 snapshot's srbench loader, read-only, WITHOUT loading the oracle (Naveh
    2026-10-08, [coverage_host]: the fits run on WEXAC). The coverage fits use the port, not the frozen method, so
    the oracle's host check (byte identity is claimed only on the snapshot host) does not apply; the rest of its
    preflight does: the snapshot's files against its MANIFEST, the package pins, the Python version. The library
    checks its cohort checksums on load. Its root is oracle_jax2.LIBRARY_V2 (SRAGING_J2_ROOT moves it)."""
    global _LIB
    if _LIB is None:
        import importlib.metadata as md
        import oracle_jax2 as OJ
        import oracle_v033 as V
        problems = [] if sys.pycache_prefix else ["sys.pycache_prefix is not set"]
        man = json.load(open(V.MANIFEST))
        for f in man["files"]:
            fp = os.path.join(V.SNAP, f["repo"], f["path"])
            if not os.path.exists(fp) or V._md5(fp) != f["md5"]:
                problems.append(f"snapshot file missing or changed: {f['repo']}/{f['path']}")
        for pkg, want in man["environment_pins"].items():
            try:
                got = md.version(pkg)
            except md.PackageNotFoundError:
                got = "not installed"
            if got != want:
                problems.append(f"{pkg} {got} != pinned {want}")
        if sys.version.split()[0] != man["python"].split()[0]:
            problems.append(f"python {sys.version.split()[0]} != snapshot {man['python'].split()[0]}")
        if problems:
            raise RuntimeError("the library cannot be loaded here: " + "; ".join(problems))
        if V.SNAP_TESTER not in sys.path:
            sys.path.insert(0, V.SNAP_TESTER)
        from srbench.library.load import Library
        _LIB = Library(root=OJ.LIBRARY_V2, version="v2")
    return _LIB


def host() -> dict:
    """Where a fit ran: oracle_v033's host facts (the hardware UUID on macOS), the CPU model and the LSF job."""
    import oracle_v033 as V
    cpu = None
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=10).stdout.strip() or None
        else:
            cpu = next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")),
                       None)
    except (OSError, subprocess.SubprocessError):
        pass
    return {**V.host_identity(), "cpu": cpu, "lsf_job": os.environ.get("LSB_JOBID"),
            "lsf_index": os.environ.get("LSB_JOBINDEX")}


def _jsonable(x):
    import numpy as np
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    raise TypeError(type(x).__name__)


def error_origin(ex: BaseException) -> str:
    """'frozen' when the innermost sraging frame of `ex`, or of any exception in its cause/context chain, lies in
    compat/jax2 (the frozen code, ported); 'sraging' when it lies elsewhere in src/sraging; else 'harness'."""
    seen, origin, e = set(), "harness", ex
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        frames = [f.filename for f in traceback.extract_tb(e.__traceback__)]
        mine = [f for f in frames if os.path.abspath(f).startswith(SRAGING)]
        if mine and os.path.abspath(mine[-1]).startswith(FROZEN):
            return "frozen"
        if mine:
            origin = "sraging"
        e = e.__cause__ or e.__context__
    return origin


def negligible_floor(r, cohort) -> dict:
    """The negligibility floor the label code would use on this fit, in r-bar/eta's raw units (floor per L_SR times
    L_SR at theta-hat), via profile_jax2's own `negligible_regime` and the frozen floors; {} when not computable."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.claims import shapes as SH
    floors = PJ.load_floors()
    lsr = ((r.diagnostics or {}).get("L_SR") or {}).get("L_SR")
    if floors is None or lsr is None or r.theta_hat is None:
        return {"why": "no frozen floors, L_SR or theta-hat"}
    eps = r.claims.get("epsilon") if hasattr(r.claims, "get") else None
    regime, why = PJ.negligible_regime(eps, r.theta_hat, lsr, floors)
    if regime is None:
        return {"why": why}
    per_lsr = floors[regime]["c_rbar"] / math.sqrt(SH.n_eff(cohort.n_deaths))
    return {"regime": regime, "why": why, "L_SR": lsr, "floor_per_lsr": per_lsr, "floor": per_lsr * lsr}


def fit_worker(cid, calibrator, head):
    row = {"cohort_id": cid, "calibrator": calibrator, "head": head, "host": host()}
    try:
        lib = library()
        sp = lib.spec(cid)
        row.update(arm=sp.arm, cluster=sp.cluster_key, informative=not sp.het_var, truth=lib.truth(cid))
        obs = lib.observation(cid)
    except Exception as ex:  # noqa: BLE001
        return {**row, "error": f"{type(ex).__name__}: {ex}", "error_origin": "harness",
                "trace": traceback.format_exc()[-1500:]}
    try:
        import sraging
        from sraging.adapters.observation import cohort_from_observation
        cohort = cohort_from_observation(obs)
        c0 = time.process_time()
        # all six are scored before any ships (US2); the lead pair's claims equal a default fit's by construction
        # (profile_jax2.combination_claims widens them over the archive as it stood after the lead walks)
        r = sraging.fit_cohort(cohort, calibrator=calibrator, combinations="all")
        cpu = time.process_time() - c0
    except Exception as ex:  # noqa: BLE001
        return {**row, "error": f"{type(ex).__name__}: {ex}", "error_origin": error_origin(ex),
                "trace": traceback.format_exc()[-1500:]}
    out = {**row, "status": r.status, "n_deaths": r.cohort["n_deaths"], "claims": [c.to_dict() for c in r.claims],
           "combinations": {k: v for k, v in (r.combinations or {}).items() if k != "walks"}, "cpu_s": cpu,
           "gap_nats": r.gap_nats, "warnings": [w.code for w in r.warnings]}
    try:
        out["negligible_floor"] = negligible_floor(r, cohort)
    except Exception as ex:  # noqa: BLE001 -- recorded; the negligible scores then say "not measured"
        out["negligible_floor"] = {"why": f"{type(ex).__name__}: {ex}"}
    return out


def _load_rows(head):
    """Rows of this HEAD: ok fits, and fits that raised in the frozen code (declines); other errors are re-run."""
    rows = {}
    if os.path.exists(ROWS):
        for line in open(ROWS):
            r = json.loads(line)
            if r.get("head") == head and ("error" not in r or r.get("error_origin") == "frozen"):
                rows[(r["cohort_id"], r["calibrator"])] = r
    return rows


# ------------------------------------------------------------------------------------------------ scoring
def failed(r) -> bool:
    return "error" in r or r.get("status") != "ok"


def claim_of(r, q):
    """The row's claim on `q`, None when it has none; 'failed' when the fit failed (a decline of every quantity)."""
    if failed(r):
        return "failed"
    return next((c for c in r["claims"] if c["parameter"] == q), None)


def hit(claim, truth):
    """True/False when the claim states a bound the truth can be checked against; None for a decline."""
    s, lo, hi = claim["shape"], claim.get("lower"), claim.get("upper")
    if s == "closed" and lo is not None and hi is not None:
        return lo <= truth <= hi
    if s == "upper_bound" and hi is not None:
        return truth <= hi
    if s == "lower_bound" and lo is not None:
        return truth >= lo
    return None


def truth_value(q, truth):
    from sraging.claims import combinations as K
    if q in PARAMS:
        return truth[q]
    return K.value(q, truth)


def label_kinds(c) -> set:
    """Every label kind a claim carries: label_info holds only the first that applied, the text all of them."""
    kinds = {(c.get("label_info") or {}).get("kind")} - {None}
    return kinds | {k for k, m in LABEL_MARKERS.items() if m in (c.get("label") or "")}


def shape_mix(rows, q) -> dict:
    mix = {"closed": 0, "upper_bound": 0, "lower_bound": 0, "none": 0, "failed": 0}
    for r in rows:
        c = claim_of(r, q)
        mix["failed" if c == "failed" else "none" if c is None else c["shape"]] += 1
    return mix


def width_dex(c):
    lo, hi = c.get("lower"), c.get("upper")
    return math.log10(hi / lo) if c.get("shape") == "closed" and lo and hi and lo > 0 else None


def widths(rows, q):
    """Widths (dex) of the rows' closed claims on `q`."""
    cs = [claim_of(r, q) for r in rows]
    return [w for w in (width_dex(c) for c in cs if isinstance(c, dict)) if w is not None]


def hits_of(rows, q, cls):
    """{cohort: hit or None}: None for a failed fit, no claim, another shape class, or a box-limited claim."""
    h = {}
    for r in rows:
        c = claim_of(r, q)
        ok = isinstance(c, dict) and (c["shape"] == cls if cls != "one_sided" else
                                      c["shape"] in ("upper_bound", "lower_bound"))
        h[r["cohort_id"]] = (hit(c, truth_value(q, r["truth"])) if ok and "box_limited" not in label_kinds(c)
                             else None)
    return h


def widened_scores(rows, q):
    """The widened subset against its own claims_profile (the claim before widening), on the same cohorts. A union
    only raises coverage, so the gain is not evidence; the width and shape it costs are reported."""
    from collections import Counter
    pairs = []
    for r in rows:
        c = claim_of(r, q)
        if isinstance(c, dict) and (c.get("widening") or {}).get("claims_profile"):
            pairs.append((r, c, c["widening"]["claims_profile"]))
    if not pairs:
        return {"n": 0}

    def cov(which):
        hs = [hit((c, p)[which], truth_value(q, r["truth"])) for r, c, p in pairs]
        hs = [h for h in hs if h is not None]
        return {"n": len(hs), "coverage": (sum(hs) / len(hs)) if hs else None}
    wd = [(width_dex(c), width_dex(p)) for _, c, p in pairs]
    both = [(a, b) for a, b in wd if a is not None and b is not None]
    return {"n": len(pairs), "widened": cov(0), "claims_profile": cov(1),
            "shape_moves": dict(Counter(f"{p.get('shape')}->{c['shape']}" for _, c, p in pairs)),
            "median_width_dex": {"widened": st.median([a for a, _ in both]) if both else None,
                                 "claims_profile": st.median([b for _, b in both]) if both else None},
            "median_upper_moved_dex": st.median([(c["widening"].get("upper_moved_dex") or 0.0) for _, c, _ in pairs]),
            "median_lower_moved_dex": st.median([(c["widening"].get("lower_moved_dex") or 0.0) for _, c, _ in pairs])}


def negligible_scores(rows, q="rbar_over_eta"):
    """SC-006 and the spec's asymmetric scoring of "negligible": the costly error is a truth above COSTLY_X times the
    floor on a claim labelled negligible. Rates: costly / labelled, and SC-006's labelled / truths above 3x."""
    lab, costly, above, no_floor, mismatch = 0, 0, 0, 0, 0
    for r in rows:
        c = claim_of(r, q)
        if not isinstance(c, dict):
            continue
        fl = (r.get("negligible_floor") or {}).get("floor")
        labelled = "negligible" in label_kinds(c)
        li = c.get("label_info") or {}
        if labelled and li.get("kind") == "negligible" and c.get("upper"):     # the label's own floor, cross-checked
            own = c["upper"] / 10 ** li["log10_cap_over_floor"]
            mismatch += fl is not None and abs(math.log10(own / fl)) > 1e-6
            fl = own
        if fl is None:
            no_floor += labelled
            continue
        hi_truth = truth_value(q, r["truth"]) > COSTLY_X * fl
        lab += labelled
        above += hi_truth
        costly += labelled and hi_truth
    return {"labelled": lab, "costly_errors": costly, "costly_error_rate": costly / lab if lab else None,
            "truths_above_3x_floor": above, "labelled_rate_among_truths_above_3x": costly / above if above else None,
            "labelled_without_floor": no_floor, "floor_mismatch": mismatch, "x": COSTLY_X}


DECISIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "acceptance",
                         "decisions_003.toml")


def gate_decision():
    """Naveh's decision on the non-inferiority margin (tests/acceptance/decisions_003.toml [coverage_gate]: margin --
    a number, or "none" for the Clarification rule z < -2 alone -- decision, date, by), or None if not recorded.
    The margin came from the sr-naveh implementation review, not from Naveh; its power is computed first."""
    import tomllib
    if not os.path.exists(DECISIONS):
        return None
    d = tomllib.load(open(DECISIONS, "rb")).get("coverage_gate")
    if not d or "naveh" not in str(d.get("by", "")).lower() or "margin" not in d:
        return None
    return dict(d, margin=None if str(d["margin"]).lower() == "none" else float(d["margin"]))


def gate_margin():
    from sraging.scoring.paired import NI_MARGIN
    d = gate_decision()
    return (NI_MARGIN, "default, NOT decided") if d is None else (d["margin"], f"Naveh {d.get('date')}")


def gate_record(inf, q, clusters, underpowered_cell):
    """The gate of combination `q` against eta and epsilon in one gated cell, with the tester's other two axes."""
    from sraging.scoring.paired import availability_axis, gate, information_axis, paired_gap
    closed_q = hits_of(inf, q, "closed")
    out, axes = {}, {}
    margin, source = gate_margin()
    for ref in ("eta", "epsilon"):
        gap = paired_gap(closed_q, hits_of(inf, ref, "closed"), clusters)
        g = dict(gate(gap, margin=margin), margin=margin, margin_source=source)
        if underpowered_cell:
            g = {"verdict": "underpowered", "reasons": [underpowered_cell] + g["reasons"]}
        out[f"vs_{ref}"] = {**gap, **g}
        axes[f"vs_{ref}"] = {"availability": availability_axis(shape_mix(inf, q), shape_mix(inf, ref)),
                             "information": information_axis(widths(inf, q), widths(inf, ref), shape_mix(inf, q),
                                                             shape_mix(inf, ref))}
    verdicts = [v["verdict"] for v in out.values()]
    worst = next((v for v in ("loses", "inferior", "underpowered") if v in verdicts), "no loss")
    return {**out, "verdict": worst}, {**axes, "gating": False,
                                       "note": "the tester's axes (verdict.py:137-178), reported beside the gate"}


def validity(tester_root) -> dict:
    """The tester's validity flags (reports/validity/validity.csv, untracked in the tester; the kmax panel reads the
    same file): {valid: cohort ids not flag_invalid, source, sha256, n}."""
    import csv
    import hashlib
    path = os.path.join(tester_root, VALIDITY)
    rows = list(csv.DictReader(open(path)))
    return {"valid": sorted(r["cohort_id"] for r in rows if r.get("flag_invalid") in ("False", "false", "0")),
            "source": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "n": len(rows)}


def score(rows, box, valid=None):
    """Per (calibrator, arm) cell and quantity; with `valid` (cohort ids), the absolute coverage again on those."""
    from sraging.claims import combinations as K
    from sraging.scoring.paired import availability, wilson
    out = {}
    by = {}
    for r in rows:
        by.setdefault((r["calibrator"], r["arm"]), []).append(r)
    for (cal, arm), rs in sorted(by.items()):
        inf = [r for r in rs if r["informative"]]
        gated = (cal, arm) in GATED
        cell = {"n": len(rs), "n_informative": len(inf), "n_failed": sum(failed(r) for r in inf), "gated": gated,
                "quantities": {}}
        under = (f"{len(inf)} informative cohorts < {MIN_INFORMATIVE}" if gated and len(inf) < MIN_INFORMATIVE
                 else None)
        clusters = {r["cohort_id"]: r["cluster"] for r in inf}
        for q in PARAMS + K.KEYS:
            mix = shape_mix(inf, q)
            present = [c for c in (claim_of(r, q) for r in inf) if isinstance(c, dict)]
            if not present and not (gated and q in K.KEYS):
                continue
            closed = hits_of(inf, q, "closed")
            k = sum(1 for v in closed.values() if v)
            n = sum(1 for v in closed.values() if v is not None)
            one = hits_of(inf, q, "one_sided")
            k1 = sum(1 for v in one.values() if v)
            n1 = sum(1 for v in one.values() if v is not None)
            lo, hi = wilson(k, n)
            qd = {"closed_coverage": None if n == 0 else k / n, "closed_n": n, "wilson95": [lo, hi],
                  **{f: availability(mix)[f] for f in ("decline_rate", "failure_rate")},
                  "shape_mix": mix, "box_limited_n": sum("box_limited" in label_kinds(c) for c in present),
                  "labels": {kind: sum(kind in label_kinds(c) for c in present) for kind in LABEL_MARKERS},
                  "one_sided_coverage": (None if n1 == 0 else k1 / n1), "one_sided_n": n1,
                  "widened": widened_scores(inf, q)}
            qd["label_rates"] = {kind: (v / len(present) if present else None) for kind, v in qd["labels"].items()}
            if q in K.KEYS:
                g_lo, g_hi = K.box_image(q, box)
                dist = [math.log10(g_hi / c["upper"]) for c in present if c["shape"] == "upper_bound" and c["upper"]]
                qd["upper_bound_dex_inside_null"] = st.median(dist) if dist else None
                widths = [w for w in (width_dex(c) for c in present) if w is not None]
                qd["closed_median_width_dex"] = st.median(widths) if widths else None
                qd["null_width_dex"] = math.log10(g_hi / g_lo)
                if q == "rbar_over_eta":
                    qd["negligible"] = negligible_scores(inf, q)
                if gated:
                    qd["gate"], qd["axes"] = gate_record(inf, q, clusters, under)
            if valid is not None:
                kv = sum(1 for c, v in closed.items() if v and c in valid)
                nv = sum(1 for c, v in closed.items() if v is not None and c in valid)
                qd["valid"] = {"closed_coverage": None if nv == 0 else kv / nv, "closed_n": nv,
                               "wilson95": list(wilson(kv, nv))}
            cell["quantities"][q] = qd
        if valid is not None:
            cell["n_valid"] = sum(1 for r in inf if r["cohort_id"] in valid)
        out[f"{cal}|{arm}"] = cell
    return out


def gate_outcome(sc) -> dict:
    """{losses, underpowered, absent}: the gated (cell, quantity, side)s that read a loss (loses or inferior), those
    underpowered, and the gated cells with no rows at all."""
    from sraging.claims import combinations as K
    losses, under = [], []
    for cal, arm in GATED:
        d = sc.get(f"{cal}|{arm}")
        for q in K.KEYS if d else ():
            g = d["quantities"].get(q, {}).get("gate")
            for side in ("vs_eta", "vs_epsilon"):
                v = (g or {}).get(side, {}).get("verdict", "underpowered")
                if v in ("loses", "inferior"):
                    losses.append((f"{cal}|{arm}", q, side, v))
                elif v == "underpowered":
                    under.append((f"{cal}|{arm}", q, side, (g or {}).get(side, {}).get("reasons", ["no gate"])))
    return {"losses": losses, "underpowered": under,
            "absent": [f"{c}|{a}" for c, a in GATED if f"{c}|{a}" not in sc]}


# ------------------------------------------------------------------------------------------------ before the fits
def power(lib, panel) -> dict:
    """The gate's power from the panel alone (free; sr-manager gate 5): per gated cell, the informative cohorts'
    cluster sizes, and `gate_power` over POWER_GRID, every cohort a paired cell (an upper bound)."""
    from sraging.scoring.paired import MIN_CELLS, MIN_CLUSTERS, gate_power
    margin, source = gate_margin()
    out = {}
    for cal, arm in GATED:
        ids = [c for c in panel.get(cal, []) if lib.spec(c).arm == arm and not lib.spec(c).het_var]
        sizes = {}
        for c in ids:
            sizes[lib.spec(c).cluster_key] = sizes.get(lib.spec(c).cluster_key, 0) + 1
        cs = sorted(sizes.values(), reverse=True)
        grid = []
        for d, p, icc, cc in itertools.product(*POWER_GRID.values()):
            pw = gate_power(cs, delta=d, p_ref=p, icc=icc, concordance=cc, reps=POWER_REPS,
                            margin=margin) if cs else None
            grid.append({"delta": d, "p_ref": p, "icc": icc, "concordance": cc, **(pw or {})})
        out[f"{cal}|{arm}"] = {"n_cohorts": len(ids), "n_clusters": len(cs), "cluster_sizes": cs,
                               "min_cells_fraction": (MIN_CELLS / len(ids)) if ids else None,
                               "clusters_ok": len(cs) >= MIN_CLUSTERS, "grid": grid}
    return {"cells": out, "reps": POWER_REPS, "margin": margin, "margin_source": source,
            "model": "every informative cohort a paired cell (an upper bound); the reference covers with p_ref, the "
                     "combination with p_ref + delta; concordance c: P(both) = (1-c) p_a p_ref + c min(p_a, p_ref); "
                     "icc: within-cluster copy probability; 'power' = the fraction that reads a loss (z < -2, or a "
                     "CI lower end <= -margin when a margin is set); at delta = 0 it is the chance of failing a "
                     "combination that loses nothing"}


def cost_decided(w) -> bool:
    """Naveh's combination-cost decision as the walker result carries it (`cost_decision`, from
    tests/acceptance/decisions_003.toml [combination_cost]: decision, date, by)."""
    cd = w.get("cost_decision")
    return (isinstance(cd, dict) and bool(cd.get("decision")) and bool(cd.get("date"))
            and "naveh" in str(cd.get("by", "")).lower())


def walker_problems(w) -> list[str]:
    p = []
    ch = w.get("checks") or {}
    if (not (ch.get("total", 0) > 0 and ch.get("passed") == ch.get("total")) or w.get("failures") or w.get("errors")
            or w.get("archive_failures")):
        p.append(f"the walker result does not pass SC-004 (checks {ch.get('passed')}/{ch.get('total')}, "
                 f"{len(w.get('failures') or [])} failures, {len(w.get('archive_failures') or [])} archive failures, "
                 f"{len(w.get('errors') or [])} errors)")
    r, lim = w.get("cost_ratio_median"), w.get("cost_limit")
    if r is None or lim is None:
        p.append("the walker result has no measured cost ratio")
    elif r > lim and not cost_decided(w):
        p.append(f"combinations cost {r:.2f}x the fit (> {lim}x) and Naveh's decision is not recorded in the walker "
                 f"result (`cost_decision`)")
    return p


def prerequisites(A) -> list[str]:
    """sr-manager gate 5: the refusals checked before any fit starts."""
    problems = []

    def result(name):
        rel = f"tests/acceptance/results/{name}_003.json"
        if not os.path.exists(os.path.join(A.REPO, rel)):
            problems.append(f"{rel} does not exist")
            return None
        if not A.committed(rel):
            problems.append(f"{rel} is not committed unchanged")
        d = json.load(open(os.path.join(A.REPO, rel)))
        if d.get("status_for_acceptance") != "acceptance" or str(d.get("status", "")).startswith("smoke"):
            problems.append(f"{rel} is not an acceptance run ({d.get('status_for_acceptance')}, {d.get('status')})")
        return d

    def frozen(rel):
        if not os.path.exists(os.path.join(A.REPO, rel)) or not A.committed(rel):
            problems.append(f"{rel} is not committed")
            return None
        return json.load(open(os.path.join(A.REPO, rel)))
    w = result("walker")
    if w is not None:
        problems += walker_problems(w)
    if gate_decision() is None:
        problems.append("Naveh's decision on the non-inferiority margin is not recorded in "
                        "tests/acceptance/decisions_003.toml [coverage_gate] (margin = a number, or \"none\")")
    o, f = result("order"), result("neg_floor")
    for name, d in (("order", o), ("neg_floor", f)):
        if d is not None and d.get("status") != "written":
            problems.append(f"results/{name}_003.json status is {d.get('status')!r}, not 'written'")
    fo, ff = frozen("src/sraging/claims/order.json"), frozen("src/sraging/claims/floors.json")
    if o is not None and fo is not None and list(fo.get("order", [])) != list(o.get("order", [])):
        problems.append("src/sraging/claims/order.json is not results/order_003.json's order")
    if f is not None and ff is not None:
        keys = ("rho_eps", "kappa", "c_rbar")
        a = {r: {k: v.get(k) for k in keys} for r, v in (ff.get("regimes") or {}).items()}
        b = {r: {k: v.get(k) for k in keys} for r, v in (f.get("regimes") or {}).items()}
        if a != b:
            problems.append("src/sraging/claims/floors.json is not results/neg_floor_003.json's regimes")
    return problems


def price(A, n_todo) -> str:
    p = os.path.join(A.RESULTS, "walker_step1_003.json")      # step 1 holds the fits' and walks' costs
    if not os.path.exists(p):
        return f"{n_todo} fits to run (all six combinations); UNPRICED: no walker step 1 result"
    rows = [r for r in json.load(open(p)).get("rows", []) if r.get("fit_cpu_s") and r.get("combo_cpu_s") is not None]
    if not rows:
        return f"{n_todo} fits to run (all six combinations); UNPRICED: the walker result has no costed rows"
    per = st.median([r["fit_cpu_s"] + r["combo_cpu_s"] for r in rows])
    return (f"{n_todo} fits to run (all six combinations) at the walker pilot's median {per:.0f} CPU-s a fit (n = "
            f"{len(rows)}): about {n_todo * per / 3600:.1f} CPU-h")


def panel_jobs(panel=None):
    """The phase's (cohort, calibrator) jobs in their fixed order: profile_jax2's cohorts, then profile_jax2_h's."""
    if panel is None:
        import tomllib
        panel = tomllib.load(open(os.path.join(REPO, PANEL_REL), "rb"))["coverage"]
    return [(c, "profile_jax2") for c in panel["profile_jax2"]] + [(c, "profile_jax2_h") for c in panel["profile_jax2_h"]]


def shard_jobs(jobs, index, of):
    """Shard `index` of `of` (1-based, as LSB_JOBINDEX): jobs index-1, index-1+of, ... -- the shards partition jobs."""
    if not 1 <= index <= of:
        raise ValueError(f"shard {index} of {of}")
    return jobs[index - 1::of]


def _git(*args) -> str:
    """git in this checkout, never taking the index lock (215 cluster jobs share one checkout); "" on failure."""
    p = subprocess.run(["git", "--no-optional-locks", "-C", REPO, *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def fit_shard(index, of, rows_out):
    """One cluster job (Naveh 2026-10-08, [coverage_host]): fit shard `index` of `of` at this checkout's HEAD (a
    clean tracked tree, sraging imported from it), appending one row per fit, with its host, to `rows_out`; rows of
    this HEAD already there are kept (a resubmitted job resumes). No scoring: the Mac scores at the same commit."""
    _init()
    head = _git("rev-parse", "HEAD")
    st_ = subprocess.run(["git", "--no-optional-locks", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True)
    if not head or st_.returncode != 0 or st_.stdout.strip():
        why = ("no git HEAD" if not head else f"git status failed: {st_.stderr.strip()[:200]}" if st_.returncode
               else "the tracked tree is dirty")
        sys.exit(f"[coverage shard] REFUSED: {why} in {REPO}")
    import sraging
    if not os.path.realpath(sraging.__file__).startswith(os.path.realpath(SRAGING)):
        sys.exit(f"[coverage shard] REFUSED: sraging imports from {sraging.__file__}, not {SRAGING}")
    jobs = shard_jobs(panel_jobs(), index, of)
    have = set()
    if os.path.exists(rows_out):
        for line in open(rows_out):
            r = json.loads(line)
            if r.get("head") == head and ("error" not in r or r.get("error_origin") == "frozen"):
                have.add((r["cohort_id"], r["calibrator"]))
    os.makedirs(os.path.dirname(os.path.abspath(rows_out)), exist_ok=True)
    for c, cal in jobs:
        if (c, cal) in have:
            continue
        r = fit_worker(c, cal, head)
        with open(rows_out, "a") as fh:
            fh.write(json.dumps(r, default=_jsonable) + "\n")
        print(f"[coverage shard {index}/{of}] {c} {cal}: {'ERROR ' + r['error'][:100] if 'error' in r else 'ok'} "
              f"({r.get('cpu_s', 0):.0f} CPU-s)", flush=True)


def import_rows(paths, head=None) -> dict:
    """The cluster's shard rows into ROWS for the scoring run at the same commit. Every row must carry this HEAD and
    a host. Rows `_load_rows` would keep (ok fits; fits that raised in the frozen code) are appended unless their job
    is already in ROWS at this HEAD; harness and sraging errors are listed, to be re-submitted, never imported."""
    head = head or _git("rev-parse", "HEAD")
    rows = [json.loads(ln) for p in paths for ln in open(p) if ln.strip()]
    wrong = sorted({(r.get("cohort_id"), r.get("calibrator")) for r in rows
                    if r.get("head") != head or not (r.get("host") or {}).get("node")})
    if wrong:
        raise SystemExit(f"[coverage import] REFUSED: {len(wrong)} rows of another HEAD or without a host, e.g. "
                         f"{wrong[:3]}")
    have, new, errors = _load_rows(head), {}, []
    for r in rows:
        key = (r["cohort_id"], r["calibrator"])
        if "error" in r and r.get("error_origin") != "frozen":
            errors.append([*key, r.get("error_origin"), r["error"][:200]])
        elif key not in have and key not in new:
            new[key] = r
    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    with open(ROWS, "a") as fh:
        for r in new.values():
            fh.write(json.dumps(r, default=_jsonable) + "\n")
    done = set(have) | set(new)
    return {"imported": len(new), "already_present": len(set(have) & {(r["cohort_id"], r["calibrator"]) for r in rows}),
            "errors": errors, "missing": [list(j) for j in panel_jobs() if j not in done]}


def fit_hosts(rows) -> dict:
    """Where the scored fits ran: distinct nodes, rows per CPU model, rows without a host."""
    cpus, nodes = {}, set()
    for r in rows:
        h = r.get("host") or {}
        if h.get("node"):
            nodes.add(h["node"])
            cpus[h.get("cpu") or "unknown"] = cpus.get(h.get("cpu") or "unknown", 0) + 1
    return {"n_nodes": len(nodes), "rows_by_cpu": dict(sorted(cpus.items())),
            "rows_without_host": sum(1 for r in rows if not (r.get("host") or {}).get("node"))}


def xhost_jobs(jobs, n=XHOST_N, seed=XHOST_SEED):
    """The Mac subset: `n` jobs per gated cell, the arm read from the cohort id's prefix, in a seeded shuffle."""
    import random
    rng, out = random.Random(seed), []
    for cal, arm in GATED:
        ids = sorted(c for c, k in jobs if k == cal and c.split("-")[0] == arm)
        rng.shuffle(ids)
        out += [(c, cal) for c in ids[:n]]
    return out


def xhost_compare(pairs) -> dict:
    """Per quantity over (Mac row, cluster row) pairs of the same job: claims present on both, shapes that differ,
    hits on the truth that flip, how far the ends moved (dex, ends finite on both), and fits failed on one host."""
    from sraging.claims import combinations as K
    out = {}
    for q in PARAMS + K.KEYS:
        n = shape = flips = one_failed = 0
        shifts = []
        for a, b in pairs:
            ca, cb = claim_of(a, q), claim_of(b, q)
            if (ca == "failed") != (cb == "failed"):
                one_failed += 1
            if not isinstance(ca, dict) or not isinstance(cb, dict):
                continue
            n += 1
            shape += ca["shape"] != cb["shape"]
            for end in ("lower", "upper"):
                x, y = ca.get(end), cb.get(end)
                if x and y and x > 0 and y > 0 and math.isfinite(x) and math.isfinite(y):
                    shifts.append(abs(math.log10(x / y)))
            ha, hb = hit(ca, truth_value(q, a["truth"])), hit(cb, truth_value(q, b["truth"]))
            flips += ha is not None and hb is not None and ha != hb
        out[q] = {"n": n, "shape_differs": shape, "hit_flips": flips, "failed_on_one_host": one_failed,
                  "n_ends": len(shifts), "end_shift_dex_median": st.median(shifts) if shifts else None,
                  "end_shift_dex_max": max(shifts) if shifts else None}
    return out


def phase_xhost(args):
    """The Mac subset (Naveh 2026-10-08, [coverage_host]; reported, not gated): XHOST_N jobs per gated cell fitted
    on this Mac at this HEAD into XHOST_ROWS, compared row by row with the cluster's rows in ROWS (imported first)."""
    import acceptance_003 as A
    head = A.git("rev-parse", "HEAD")
    jobs = xhost_jobs(panel_jobs(A.load_toml(A.PANEL)["coverage"]))
    mac = {}
    if os.path.exists(XHOST_ROWS):
        for line in open(XHOST_ROWS):
            r = json.loads(line)
            if r.get("head") == head and ("error" not in r or r.get("error_origin") == "frozen"):
                mac[(r["cohort_id"], r["calibrator"])] = r
    todo = [j for j in jobs if j not in mac]
    run = A.Run("coverage-xhost", manifests=[A.PANEL], dev=args.dev,
                cost=f"{len(todo)} Mac fits of {len(jobs)} (fit + all six walks, ~5140 CPU-s median each), "
                     f"{args.workers} workers")
    os.makedirs(os.path.dirname(XHOST_ROWS), exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = {ex.submit(fit_worker, c, cal, head): (c, cal) for c, cal in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"cohort_id": futs[f][0], "calibrator": futs[f][1], "head": head, "error": f"crash: {e}",
                     "error_origin": "harness"}
            with open(XHOST_ROWS, "a") as fh:
                fh.write(json.dumps(r, default=A._jsonable) + "\n")
            if "error" not in r or r.get("error_origin") == "frozen":
                mac[(r["cohort_id"], r["calibrator"])] = r
            print(f"  [xhost {i}/{len(todo)}] {r['cohort_id']} {r['calibrator']}: "
                  f"{'ERROR ' + r['error'][:80] if 'error' in r else 'ok'}", flush=True)
    cluster = _load_rows(head)
    pairs = [(mac[j], cluster[j]) for j in jobs if j in mac and j in cluster]
    missing = {"mac": [list(j) for j in jobs if j not in mac], "cluster": [list(j) for j in jobs if j not in cluster]}
    run.write({"status": "reported" if len(pairs) == len(jobs) else "incomplete", "gating": False,
               "jobs": [list(j) for j in jobs], "per_cell": XHOST_N, "seed": XHOST_SEED, "n_pairs": len(pairs),
               "missing": missing, "hosts": {"mac": fit_hosts(list(mac.values())),
                                             "cluster": fit_hosts([b for _, b in pairs])},
               "compare": xhost_compare(pairs)}, name="coverage_xhost")


def phase_coverage(args):
    import acceptance_003 as A
    panel = A.load_toml(A.PANEL)["coverage"]
    jobs = panel_jobs(panel)
    if args.only:
        jobs = [j for j in jobs if j[0] in args.only]
    head = A.git("rev-parse", "HEAD")
    done = _load_rows(head)
    todo = [j for j in jobs if j not in done]
    run = A.Run("coverage", manifests=[A.PANEL], dev=args.dev or bool(args.only),
                cost=f"{price(A, len(todo))} of {len(jobs)}, {args.workers} workers")
    pw = power(library(), panel)
    for cell, d in pw["cells"].items():
        at = {(g["delta"], g["concordance"]): g.get("power") for g in d["grid"] if g["p_ref"] == 0.95 and g["icc"] == 0}
        print(f"  power {cell}: {d['n_cohorts']} cohorts, {d['n_clusters']} clusters; at p 0.95, icc 0: P(loss | no "
              f"loss) = {at.get((0.0, 0.0))} (independent hits), {at.get((0.0, 0.9))} (concordance 0.9); P(loss | "
              f"delta -0.05) = {at.get((-0.05, 0.0))}, {at.get((-0.05, 0.9))}", flush=True)
    problems = prerequisites(A)
    if problems:
        print("[coverage] " + ("REFUSED" if not args.dev else "would refuse (--dev runs anyway)") + ":\n  "
              + "\n  ".join(problems), flush=True)
        if not args.dev:
            run.write({"status": "not_applicable", "reasons": problems, "power": pw})
            sys.exit(3)
    if getattr(args, "score_only", False):         # the cluster's rows (import_rows): never fit here, so the
        todo = []                                  # scored fits all ran on one host family; a missing one fails
    else:
        A.preflight_or_exit(run)
    os.makedirs(os.path.dirname(ROWS), exist_ok=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"), initializer=_init) as ex:
        futs = {ex.submit(fit_worker, c, cal, head): (c, cal) for c, cal in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"cohort_id": futs[f][0], "calibrator": futs[f][1], "head": head, "error": f"crash: {e}",
                     "error_origin": "harness"}
            with open(ROWS, "a") as fh:
                fh.write(json.dumps(r, default=A._jsonable) + "\n")
            if "error" not in r or r.get("error_origin") == "frozen":
                done[(r["cohort_id"], r["calibrator"])] = r
            print(f"  [{i}/{len(todo)}] {r['cohort_id']} {r['calibrator']}: "
                  f"{'ERROR (' + r.get('error_origin', '?') + ') ' + r['error'][:80] if 'error' in r else 'ok'}",
                  flush=True)
    from sraging.compat.jax2 import estimator as JE
    run.write(summarise(jobs, done, JE.box_for(300.0), problems, pw, smoke=bool(args.only), valid=validity(A.T)))


def summarise(jobs, done, box, problems, pw, smoke=False, valid=None) -> dict:
    """The phase's verdict and payload from the rows `done` {(cohort, calibrator): row} of the `jobs`: fail on a loss
    or a missing fit; not_applicable when a gated cell is absent or underpowered, or a prerequisite was refused
    (a --dev run); else pass."""
    rows = [done[j] for j in jobs if j in done]
    sc = score(rows, box, None if valid is None else set(valid["valid"]))
    g = gate_outcome(sc)
    missing = [list(j) for j in jobs if j not in done]
    if g["losses"] or missing:
        verdict = "fail"
    elif g["underpowered"] or g["absent"] or problems:
        verdict = "not_applicable"
    else:
        verdict = "pass"
    reasons = ([f"gated cell absent: {c}" for c in g["absent"]] + [f"underpowered: {u[0]} {u[1]} {u[2]}: {u[3]}"
                                                                  for u in g["underpowered"]] + list(problems))
    ok_rows = [r for r in rows if not failed(r)]
    ratios = [x for x in ((r.get("combinations") or {}).get("ratio_to_fit") for r in ok_rows) if x]
    return {"status": f"smoke-{verdict}" if smoke else verdict, "reasons": reasons, "losses": g["losses"],
            "underpowered": g["underpowered"], "absent_gated_cells": g["absent"], "missing": missing,
            "frozen_code_failures": [(r["cohort_id"], r["calibrator"], r["error"][:200]) for r in rows if "error" in r],
            "scores": sc, "power": pw, "prerequisites": list(problems),
            "validity": None if valid is None else {k: valid[k] for k in ("source", "sha256", "n")} | {
                "n_valid": len(valid["valid"]), "gate_counts": "all cohorts (Naveh 2026-10-08)"},
            "gate": {"cells": [list(c) for c in GATED], "min_informative": MIN_INFORMATIVE,
                     "rule": "per combination vs eta and vs epsilon: loss if z < -2" + (
                         "" if gate_margin()[0] is None else f" or 95% CI lower end <= -{gate_margin()[0]:g}")
                             + "; underpowered below 30 paired cells or 10 clusters (sraging.scoring.paired.gate)"},
            "fit_hosts": fit_hosts(rows),
            "cost": {"fits": len(ok_rows), "cpu_s_total": sum(r.get("cpu_s") or 0.0 for r in ok_rows),
                     "combination_ratio_median": st.median(ratios) if ratios else None}}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="coverage on the cluster: fit one shard, or import the shards' rows")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("shard", help="fit shard INDEX of OF at this checkout's HEAD (one LSF array element)")
    sh.add_argument("--index", type=int, default=int(os.environ.get("LSB_JOBINDEX", "0")))
    sh.add_argument("--of", type=int, required=True)
    sh.add_argument("--rows-out", required=True)
    im = sub.add_parser("import-rows", help="append the shards' rows to the local row log, for --score-only")
    im.add_argument("paths", nargs="+")
    a = ap.parse_args()
    if a.cmd == "shard":
        fit_shard(a.index, a.of, a.rows_out)
    else:
        _init()
        print(json.dumps(import_rows(a.paths), indent=1))
