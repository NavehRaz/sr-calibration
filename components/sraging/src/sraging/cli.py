"""`sraging fit-cohort` (T034; contracts/cli.md), and spec 005's `fit-set` and `clean` (specs/005-data-plots-cli/
contracts/cli.md).

fit-cohort's exit codes: 0 fit completed (warnings may have fired) | 2 usage error | 3 CSV refused | 4 unsupported
calibrator, option or runtime | 5 the fit failed (status error:*; JSON still written if requested).
fit-set and clean: 0 ok | 2 usage or a refused option | 3 data refused | 4 a fit failed (fit-set) | 5 other.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import sys

_DELTA_WHY = ("delta (the recording resolution) changes how every death is scored: with it, a death "
              "counts as happening somewhere inside its recording interval; without it, at exactly that "
              "age. Pass --delta D in the ages' units, or --continuous if the ages are exact times.")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        if "--delta" in message or "--continuous" in message:
            message = f"{message}\n{_DELTA_WHY}"
        super().error(message)


def build_parser() -> argparse.ArgumentParser:
    from sraging import registry
    p = _Parser(prog="sraging", description="The SR (saturating-removal) ageing model.")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser)
    f = sub.add_parser("fit-cohort", help=f"fit one cohort (default calibrator {registry.DEFAULT})",
                       description=f"Fit one survival cohort (CSV: age, event) with a calibrator (default "
                                   f"{registry.DEFAULT}). See contracts/cli.md.")
    f.add_argument("--csv", required=True, help="two columns: age, event (1 death, 0 right-censored)")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--delta", type=float, help="recording resolution, in the ages' units (> 0)")
    g.add_argument("--continuous", action="store_true", help="the ages are exact, continuous times")
    f.add_argument("--json", help="also write the full FitResult as JSON here")
    f.add_argument("--name", help="cohort label (default: the CSV file name)")
    f.add_argument("--time-unit", default="input units", help="label for theta-hat's time unit")
    f.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"),
                   help="fit the time range [LO, HI] (HI may be inf), as the tester's window rule applies it; only "
                        "profile_jax2_w (h = 0) and profile_jax2_h (h fitted) fit a window (spec 006)")
    f.add_argument("--calibrator", default=registry.DEFAULT, help=f"NAME[@VERSION] (default {registry.DEFAULT}; also "
                                                                    "profile_jax2_h, profile_v033)")
    cg = f.add_mutually_exclusive_group()
    cg.add_argument("--combinations", choices=("lead", "all"), default=None,
                    help="jax2 calibrators: 'lead' (the default, the frozen lead group) or 'all' (all six); "
                         "profile_v033 gives none")
    cg.add_argument("--no-combinations", action="store_true",
                    help="skip the combination claims (jax2 calibrators; profile_v033 gives none)")
    _add_fit_set(sub)
    _add_clean(sub)
    return p


def _add_input(q, *, cage):
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="one long CSV (columns age/death times, event/events, and the cohort column)")
    src.add_argument("--dir", help="a directory of CSVs (one cohort per file, named by its stem, unless --cohort-col)")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--delta", type=float, help="recording resolution, in the ages' units (> 0)")
    g.add_argument("--continuous", action="store_true", help="the ages are exact, continuous times")
    q.add_argument("--time-unit", required=True, help="the ages' unit, passed into every cohort read (FR-009)")
    q.add_argument("--cohort-col", help="the column naming each row's cohort")
    if cage:
        q.add_argument("--cage-col", required=True, help="the column naming each row's cage")


def _add_fit_set(sub):
    from sraging import registry
    q = sub.add_parser("fit-set", help="fit every cohort of a set, one spawned process per fit (spec 005)",
                       description="Fit each cohort of a long CSV or a directory in its own process, W at a time; "
                                   "results do not depend on W. See specs/005-data-plots-cli/contracts/cli.md.")
    _add_input(q, cage=False)
    q.add_argument("--calibrator", default=registry.DEFAULT, help=f"NAME[@VERSION] (default {registry.DEFAULT})")
    q.add_argument("--window", nargs=2, type=float, metavar=("LO", "HI"),
                   help="fit every cohort on [LO, HI] (HI may be inf); profile_jax2_w or profile_jax2_h (spec 006)")
    q.add_argument("--workers", type=int, default=1, help="processes at a time (default 1); provenance only")
    q.add_argument("--out", required=True, help="writes <cohort>.json, summary.csv and run.json here")


def _add_clean(sub):
    q = sub.add_parser("clean", help="the MAD cage cleaning: flags, removes only with --drop (spec 005)",
                       description="Judge each cage against the others of its cohort (sraging.cleaning). Writes "
                                   "cleaning.csv and cleaning.json; a cleaned set only with --drop.")
    _add_input(q, cage=True)
    q.add_argument("--k", type=float, required=True, help="the radius in robust sigmas (required: no default)")
    q.add_argument("--axes", choices=("both", "median"), default="both", help="'median': judge on the median alone")
    q.add_argument("--floors", help="PHI_ML,PHI_ST given (default: pooled over cohorts with >= --min-cages cages)")
    q.add_argument("--min-cages", type=int, default=10, help="cages a cohort needs to enter the pooled floors")
    q.add_argument("--spread-by", metavar="COLUMN",
                   help="pool the own-spread null over the cohorts sharing this design column's value (default: "
                        "every cohort)")
    q.add_argument("--seed", type=int, required=True, help="the noise baseline's seed (required)")
    q.add_argument("--out", required=True, help="writes cleaning.csv and cleaning.json here")
    q.add_argument("--drop", action="store_true", help="also write cleaned/<cohort>.csv without the flagged cages")


def _fit(cohort, calibrator, version, combinations=None):
    import sraging
    return sraging.fit_cohort(cohort, calibrator=calibrator, version=version, combinations=combinations)


def _fmt(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "" if v is None else repr(v)
    return f"{v:.5g}"


def _print_result(r, args, out):
    from sraging.calibrators.profile_v033 import REFERENCE_COST_SECONDS as cost
    import sraging
    c = r.cohort
    res = "continuous" if c.get("continuous") else f"delta={c.get('delta'):g}"
    cal = r.provenance.get("calibrator", {})
    if cal.get("name") == "profile_v033":
        cost_text = (f"reference cost median {cost['median']:.0f} CPU-s, range {cost['min']:.0f}–{cost['max']:.0f} "
                     f"CPU-s")
    else:                              # spec 003: the calibrator's own measured cost (no declared reference cost)
        cost_text = f"fit cost {_fmt(r.timing.get('cpu_s'))} CPU-s (measured)"
    print(f"sraging {sraging.__version__} · calibrator {cal.get('name')}@{cal.get('version')} · "
          f"n={c['n']} ({c['n_deaths']} deaths, {c['n_censored']} censored) · {res} · {cost_text}",
          file=out)
    th = r.theta_hat.as_dict() if r.theta_hat is not None else {}
    print(f"{'parameter':10s} {'claim':12s} {'theta-hat':>12s} {'lower':>12s} {'upper':>12s}  label",
          file=out)
    for cl in r.claims:
        print(f"{cl.parameter:10s} {cl.shape:12s} {_fmt(th.get(cl.parameter, cl.point)):>12s} {_fmt(cl.lower):>12s} "
              f"{_fmt(cl.upper):>12s}  {cl.label}", file=out)
    print(f"loglik {r.loglik!r}  (wall {_fmt(r.timing.get('wall_s'))} s, cpu {_fmt(r.timing.get('cpu_s'))} s)",
          file=out)
    wl = window_line(r)
    if wl:
        print(wl, file=out)
    _print_jax2(r, out)
    if cal.get("name") == "profile_v033":       # spec 003 FR-018
        print("note: " + sraging.NO_COMBINATIONS_NOTE.format(name="profile_v033"), file=out)
    for w in r.warnings:
        print(f"WARNING {w.code} {w.message}", file=out)
    print("checks: " + " ".join(f"{k.code}={'ran' if k.ran else 'NOT-RUN'}/{'fired' if k.fired else 'quiet'}"
                                for k in r.checks), file=out)
    if r.status != "ok":
        print(f"RESULT FAILED — {r.status}: {r.diagnostics.get('exception', '')}", file=out)
    elif r.all_starts_failed:
        print("RESULT FAILED — every optimiser start failed (D-015): v0.3.3 still reports status ok with four "
              "`none` claims, but nothing here is an estimate", file=out)
    elif r.clean:
        print("RESULT ok — no warnings fired", file=out)
    elif r.warnings:
        print(f"RESULT ok — {len(r.warnings)} WARNING(S): {', '.join(w.code for w in r.warnings)} — read "
              f"before using these numbers", file=out)
    elif r.leverage_not_run:
        print("RESULT ok — the leverage check NOT RUN (this result was written before spec 003f)", file=out)
    else:
        print("RESULT ok — warning checks NOT RUN — this result has not been checked for the known "
              "v0.3.3 failure signatures", file=out)
    p = r.provenance
    pk, g = p.get("package", {}), p.get("grid", {})
    print(f"provenance: sraging {pk.get('version')} @ {str(pk.get('git_sha'))[:12]}"
          f"{' (dirty)' if pk.get('dirty') else ''} · numerics {p.get('numerics')} · engine {p.get('engine')} · "
          f"grid N={g.get('N')} dt={g.get('dt')} tau_max={g.get('tau_max')} dt_ceiling={g.get('dt_ceiling')}",
          file=out)


#: spec 006 (FR-024): profile_v033 runs its own leverage check; a v033 result without the block predates it
V033_LEVERAGE_LINE = "leverage: NOT RUN (a profile_v033 result written before spec 006)"


def window_refused(calibrator: str):
    """Spec 006 FR-012: why `calibrator` ("NAME[@VERSION]") cannot fit a window, or None when it can (profile_jax2_w,
    and profile_jax2_h from 1.0.1) or is not registered (the fit names that)."""
    from sraging import registry
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.errors import UnknownCalibratorError
    name, _, version = calibrator.partition("@")
    try:
        spec = registry.get(name, version or None)
    except UnknownCalibratorError:
        return None
    if spec.name in PJ.WINDOW_FITTERS and spec.version != "1.0.0":
        return None
    return (f"{spec.name}@{spec.version} does not fit a window; use profile_jax2_w (h = 0) or profile_jax2_h "
            "(h fitted), which apply the tester's window rule (spec 006)")


def window_line(r):
    """The report's window line (spec 006, contracts/cli.md): the window used, the hi mapping, the rows dropped below
    lo and the deaths the window's edge censored; None for a fit with no window."""
    w = (r.diagnostics or {}).get("window")
    if not w:
        return None
    lo, hi = w["window_eff"]
    given = ", ".join(x if isinstance(x, str) else f"{x:g}" for x in w["window_given"])
    mapped = f"; hi = inf mapped to {w['hi_used']:g}" if w.get("hi_mapped_from") else ""
    return (f"window: [{lo:g}, {hi:g}] used (given [{given}]{mapped}) · {w['n_dropped_below_lo']} row(s) dropped "
            f"below lo · {w['n_deaths_censored']} death(s) censored at the window's edge "
            f"({w['n_censored_at_hi_eff']} record(s) at hi_eff)")


def _leverage_line(r):
    """Spec 003f: the leverage check's one line for a jax2 result, and since spec 006 (FR-024) for profile_v033's own;
    a result without the block says it was not run ("no warnings fired" must not read as "the leverage check
    passed"); None for anything else."""
    v033 = ((r.provenance or {}).get("calibrator") or {}).get("name") == "profile_v033"
    if (r.provenance or {}).get("numerics") != "jax2" and not v033:
        return None
    b = (r.diagnostics or {}).get("leverage")
    if b is None:
        return V033_LEVERAGE_LINE if v033 else "leverage: NOT RUN (result written before spec 003f)"
    c = b.get("cost") or {}
    share = c.get("share_of_fit")
    cost = f"cost {_fmt(c.get('cpu_s'))} s CPU" + (f" ({share:.0%} of the fit)" if share is not None else "")
    g = b.get("gap_sd") or {}
    gap = f"; gap/sd {_fmt(g.get('g'))} > 2 (an isolated latest death)" if g.get("flagged") else ""
    d, ra = b.get("dropped"), b.get("ratios")
    if not d or not ra:
        return f"leverage: {b.get('label')} -- {b.get('reason')}{gap}; {cost}"
    q = {k: (ra.get(k) or {}).get("ratio") for k in ("eta_over_eps", "rbar_over_eta", "eta", "epsilon")}
    kind = "death" if d.get("event") == 1 else "censored"
    start = {"mle_stage": "the MLE stage", "warm": "the warm start"}.get((b.get("refit") or {}).get("start"), "?")
    tie = f", tied with {b['n_tied_top'] - 1} more" if (b.get("n_tied_top") or 1) > 1 else ""
    return (f"leverage: {b.get('label')} -- dropped row {d.get('row')} (age {d.get('age'):g}, {kind}{tie}); "
            f"eta/eps {_fmt(q['eta_over_eps'])}x, rbar/eta {_fmt(q['rbar_over_eta'])}x, eta {_fmt(q['eta'])}x, "
            f"eps {_fmt(q['epsilon'])}x; refit gains {_fmt(b.get('delta_total'))} nats (by {start}){gap}; {cost}")


def _print_jax2(r, out):
    """Spec 003 (contracts/cli.md): what only a jax2 fit has, and its notes; for profile_v033 only its leverage line
    (its own check since spec 006)."""
    line = _leverage_line(r)
    if line is not None:
        print(line, file=out)
    for n in (r.diagnostics or {}).get("notes") or []:        # spec 003b: notes are not warnings
        if n.get("code") != "SHARP_COHORT_LIMIT":              # printed on the DT_RULE line (spec 003g N2)
            print(f"note: {n.get('message')}", file=out)
    if r.mechanisms is None and r.nuisance is None:
        return
    assumption = (r.diagnostics or {}).get("assumptions")
    if assumption:
        print(f"note: {assumption}", file=out)
    if r.nuisance is not None:
        n = r.nuisance
        print(f"external hazard (nuisance, not scored): h={_fmt(n.get('h_ext'))} per time unit, "
              f"h*L={_fmt(n.get('hL'))}, ExtH={_fmt(n.get('ExtH'))}", file=out)
    m = r.mechanisms or {}
    bf, dtf = m.get("basin_refit") or {}, m.get("dt_floor") or {}
    print(f"mechanisms: N={m.get('N')} · basin re-fit {'fired (' + str(bf.get('rounds')) + ' rounds)' if bf.get('fired') else 'not fired'}"
          f" · censor-tie relabel {m.get('relabelled_rows', 0)} rows · floored dt {'binds' if dtf.get('binds') else 'no'}"
          f" · widened {', '.join(m.get('widened') or []) or 'none'} · Nelder-Mead fallbacks {m.get('n_fallback', 0)}",
          file=out)
    rule = ((r.diagnostics or {}).get("dt_floor") or {}).get("grid_rule") or {}
    if rule.get("dt_in_scope"):                                # spec 003g FR-013
        print(f"DT_RULE: in scope (CV_c {rule.get('cv_c'):.4g} < 0.05); dt capped at CV/80", file=out)
        for n in (r.diagnostics or {}).get("notes") or []:
            if n.get("code") == "SHARP_COHORT_LIMIT":
                print(f"  limits in this regime: {n.get('message')}", file=out)
    from sraging.claims.combinations import KEYS
    co = r.combinations or {}
    if not co.get("on"):
        print("combination claims: off", file=out)
    else:
        lead, n = co.get("lead_group") or [], len(co.get("order") or [])
        which = (f"the lead group: {', '.join(lead)}; --combinations all adds the other {len(KEYS) - len(lead)}"
                 if co.get("mode") == "lead" else f"all {n}")
        ratio = co.get("ratio_to_fit")
        print(f"combination claims ({which}): {_fmt(co.get('cost_s'))} CPU-s"
              f"{f' ({ratio:.2g}x the fit)' if ratio else ''}; order {co.get('order_source')}", file=out)
    lc = co.get("combination_claims_contradicted") or []
    if lc:
        ends = ", ".join(f"{x['parameter']} {x['side']}" for x in lc)
        print(f"note: the later walks evaluated points beyond {len(lc)} lead claim end(s) within the threshold "
              f"({ends}); the lead claims are the default fit's, unchanged", file=out)
    pc = co.get("param_claims_contradicted") or []
    if pc:
        ends = ", ".join(f"{x['parameter']} {x['side']}" for x in pc)
        print(f"note: the combination walks evaluated points beyond {len(pc)} per-parameter claim end(s) within the "
              f"threshold ({ends}); the per-parameter claims are the port's, unchanged", file=out)
    lsr, bl = (r.diagnostics or {}).get("L_SR") or {}, (r.diagnostics or {}).get("box_lsr") or {}
    if lsr.get("L_SR"):
        sym = {"eta": "eta*L^2", "beta": "beta*L", "epsilon": "eps*L", "kappa": "kappa", "h": "h*L"}
        print(f"L_SR = {_fmt(lsr['L_SR'])} {lsr.get('time_unit') or ''} (the SR mean lifetime at theta-hat); box in "
              f"L_SR units: " + ", ".join(f"{sym[p]} in [{_fmt(v[0])}, {_fmt(v[1])}]" for p, v in bl.items() if p in sym),
              file=out)
    if r.gap_nats:
        print(f"gap: the best point the fit evaluated beats theta-hat by {r.gap_nats:.3g} nats "
              f"({'above' if r.gap_nats > 1.92 else 'below'} the 1.92-nat threshold)", file=out)


def main(argv=None) -> int:
    from sraging.errors import (CohortDataError, CohortSpecError, RuntimeConfigError,
                                UnknownCalibratorError, UnsupportedOptionError)
    from sraging.cohort import Cohort

    args = build_parser().parse_args(argv)
    if args.delta is not None and not (math.isfinite(args.delta) and args.delta > 0):
        build_parser().error(f"argument --delta: must be a finite number > 0, got {args.delta!r}")
    if args.command == "fit-set":
        return _main_fit_set(args)
    if args.command == "clean":
        return _main_clean(args)
    name, _, version = args.calibrator.partition("@")
    why = window_refused(args.calibrator) if args.window else None
    if why:                                    # spec 006 FR-012: before any read or fit
        print(f"sraging: {why}", file=sys.stderr)
        return 4
    try:
        cohort = Cohort.from_csv(args.csv, delta=args.delta, continuous=args.continuous, name=args.name,
                                 window=None if args.window is None else tuple(args.window))
    except (CohortDataError, CohortSpecError, OSError) as ex:
        print(f"sraging: CSV refused: {ex}", file=sys.stderr)
        return 3
    combos = False if args.no_combinations else args.combinations
    if combos is not None and name == "profile_v033":
        flag = "--no-combinations" if args.no_combinations else f"--combinations {args.combinations}"
        print(f"note: {flag} ignored: profile_v033 gives no combination claims", file=sys.stderr)
        combos = None
    try:
        r = _fit(cohort, name, version or None, combos)
    except (UnknownCalibratorError, UnsupportedOptionError, RuntimeConfigError) as ex:
        print(f"sraging: {ex}", file=sys.stderr)
        return 4
    except Exception as ex:  # noqa: BLE001 -- a jax2 calibrator re-raises what the frozen code re-raises
        if str(ex).startswith("REFUSING"):
            print(f"sraging: {ex}", file=sys.stderr)
            return 4
        print(f"sraging: the fit failed on cohort {cohort.name!r}: {type(ex).__name__}: {ex}", file=sys.stderr)
        return 5
    if r.theta_hat is not None and args.time_unit != r.theta_hat.time_unit:
        r = dataclasses.replace(r, theta_hat=dataclasses.replace(r.theta_hat, time_unit=args.time_unit))
    _print_result(r, args, sys.stdout)
    if args.json:
        r.to_json(args.json)
        print(f"-> {args.json}")
    return 0 if r.status == "ok" and not r.all_starts_failed else 5


def _read_set(args, cage_col=None):
    """(CohortSet, None) or (None, exit code 3) after printing why: a refused file, or any refused row."""
    from sraging.cohorts import read_cohorts
    from sraging.errors import CohortDataError, CohortSpecError
    try:
        s, refused = read_cohorts(args.csv or args.dir, delta=args.delta, continuous=args.continuous,
                                  time_unit=args.time_unit, cohort_col=args.cohort_col, cage_col=cage_col,
                                  window=tuple(args.window) if getattr(args, "window", None) else None)
    except (CohortDataError, CohortSpecError, OSError) as ex:
        print(f"sraging: data refused: {ex}", file=sys.stderr)
        return None, 3
    if refused:
        print(f"sraging: data refused: {len(refused)} row(s) cannot be used (fix the file; nothing was skipped):",
              file=sys.stderr)
        for r in refused:
            print(f"  {r.file}:{r.row}: {r.reason}", file=sys.stderr)
        return None, 3
    for n in s.notes:
        print(f"note: {n}", file=sys.stderr)
    return s, None


def _main_fit_set(args) -> int:
    """`sraging fit-set` (FR-021, FR-009): exit 4 when any cohort's fit failed (the others complete)."""
    from sraging.errors import UnknownCalibratorError, UnsupportedOptionError
    from sraging.fitset import fit_set
    if args.workers < 1:
        build_parser().error(f"argument --workers: must be >= 1, got {args.workers}")
    why = window_refused(args.calibrator) if args.window else None
    if why:                                    # spec 006 FR-012: before any read or fit
        print(f"sraging: {why}", file=sys.stderr)
        return 2
    s, code = _read_set(args)
    if s is None:
        return code
    try:
        run = fit_set(s, workers=args.workers, calibrator=args.calibrator, out_dir=args.out)
    except (UnknownCalibratorError, UnsupportedOptionError) as ex:
        print(f"sraging: {ex}", file=sys.stderr)
        return 2
    for f in run.fits:
        print(f"{f.name}: {f.status}" + (f" -- {f.error}" if f.error else f" -> {f.path}"))
    print(f"summary -> {run.summary_path} ({sum(f.status == 'ok' for f in run.fits)} of {len(run.fits)} ok)")
    return 0 if run.status == "ok" else 4


def _main_clean(args) -> int:
    """`sraging clean` (FR-022): the report always; the cleaned set only with --drop."""
    import csv
    import json
    import os
    from dataclasses import fields
    from sraging.cleaning import CageVerdict, clean, drop
    from sraging.errors import CohortDataError, CohortSpecError, UnsupportedOptionError
    if args.csv and not args.cohort_col:
        build_parser().error("argument --cohort-col: required with --csv (the column naming each row's cohort)")
    floors = None
    if args.floors is not None:
        try:
            floors = tuple(float(x) for x in args.floors.split(","))
            assert len(floors) == 2
        except (ValueError, AssertionError):
            build_parser().error(f"argument --floors: PHI_ML,PHI_ST, got {args.floors!r}")
    s, code = _read_set(args, cage_col=args.cage_col)
    if s is None:
        return code
    try:
        report = clean(s, by="cohort", k=args.k, axes=args.axes, floors=floors, min_cages=args.min_cages,
                       spread_by=args.spread_by, seed=args.seed)
    except UnsupportedOptionError as ex:
        print(f"sraging: {ex}", file=sys.stderr)
        return 2
    except (CohortDataError, CohortSpecError) as ex:
        print(f"sraging: data refused: {ex}", file=sys.stderr)
        return 3
    os.makedirs(args.out, exist_ok=True)
    cols = [f.name for f in fields(CageVerdict)]
    with open(os.path.join(args.out, "cleaning.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in report.cages:
            w.writerow([repr(v) if isinstance(v, float) else v for v in (getattr(r, c) for c in cols)])
    d = report.as_dict()
    d.pop("cages")
    import sraging
    d.update(report_sha256=report.sha256, provenance={"sraging": sraging.__version__, "input": args.csv or args.dir,
                                                      "cage_col": args.cage_col, "cohort_col": args.cohort_col,
                                                      "time_unit": args.time_unit})
    with open(os.path.join(args.out, "cleaning.json"), "w") as fh:
        json.dump(d, fh, indent=1, allow_nan=True, default=str)
        fh.write("\n")
    flagged = report.flagged
    print(f"cleaning: {len(report.cages)} cages in {len(report.cohorts)} cohorts, {len(flagged)} flagged at k = "
          f"{report.k:g} ({report.axes}); floors {report.floors} ({report.floors_source}); report sha256 "
          f"{report.sha256}")

    def expected(n):
        return f"{n['expected_flags']:.2f}" if math.isfinite(n["expected_flags"]) else n["reason"]
    for c in report.cohorts:
        f, o = c.noise
        pool = "all cohorts" if o["spread_by"] is None else f"{o['spread_by']} = {o['pool']}"
        print(f"  {c.cohort}: {c.n_flagged} of {c.n_cages} flagged; expected from scatter like the floor: "
              f"{expected(f)}; like the group's own spread ({pool}: {o['pool_cohorts']} cohorts, "
              f"{o['pool_cages']} cages): {expected(o)}")
    if args.drop:
        kept = drop(s, report)
        out = os.path.join(args.out, "cleaned")
        os.makedirs(out, exist_ok=True)
        for co, sub in kept.by("cohort").items():
            with open(os.path.join(out, f"{str(co).replace(os.sep, '__')}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["age", "event", "cage"])
                for key in sub:
                    cage = sub.design[key].get("cage", key)
                    for a, e in zip(sub[key].ages, sub[key].events):
                        w.writerow([repr(float(a)), int(e), cage])
        print(f"cleaned set -> {out} (dropped {len(flagged)}; report sha256 {report.sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
