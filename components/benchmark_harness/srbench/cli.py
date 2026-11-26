"""The command line.

INVARIANTS ACROSS EVERY COMMAND, enforced here rather than remembered:

  * Nothing expensive is computed twice. Every command that could run calibrations prints
    RE-USED vs COMPUTED, before and after.
  * Only calibration costs anything. `score`, `report`, `gates`, `reference set` and friends run
    in seconds from the store and say `0 calibrations performed`.
  * Every coverage figure appears with its decline rate and its denominator.
  * Every deliverable names the reference it used, with version and run id.

Exit codes: 0 success · 1 a check or gate failed · 2 invalid usage or contract violation ·
3 refused before running (budget guard, version mismatch, wrong code location).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED = 0, 1, 2, 3


def _library(args):
    from .library.load import Library, LibraryError
    try:
        return Library(root=getattr(args, "library_root", None) or None)
    except LibraryError as ex:
        print(f"error: {ex}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _library_for(manifest, args=None):
    """The library a RUN was built against -- NOT the default.

    Every command that reads a stored run used a bare `Library()`, so it paired the run's cohorts
    with whatever library happened to be the default. Measured by the mle_profile_jax2 session on
    2026-10-27: under a v5 default, a v6 run's explorer took v5's cv on 1,752 of 1,776 cohorts and
    validity_v5.csv, stamped "library v6", with no warning. Flipping the default to v6 did not fix
    it, it reversed it -- a v5 run then takes v6's cv.

    Nothing failed, which is the problem: the page renders, the numbers are plausible, and the
    stamp says the right library while the data underneath comes from another one. It was being
    worked around by hand in a CONTEXT-SYNC ("pass --no-explorer") by four sessions.

    `--library-root` still wins when given, because that is an explicit instruction. Otherwise the
    manifest decides, and a manifest with no `library_version` falls back to the default rather
    than guessing.
    """
    from .library.load import Library, LibraryError
    root = getattr(args, "library_root", None) or None
    if root:
        try:
            return Library(root=root)
        except LibraryError as ex:
            print(f"error: {ex}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
    want = (manifest or {}).get("library_version")
    try:
        return Library(version=want) if want else Library()
    except LibraryError as ex:
        print(f"error: run names library {want!r}, which is not built here: {ex}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


def _report_for_run(manifest, explicit=None, default=os.path.join("reports", "comparison.md")):
    """The report G10 may check for this RUN: (path, None), or (None, why not).

    `gates` read `reports/comparison.md` whenever --report was not given. That is only where
    `compare` writes without --out, so it holds whichever comparison ran last: on 2026-10-28 a
    2026-09-24 report of mle_profile 1.0.0 against itself on library v1, which G10 then "passed"
    against the v6 reference run. A gate reporting PASS on evidence it never looked at.

    A report belongs to a run when the comparison.json `compare` writes beside it names that run
    and its library. A default report that does not is skipped with the reason, and G10 reports
    not applicable. An explicit --report that does not, or cannot be traced, raises ValueError:
    it was asked for by name, so dropping it would hide the mistake. An explicit --report that
    does not exist stays not applicable, as before -- the mle_profile_jax2 switch recipe passes a
    missing one, and its recorded 6/10/0 must keep reproducing.
    """
    path = explicit or default
    if not os.path.exists(path):
        return None, None
    cj = os.path.join(os.path.dirname(path), "comparison.json")
    try:
        with open(cj, encoding="utf-8") as fh:
            meta = json.load(fh)
        owner = (meta.get("method") or {}).get("run_id")
        lib = meta.get("library_version")
    except (OSError, ValueError, AttributeError):
        why = f"{path} has no readable {cj} beside it, so which run it belongs to is unknown"
    else:
        if owner == manifest.get("run_id") and lib == manifest.get("library_version"):
            return path, None
        why = (f"{path} is the report of run {owner!r} on library {lib}, not of run "
               f"{manifest.get('run_id')!r} on library {manifest.get('library_version')}")
    if explicit:
        raise ValueError(why)
    return None, why


def _store(args):
    from .store.store import ResultStore, default_store_root
    return ResultStore(getattr(args, "store", None) or default_store_root())


# --------------------------------------------------------------------------- methods

def cmd_methods(args) -> int:
    from .contract import method as M
    from .contract.types import ContractError
    try:
        reg = M.registered()
    except ContractError as ex:
        print(f"contract violation: {ex}", file=sys.stderr)
        return EXIT_USAGE
    if not reg:
        print("no methods registered. Add a package under `methods/` that calls "
              "`srbench.contract.method.register(...)` at import time.")
        return EXIT_OK
    bad = 0
    print(f"{'name':<16} {'version':<10} {'contract':<9} {'models':<28} description")
    print("-" * 100)
    for name, m in sorted(reg.items()):
        d = m.descriptor
        try:
            M.check(m)
            status = ""
        except ContractError as ex:
            status, bad = f"  INVALID: {ex}", bad + 1
        caps = ", ".join(d.capabilities) or "(nothing)"
        print(f"{name:<16} {d.version:<10} {d.contract_version:<9} {caps:<28} {d.description}")
        if status:
            print(status)
    return EXIT_FAILED if bad else EXIT_OK


# --------------------------------------------------------------------------- library

def cmd_library(args) -> int:
    lib = _library(args)
    if args.action == "info":
        m = lib.manifest
        print(f"library     {lib.version}   {m['n_cohorts']} cohorts")
        print(f"built       {m['built_at']}  by {m['harness'].get('revision') or '?'}")
        print(f"generator   {m['generator']['resolved_path']} @{m['generator'].get('revision')}")
        print(f"hazard row  {m.get('hazard_row')}")
        print(f"kappas      {m['kappas']}  ({m['n_per_kappa']} points each)")
        print(f"tiers       " + "  ".join(f"{k}={v}" for k, v in m["tiers"].items()))
        print(f"arms        " + "  ".join(f"{k}={v}" for k, v in m["arm_counts"].items()))
        groups = m.get("group_counts") or lib.groups()
        if len(groups) > 1:
            print("groups      " + "  ".join(f"{k}={v}" for k, v in sorted(groups.items())))
            opt = [k for k in sorted(groups) if k != "main"]
            print(f"            `main` is the tiered body; {', '.join(opt)} "
                  f"{'are' if len(opt) > 1 else 'is'} optional -- select with --group NAME")
        if m.get("n_non_binding_windows"):
            print(f"note        {m['n_non_binding_windows']} real fit windows contain all of "
                  "their cohort's deaths (non-binding, and that is the real model)")
        return EXIT_OK

    print(f"verifying {len(lib)} cohorts against the manifest ...")
    res = lib.verify_all(progress=lambda i, n: print(f"  {i}/{n}", flush=True))
    print(f"\nok {res['ok']}   mismatched {res['mismatched']}   missing {res['missing']}")
    if res["mismatched"]:
        print(f"MISMATCHED (first few): {res['mismatched_ids']}")
        print("A mismatch is a FAILURE, not a new version: it is different data wearing the same "
              "identity.")
        return EXIT_FAILED
    if res["missing"]:
        print("Cohort arrays are not committed (large, deterministic). Rebuild with:")
        print("    python -m srbench.library.build")
        return EXIT_FAILED
    return EXIT_OK


# --------------------------------------------------------------------------- store

def cmd_store(args) -> int:
    store = _store(args)
    if args.action == "info":
        n = store.count()
        by_method: dict[str, int] = {}
        for r in store.iter_records():
            k = f"{r.key.method_name} v{r.key.method_version}"
            by_method[k] = by_method.get(k, 0) + 1
        print(f"store   {store.root}")
        print(f"records {n}")
        for k, v in sorted(by_method.items()):
            print(f"  {k:<32} {v}")
        return EXIT_OK

    lib = _library(args)
    from .contract import method as M
    from .run.execute import plan_run
    try:
        meth = M.get(args.method)
    except Exception as ex:  # noqa: BLE001
        print(f"error: {ex}", file=sys.stderr)
        return EXIT_USAGE
    ids = [s.cohort_id for s in _select(lib, args)]
    plan = plan_run(meth, lib, ids, store)
    print(f"requested      {len(ids)} cohorts")
    print(f"already stored {plan.n_reuse}   (this method at v{plan.method_version}, "
          f"library {plan.library_version})")
    print(f"to compute     {plan.n_compute}")
    return EXIT_OK


# --------------------------------------------------------------- merge / migrate

def cmd_merge(args) -> int:
    """Bring cluster results home. The source store is never modified."""
    from .store.merge import merge
    dst = _store(args)
    try:
        report = merge(args.source, dst, dry_run=args.dry_run)
    except (FileNotFoundError, ValueError) as ex:
        print(f"error: {ex}", file=sys.stderr)
        return EXIT_USAGE
    print(f"source      {args.source}   (read-only)")
    print(f"destination {dst.root}")
    if args.dry_run:
        print("DRY RUN -- nothing was written\n")
    print(report.format())
    return EXIT_OK if report.ok else EXIT_REFUSED


def cmd_migrate(args) -> int:
    """Re-key a store in place after a key-schema change. Costs zero calibrations."""
    from .store.migrate import migrate, migrate_manifests
    store = _store(args)

    if args.manifests:
        # Run manifests are lists of digests, so re-keying the store invalidates every one of them
        # until they are repointed. Nothing is recomputed: the results exist, their addresses moved.
        reports = migrate_manifests(args.manifests, store, dry_run=args.dry_run)
        print(f"manifests in {args.manifests}")
        if args.dry_run:
            print("DRY RUN -- nothing was written")
        bad = 0
        for r in reports:
            name = os.path.basename(r.path)
            flag = "" if r.ok else f"   UNRESOLVED {r.unresolved}"
            print(f"  {name:<36} {r.total:>5} keys   "
                  f"remapped {r.remapped:>5}   current {r.already_current:>5}{flag}")
            if not r.ok:
                bad += 1
                for c in r.unresolved_ids[:3]:
                    print(f"      unresolved: {c}")
        print()
        print("0 calibrations performed -- a manifest holds addresses, not results.")
        if bad:
            print(f"\n{bad} manifest(s) have entries pointing at neither a current nor a known old\n"
                  "address. They were left UNCHANGED: repointing them to a guess would silently\n"
                  "attach a run to the wrong result.", file=sys.stderr)
            return EXIT_REFUSED
        return EXIT_OK

    report = migrate(store, dry_run=args.dry_run)
    print(f"store {store.root}")
    if args.dry_run:
        print("DRY RUN -- nothing was written\n")
    print(report.format())
    return EXIT_OK if report.conflicts == 0 else EXIT_REFUSED



# --------------------------------------------------------------------------- run

def _select(lib, args):
    # An optional group is selected BY NAME, never as a prefix, so asking for one cannot be
    # confused with asking for a tier. Mixing the two is refused rather than silently resolved.
    # `--group` may be repeated, and may accompany a tier or a cohort count. The result is the
    # UNION, in library order, de-duplicated.
    #
    # An earlier version refused to combine them, on the grounds that a group is not a prefix.
    # That was right about the mechanics and wrong about the need: a run over "the full tier plus
    # the bign block" is a perfectly well-defined set, and the paired v5 gate asks for exactly
    # that. Forcing two manifests would mean two comparisons that cannot be paired across.
    groups = getattr(args, "group", None) or []
    if isinstance(groups, str):
        groups = [groups]
    if groups:
        from .library.load import LibraryError
        picked = []
        for g in groups:
            try:
                picked.extend(lib.group(g))
            except LibraryError as ex:
                print(f"error: {ex}", file=sys.stderr)
                raise SystemExit(EXIT_USAGE)
        if not (getattr(args, "tier", None) or getattr(args, "cohorts", None)
                or getattr(args, "budget", None)):
            return _dedup(picked)
        base = _resolve_prefix(lib, args)
        return _dedup(list(base) + picked)
    return _resolve_prefix(lib, args)


def _resolve_prefix(lib, args):
    """The prefix half of a selection: a tier, a cohort count, or a budget."""
    from .run.sizing import BudgetRefusal, SizingError, resolve
    per = getattr(args, "_per_cohort_seconds", None)
    budget = None
    if getattr(args, "budget", None):
        budget = _parse_budget(args.budget)
    try:
        sz = resolve(len(lib), tier=getattr(args, "tier", None),
                     cohorts=getattr(args, "cohorts", None),
                     budget_units=budget, per_cohort_units=per,
                     tiers=getattr(lib, "tiers", None))
    except BudgetRefusal as ex:
        print(str(ex), file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)
    except SizingError as ex:
        print(f"error: {ex}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)
    return lib.select(sz.n)


def _dedup(specs):
    """Library order, each cohort once. A cohort named by both a tier and a group is ONE cohort;
    scoring it twice would double its weight in every denominator."""
    seen, out = set(), []
    for sp in sorted(specs, key=lambda x: x.position):
        if sp.cohort_id not in seen:
            seen.add(sp.cohort_id)
            out.append(sp)
    return out


def _parse_budget(s: str) -> float:
    """Budget in CPU-seconds. Accepts `200h`, `30m`, `1200s` or a bare number of hours."""
    s = str(s).strip().lower()
    if s.endswith("h"):
        return float(s[:-1]) * 3600
    if s.endswith("m"):
        return float(s[:-1]) * 60
    if s.endswith("s"):
        return float(s[:-1])
    return float(s) * 3600


def cmd_cost(args) -> int:
    from .contract import method as M
    from .run.probe import format_forecast, measure
    lib = _library(args)
    store = _store(args)
    try:
        meth = M.get(args.method)
    except Exception as ex:  # noqa: BLE001
        print(f"error: {ex}", file=sys.stderr)
        return EXIT_USAGE
    ids = [s.cohort_id for s in lib.select(min(len(lib), args.probe_from))]
    rep = measure(meth, lib, ids, store=store, n_probes=args.n_probes)
    print(rep.format())
    print()
    print(format_forecast(rep.per_cohort_seconds, lib.tiers))
    return EXIT_OK


def cmd_run(args) -> int:
    from .contract import method as M
    from .run.execute import run, save_manifest
    from .run.probe import measure
    from .run.sizing import claim_note

    lib = _library(args)
    store = _store(args)
    try:
        meth = M.get(args.method)
        M.check(meth)
    except Exception as ex:  # noqa: BLE001
        print(f"error: {ex}", file=sys.stderr)
        return EXIT_USAGE

    # A quarantined method is refused HERE as well as in `run.execute.run`. The deeper guard is the
    # authoritative one (it covers programmatic callers); this one exists so the CLI answers with a
    # readable refusal and EXIT_REFUSED instead of a traceback, before any planning output is
    # printed that would suggest the run is about to proceed.
    try:
        M.ensure_runnable(meth)
    except Exception as ex:  # noqa: BLE001
        print(f"refused: {ex}", file=sys.stderr)
        return EXIT_REFUSED

    # A budget needs a MEASURED cost, never a declared one.
    if args.budget:
        probe_ids = [s.cohort_id for s in lib.select(min(len(lib), 50))]
        rep = measure(meth, lib, probe_ids, store=store, n_probes=args.n_probes)
        print(rep.format())
        print()
        args._per_cohort_seconds = rep.per_cohort_seconds

    specs = _select(lib, args)
    if getattr(args, "shard", None):
        # Shard a selection across processes. The store is content-addressed and written
        # atomically, so concurrent shards cannot collide: each writes different keys, and a
        # duplicate write of the same key is a no-op. This is the concurrency the store was
        # designed for, used for real.
        i, n = (int(x) for x in args.shard.split("/"))
        if not (1 <= i <= n):
            print(f"error: --shard must be i/n with 1 <= i <= n", file=sys.stderr)
            return EXIT_USAGE
        specs = [s for j, s in enumerate(specs) if j % n == (i - 1)]
        print(f"shard     {i}/{n}  -> {len(specs)} cohorts")
    from .run.execute import plan_run
    if getattr(args, "only_stored", False):
        # Score what is ALREADY computed, without running anything more. Useful when an expensive
        # batch is still in flight.
        #
        # ⚑ THIS CAN BIAS THE RESULT, and the omission is never silent. Cohorts go missing because
        # they were HARD -- they time out, they get preempted, they are the stiff corner of the
        # library -- so dropping them systematically flatters the method. The count, the arms and
        # the reason are printed here and carried into the run manifest so every downstream figure
        # inherits them.
        full = list(specs)
        plan0 = plan_run(meth, lib, [s.cohort_id for s in full], store)
        have = set(plan0.reuse)
        specs = [s for s in full if s.cohort_id in have]
        dropped = [s for s in full if s.cohort_id not in have]
        if dropped:
            from collections import Counter
            print(f"--only-stored: scoring {len(specs)} of {len(full)} cohorts; "
                  f"{len(dropped)} NOT YET COMPUTED and omitted.")
            print(f"    omitted by arm: {dict(Counter(s.arm for s in dropped))}")
            print("    ⚑ cohorts are missing because they were HARD (timeouts, preemption, the "
                  "stiff corner), so this selection is BIASED IN THE METHOD'S FAVOUR.")
            print("    Re-score once they land -- it costs zero calibrations.")
    plan = plan_run(meth, lib, [s.cohort_id for s in specs], store)
    print(f"method    {meth.descriptor.name} v{meth.descriptor.version}")
    print(f"library   {lib.version}")
    print(f"selection {plan.format()}")
    print(claim_note(specs))
    print()

    t0 = time.time()
    manifest = run(meth, lib, [s.cohort_id for s in specs], store,
                   progress=lambda i, n, el: print(f"  {i}/{n}  ({el:.0f}s)", flush=True))
    manifest["run_id"] = args.run_id or f"{meth.descriptor.name}-{int(t0)}"
    if getattr(args, "only_stored", False):
        manifest["only_stored"] = True
        manifest["omitted_cohorts"] = [s.cohort_id for s in dropped] if "dropped" in dir() else []
    out = args.out or os.path.join("runs", f"{manifest['run_id']}.json")
    save_manifest(manifest, out)
    print(f"\nreused {manifest['n_reused']}   computed {manifest['n_computed']}   "
          f"failed {manifest['n_failed']}   in {manifest['wall_seconds']:.1f}s")
    print(f"run manifest -> {out}")
    _autobuild_explorer(manifest, store, args)
    return EXIT_OK


# ----------------------------------------------------------------- automatic explorer

def _is_the_reference(manifest: dict) -> bool:
    """Is this run the one the reference pointer names?"""
    try:
        from .reference import pointer as P
        ptr = P.get()
        return bool(ptr and ptr.run_id and ptr.run_id == manifest.get("run_id"))
    except Exception:                      # noqa: BLE001
        return False


def _autobuild_explorer(manifest: dict, store, args) -> None:
    """Build the explorer for a run that just finished. NEVER fails the command.

    WHY IT IS AUTOMATIC. The explorer is a deliverable, it costs zero calibrations, and it was a
    step people forgot -- so the page beside a run was routinely older than the run.

    WHERE IT GOES, and why not simply `reports/explorer.html`: several sessions benchmark different
    methods against this repository at once, and a single canonical path means the last run to
    finish silently overwrites everyone else's page. Each run therefore gets
    `reports/<run-id>/explorer.html`, and the canonical `reports/explorer.html` is refreshed only
    for the run the REFERENCE names -- so "the explorer" always means the reference's.

    SKIPPED FOR A SHARD. A cluster element holds a fraction of the cohorts, so 360 of them would
    each write a partial page over the real one.
    """
    if getattr(args, "no_explorer", False):
        return
    if getattr(args, "shard", None):
        print("explorer: skipped for a shard (each element holds only part of the run)")
        return

    from .library.load import Library
    from .explorer import assemble as ASM
    from .explorer import payload as PL
    from .run.execute import load_results

    try:
        ptr = None
        ref_sets = None
        try:
            ptr = P.get()
            if ptr is not None:
                ref_path = os.path.join("runs", f"{ptr.run_id}.json")
                if os.path.exists(ref_path):
                    with open(ref_path, encoding="utf-8") as fh:
                        ref_sets = load_results(json.load(fh), store)
        except Exception:                  # noqa: BLE001 - a missing reference is not fatal here
            ref_sets = None

        sets = load_results(manifest, store)
        p = PL.build_payload(_library_for(manifest, args), sets, manifest, ptr, ref_sets)
        out = os.path.join("reports", str(manifest.get("run_id") or "run"))
        PL.write(p, out)
        ASM.assemble(p, out)
        print(f"explorer -> {os.path.join(out, 'explorer.html')}   (0 calibrations)")
        if _is_the_reference(manifest):
            PL.write(p, "reports")
            ASM.assemble(p, "reports")
            print("explorer -> reports/explorer.html   (this run IS the reference)")
    except Exception as ex:                # noqa: BLE001
        # A finished calibration must not be reported as a failure because a figure could not be
        # drawn. Say what happened and leave the run's own exit code alone.
        print(f"explorer: not built ({type(ex).__name__}: {str(ex)[:160]}). "
              f"The run itself is fine; build it later with `srbench explorer --run {out_hint(args)}`.",
              file=sys.stderr)


def out_hint(args) -> str:
    return getattr(args, "out", None) or "runs/<run-id>.json"


# --------------------------------------------------------------------------- reference

def cmd_reference(args) -> int:
    from .reference import pointer as P
    if args.action == "show":
        ptr = P.get()
        if ptr is None:
            print("no reference set. `srbench reference set --run <manifest.json>`")
            return EXIT_OK
        print(ptr.stamp())
        if ptr.note:
            print(f"note: {ptr.note}")
        return EXIT_OK
    with open(args.run, encoding="utf-8") as fh:
        manifest = json.load(fh)
    ptr = P.set_reference(manifest, note=args.note or "")
    print(f"reference set -> {ptr.stamp()}")
    print("0 calibrations performed: re-pointing the reference is a DATA edit, and every affected "
          "comparison re-scores from the store.")
    # The canonical explorer is the REFERENCE's, so re-pointing must refresh it -- otherwise
    # `reports/explorer.html` silently keeps describing the method that used to be the reference.
    class _A:
        no_explorer = getattr(args, "no_explorer", False)
        shard = None
        out = args.run
    _autobuild_explorer(manifest, _store(args), _A())
    return EXIT_OK


# --------------------------------------------------------------------------- score

def cmd_score(args) -> int:
    from .library.load import Library
    from .reference import pointer as P
    from .run.execute import load_results
    from .score import floors as F
    from .score import verdict as V

    store = _store(args)
    with open(args.run, encoding="utf-8") as fh:
        run_manifest = json.load(fh)
    ref_path = args.reference
    if not ref_path:
        ptr = P.get()
        if ptr is None:
            print("no reference set and none given. `srbench reference set --run <manifest>`",
                  file=sys.stderr)
            return EXIT_USAGE
        ref_path = os.path.join("runs", f"{ptr.run_id}.json")
    with open(ref_path, encoding="utf-8") as fh:
        ref_manifest = json.load(fh)

    try:
        P.require_comparable(run_manifest, ref_manifest)
    except ValueError as ex:
        print(f"REFUSED: {ex}", file=sys.stderr)
        return EXIT_REFUSED

    lib = _library_for(run_manifest, args)
    a = load_results(run_manifest, store)
    b = load_results(ref_manifest, store)
    ids = sorted({cs.cohort_id for cs in a} & {cs.cohort_id for cs in b})
    truths = lib.truths(ids)
    clusters = lib.cluster_map(ids)

    ptr = P.get()
    print(f"method     {run_manifest['method']} v{run_manifest['method_version']}")
    # The reference ACTUALLY USED, not the pointer. `--reference X` scored X while this line
    # printed whatever the pointer named: after the v6 switch, `score --reference
    # runs/jax2-v5-full-bign.json` reported "mle_profile_jax2 v1.0.1 (run jax2-v6-full-bign,
    # library v6)" over v5 numbers. pointer.py's own docstring forbids exactly that, and the
    # 2026-10-27 CONTEXT-SYNC had to tell four sessions to remember the real one themselves.
    _explicit_ref = getattr(args, "reference", None)
    if ptr and not _explicit_ref:
        print(ptr.stamp())
    else:
        print(f"reference  {ref_manifest['method']} v{ref_manifest['method_version']} "
              f"(run {ref_manifest.get('run_id', '?')}, library {ref_manifest['library_version']})")
    print(f"library    {run_manifest['library_version']}   paired cohorts {len(ids)}")
    print("0 calibrations performed -- scoring reads the store.")
    print()

    from .library import order as O
    specs = [lib.spec(c) for c in ids]
    strength = O.claim_strength(specs)

    # From the data, not a literal: a hardcoded tuple here diverges from the scored set the
    # moment an optional quantity exists, and the divergence is silent.
    from .score import claims as _CL
    for param in _CL.scored_quantities(truths):
        axes = {
            "calibration": V.calibration_axis(a, b, truths, param, clusters,
                                              cohort_set=f"{len(ids)} paired"),
            "calibration_one_sided": V.calibration_one_sided_axis(a, b, truths, param, clusters,
                                                                  f"{len(ids)} paired"),
            "availability": V.availability_axis(a, b, param, f"{len(ids)} paired", truths),
            "information": V.information_axis(a, b, param, f"{len(ids)} paired"),
            # floors applied (they were `{}` until 2026-09-25 -- see score/floors.cohort_floors)
            "accuracy": V.accuracy_axis(a, b, truths, param,
                                        F.cohort_floors(dict(zip(ids, specs)), param),
                                        f"{len(ids)} paired"),
            "cost": V.cost_axis(a, b, f"{len(ids)} paired"),
        }
        overall, reasons = V.decide(axes)
        print(f"=== {param} ===  {overall.upper()}")
        for ax in axes.values():
            print("   " + ax.format())
        for r in reasons:
            print(f"   - {r}")
        print()

    if strength["arms_absent"]:
        print(f"UNTESTED REGIMES in this run: {', '.join(strength['arms_absent'])}")
    if not strength["supports_per_arm"]:
        print("This size supports a POOLED verdict only: no per-arm or per-stratum claim is made.")
    return EXIT_OK


# --------------------------------------------------------------------------- deliverables

def _load_pair(args, store):
    """(run manifest, reference manifest, method sets, reference sets) or raise SystemExit."""
    from .reference import pointer as P
    from .run.execute import load_results
    with open(args.run, encoding="utf-8") as fh:
        run_manifest = json.load(fh)
    ref_path = getattr(args, "reference", None)
    ptr = P.get()
    if not ref_path:
        if ptr is None:
            print("no reference set. `srbench reference set --run <manifest>`", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        ref_path = os.path.join("runs", f"{ptr.run_id}.json")
    with open(ref_path, encoding="utf-8") as fh:
        ref_manifest = json.load(fh)
    try:
        P.require_comparable(run_manifest, ref_manifest)
    except ValueError as ex:
        print(f"REFUSED: {ex}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)
    return run_manifest, ref_manifest, load_results(run_manifest, store), \
        load_results(ref_manifest, store), ptr


def cmd_compare(args) -> int:
    """Deliverable A: five axes vs the reference, with run time as a first-class result."""
    from .library.load import Library
    from .report import comparison as CMP
    store = _store(args)
    run_m, ref_m, a, b, ptr = _load_pair(args, store)
    nulls = None
    if args.null_run and os.path.exists(args.null_run):
        from .run.execute import load_results
        with open(args.null_run, encoding="utf-8") as fh:
            nulls = load_results(json.load(fh), store)
    cmp_ = CMP.build_comparison(_library_for(run_m, args), a, b, run_m, ref_m, ptr, nulls)
    jp, mp = CMP.write(cmp_, args.out or "reports")
    print(CMP.render_markdown(cmp_))
    print(f"\n-> {jp}\n-> {mp}")
    print("0 calibrations performed.")
    return EXIT_OK


def cmd_explorer(args) -> int:
    """Deliverable B: a self-contained explorer plus its data as a separate file."""
    from .library.load import Library
    from .explorer import assemble as ASM
    from .explorer import payload as PL
    store = _store(args)
    run_m, ref_m, a, b, ptr = _load_pair(args, store)
    lib = _library_for(run_m, args)
    p = PL.build_payload(lib, a, run_m, ptr, b,
                         validity_path=getattr(args, "validity", None),
                         reference_manifest=ref_m)
    out = args.out or "reports"
    dp = PL.write(p, out)
    hp = ASM.assemble(p, out)
    print(f"payload  -> {dp}   ({os.path.getsize(dp)/1024:.0f} KB)")
    print(f"explorer -> {hp}   ({os.path.getsize(hp)/1024:.0f} KB, self-contained)")
    print("The payload is written separately so a new view can be built against the same data "
          "without regenerating anything.")
    print("0 calibrations performed.")
    return EXIT_OK


def cmd_crosscheck(args) -> int:
    """Re-score the predecessor's stored results with THIS scorer and diff. No calibration."""
    import os as _os
    from .score import crosscheck as X
    root = args.against
    results = _os.path.join(root, "inputs", "mle_k300.json")
    tables = _os.path.join(root, "tables")
    if not (_os.path.exists(results) and _os.path.isdir(tables)):
        print(f"error: expected {results} and {tables}", file=sys.stderr)
        return EXIT_USAGE
    rows = X.compare(results, tables, method=args.method, level=args.level)
    print(X.format_report(rows, results))
    like = [r for r in rows if r.shape_class == "any"]
    bad = [r for r in like if r.delta is None or abs(r.delta) > 5e-4]
    return EXIT_FAILED if bad else EXIT_OK


def cmd_verify(args) -> int:
    """Recompute every claim in a generated document, independently of the generator."""
    from .library.load import Library
    from .report import verify as VER
    store = _store(args)
    md = args.report
    cj = args.comparison or os.path.join(os.path.dirname(md), "comparison.json")
    if not (os.path.exists(md) and os.path.exists(cj)):
        print(f"error: need both {md} and {cj}", file=sys.stderr)
        return EXIT_USAGE
    # The comparison records the library it was built on; recomputing against a different one
    # would "verify" a document by checking it against other cohorts.
    try:
        with open(cj, encoding="utf-8") as fh:
            _cmp_meta = json.load(fh)
    except (OSError, ValueError):
        _cmp_meta = {}
    problems = VER.verify_markdown(md, cj, _library_for(_cmp_meta, args), store)
    print(VER.format_report(problems, md))
    return EXIT_FAILED if problems else EXIT_OK


# --------------------------------------------------------------------------- gates

def cmd_gates(args) -> int:
    """Run the gate registry. Passes, warnings and failures are counted SEPARATELY."""
    from .gates import registry as R
    if getattr(args, "run", None):
        # Against a REAL run: this is where the gates actually guard a result.
        from .library.load import Library
        from .run.execute import load_results
        store = _store(args)
        with open(args.run, encoding="utf-8") as fh:
            mf = json.load(fh)
        lib = _library_for(mf, args)
        # Before loading anything: refusing a --report that is not this run's costs nothing here.
        try:
            rep, why_not = _report_for_run(mf, getattr(args, "report", None))
        except ValueError as ex:
            print(f"REFUSED: {ex}", file=sys.stderr)
            return EXIT_REFUSED
        if why_not:
            print(f"NOTE  G10 did not run: {why_not}. Pass --report.\n", file=sys.stderr)
        nulls = None
        if getattr(args, "null_run", None) and os.path.exists(args.null_run):
            with open(args.null_run, encoding="utf-8") as fh:
                nulls = load_results(json.load(fh), store)
        from .reference import pointer as P      # needed on BOTH paths below, not just the
        refs = None                                 # pointer one -- see require_comparable
        ref_path = getattr(args, "reference_run", None)
        if not ref_path:
            # Default to the registered reference, so G5 and G6 run without being asked for.
            from .reference import pointer as P
            ptr = P.get()
            cand = os.path.join(os.path.dirname(args.run), f"{ptr.run_id}.json") if ptr else None
            ref_path = cand if cand and os.path.exists(cand) else None
        if ref_path and os.path.exists(ref_path) and os.path.abspath(ref_path) != os.path.abspath(args.run):
            with open(ref_path, encoding="utf-8") as fh:
                _ref_mf = json.load(fh)
            # G5 and G6 compare the run against this reference, so pairing across libraries
            # compares different cohorts and calls the result a gate verdict. Every other command
            # refuses it (require_comparable); `gates` did not, so after the v6 switch a v5 run
            # "passed" G5/G6 against the v6 reference.
            try:
                P.require_comparable(mf, _ref_mf)
            except ValueError as ex:
                print(f"REFUSED: {ex}", file=sys.stderr)
                return EXIT_REFUSED
            refs = load_results(_ref_mf, store)
        md = None
        if rep:
            with open(rep, encoding="utf-8") as fh:
                md = R.markdown_table_blocks(fh.read())
        outs = R.run_against(lib, load_results(mf, store), mf, reference_sets=refs,
                             null_sets=nulls, report_markdown=md)
        if refs is None:
            print("NOTE  no reference run supplied or found, so G5 and G6 did not run. "
                  "Pass --reference-run.\n", file=sys.stderr)
        print(R.format_report(outs))
        print(f"\nchecked against run {mf.get('run_id','?')} "
              f"({mf['n_cohorts']} cohorts).  0 calibrations performed.")
        return EXIT_FAILED if R.summarise(outs)["failed"] else EXIT_OK

    gates = R.all_gates()
    outs = []
    for gid, g in gates.items():
        # Without a run to check, each gate is exercised on its own clean fixture -- which also
        # demonstrates it is wired up at all.
        try:
            outs.append(g.run(**g.passing_fixture()))
        except Exception as ex:  # noqa: BLE001
            outs.append(R.GateOutcome(gid, g.title, R.FAIL, f"gate raised: {ex}",
                                      g.motivating_failure, g.blocking))
    print(R.format_report(outs))
    print("\n0 calibrations performed.")
    s = R.summarise(outs)
    return EXIT_FAILED if s["failed"] else EXIT_OK


# --------------------------------------------------------------------------- entry

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="srbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=None, help="result-store root")
    ap.add_argument("--library-root", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("methods", help="list and validate registered methods").set_defaults(
        func=cmd_methods)

    p = sub.add_parser("library", help="inspect or verify the frozen library")
    p.add_argument("action", choices=["info", "verify"])
    p.set_defaults(func=cmd_library)

    p = sub.add_parser("store", help="what is stored; what a run would reuse vs compute")
    p.add_argument("action", choices=["info", "plan"])
    p.add_argument("--method")
    p.add_argument("--tier", choices=["quick", "medium", "full"])
    p.add_argument("--cohorts", type=int)
    # `action="append"` to match `run`. Without it argparse keeps only the LAST --group, so
    # `--group sharp --group nladder` planned 300 cohorts while `run` would compute 632 --
    # the planner answering "what will this cost me?" with the wrong number, silently.
    p.add_argument("--group", action="append", metavar="NAME",
                   help="an optional block by name, e.g. nladder (not a tier). Repeatable; "
                        "several are their union, as in `run`.")
    p.set_defaults(func=cmd_store, budget=None)

    p = sub.add_parser("merge", help="merge another store (e.g. the cluster's) into this one")
    p.add_argument("source", help="path to the source store root -- never modified")
    p.add_argument("--dry-run", action="store_true", help="report what would be merged")
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("migrate", help="re-key this store after a key-schema change")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manifests", metavar="DIR",
                   help="instead of the store, repoint run manifests in DIR at current addresses")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("cost", help="measure what one calibration costs, without committing")
    p.add_argument("--method", required=True)
    p.add_argument("--n-probes", type=int, default=5)
    p.add_argument("--group", action="append", metavar="NAME",
                   help="probe from an optional block, e.g. nladder. Repeatable; several are "
                        "their union, as in `run`.")
    p.add_argument("--probe-from", type=int, default=50,
                   help="draw probes from the first N cohorts")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("run", help="run a method over a cohort selection")
    p.add_argument("--method", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tier", choices=["quick", "medium", "full"])
    g.add_argument("--cohorts", type=int)
    p.add_argument("--group", action="append", metavar="NAME",
                   help="an OPTIONAL block by name, e.g. nladder or bign. Selected WHOLE, never "
                        "as a prefix, so adding one never changes what a tier means. Repeatable, "
                        "and may accompany a tier: `--tier full --group bign` is their union.")
    p.add_argument("--budget", help="e.g. 200h, 30m, 1200s; with a size it acts as a GUARD")
    p.add_argument("--n-probes", type=int, default=5)
    p.add_argument("--shard", help="i/n -- run every n-th cohort of the selection, for parallelism")
    p.add_argument("--only-stored", action="store_true",
                   help="score only cohorts already computed; prints what it omitted and warns "
                        "that the omission is biased toward easy cohorts")
    p.add_argument("--run-id")
    p.add_argument("--no-explorer", action="store_true",
                   help="do not build the explorer for this run (it is built by default)")
    p.add_argument("--out")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("reference", help="show or re-point the reference role")
    p.add_argument("--no-explorer", action="store_true")
    p.add_argument("action", choices=["show", "set"])
    p.add_argument("--run")
    p.add_argument("--note")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("compare", help="deliverable A: five axes vs the reference")
    p.add_argument("--run", required=True)
    p.add_argument("--reference")
    p.add_argument("--null-run", help="a null_prior run, scored beside every coverage")
    p.add_argument("--out")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("explorer", help="deliverable B: standalone explorer + its data file")
    p.add_argument("--validity", metavar="CSV",
                   help="a validity audit to filter by; defaults to reports/validity/"
                        "validity_v6.csv when present")
    p.add_argument("--run", required=True)
    p.add_argument("--reference")
    p.add_argument("--out")
    p.set_defaults(func=cmd_explorer)

    p = sub.add_parser("crosscheck", help="re-score the predecessor's results with this scorer")
    p.add_argument("--against", required=True, help="a final_matched/ directory")
    p.add_argument("--method", default="mle_profile")
    p.add_argument("--level", type=float, default=0.95)
    p.set_defaults(func=cmd_crosscheck)

    p = sub.add_parser("verify", help="recompute every claim in a generated document")
    p.add_argument("--report", required=True)
    p.add_argument("--comparison")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("gates", help="run the gate registry")
    p.add_argument("--run", help="check against a real run rather than the gates' own fixtures")
    p.add_argument("--null-run", help="a null_prior run, so G15 can check coverage against the null")
    p.add_argument("--report", help="the run's rendered comparison.md, so G10 can check its "
                                    "table column counts; refused unless the comparison.json "
                                    "beside it names this run. Defaults to "
                                    "reports/comparison.md when that is this run's")
    p.add_argument("--reference-run", help="a run to compare against, so G5 (fixed-floor "
                                           "accuracy) and G6 (paired cohort set) can run; "
                                           "defaults to the registered reference")
    p.set_defaults(func=cmd_gates)

    p = sub.add_parser("score", help="score a run against the reference")
    p.add_argument("--run", required=True)
    p.add_argument("--reference")
    p.set_defaults(func=cmd_score)
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except SystemExit as ex:
        return int(ex.code or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
