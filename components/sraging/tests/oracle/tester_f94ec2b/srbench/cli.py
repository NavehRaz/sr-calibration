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
    from .run.sizing import BudgetRefusal, SizingError, resolve
    # An optional group is selected BY NAME, never as a prefix, so asking for one cannot be
    # confused with asking for a tier. Mixing the two is refused rather than silently resolved.
    group = getattr(args, "group", None)
    if group:
        if getattr(args, "tier", None) or getattr(args, "cohorts", None):
            print("give either --group or a tier/--cohorts, not both: a group is not a prefix, "
                  "so there is no sensible way to combine them.", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
        from .library.load import LibraryError
        try:
            return lib.group(group)
        except LibraryError as ex:
            print(f"error: {ex}", file=sys.stderr)
            raise SystemExit(EXIT_USAGE)
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
            from .reference import pointer as P
            ptr = P.get()
            if ptr is not None:
                ref_path = os.path.join("runs", f"{ptr.run_id}.json")
                if os.path.exists(ref_path):
                    with open(ref_path, encoding="utf-8") as fh:
                        ref_sets = load_results(json.load(fh), store)
        except Exception:                  # noqa: BLE001 - a missing reference is not fatal here
            ref_sets = None

        sets = load_results(manifest, store)
        p = PL.build_payload(Library(), sets, manifest, ptr, ref_sets)
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

    lib = Library()
    a = load_results(run_manifest, store)
    b = load_results(ref_manifest, store)
    ids = sorted({cs.cohort_id for cs in a} & {cs.cohort_id for cs in b})
    truths = lib.truths(ids)
    clusters = lib.cluster_map(ids)

    ptr = P.get()
    print(f"method     {run_manifest['method']} v{run_manifest['method_version']}")
    print(f"{ptr.stamp() if ptr else 'reference  ' + ref_manifest['method']}")
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
    cmp_ = CMP.build_comparison(Library(), a, b, run_m, ref_m, ptr, nulls)
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
    lib = Library()
    p = PL.build_payload(lib, a, run_m, ptr, b)
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
    problems = VER.verify_markdown(md, cj, Library(), store)
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
        lib = Library()
        nulls = None
        if getattr(args, "null_run", None) and os.path.exists(args.null_run):
            with open(args.null_run, encoding="utf-8") as fh:
                nulls = load_results(json.load(fh), store)
        outs = R.run_against(lib, load_results(mf, store), mf, null_sets=nulls)
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
    p.add_argument("--group", help="an optional block by name, e.g. nladder (not a tier)")
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
    p.add_argument("--group", help="probe from an optional block, e.g. nladder")
    p.add_argument("--probe-from", type=int, default=50,
                   help="draw probes from the first N cohorts")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("run", help="run a method over a cohort selection")
    p.add_argument("--method", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tier", choices=["quick", "medium", "full"])
    g.add_argument("--cohorts", type=int)
    g.add_argument("--group", help="an OPTIONAL block by name, e.g. nladder. Not a tier and not a "
                                   "prefix: it is selected whole, so adding one never changes "
                                   "what a tier means.")
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
