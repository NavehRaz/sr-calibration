"""Spec 003d T045 (FR-018, R13, SC-011): the re-test of spec 003's censored coverage, and the truth witness.

    .venv/bin/python tools/retest_003d.py generate          # Mac: the 55 + 12 cohorts by the tagged generator
    .venv/bin/python tools/retest_003d.py check             # every array against the manifest (also on WEXAC)
    python tools/retest_003d.py fit --index I --of N --out shard.jsonl    # WEXAC: one LSF element's fits

generate. Spec 003's 55 censored coverage cohorts (panel_003.toml [coverage].profile_jax2, the censor arm; their specs,
seeds and truth from J2's library v2) and the 12 witness cohorts (panel_003d.toml [witness], library v5), each
materialised by `sraging.generator` with the generator's bytes as tagged (generator.py, its point table, the engine
and compat equal to `sraging-gen-1`'s, else refused), into .cache/out/retest_003d/<id>.npz with a manifest: digest
(the tester's cohort_digest), truth (the tester's single conversion: eta = rho_eta / L^2, beta = rho_beta / L,
epsilon = rho_eps / L), calibrator, caveats, realised censored fraction. The 55 hazard cohorts wait for _h 1.0.1
(amended 2026-10-14, Naveh).

fit. Spec 003's coverage fit (`acceptance_003_coverage.fit_worker`'s row: claims, combinations without the walks,
CPU, gap, warnings, the negligible floor) on the arrays instead of the v2 loader, each array sha-checked against the
manifest first, with combinations "first3" (D1). Beside it the witness: the log-likelihood at the truth on the fit's
own main objective, and D = 2 (ll_fit - ll_truth) (SC-011).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))

OUT = os.path.join(REPO, ".cache", "out", "retest_003d")
MANIFEST = os.path.join(OUT, "manifest.json")
TAG = "sraging-gen-1"
GENERATOR_PATHS = ("src/sraging/generator.py", "src/sraging/generator_point_table.json", "src/sraging/engine",
                   "src/sraging/compat", "src/sraging/errors.py")
MODE = "first3"
CALIBRATOR = "profile_jax2"


def _panel(rel):
    return tomllib.load(open(os.path.join(REPO, rel), "rb"))


def panel_ids() -> list[dict]:
    """The re-test's cohorts in order: [{cohort_id, kind, source}]. kind "retest" (spec 003's censored coverage
    cohorts, library v2) or "witness" (R11's 12, library v5)."""
    cov = _panel("tests/acceptance/panel_003.toml")["coverage"]
    p3d = _panel("tests/acceptance/panel_003d.toml")
    if p3d["retest"]["now"] != "censor":
        raise ValueError(f"retest: panel_003d [retest].now is {p3d['retest']['now']!r}; this tool runs 'censor'")
    retest = [c for c in cov[CALIBRATOR] if c.startswith("censor-")]
    witness = list(p3d["witness"]["sharp"]) + list(p3d["witness"]["nladder"])
    return ([{"cohort_id": c, "kind": "retest", "source": "v2"} for c in retest] +
            [{"cohort_id": c, "kind": "witness", "source": "v5"} for c in witness])


def physical_truth(row) -> dict:
    """The tester's single conversion site (srbench/library/spec.py physical_truth), at xc = 1 and L = L_phys."""
    L = float(row["L_phys"])
    return {"eta": row["rho_eta"] / (L * L), "beta": row["rho_beta"] / L, "epsilon": row["rho_eps"] / L,
            "kappa": float(row["kappa"])}


def _git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True)


def generator_as_tagged() -> list[str]:
    """Every generator path that differs from the tag (empty: the tag's bytes)."""
    r = _git("diff", "--name-only", TAG, "HEAD", "--", *GENERATOR_PATHS)
    if r.returncode != 0:
        return [f"git diff against {TAG} failed: {r.stderr.strip()}"]
    dirty = _git("status", "--porcelain", "--untracked-files=no", "--", *GENERATOR_PATHS).stdout.split("\n")
    return [ln for ln in r.stdout.split("\n") if ln] + [ln for ln in dirty if ln]


def library_rows() -> dict:
    """{source: {cohort_id: manifest row}} for libraries v2 (J2) and v5 (tester main), read-only."""
    import oracle_jax2 as OJ
    from v5_reproduce_003d import V5
    out = {}
    for src, root in (("v2", OJ.LIBRARY_V2), ("v5", V5)):
        out[src] = {s["cohort_id"]: s for s in json.load(open(os.path.join(root, "manifest.json")))["specs"]}
    return out


def generate(args):
    from acceptance_003d import spec_from_library
    from v5_reproduce_003d import cohort_digest
    if _git("status", "--porcelain", "--untracked-files=no").stdout.strip() and not args.dev:
        sys.exit("REFUSED: the tracked tree is dirty (commit first, or --dev for a run that never counts)")
    moved = generator_as_tagged()
    if moved and not args.dev:
        sys.exit(f"REFUSED: the generator differs from {TAG}: {moved}")
    from sraging import generator as G
    libs = library_rows()
    os.makedirs(OUT, exist_ok=True)
    rows, c0 = [], time.process_time()
    for p in panel_ids():
        lib = libs[p["source"]][p["cohort_id"]]
        co = G.materialize(spec_from_library(lib))
        np.savez(os.path.join(OUT, f"{p['cohort_id']}.npz"), ages=co.death_times, events=co.events)
        rows.append({**p, "calibrator": CALIBRATOR, "arm": lib["arm"], "cluster": lib.get("cluster_key"),
                     "n": int(lib["n"]), "delta": float(lib["delta"]), "L_phys": float(lib["L_phys"]),
                     "censor_frac": lib.get("censor_frac") or 0.0, "truth": physical_truth(lib),
                     "digest": cohort_digest(co.death_times, co.events), "censor_time": co.censor_time,
                     "realised_censored_fraction": co.realised_censored_fraction, "caveats": list(co.caveats),
                     "generator_version": co.provenance["generator_version"]})
        print(f"  {p['cohort_id']}: n {co.events.size}, censored {co.realised_censored_fraction:.3f}", flush=True)
    man = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "repo_head": _git("rev-parse", "HEAD").stdout.strip(),
           "tag": TAG, "tag_commit": _git("rev-parse", f"{TAG}^{{commit}}").stdout.strip(),
           "generator_differs_from_tag": moved, "dev": bool(args.dev), "mode": MODE, "cpu_s": time.process_time() - c0,
           "cohorts": rows}
    with open(MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    print(f"wrote {len(rows)} cohorts and {os.path.relpath(MANIFEST, REPO)}")


def load_checked(row, root=OUT):
    """The cohort's arrays, refused unless their digest equals the manifest's."""
    from v5_reproduce_003d import cohort_digest
    z = np.load(os.path.join(root, f"{row['cohort_id']}.npz"))
    ages, events = z["ages"].astype(float), z["events"].astype(int)
    got = cohort_digest(ages, events)
    if got != row["digest"]:
        raise ValueError(f"sha check failed for {row['cohort_id']}: {got[:12]} != manifest {row['digest'][:12]}")
    return ages, events


def check(args):
    man = json.load(open(MANIFEST))
    bad = []
    for row in man["cohorts"]:
        try:
            load_checked(row)
        except (OSError, ValueError) as ex:
            bad.append(f"{row['cohort_id']}: {ex}")
    print("\n".join(bad) or f"all {len(man['cohorts'])} arrays match the manifest (generated at "
                                f"{man['repo_head'][:7]}, {man['tag']})")
    sys.exit(1 if bad else 0)


def fit_one(row, head, root=OUT):
    """Spec 003's coverage row for one manifest cohort, fitted from its sha-checked arrays in mode "first3", with the
    truth witness on the fit's own main objective."""
    import acceptance_003_coverage as COV
    out = {"cohort_id": row["cohort_id"], "calibrator": row["calibrator"], "head": head, "host": COV.host(),
           "kind": row["kind"], "arm": row["arm"], "cluster": row["cluster"], "informative": True,
           "truth": row["truth"], "digest": row["digest"]}
    try:
        ages, events = load_checked(row, root)
        import sraging
        from sraging import Cohort
        from sraging.calibrators import profile_jax2 as PJ
        cohort = Cohort.from_arrays(ages, events, delta=row["delta"], name=row["cohort_id"])
    except Exception as ex:  # noqa: BLE001
        return {**out, "error": f"{type(ex).__name__}: {ex}", "error_origin": "harness",
                "trace": traceback.format_exc()[-1500:]}
    try:
        c0 = time.process_time()
        with PJ.capture_objectives() as objs:
            r = sraging.fit_cohort(cohort, calibrator=row["calibrator"], combinations=MODE)
        cpu = time.process_time() - c0
    except Exception as ex:  # noqa: BLE001
        return {**out, "error": f"{type(ex).__name__}: {ex}", "error_origin": COV.error_origin(ex),
                "trace": traceback.format_exc()[-1500:]}
    out.update({"status": r.status, "n_deaths": r.cohort["n_deaths"], "claims": [c.to_dict() for c in r.claims],
                "combinations": {k: v for k, v in (r.combinations or {}).items() if k != "walks"}, "cpu_s": cpu,
                "gap_nats": r.gap_nats, "warnings": [w.code for w in r.warnings], "loglik": r.loglik})
    try:
        out["negligible_floor"] = COV.negligible_floor(r, cohort)
    except Exception as ex:  # noqa: BLE001 -- recorded; the negligible scores then say "not measured"
        out["negligible_floor"] = {"why": f"{type(ex).__name__}: {ex}"}
    obj = next((o for o in objs if o.NP == len(PJ.PARAMS)), None)       # the fit's main objective (first created)
    if obj is not None and r.loglik is not None:
        ll_t = -float(obj.nll(np.log([float(row["truth"][p]) for p in PJ.PARAMS])))
        out["witness"] = {"loglik_truth": ll_t, "D": 2.0 * (float(r.loglik) - ll_t)}
    else:
        out["witness"] = {"why": "no main objective or no loglik"}
    return out


def fit(args):
    import acceptance_003_coverage as COV
    COV._init()
    man = json.load(open(MANIFEST))
    rows = sorted(man["cohorts"], key=lambda r: r["cohort_id"])
    mine = [r for i, r in enumerate(rows) if i % args.of == args.index - 1]
    head = _git("rev-parse", "HEAD").stdout.strip()
    for row in mine:
        res = fit_one(row, head)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(res, default=COV._jsonable) + "\n")
        print(f"{row['cohort_id']}: {'ERROR ' + res['error'] if 'error' in res else 'ok, cpu %.0f s' % res['cpu_s']}",
              flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["generate", "check", "fit"])
    ap.add_argument("--dev", action="store_true", help="generate on a dirty tree or a moved generator (never counts)")
    ap.add_argument("--index", type=int, help="fit: this element (1-based)")
    ap.add_argument("--of", type=int, help="fit: the number of elements")
    ap.add_argument("--out", help="fit: the JSONL file rows are appended to")
    args = ap.parse_args(argv)
    {"generate": generate, "check": check, "fit": fit}[args.phase](args)


if __name__ == "__main__":
    main()
