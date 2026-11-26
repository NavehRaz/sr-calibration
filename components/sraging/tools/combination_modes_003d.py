"""Spec 003d T042-T043 (FR-022, D1): the reference outputs proving that the additive "first3" mode leaves
`profile_jax2.combination_claims` byte-identical in modes "lead" and "all".

    .venv/bin/python tools/combination_modes_003d.py capture     # at the parent commit of T044, in a worktree

For each quick cohort and mode: a fresh fit (`run_port`, objectives captured, as the public fit runs it), then
`combination_claims` in that mode. The output is the canonical JSON (`sraging.result.canon`: floats as float.hex) of
everything `combination_claims` returns (claims, walks, extents, counts, order) except timings (`cost_s` and the
tester's timing leaves), with the fit's theta-hat and loglik beside it. The quick cohorts are the 300-death test
cohort and the same cohort censored at its 80th-percentile age (the re-test is censored; the deaths tied with the
censoring age take the relabel path). Byte identity is claimed on this host and the pinned environment only, as for
every parity test.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "tests", "unit", "data", "combination_modes_003d")
SMALL = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")
COHORTS = ("small", "small_censored")
MODES = ("lead", "all")
NAME = "profile_jax2"


def cohort(name):
    from sraging import Cohort
    c = Cohort.from_csv(SMALL, delta=1.0)
    if name == "small":
        return c
    if name == "small_censored":
        a = np.asarray(c.ages, float)
        C = float(np.sort(a)[int(0.8 * a.size)])
        ev = np.where(a > C, 0, np.asarray(c.events, int))
        return Cohort.from_arrays(np.minimum(a, C), ev, delta=1.0, name="small_censored")
    raise ValueError(f"unknown cohort {name!r}")


def output(name, mode) -> str:
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.compat.v033 import runtime
    from sraging.result import canon
    runtime.check()
    out = PJ.run_port(cohort(name), NAME, capture=True)
    cc = PJ.combination_claims(out, NAME, None, mode=mode)
    rec = {"theta": list(out["result"]["theta"]), "loglik": out["result"]["loglik"],
           "combinations": {**cc, "claims": [c.to_dict() for c in cc["claims"]]}}
    return json.dumps(canon(PJ.strip_timing(rec, skip=("cost_s",))), sort_keys=True)


def capture(args):
    git = lambda *a: subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    if git("status", "--porcelain", "--untracked-files=no") and not args.dev:
        sys.exit("REFUSED: the tracked tree is dirty (the reference must be captured at a committed parent)")
    os.makedirs(DATA, exist_ok=True)
    import importlib.metadata as md
    import platform
    import time
    rec = {"repo_head": git("rev-parse", "HEAD"), "tree_dirty": bool(git("status", "--porcelain",
                                                                          "--untracked-files=no")),
           "profile_jax2_sha": git("rev-parse", "HEAD:src/sraging/calibrators/profile_jax2.py"),
           "python": sys.version.split()[0], "packages": {p: md.version(p) for p in ("jax", "jaxlib", "numpy", "scipy")},
           "host": platform.node(), "machine": platform.machine(), "outputs": {}}
    for name in COHORTS:
        for mode in MODES:
            c0 = time.process_time()
            s = output(name, mode)
            with open(os.path.join(DATA, f"{name}__{mode}.json"), "w") as fh:
                fh.write(s)
            rec["outputs"][f"{name}__{mode}"] = {"bytes": len(s), "cpu_s": time.process_time() - c0}
            print(f"{name} {mode}: {len(s)} bytes, {rec['outputs'][f'{name}__{mode}']['cpu_s']:.0f} CPU-s", flush=True)
    with open(os.path.join(DATA, "capture.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["capture"])
    ap.add_argument("--dev", action="store_true", help="capture on a dirty tree (never a reference)")
    args = ap.parse_args(argv)
    capture(args)


if __name__ == "__main__":
    main()
