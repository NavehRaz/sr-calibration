"""L0: the floored dt ceiling, ported into `solver_grid` / `floor_info`: on every library cohort the package's
grid (L, solver kwargs, ceiling, floor record) is rc1's exactly, and the floor binds on exactly the 48
E. coli / C. elegans cohorts of library v1 (Phase A's measured census). Runs in a SUBPROCESS: it imports SRK
(kaplan_meier, auto_grid), which the tester suite's guard forbids in the pytest process."""
from __future__ import annotations

import os

import pytest

import _jax2_testlib as T

pytestmark = pytest.mark.sut


def test_grid_and_floor_match_rc1_on_every_cohort():
    if not os.path.isdir(os.path.join(T.ROOT, "data", "library", "v2", "cohorts")):
        pytest.skip("library v2 cohorts are not built in this checkout")
    out = T.run_py("""
import sys, json, os, importlib.util
import numpy as np
from srbench import npcompat
def load(k, p):
    s = importlib.util.spec_from_file_location(k, p); m = importlib.util.module_from_spec(s)
    sys.modules[k] = m; s.loader.exec_module(m); return m
new = load("g.new", os.path.join("methods", "mle_profile_jax2", "estimator.py"))
old = load("g.rc1", os.path.join("methods", "mle_profile_jax2", "tests", "fixtures", "candidate_rc1", "code", "estimator.py"))
from srbench.library.load import Library
lib = Library(version="v2"); v1 = {s.cohort_id for s in Library(version="v1").specs}
n = mism = 0; binds = []; three = True
for s in lib.specs:
    t, e = lib.data(s.cohort_id); lib._cache.clear()
    d = s.delta if s.delta else None
    a = new._grid(np.asarray(t, float), np.asarray(e, int), d)
    b = old._grid(np.asarray(t, float), np.asarray(e, int), d)
    n += 1
    if repr(a) != repr(b):
        mism += 1
    if a[3]["binds"]:
        binds.append(s.cohort_id)
    if s.cohort_id in ("realfull-real-ecoli-c0-s0", "pure-k0.01-p00-c0-s0"):
        sg = new.solver_grid(t, e, d); fi = new.floor_info(t, e, d)
        three = three and len(sg) == 3 and repr(sg) == repr(a[:3]) and repr(fi) == repr(a[3])
print(json.dumps(dict(n=n, mism=mism, binds_v1=sorted(c for c in binds if c in v1),
                      binds_other=sorted(c for c in binds if c not in v1), three=three)))
""", timeout=1800)
    assert out["_rc"] == 0, out["_stderr"]
    assert out["n"] == 2290 and out["mism"] == 0
    assert out["three"]
    assert len(out["binds_v1"]) == 48, out["binds_v1"]
    assert all(("ecoli" in c) or ("celegans" in c) for c in out["binds_v1"]), out["binds_v1"]
    print(f"\nfloor binds: v1 {len(out['binds_v1'])}, v2-only {len(out['binds_other'])}")
