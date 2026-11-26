"""fit_set: one spawned process per fit, W at a time (spec 005 US6; FR-021; research R8; HR-6 in small; tasks T033).
Two tiny synthetic cohorts and a planted failing one, with tests/unit/fitset_fast_calibrator.py (seconds per fit): W =
1 and W = 2 give identical results excluding timing (and equal the in-process fit); the failure is recorded with its
error while the others complete; W and the pids are provenance only."""
import csv
import json
import os

import numpy as np
import pytest

import fitset_fast_calibrator as FAST
from sraging.cohort import Cohort
from sraging.cohorts import CohortSet
from sraging.errors import UnsupportedOptionError
from sraging.fitset import FitSetRun, fit_set


def _cohort(name, scale):
    q = (np.arange(60) + 0.5) / 60
    ages = np.round(scale * (-np.log1p(-q)) ** 0.25) + 1.0
    return Cohort.from_arrays(ages, np.where(np.arange(60) % 9 == 8, 0, 1), delta=1.0, name=name, time_unit="days")


SET = CohortSet.from_members([(n, _cohort(n, s), {"cohort": n}) for n, s in (("a", 50.0), ("bad", 55.0),
                                                                             ("b", 70.0))], time_unit="days")


def _strip(d):
    """A result JSON without what may differ between processes: timing."""
    return {k: v for k, v in d.items() if k != "timing"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = {}
    for w in (1, 2):
        d = tmp_path_factory.mktemp(f"w{w}")
        out[w] = (fit_set(SET, workers=w, calibrator=FAST.SPEC, out_dir=str(d)), d)
    return out


def test_w1_and_w2_give_identical_results_equal_to_the_in_process_fit(runs):
    (r1, d1), (r2, d2) = runs[1], runs[2]
    for name in ("a", "b"):
        j1, j2 = (json.load(open(os.path.join(d, f"{name}.json"))) for d in (d1, d2))
        assert _strip(j1) == _strip(j2), name
        inproc = json.loads(FAST.fit(SET[name]).to_json())
        assert _strip(j1) == _strip(inproc), name
    assert [f.name for f in r1.fits] == ["a", "bad", "b"]


def test_a_failing_cohort_is_recorded_with_its_error_and_the_others_complete(runs):
    r, d = runs[2]
    assert isinstance(r, FitSetRun) and r.status == "error"
    by = {f.name: f for f in r.fits}
    assert by["bad"].status == "error" and "planted failure" in by["bad"].error and by["bad"].path is None
    assert by["a"].status == by["b"].status == "ok" and not os.path.exists(os.path.join(d, "bad.json"))
    rows = list(csv.DictReader(open(os.path.join(d, "summary.csv"))))
    assert [x["cohort"] for x in rows] == ["a", "bad", "b"] and rows[1]["status"] == "error"
    assert "planted failure" in rows[1]["error"] and float(rows[0]["loglik"]) < 0


def test_w_and_the_pids_are_provenance_only(runs):
    (r1, d1), (r2, d2) = runs[1], runs[2]
    assert r1.provenance["workers"] == 1 and r2.provenance["workers"] == 2
    assert set(r2.provenance["pids"]) == {"a", "bad", "b"} and len(set(r2.provenance["pids"].values())) == 3
    assert os.getpid() not in r2.provenance["pids"].values()                       # each fit in its own process
    for d in (d1, d2):
        text = open(os.path.join(d, "summary.csv")).read() + open(os.path.join(d, "a.json")).read()
        assert "pid" not in text and "workers" not in text
    assert json.load(open(os.path.join(d2, "run.json")))["workers"] == 2


def test_bad_options_are_refused_before_any_process_starts(tmp_path):
    with pytest.raises(UnsupportedOptionError, match="workers"):
        fit_set(SET, workers=0, calibrator=FAST.SPEC, out_dir=str(tmp_path))
    with pytest.raises(Exception, match="no_such"):
        fit_set(SET, calibrator="no_such", out_dir=str(tmp_path))
    with pytest.raises(TypeError, match="out_dir"):
        fit_set(SET, calibrator=FAST.SPEC)


def test_a_registered_calibrator_reaches_a_spawned_worker_by_name_and_version():
    """Pilot P1 (2026-11-08): the registry's fits are closures (`_make_fit.<locals>.fit`), which do not pickle, so every
    spawned fit of the default calibrator failed while the fast test calibrator (a top-level function) passed. A
    registered spec now travels as (name, version) and the worker resolves it from the registry."""
    import multiprocessing as mp
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    from sraging import registry
    from sraging.fitset import _carried
    spec = registry.get(registry.DEFAULT)
    with pytest.raises(Exception, match="local object"):                 # why: the closure itself does not pickle
        pickle.dumps(spec.fit)
    carried = _carried(spec)
    assert carried == (spec.name, spec.version) and pickle.loads(pickle.dumps(carried)) == carried
    assert _carried(FAST.SPEC) is FAST.fit                                # an unregistered spec travels as its fit
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        got = ex.submit(FAST.describe, carried).result()
    assert got == (spec.fit.__module__, spec.fit.__qualname__)


def test_an_unregistered_calibrator_whose_fit_cannot_pickle_is_refused_before_any_process_starts(tmp_path):
    from sraging.registry import CalibratorSpec
    local = CalibratorSpec(name="local", version="1", numerics="test", settings={}, fit=lambda c: FAST.fit(c))
    out = tmp_path / "out"
    with pytest.raises(UnsupportedOptionError, match="pickle"):
        fit_set(SET, calibrator=local, out_dir=str(out))
    assert not out.exists()
