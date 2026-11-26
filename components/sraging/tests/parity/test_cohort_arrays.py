"""Cohort arrays are byte-identical to the tester's Observation arrays, on both paths (T038; FR-023, SC-002).

Direct: `cohort_from_observation(obs)` vs `np.asarray(obs.ages, float)` / `np.asarray(obs.events, int)`.
CSV: the observation written with repr floats and read back by `Cohort.from_csv`; and the reference
reader (`SRK.fit_cohort.read_csv`) on the same file gives the same bytes (FR-003).
"""
import numpy as np
import pytest

import oracle_v033 as O
from parity_checks import arrays_diff
from sraging.adapters.observation import cohort_from_observation
from sraging.cohort import Cohort

pytestmark = pytest.mark.parity


def write_csv(path, obs):
    with open(path, "w") as fh:
        fh.write("death times,events\n")
        for a, e in zip(obs.ages, obs.events):
            fh.write(f"{float(a)!r},{int(e)}\n")


@pytest.fixture(scope="module")
def quick(oracle_v033):
    return O.quick_tier()


def test_quick_tier_is_the_30_reference_cohorts(quick):
    assert len(quick) == 30 and quick[0] == "pure-k0.01-p00-c0-s0" and quick[-1] == "pure-k0.01-p07-c0-s1"


@pytest.mark.parametrize("i", range(30))
def test_direct_and_csv_paths(oracle_v033, quick, tmp_path, i):
    obs = O.observation(quick[i])
    ref_a, ref_e = np.asarray(list(obs.ages), float), np.asarray(list(obs.events), int)
    c = cohort_from_observation(obs)
    assert arrays_diff(c.ages, c.events, ref_a, ref_e, "direct") == []
    p = tmp_path / f"{obs.cohort_id}.csv"
    write_csv(p, obs)
    delta = obs.delta if obs.delta else None
    c2 = Cohort.from_csv(p, delta=delta, continuous=delta is None)
    assert arrays_diff(c2.ages, c2.events, ref_a, ref_e, "csv") == []
    ra, re_ = oracle_v033.read_csv(str(p))                       # the reference reader, same file
    assert arrays_diff(c2.ages, c2.events, ra, re_, "reference reader") == []
