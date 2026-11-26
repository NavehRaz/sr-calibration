"""Seeded damage paths at requested ages (tasks T091; research R24, R25; data-model "LangevinSample"): X[i, j] is NaN
iff individual i is dead by age j, finite values lie in [0, xc), recording draws nothing, the paths are the same for
any worker count, the ages sit on the dt grid, the memory is capped, the sample saves with a sidecar, and the clamp's
lump at the wall is measured and warned.

W = 2 here, not T091's 7: worker processes are kept at W <= 2 while hr3 runs (the coordinator's load rule)."""
import hashlib
import json
import math

import numpy as np
import pytest

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

from sraging import SRParams  # noqa: E402
from sraging.errors import UnsupportedOptionError  # noqa: E402
from sraging.langevin import grid_dt, simulate  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1, h_ext=2e-3)
AGES = [20.0, 100.0, 130.0]


@pytest.fixture(scope="module")
def smp():
    return simulate(P, n=1_003, key=9, dt=0.5, record_ages=AGES, error="ks")


def test_x_is_nan_iff_dead_external_deaths_included(smp):
    X = smp.paths.X
    assert X.shape == (1_003, 3) and smp.paths.ages.tolist() == AGES
    dead = smp.times[:, None] <= np.asarray(AGES)[None, :]
    assert np.array_equal(np.isnan(X), dead)
    t_max = smp.provenance["t_max"]
    assert np.any(dead & (smp.times[:, None] < t_max))


def test_finite_values_lie_in_zero_to_xc(smp):
    x = smp.paths.X[np.isfinite(smp.paths.X)]
    assert x.size and np.all(x >= 0) and np.all(x < P.xc)


def test_recording_draws_nothing():
    a = simulate(P, n=500, key=9, dt=0.5, error="ks")
    b = simulate(P, n=500, key=9, dt=0.5, record_ages=AGES, error="ks")
    assert a.times.tobytes() == b.times.tobytes() and a.ages.tobytes() == b.ages.tobytes()


def test_the_paths_are_the_same_for_any_worker_count(smp):
    b = simulate(P, n=1_003, key=9, dt=0.5, record_ages=AGES, error="ks", workers=2, chunk=300)
    assert b.paths.X.tobytes() == smp.paths.X.tobytes() and b.times.tobytes() == smp.times.tobytes()


def test_ages_off_the_dt_grid_are_refused_naming_them_and_dt():
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, n=10, key=1, dt=0.3, record_ages=[10.0, 10.05], error="ks")
    assert ex.value.option == "record_ages" and "10.05" in str(ex.value) and "0.3" in str(ex.value)
    simulate(P, n=10, key=1, dt=0.3, record_ages=[0.0, 9.9, 30.0], error="ks")                    # the twin
    assert abs(35 / grid_dt([35, 70], 0.01) - round(35 / grid_dt([35, 70], 0.01))) < 1e-9


def test_a_negative_age_or_one_past_t_max_is_refused():
    with pytest.raises(UnsupportedOptionError):
        simulate(P, n=10, key=1, dt=0.5, record_ages=[-0.5], error="ks")
    with pytest.raises(UnsupportedOptionError):
        simulate(P, n=10, key=1, dt=0.5, t_max=50.0, record_ages=[60.0], error="ks")


def test_paths_above_the_memory_cap_are_refused_naming_it():
    with pytest.raises(UnsupportedOptionError) as ex:
        simulate(P, n=1000, key=1, dt=0.5, record_ages=AGES, max_path_bytes=1000 * 3 * 8 - 1, error="ks")
    assert ex.value.option == "max_path_bytes" and "24,000" in str(ex.value)
    simulate(P, n=1000, key=1, dt=0.5, record_ages=AGES, max_path_bytes=1000 * 3 * 8, error="ks")   # the twin
    with pytest.raises(UnsupportedOptionError):                         # doubled while the coupled rung's exist
        simulate(P, n=1000, key=1, dt=0.5, record_ages=AGES, max_path_bytes=1000 * 3 * 8)


def test_save_writes_an_npz_and_a_sidecar_with_its_sha256(smp, tmp_path):
    out = smp.save(tmp_path / "s")
    npz, side = out["npz"], out["sidecar"]
    meta = json.load(open(side))
    assert meta["npz_sha256"] == hashlib.sha256(open(npz, "rb").read()).hexdigest()
    assert meta["provenance"]["engine"] == "langevin"
    z = np.load(npz)
    assert set(z.files) == {"times", "ages", "events", "record_ages", "X"} and np.array_equal(
        np.isnan(z["X"]), np.isnan(smp.paths.X))


def test_the_clamps_lump_is_measured_and_warned_and_reflect_has_none():
    young = [4.0]
    c = simulate(SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1), n=4000, key=3, dt=0.5, record_ages=young,
                 error="ks")
    r = simulate(SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1), n=4000, key=3, dt=0.5, record_ages=young,
                 error="ks", wall="reflect")
    (pc,), (pr,) = c.error.paths, r.error.paths
    assert pc["p0"] > 0 and pr["p0"] == 0.0 and pc["wall"] == "clamp" and pr["wall"] == "reflect"
    atom = [w for w in c.warnings if w.code == "LANGEVIN_PATHS_WALL_ATOM"]
    assert (pc["p0"] > pc["crit"] / 10) == bool(atom) and not any(w.code == "LANGEVIN_PATHS_WALL_ATOM"
                                                                    for w in r.warnings)
    assert pc["x_c"] >= 5 * math.sqrt(2 * 8e-3 * 0.5) and pc["u_c"] == max(0.01, 3 * pc["p0"])


def test_the_paths_statement_per_age(smp):
    rows = smp.error.paths
    assert [r["age"] for r in rows] == AGES
    for r in rows:
        assert set(r) >= {"age", "n_alive", "wall", "p0", "u_c", "x_c", "ks_D_censored", "ks_D_full", "crit",
                          "p_value", "verdict", "step_bias", "reference_grid_error"}
        assert r["n_alive"] == int(np.sum(np.isfinite(smp.paths.X[:, AGES.index(r["age"])])))
        assert r["ks_D_censored"] <= r["ks_D_full"] + 1e-15
        assert r["step_bias"] is None                                           # error="ks": no coupled rung
