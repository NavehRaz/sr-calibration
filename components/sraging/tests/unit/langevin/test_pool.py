"""Multi-core Langevin (tasks T060; research R15, R17): a sample depends only on (key, i), so it is bit-identical for
any worker count and chunk size, nested in n; HR-8's plant (B4: drop n mod W) differs; simulate_many keeps input order.

W = 2 here, not T060's 7: worker processes are kept at W <= 2 while hr3 runs on this Mac (the coordinator's load rule,
2026-10-30); the invariance holds by construction for any W, and HR-8 (tasks T123) measures W in {1, 7, 15}."""
import numpy as np
import pytest

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

from sraging import SRParams  # noqa: E402
from sraging.langevin import LangevinConfig, simulate, simulate_many  # noqa: E402
from sraging.langevin import pool as LP  # noqa: E402

P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
W = 2


def sample_bytes(s):
    return s.times.tobytes(), s.ages.tobytes(), s.events.tobytes()


@pytest.mark.parametrize("n", [1_003, 25_013])
def test_one_worker_and_two_are_bit_identical(n):
    a = simulate(P, n=n, key=17, dt=1.0, error="ks", workers=1)
    b = simulate(P, n=n, key=17, dt=1.0, error="ks", workers=W)
    assert sample_bytes(a) == sample_bytes(b) and b.provenance["workers"] == W


def test_chunk_1000_and_333_are_bit_identical():
    a = simulate(P, n=2_500, key=17, dt=1.0, error="ks", chunk=1000)
    b = simulate(P, n=2_500, key=17, dt=1.0, error="ks", chunk=333, workers=W)
    assert sample_bytes(a) == sample_bytes(b)


def test_the_first_50_of_a_2000_draw_are_the_50_draw():
    a = simulate(P, n=2_000, key=23, dt=1.0, error="ks")
    b = simulate(P, n=50, key=23, dt=1.0, error="ks")
    assert a.times[:50].tobytes() == b.times.tobytes()


def test_hr8s_plant_dropping_n_mod_w_differs():
    cfg = LangevinConfig(params=P, n=1_003, key=17, dt=1.0, error="ks")
    real = simulate(P, n=1_003, key=17, dt=1.0, error="ks")
    plant = LP.drop_remainder_times(cfg, 4)                                     # B4: 1,000 real + 3 padded
    assert plant.tobytes() != real.times.tobytes() and plant[:1000].tobytes() == real.times[:1000].tobytes()
    twin = LP.drop_remainder_times(LangevinConfig(params=P, n=1_000, key=17, dt=1.0, error="ks"), 4)
    assert twin.tobytes() == real.times[:1000].tobytes()                        # nothing to drop: the twin


def test_simulate_many_returns_in_input_order_each_config_its_own_key():
    cfgs = [LangevinConfig(params=P, n=300, key=k, dt=1.0, error="ks") for k in (5, 1, 9)]
    out = simulate_many(cfgs, workers=W)
    for c, s in zip(cfgs, out):
        assert s.provenance["key"] == c.key
        assert s.times.tobytes() == simulate(P, n=300, key=c.key, dt=1.0, error="ks").times.tobytes()
    assert len({s.times.tobytes() for s in out}) == 3
