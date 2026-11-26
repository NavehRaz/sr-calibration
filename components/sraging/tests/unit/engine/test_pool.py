"""Nodes over worker processes (tasks T027; research R6; data-model "Node pool"; contracts/python-api.md §3).

`NodePool(ev, workers=W)` spawns W processes, each rebuilding the evaluator's per-node programs from its construction
arguments. Node k goes to worker k mod W; the parent folds strictly in k = 0..M-1 (`pool.in_order`), so a pooled
value and gradient are byte-identical to the in-process ones for every W, M not divisible by W included. The pool is
bound to one evaluator configuration, refuses W above the cores, a closed pool, and a reply set that is not range(M)
each once (B4's failure mode, for nodes)."""
import os

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine import mixture as MIX
from sraging.engine import pool as POOL
from sraging.engine import quadrature as Q
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity
from sraging.engine.pool import NodePool
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 300)) + 1.0)
E = (RNG.random(300) < 0.9).astype(int)
C = Cohort.from_arrays(T, E, delta=1.0)
GRID = GridPolicy("fixed", N=200, dt=2.0 ** -8, tau_max=2.5, L=124.0)
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
HET = Heterogeneity(parameter="xc", sd=0.2, M=7, check=False)            # M = 7: divisible by neither 2 nor 3


def ev(het=HET, **kw):
    return Evaluator(C, numerics="v1", grid=GRID, heterogeneity=het, **kw)


def as_bytes(r):
    return r.value.hex(), {k: v.hex() for k, v in r.grad.items()}


@pytest.fixture(scope="module")
def serial():
    e = ev()
    return e, e.mixture_value(P, grad=True, het_sd_grad=True)


@pytest.mark.parametrize("W", [1, 2, 3])
def test_a_pooled_value_and_gradient_are_byte_identical_to_in_process(serial, W):
    e, r0 = serial
    with NodePool(e, workers=W) as pool:
        r = e.mixture_value(P, grad=True, het_sd_grad=True, pool=pool)
        assert as_bytes(r) == as_bytes(r0)
        assert r.cost_record["workers"] == W and r.cost_record["worker_cpu_s"] > 0
        assert set(r.cost_record["compile_cpu_s_workers"]) == set(range(W))
        assert pool.workers == W and len(pool.started) == W and pool.evaluator_digest == e.digest()
        rt = pool.runtime
    assert all(x["PJRT_NPROC"] == "1" and x["x64"] for x in rt) and len(rt) == W     # each worker's runtime
    assert pool.closed and not any(POOL._alive(p) for p in pool.started)             # no process left behind


@pytest.mark.parametrize("W", [1, 2, 3])
def test_dropping_the_remainder_nodes_changes_the_bytes(serial, W):
    """HR-4's plant (B4's failure mode): the mixture over the first M - (M mod W) nodes only."""
    e, r0 = serial
    rule = Q.truncated_normal_rule(0.2, 7)
    plant = POOL.drop_remainder_value(e, P, rule, W)
    if 7 % W:
        assert plant.hex() != r0.value.hex() and abs(plant - r0.value) > 1e-3     # 0.72 nats here [M]
    else:
        assert plant.hex() == r0.value.hex()                     # nothing to drop: the twin of the plant


def test_the_parent_refuses_a_reply_set_that_is_not_range_m_each_once():
    p = {"S_i": np.ones(3)}
    good = [(0, p, 0.0), (2, p, 0.0), (1, p, 0.0)]                                   # out of order: buffered
    assert [k for k, *_ in POOL.in_order(good, 3)] == [0, 1, 2]
    with pytest.raises(POOL.NodeIndicesError, match="node 1 twice"):
        list(POOL.in_order([(0, p, 0.0), (1, p, 0.0), (1, p, 0.0), (2, p, 0.0)], 3))
    with pytest.raises(POOL.NodeIndicesError, match=r"missing \[2\]"):
        list(POOL.in_order([(0, p, 0.0), (1, p, 0.0)], 3))
    with pytest.raises(POOL.NodeIndicesError, match="outside"):
        list(POOL.in_order([(0, p, 0.0), (3, p, 0.0)], 3))


def test_a_pool_is_bound_to_one_evaluator_configuration(serial):
    e, _ = serial
    other = ev(Heterogeneity(parameter="eta", sd=0.2, M=7, check=False))
    same = ev()
    assert same.digest() == e.digest() and other.digest() != e.digest()
    with NodePool(e, workers=1) as pool:
        with pytest.raises(UnsupportedOptionError) as ex:
            other.mixture_value(P, pool=pool)
        assert ex.value.option == "pool" and "another evaluator" in str(ex.value)
        assert same.mixture_value(P, pool=pool).value == e.mixture_value(P).value      # the twin: same configuration
    with pytest.raises(UnsupportedOptionError, match="closed"):
        e.mixture_value(P, pool=pool)                                                 # a closed pool


@pytest.mark.parametrize("W", [0, -1, 2.0, "3", True])
def test_a_worker_count_that_is_not_a_positive_integer_is_refused(W):
    with pytest.raises(UnsupportedOptionError) as ex:
        NodePool(ev(), workers=W)
    assert ex.value.option == "workers"


def test_more_workers_than_cores_is_refused():
    with pytest.raises(UnsupportedOptionError, match="os.cpu_count"):
        NodePool(ev(), workers=os.cpu_count() + 1)


def test_the_digest_is_of_the_construction_arguments():
    a, b = ev(), ev()
    assert a.digest() == b.digest() and len(a.digest()) == 64
    assert ev(free=("eta", "beta", "epsilon"), fixed={"kappa": 0.1}).digest() != a.digest()
    assert Evaluator(C, numerics="v033", grid=GRID, heterogeneity=HET).digest() != a.digest()
    c2 = Cohort.from_arrays(T + 1.0, E, delta=1.0)
    assert Evaluator(c2, numerics="v1", grid=GRID, heterogeneity=HET).digest() != a.digest()
    assert MIX.programs(a) is not MIX.programs(b)                                    # each evaluator its own programs
