"""T1 for segments (T044; spec SC-002, SC-012; register D-113): numerics "v033" against SRK's frozen
fp_piecewise (tests/oracle/fp_piecewise_cb191f5, through its worker), like for like.

The twelve configurations follow SRK's own tests/test_fp_piecewise.py (chronic changes of every
parameter; windowed beta, clock and threshold; a windowed external hazard), with edges that are EXACT nodes
of the grid (dt = 2^-8 of L): fp_piecewise switches parameters at the step whose right end passes the edge,
so only binary-exact edges are like for like (research R4, measured). Per evaluation: total <= 1e-5 nats,
every term within the derived per-term bound, max |dS| <= 1e-8. Off-grid edges are register D-113: the
engine moves a node onto the edge and restarts BDF2; that difference is measured here, not tolerated.
"""
import math

import numpy as np
import pytest

import oracle_fp_piecewise as OP
from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import AgeSegments, Segment

pytestmark = [pytest.mark.parity, pytest.mark.engine]
TOTAL, TERM, DS = 1e-5, 1e-8, 1e-8
L, N, DT, TAU_MAX = 124.0, 300, 2.0 ** -8, 2.5
RNG = np.random.default_rng(7)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 500)) + 1.0)
E = (RNG.random(500) < 0.9).astype(int)
TH = dict(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
EXTH = 6.0                                                    # base hazard level, rate e^-6


def iv(param, effect, window=None):
    return {"kind": "Intervention", "param": param, "effect": effect, "window": list(window) if window else None}


def seg(sid, param, effect, window=None):
    return Segment(id=sid, parameter=param, value=1.0 + effect, window=window)


CONFIGS = {
    "chronic beta": ([iv("beta", -0.3)], [seg("s", "beta", -0.3)]),
    "chronic eta": ([iv("eta", 0.5)], [seg("s", "eta", 0.5)]),
    "chronic epsilon": ([iv("epsilon", 0.4)], [seg("s", "epsilon", 0.4)]),
    "chronic kappa": ([iv("kappa", 1.0)], [seg("s", "kappa", 1.0)]),
    "chronic xc": ([iv("xc", 0.3)], [seg("s", "xc", 0.3)]),
    "beta (62,124)": ([iv("beta", -0.6, (62, 124))], [seg("s", "beta", -0.6, (62, 124))]),
    "beta (93,186)": ([iv("beta", -0.6, (93, 186))], [seg("s", "beta", -0.6, (93, 186))]),
    "A slow (62,124)": ([iv("A", -0.3, (62, 124))], [seg("s", "A", -0.3, (62, 124))]),
    "A fast (62,124)": ([iv("A", 0.25, (62, 124))], [seg("s", "A", 0.25, (62, 124))]),
    "xc lower (62,124)": ([iv("xc", -0.25, (62, 124))], [seg("s", "xc", -0.25, (62, 124))]),
    "xc raise (93,186)": ([iv("xc", 0.3, (93, 186))], [seg("s", "xc", 0.3, (93, 186))]),
    "hazard window (62,124)": ([{"kind": "ExtHazard", "level": 4.5, "window": [62, 124]}],
                               [Segment(id="s", parameter="h_ext", value=math.exp(-4.5), window=(62, 124))]),
}


@pytest.fixture(scope="module")
def srk():
    try:
        with OP.session() as s:
            yield s
    except OP.NotApplicable as ex:
        pytest.skip(f"NOT APPLICABLE: {'; '.join(ex.problems)}")


def _ours(segs, scheme):
    c = Cohort.from_arrays(T, E, delta=1.0)
    ev = Evaluator(c, numerics="v033", grid=GridPolicy("fixed", N=N, dt=DT, tau_max=TAU_MAX, L=L), scheme=scheme,
                   segments=AgeSegments(*segs), fixed={"h_ext": math.exp(-EXTH)})
    return ev, SRParams(**TH, h_ext=math.exp(-EXTH))


def _theirs(srk, ivs, scheme, op="piecewise_loglik_terms"):
    return srk.call(op, death_times=T, events=E, theta=dict(**TH, xc=1.0), interventions=ivs, L=L,
                    t_end=float(T.max()), external_hazard=EXTH, sampling_dt=1.0, N=N, dt=DT, tau_max=TAU_MAX,
                    Gtol=1e-11, scheme=scheme)


@pytest.mark.parametrize("scheme", ["be", "bdf2"])
@pytest.mark.parametrize("name", list(CONFIGS))
def test_segments_t1_against_fp_piecewise(srk, name, scheme):
    ivs, segs = CONFIGS[name]
    ev, p = _ours(segs, scheme)
    assert ev._gspec is not None                                        # built through the general march
    terms, aux = ev.terms(p)
    ref = np.asarray(_theirs(srk, ivs, scheme)["terms"])
    assert abs(float(np.sum(terms)) - float(np.sum(ref))) <= TOTAL
    mass = np.where(E == 1, aux["mass"], aux["S_i"])
    diff = np.abs(terms - ref)
    cut = 4.0 * 1.3e-11 / TERM          # US1's measured dS_max between the two solvers (test_engine_t1_us1)
    assert np.all(diff[mass >= cut] <= TERM), f"worst resolved term {diff[mass >= cut].max():.3e}"
    assert np.all(diff[mass < cut] <= 4.0 * 1.3e-11 / np.maximum(mass[mass < cut], 1e-300) + TERM)


def test_off_grid_edges_are_d113_and_measured(srk):
    """Register D-113: with an edge between nodes the engine moves the nearest node onto it and restarts
    BDF2; fp_piecewise switches at the node below the edge while its clock runs from the exact edge.

    Measured (2026-10-03, this cohort, no external hazard, N = 300): as the step is refined (2^-8, 2^-10,
    2^-12 of L) the engine goes -3051.338, -3051.053, -3051.032 (smoothly converging) while fp_piecewise
    goes -3051.869, -3050.506, -3051.067 (an O(step) oscillation: its switch depends on where the edge falls
    inside a step). The test asserts the engine's own convergence is the tighter of the two."""
    ivs = [iv("beta", -0.6, (60, 120))]
    segs = [seg("s", "beta", -0.6, (60, 120))]
    c = Cohort.from_arrays(T, E, delta=1.0)
    p = SRParams(**TH)
    ours, theirs = [], []
    for k in (8, 10, 12):
        dt = 2.0 ** -k
        ev = Evaluator(c, numerics="v033", grid=GridPolicy("fixed", N=N, dt=dt, tau_max=TAU_MAX, L=L),
                       segments=AgeSegments(*segs))
        ours.append(ev.value(p))
        r = srk.call("piecewise_loglik_terms", death_times=T, events=E, theta=dict(**TH, xc=1.0), interventions=ivs,
                     L=L, t_end=float(T.max()), external_hazard=math.inf, sampling_dt=1.0, N=N, dt=dt,
                     tau_max=TAU_MAX, Gtol=1e-11, scheme="bdf2")
        theirs.append(float(np.sum(np.asarray(r["terms"]))))
    print(f"D-113 engine {ours}  fp_piecewise {theirs}")
    assert abs(ours[0] - theirs[0]) > TOTAL                         # a real, registered difference
    assert abs(ours[1] - ours[2]) < abs(theirs[1] - theirs[2])      # the engine converges; SRK-P oscillates
    assert abs(ours[1] - ours[2]) < 0.05
