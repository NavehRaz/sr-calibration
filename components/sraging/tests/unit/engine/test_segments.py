"""Age segments (T042, T043, T045; spec FR-002, FR-014 to FR-016; register D-110, D-111, D-113, D-114).

Inputs and refusals, the identities (a full-life segment is the chronic value; a chronic clock is eta
scaled; splitting a regime changes nothing), the clock outliving its window, a window past the last age,
and the gradient in segment values (checkpointed autodiff of the general march) against central
differences.
"""
import math

import numpy as np
import pytest

from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy, insert_edges
from sraging.engine.inputs import AgeSegments, Segment, SegmentConflictError
from sraging.errors import UnsupportedOptionError

pytestmark = pytest.mark.engine
RNG = np.random.default_rng(1)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 200)) + 1.0)
E = (RNG.random(200) < 0.9).astype(int)
C = Cohort.from_arrays(T, E, delta=1.0)
GRID = GridPolicy("fixed", N=160, dt=2.0 ** -8, tau_max=3.0, L=124.0)   # binary-exact: edges at k/256 of L are nodes
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)


def ev(segs=None, **kw):
    return Evaluator(C, numerics=kw.pop("numerics", "v033"), grid=GRID, segments=segs, **kw)


# -- inputs and refusals --------------------------------------------------------------------------------
def test_two_segments_on_one_parameter_at_one_age_are_refused():
    """Register D-111: fp_piecewise multiplied them silently."""
    with pytest.raises(SegmentConflictError, match=r"segments 'a' and 'b' both change beta on \[80.0, 100.0\)"):
        AgeSegments(Segment(id="a", parameter="beta", value=0.5, window=(60, 100)),
                    Segment(id="b", parameter="beta", value=2.0, window=(80, 120)))
    AgeSegments(Segment(id="a", parameter="beta", value=0.5, window=(60, 100)),     # adjacent is fine
                Segment(id="b", parameter="beta", value=2.0, window=(100, 120)))


def test_a_threshold_below_two_cells_is_refused():
    """Register D-114: fp_piecewise served it silently at 2/N."""
    segs = AgeSegments(Segment(id="x", parameter="xc", value=0.005, window=(60, 100)))
    with pytest.raises(UnsupportedOptionError, match="below two damage cells"):
        ev(segs)


@pytest.mark.parametrize("bad", [dict(parameter="gamma", value=1.0), dict(parameter="beta", value=0.0),
                                 dict(parameter="h_ext", value=-1.0),
                                 dict(parameter="beta", value=1.0, window=(5.0, 5.0))])
def test_bad_segments_are_refused(bad):
    with pytest.raises((ValueError, UnsupportedOptionError)):
        Segment(id="s", **bad)


def test_a_gradient_in_a_threshold_or_an_unknown_segment_is_refused():
    segs = AgeSegments(Segment(id="x", parameter="xc", value=0.8, window=(60, 100)))
    with pytest.raises(UnsupportedOptionError, match="has no gradient"):
        ev(segs, free=("eta", "beta", "epsilon", "kappa", "seg:x"))
    with pytest.raises(UnsupportedOptionError, match="no segment 'y'"):
        ev(segs, free=("eta", "beta", "epsilon", "kappa", "seg:y"))


def test_edges_on_the_grid_keep_the_uniform_mesh_and_others_move_the_nearest_node():
    nodes = np.arange(751) * 4e-3
    same, r0 = insert_edges(nodes, [0.5, 1.0])
    assert r0 == [] and np.array_equal(same, nodes)
    moved, r1 = insert_edges(nodes, [0.5013])
    assert r1 == [125] and moved[125] == 0.5013 and moved.size == nodes.size
    with pytest.raises(UnsupportedOptionError, match="within one grid step"):
        insert_edges(nodes, [0.5011, 0.5013])


# -- identities -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("par,m", [("eta", 0.75), ("beta", 1.3), ("epsilon", 0.75), ("kappa", 1.3)])
def test_a_full_life_segment_is_the_chronic_value(par, m):
    segs = AgeSegments(Segment(id="s", parameter=par, value=m))
    chronic = SRParams(**{**P.as_dict(), par: getattr(P, par) * m})
    assert ev(segs).value(P) == pytest.approx(ev().value(chronic), rel=1e-12)


def test_a_full_life_clock_is_eta_scaled():
    segs = AgeSegments(Segment(id="A", parameter="A", value=1.4))
    assert ev(segs).value(P) == pytest.approx(ev().value(SRParams(**{**P.as_dict(), "eta": P.eta * 1.4})),
                                             rel=1e-12)


def test_a_full_life_threshold_is_the_chronic_xc():
    segs = AgeSegments(Segment(id="x", parameter="xc", value=1.3))
    chronic = Evaluator(C, numerics="v033", grid=GRID, fixed={"xc": 1.3})
    assert ev(segs).value(P) == pytest.approx(chronic.value(SRParams(**{**P.as_dict(), "xc": 1.3})), rel=1e-12)


def test_splitting_one_regime_into_two_identical_segments_changes_nothing():
    """On grid-aligned edges (93 / 124 = 0.75 = 192 steps). An off-grid split moves a node and restarts
    BDF2, which is register D-113's (measured) difference, not an identity."""
    one = AgeSegments(Segment(id="a", parameter="beta", value=0.5, window=(62.0, 124.0)))
    two = AgeSegments(Segment(id="a", parameter="beta", value=0.5, window=(62.0, 93.0)),
                      Segment(id="b", parameter="beta", value=0.5, window=(93.0, 124.0)))
    assert ev(two).value(P) == pytest.approx(ev(one).value(P), rel=1e-13)


def test_the_clock_keeps_running_after_its_window():
    """A slowed clock on [62, 124) leaves the drift age lower forever after (fp_piecewise, the simulator):
    survival after the window is higher than with no segment, and not equal to it."""
    segs = AgeSegments(Segment(id="A", parameter="A", value=0.5, window=(62.0, 124.0)))
    tau, S_seg, _, k1 = ev(segs).survival_on_grid(P)
    _, S0, _, k0 = ev().survival_on_grid(P)
    late = (tau > 1.5) & (np.arange(tau.size) < min(k0, k1))     # before either march stops (held after)
    assert np.all(S_seg[late] >= S0[late]) and np.max(S_seg[late] - S0[late]) > 1e-3


def test_a_window_past_the_last_age_closes_where_it_says():
    """Register D-110: (100, 214) and (100, 10000) are different interventions after age 214."""
    a = AgeSegments(Segment(id="e", parameter="eta", value=0.4, window=(100.0, 214.0)))
    b = AgeSegments(Segment(id="e", parameter="eta", value=0.4, window=(100.0, 10000.0)))
    tau, Sa, _, _ = ev(a).survival_on_grid(P)
    _, Sb, _, _ = ev(b).survival_on_grid(P)
    after = tau * 124.0 > 260.0
    assert np.max(np.abs(Sa[after] - Sb[after])) > 1e-3


def test_lowering_the_threshold_absorbs_at_once_and_raising_exposes_empty_cells():
    low = AgeSegments(Segment(id="x", parameter="xc", value=0.7, window=(62.0, 124.0)))
    tau, S_low, _, _ = ev(low).survival_on_grid(P)
    _, S0, _, _ = ev().survival_on_grid(P)
    i = int(np.searchsorted(tau, 0.5))
    assert S_low[i + 1] < S0[i + 1] - 1e-3                    # a cliff at the window's start
    assert np.all(np.diff(S_low[: int(np.searchsorted(tau, 2.0))]) <= 1e-15)


# -- gradients ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("numerics", ["v033", "v1"])
def test_segment_value_gradients_match_central_differences(numerics):
    segs = AgeSegments(Segment(id="b", parameter="beta", value=0.4, window=(62.0, 124.0)),
                       Segment(id="a", parameter="A", value=0.8, window=(100.0, 140.0)),
                       Segment(id="x", parameter="xc", value=0.85, window=(130.0, 170.0)),
                       Segment(id="h", parameter="h_ext", value=5e-3, window=(40.0, 90.0)))
    p = SRParams(**{**P.as_dict(), "h_ext": 1e-3})
    e = ev(segs, numerics=numerics, free=("eta", "beta", "epsilon", "kappa", "h_ext", "seg:b", "seg:a", "seg:h"))
    v, g = e.value_and_grad(p)
    assert v == pytest.approx(e.value(p), rel=1e-13)
    f, _ = e.objective()
    z = e.layout.pack({**p.as_dict(), "seg:b": 0.4, "seg:a": 0.8, "seg:h": 5e-3})
    hstep = 1e-5
    bound = 50 * 2.2e-16 * abs(v) / hstep + 1e-6 * max(abs(x) for x in g.values())
    for i, n in enumerate(e.layout.names):
        dz = np.zeros(z.size)
        dz[i] = hstep
        assert abs((float(f(z + dz)) - float(f(z - dz))) / (2 * hstep) - g[n]) <= bound, n


@pytest.mark.parametrize("scheme", ["be", "bdf2"])
def test_lowering_the_threshold_keeps_s_plus_f_equal_to_one(scheme):
    """Review T064 (the guard on feature configurations): the absorbed mass at a lowering is a jump of F; the
    BDF2 history must carry it, or F is off by a third of the jump afterwards (found by the guard: |dm| =
    5e-3 on the deaths after a lowering). Mass balance: S + F = 1 to round-off (BE; BDF2 also adds the clip's
    mass to F, so S + F = 1 there too)."""
    import jax.numpy as jnp
    segs = AgeSegments(Segment(id="x", parameter="xc", value=0.75, window=(62.0, 124.0)))
    e = Evaluator(C, numerics="v1", grid=GRID, segments=segs, scheme=scheme)
    S_raw, F_raw = e._value_march(e._asm.theta(e._vals(jnp.asarray(e._z(P)))))
    k = int(np.flatnonzero(np.asarray(S_raw) < 1e-6)[0]) if np.any(np.asarray(S_raw) < 1e-6) else S_raw.size
    assert np.max(np.abs(np.asarray(S_raw)[:k] + np.asarray(F_raw)[:k] - 1.0)) <= 1e-12
