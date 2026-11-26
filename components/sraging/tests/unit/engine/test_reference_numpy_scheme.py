"""The independent NumPy march (task T017; research R3-R5, R11): checked against SRK's frozen `fpt_full`
and against its own invariants, so that it can serve as the oracle for what no old code has."""
import importlib

import numpy as np
import pytest

from reference import numpy_scheme as NS

SWEDEN = (3.4531226005888844, 1.8950299204066734, 0.0718812777718253, 0.24659997789670018)
STIFF = (20.0, 30.0, 0.1, 0.02)
GRID = {"be": 2e-3, "bdf2": 4e-3}           # dt per scheme, as SRK's own duplicate tests use
N = 150


@pytest.fixture(scope="module")
def fpl(oracle_v033):
    return importlib.import_module("SRK.fp_likelihood")


@pytest.mark.parametrize("theta", [SWEDEN, STIFF], ids=["sweden", "stiff"])
@pytest.mark.parametrize("scheme", ["be", "bdf2"])
def test_uniform_mesh_matches_srk_fpt_full(fpl, theta, scheme):
    dt, tau_max, Gtol = GRID[scheme], 20.0, 1e-11
    _m, _cv, _s, tau_o, S_o = fpl.fpt_full(*theta, N=N, dt=dt, tau_max=tau_max, Gtol=Gtol, scheme=scheme,
                                           return_survival=True)
    r = NS.march(N, NS.uniform_mesh(dt, tau_max), theta=theta, scheme=scheme, Gtol=Gtol)
    assert r.stopped_early and len(r.S_raw) == len(S_o), "the early stop must fall on the same node"
    assert np.array_equal(r.tau, tau_o)
    assert np.max(np.abs(r.S_raw - S_o)) <= 1e-12

    tau_d, S_d, f_d, *_ = fpl.fpt_full_density(*theta, N=N, dt=dt, tau_max=tau_max, Gtol=Gtol, scheme=scheme)
    S_m, f = NS.density_route(r.tau, r.S_raw)
    assert np.max(np.abs(S_m - S_d)) <= 1e-12
    assert np.max(np.abs(f - f_d)) <= 1e-10


def _threshold_dip(theta, frac):
    return [NS.Segment(0.0, 0.4, *theta, 1.0, 1.0), NS.Segment(0.4, 0.8, *theta, 1.0, frac),
            NS.Segment(0.8, np.inf, *theta, 1.0, 1.0)]


@pytest.mark.parametrize("case", ["uniform-be", "uniform-bdf2", "graded-bdf2", "threshold-bdf2"])
def test_F_plus_S_is_one_to_round_off(case):
    """F comes from the scheme's mass balance, not from 1 - S; the identity holds to the accumulated round-off
    of the solves (it grows with the step count and with dtau * rho_eps / h^2, hence the moderate grid)."""
    scheme = case.split("-")[1]
    if case.startswith("graded"):
        mesh = NS.geometric_mesh(3.0, 400, 30.0)
    else:
        mesh = NS.uniform_mesh(GRID[scheme], 3.0)
    if case.startswith("threshold"):
        r = NS.march(N, mesh, segments=_threshold_dip(SWEDEN, 0.6), scheme=scheme)
    else:
        r = NS.march(N, mesh, theta=SWEDEN, scheme=scheme)
    resolved = r.S_raw > 1e-6
    assert resolved.sum() > 100
    assert np.max(np.abs(r.F_raw + r.S_raw - 1.0)[resolved]) <= 1e-12
    assert np.all(np.diff(r.F_raw) >= -1e-15), "the absorbed mass never decreases"


@pytest.mark.parametrize("scheme", ["be", "bdf2"])
def test_a_full_life_segment_equals_no_segment_exactly(scheme):
    mesh = NS.uniform_mesh(GRID[scheme], 20.0)
    plain = NS.march(N, mesh, theta=SWEDEN, scheme=scheme)
    one = NS.march(N, mesh, segments=[NS.Segment(0.0, np.inf, *SWEDEN, 1.0, 1.0)], scheme=scheme)
    three = NS.march(N, mesh, segments=[NS.Segment(0.0, 0.4, *SWEDEN), NS.Segment(0.4, 0.8, *SWEDEN),
                                        NS.Segment(0.8, np.inf, *SWEDEN)], scheme=scheme)
    for r in (one, three):
        assert np.array_equal(r.S_raw, plain.S_raw) and np.array_equal(r.F_raw, plain.F_raw)
    assert not np.any(three.bdf2[:20]) and np.all(three.bdf2[20:]) == (scheme == "bdf2"), \
        "an edge in no parameter at all must not restart BDF2"


def test_a_chronic_clock_rate_is_a_chronic_eta_change():
    """eta * (A tau) = (eta A) tau: the clock multiplies the drift's age term and nothing else."""
    mesh = NS.uniform_mesh(GRID["bdf2"], 20.0)
    clock = NS.march(N, mesh, segments=[NS.Segment(0.0, np.inf, *SWEDEN, 2.5, 1.0)])
    eta = NS.march(N, mesh, theta=(SWEDEN[0] * 2.5,) + SWEDEN[1:])
    assert len(clock.S_raw) == len(eta.S_raw)
    assert np.max(np.abs(clock.S_raw - eta.S_raw)) <= 1e-12


@pytest.mark.parametrize("frac", [0.9, 0.6, 0.3])
@pytest.mark.parametrize("scheme", ["be", "bdf2"])
def test_a_lowered_then_raised_threshold_never_increases_S(scheme, frac):
    mesh = NS.uniform_mesh(GRID[scheme], 20.0)
    plain = NS.march(N, mesh, theta=SWEDEN, scheme=scheme)
    dip = NS.march(N, mesh, segments=_threshold_dip(SWEDEN, frac), scheme=scheme)
    assert np.max(np.diff(dip.S_raw)) <= 1e-13, "raising the threshold must expose empty nodes only"
    if scheme == "be":
        # BE is a monotone scheme (an M-matrix step), so the cut march stays below the plain one node by node.
        # (Not under BDF2: the BE restart at each move changes the discretisation, and at frac = 0.9 the
        # dip's tail ends up 1e-3 relatively ABOVE the plain tail at S ~ 4e-11 -- measured.)
        n = min(len(plain.S_raw), len(dip.S_raw))
        assert np.all(dip.S_raw[:n] <= plain.S_raw[:n] * (1 + 1e-12) + 1e-15)
    i = int(np.searchsorted(dip.tau, 0.4))
    assert dip.tau[i] == pytest.approx(0.4, abs=1e-12)
    assert np.array_equal(dip.S_raw[:i + 1], plain.S_raw[:i + 1]), "nothing changes before the first move"
    assert set(dip.M.tolist()) == {N, int(round(N * frac))}
    if scheme == "bdf2":
        be_steps = set(np.flatnonzero(~dip.bdf2).tolist())
        moves = {int(j) + 1 for j in np.flatnonzero(np.diff(dip.M) != 0)}
        assert be_steps == set(range(20)) | moves and len(moves) == 2, "one BE restart per threshold move"


@pytest.mark.parametrize("scheme", ["be", "bdf2"])
def test_lowering_the_threshold_absorbs_at_once_and_F_books_it(scheme):
    mesh = NS.uniform_mesh(GRID[scheme], 20.0)
    i = int(np.searchsorted(mesh, 0.4))
    drops = []
    for frac in (1.0, 0.9, 0.6, 0.3):
        r = NS.march(N, mesh, segments=_threshold_dip(SWEDEN, frac), scheme=scheme)
        drops.append(r.S_raw[i] - r.S_raw[i + 1])
        assert abs(r.F_raw[i + 1] + r.S_raw[i + 1] - 1.0) <= 1e-13
    assert drops[0] < drops[1] < drops[2] < drops[3], f"deeper cuts must absorb more, at once: {drops}"
    assert drops[3] > 1e-3


def test_a_graded_mesh_converges_to_the_uniform_fine_step_answer():
    """Variable-step BDF2 on geometric meshes (last/first step = 20) refined 100 -> 200 -> 400 steps. The
    error ratio approaches 4 (second order); constant BDF2 coefficients on the same meshes give 2.1 and a
    9x larger error at 400 steps (measured), so the check separates the two."""
    T = 1.0
    ref = NS.march(60, NS.uniform_mesh(1e-4, T), theta=SWEDEN, early_stop=False)
    errs = []
    for K in (100, 200, 400):
        r = NS.march(60, NS.geometric_mesh(T, K, 20.0), theta=SWEDEN, early_stop=False)
        assert r.tau[-1] == T and np.any(~np.isclose(np.diff(r.tau)[1:] / np.diff(r.tau)[:-1], 1.0))
        errs.append(np.max(np.abs(r.S_raw - np.interp(r.tau, ref.tau, ref.S_raw))))
    assert errs[0] > errs[1] > errs[2]
    assert errs[1] / errs[2] > 3.0, f"errors {errs}: not second order"
    assert errs[2] < 1e-3


def test_unsupported_options_are_refused():
    mesh = NS.uniform_mesh(1e-2, 1.0)
    with pytest.raises(ValueError, match="scheme"):
        NS.march(N, mesh, theta=SWEDEN, scheme="cn")
    with pytest.raises(ValueError, match="not a mesh node"):
        NS.march(N, mesh, segments=[NS.Segment(0.0, 0.405, *SWEDEN), NS.Segment(0.405, np.inf, *SWEDEN)])
    with pytest.raises(ValueError, match="need >= 2"):
        NS.march(N, mesh, segments=[NS.Segment(0.0, 0.4, *SWEDEN), NS.Segment(0.4, np.inf, *SWEDEN, 1.0, 0.004)])
    with pytest.raises(ValueError, match="contiguous"):
        NS.march(N, mesh, segments=[NS.Segment(0.0, 0.4, *SWEDEN), NS.Segment(0.5, np.inf, *SWEDEN)])
    with pytest.raises(ValueError, match="before the mesh"):
        NS.march(N, mesh, segments=[NS.Segment(0.0, 0.4, *SWEDEN)])
