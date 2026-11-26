"""T1 for heterogeneity (T052; spec SC-002): numerics "v033" mixtures against SRK's frozen
fp_loglik_het_physical (tests/oracle/fp_piecewise_cb191f5, through its worker), like for like, in each of
the five parameters, with and without a fit window (the window conditions on the MIXTURE's survival).
Totals per evaluation within 1e-5 nats (SRK returns only the total for the mixture)."""
import numpy as np
import pytest

import oracle_fp_piecewise as OP
from sraging import Cohort, SRParams
from sraging.engine.evaluator import Evaluator
from sraging.engine.grid import GridPolicy
from sraging.engine.inputs import Heterogeneity

pytestmark = [pytest.mark.parity, pytest.mark.engine]
RNG = np.random.default_rng(3)
T = np.sort(np.floor(RNG.gamma(9.0, 14.0, 400)) + 1.0)
E = (RNG.random(400) < 0.9).astype(int)
L, N, DT, TAU_MAX = 124.0, 300, 2.0 ** -8, 2.5
TH = dict(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)


@pytest.fixture(scope="module")
def srk():
    try:
        with OP.session() as s:
            yield s
    except OP.NotApplicable as ex:
        pytest.skip(f"NOT APPLICABLE: {'; '.join(ex.problems)}")


@pytest.mark.parametrize("scheme", ["bdf2", "be"])
@pytest.mark.parametrize("window", [None, (62.0, 186.0)], ids=["full", "window"])
@pytest.mark.parametrize("par,v,M", [("xc", 0.2, 7), ("eta", 0.2, 7), ("beta", 0.3, 9), ("epsilon", 0.2, 7),
                                     ("kappa", 0.4, 7)])
def test_mixture_t1_against_srk(srk, par, v, M, window, scheme):
    c = Cohort.from_arrays(T, E, delta=1.0, window=window)
    ev = Evaluator(c, numerics="v033", grid=GridPolicy("fixed", N=N, dt=DT, tau_max=TAU_MAX, L=L), scheme=scheme,
                   heterogeneity=Heterogeneity(parameter=par, definition="normal_frac_sd", v=v, M=M))
    ours = ev.value(SRParams(**TH))
    ref = srk.call("fp_loglik_het_physical", death_times=T, events=E, **TH, xc=1.0, L=L, het_param=par, het_var=v,
                   M=M, time_range=list(window) if window else None, sampling_dt=1.0, N=N, dt=DT, tau_max=TAU_MAX,
                   Gtol=1e-11, scheme=scheme)
    assert abs(ours - float(ref["value"])) <= 1e-5
