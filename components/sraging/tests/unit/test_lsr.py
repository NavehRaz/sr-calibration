"""L_SR, the SR mean lifetime (spec 003, T035; research R11): it converges under refinement, and it equals the mean of
the engine's own density (integral of t f(t)), an independent route to the same number."""
import numpy as np

from sraging import SRParams
from sraging.claims.lsr import RTOL, l_sr
from sraging.engine.curves import curves, model_L
from sraging.engine.grid import GridPolicy

P = SRParams(eta=3.6e-5, beta=0.012, epsilon=2.4e-3, kappa=0.7, time_unit="days")


def test_l_sr_converges_and_equals_the_densitys_mean():
    r = l_sr(P)
    assert r["converged"] and r["rel_change"] <= RTOL
    L = model_L(P)
    ages = np.linspace(0.0, r["tau_max"] * L, 8001)
    c = curves(P, ages, grid=GridPolicy("fixed", N=r["N"], dt=r["dt"], tau_max=r["tau_max"], L=L))
    mean = float(np.trapezoid(ages * c.f, ages))
    assert abs(mean - r["L_SR"]) <= 5 * RTOL * r["L_SR"], (mean, r["L_SR"])
