"""Theta-vector conventions shared by the SR simulation factories.

THE PROBLEM THIS SOLVES
-----------------------
Three incompatible theta layouts coexist in this codebase:

    ``[eta, beta, epsilon, kappa]``                  sr_mcmc.inv_transform (the FITTING output),
                                                     SRK.intervention.io.load_posterior
    ``[eta, beta, epsilon, xc, kappa]``              SR_hetro.getSrHetro, presets.getTheta (SIM input)
    ``[eta, beta, epsilon, xc, kappa, ExtH]``        presets.getTheta(ExtH=True)

So a bare 4- or 5-element vector is ambiguous, and the two ways of getting it wrong are both silent:

* a fit/posterior row ``[eta, beta, epsilon, kappa]`` handed to a simulation factory has its **kappa
  read as xc** while kappa itself falls back to a default -- two errors at once, no message;
* a 6-element preset theta handed to a factory silently **drops the external hazard**.

`kappa_from_theta` centralises the ``theta[4]`` convention that `SR_hetro.getSrHetro` (SR_hetro.py:
213-217) established, and warns on both mis-binding patterns. It is deliberately a tiny, dependency
-free module (numpy + warnings) so the simulator factories can use it without new imports.

Historical note: `getISR`, `getISR_periodic` and `getSrDDD` used to hard-code ``kappa=0.5`` and ignore
``theta[4]`` entirely. Measured on the mice_F max-likelihood preset, that made `getISR` report a
median lifetime of 71.6 wk where the exact FP solution at the fitted kappa is 126.1 wk (-43 %), and
made Jack_Russell produce no deaths at all. See tests/test_intervention_regression.py.
"""
import warnings

import numpy as np

# The canonical ARRAY order for simulation input. Chosen to match what getSrHetro and presets
# already use, so making it canonical requires rewriting nothing.
CANONICAL_ORDER = ("eta", "beta", "epsilon", "xc", "kappa", "external_hazard")

LEGACY_KAPPA = 0.5      # the value the pre-kappa factories assumed when theta carried no kappa

__all__ = ["CANONICAL_ORDER", "LEGACY_KAPPA", "kappa_from_theta"]


def kappa_from_theta(theta, kappa=LEGACY_KAPPA, source="", warn=True):
    """Return the kappa implied by ``theta``, mirroring ``SR_hetro.getSrHetro``.

    ``theta[4]`` wins when present; a 4-element (legacy, free-xc) vector keeps ``kappa`` unchanged,
    which is what makes threading kappa a no-op for every pre-existing caller.

    Parameters
    ----------
    theta : sequence
        ``[eta, beta, epsilon, xc]`` (legacy) or ``[eta, beta, epsilon, xc, kappa]`` (+ ExtH).
    kappa : float
        Fallback for a 4-element theta. Defaults to the historical 0.5.
    source : str
        Caller name, quoted in warnings so the message points at real code.
    warn : bool
        Set False to silence the mis-binding heuristics (they are heuristics, not proofs).

    Returns
    -------
    float
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or theta.size < 4:
        raise ValueError(
            f"theta must be a 1-D vector of at least 4 elements "
            f"[eta, beta, epsilon, xc]; got shape {theta.shape}")

    where = f"{source}: " if source else ""

    if theta.size > 4:
        kappa = float(theta[4])
    elif warn and theta[3] < 1.0:
        # xc < 1 in a 4-element vector is the fingerprint of a FIT/posterior row
        # [eta, beta, epsilon, kappa] being read as [eta, beta, epsilon, xc]: kappa is bounded to
        # ~[0.01, 1] by the prior while a legacy free-xc theta has xc >> 1 (e.g. 29.7, 42.6).
        warnings.warn(
            f"{where}4-element theta with theta[3]={theta[3]:.4g} < 1. This looks like a fitted / "
            f"posterior vector [eta, beta, epsilon, kappa] being interpreted as "
            f"[eta, beta, epsilon, xc] -- kappa would be read as xc and the real kappa replaced by "
            f"the {kappa:g} default. Pass [eta, beta, epsilon, 1.0, kappa] instead "
            f"(order: {', '.join(CANONICAL_ORDER[:5])}).",
            RuntimeWarning, stacklevel=3)

    if warn and theta.size > 5:
        warnings.warn(
            f"{where}theta has {theta.size} elements; only the first 5 "
            f"({', '.join(CANONICAL_ORDER[:5])}) are used and element 5 "
            f"({theta[5]:.4g}) is IGNORED. If that is an external hazard, pass it as the "
            f"external_hazard argument -- otherwise it is silently dropped.",
            RuntimeWarning, stacklevel=3)

    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError(f"{where}kappa must be finite and positive; got {kappa!r}")
    return float(kappa)
