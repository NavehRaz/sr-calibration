"""The truncated-normal Gauss rule (spec 004 FR-001; research R1; data-model "Rule"; register D-401).

`truncated_normal_rule(sd, M)` is Gaussian quadrature for the standard normal truncated to s = 1 + sd z > 0, gen-1's
heterogeneity law (`generator._het_z`). It is the het-tradeoff prototype (`docs/planning/het_tradeoff/evidence/hetlib.py:
tn_gauss`): the monic discretized Stieltjes procedure on 4,000 Gauss-Legendre points over [max(-1/sd, -8), 8], then
Golub-Welsch (the eigenvalues of the Jacobi matrix are the nodes, the squared first components of its eigenvectors the
weights). Its nodes and weights are smooth in sd: no node drops out as sd moves (register D-116's jump).

Written in jax.numpy, float64, one jitted program per M, so the sd-derivatives (`dz_dsd`, `dw_dsd`, by jax.jvp) come
from the same implementation as the values (spec 006 fits sd; HR-2 checks them against central differences).

Caching (constitution VIII: a memo returns what a fresh computation returns, and a test pins it): the 4,000-point base
grid is computed once per process; rules are memoised on (sd, M) and returned read-only; every rule carries the sha256
of its z || w bytes, so an evaluation records which nodes it used.
"""
from __future__ import annotations

import functools
import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

from sraging.engine import stepper as _ST  # noqa: F401  (x64 and one XLA thread, before the first JAX computation)
from sraging.errors import UnsupportedOptionError

import jax                              # noqa: E402
import jax.numpy as jnp                 # noqa: E402

BASE_POINTS = 4000                      # the discretized measure's Gauss-Legendre points (R1)
Z_HI = 8.0                              # the range's upper end, and its lower end where -1/sd is below -8
M_CAP = 192                             # the ladder's cap (sraging.engine.mixture.CAP)


@dataclass(frozen=True, eq=False)
class Rule:
    """One quadrature rule: nodes z (ascending), weights w (summing to 1), scales s = 1 + spread z, s_min = s[0], the
    sha256 of z || w, and (truncated_normal only) dz/dsd and dw/dsd. Arrays are read-only."""

    definition: str
    spread: float
    M: int
    z: np.ndarray
    w: np.ndarray
    s: np.ndarray
    s_min: float
    sha256: str
    dz_dsd: np.ndarray | None = None
    dw_dsd: np.ndarray | None = None


def _ro(a) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _sha(z, w) -> str:
    return hashlib.sha256(np.ascontiguousarray(z).tobytes() + np.ascontiguousarray(w).tobytes()).hexdigest()


@functools.lru_cache(maxsize=None)
def _base_grid(n: int = BASE_POINTS):
    """Gauss-Legendre points and weights on [-1, 1]: a pure function of n, computed once per process (~2.5 CPU-s)."""
    x, w = leggauss(n)
    return _ro(x), _ro(w)


@jax.custom_jvp
def _ratio(x, y):
    return x / y


@_ratio.defjvp
def _ratio_jvp(primals, tangents):
    """d(x/y) = (dx - (x/y) dy) / y. JAX's own rule forms x dy y^-2, which overflows once the monic norms pass ~1e154
    (M ~ 155: nrm reaches 1e191 at M = 192 [M]); this form stays at the scale of the ratio. The primal is x / y."""
    x, y = primals
    dx, dy = tangents
    q = x / y
    return q, (dx - q * dy) / y


@functools.partial(jax.jit, static_argnums=1)
def _tn_nodes(sd, M):
    """(z, w) of the M-point Gauss rule of the standard normal truncated to [max(-1/sd, -8), 8]: the prototype's
    operations in its order, the recurrence as a scan. The weights are renormalised to sum to 1 (the eigenvectors'
    first components squared sum to 1 only to a few ulps)."""
    x, wx = _base_grid(BASE_POINTS)
    a = jnp.maximum(-1.0 / sd, -Z_HI)
    z = 0.5 * (Z_HI - a) * x + 0.5 * (Z_HI + a)
    w = 0.5 * (Z_HI - a) * wx * jnp.exp(-0.5 * z * z)
    w = w / jnp.sum(w)

    def step(carry, _):
        p_prev, p, nrm_prev = carry
        nrm = jnp.sum(w * p * p)
        al = _ratio(jnp.sum(w * z * p * p), nrm)
        be = _ratio(nrm, nrm_prev)                  # be[0] is unused (J takes be[1:]) and p_prev = 0 at k = 0
        return (p, (z - al) * p - be * p_prev, nrm), (al, be)

    init = (jnp.zeros_like(z), jnp.ones_like(z), jnp.ones(()))
    _, (al, be) = jax.lax.scan(step, init, None, length=M)
    off = jnp.sqrt(be[1:])
    J = jnp.diag(al) + jnp.diag(off, 1) + jnp.diag(off, -1)
    lam, vec = jnp.linalg.eigh(J)
    wq = vec[0] ** 2
    return lam, wq / jnp.sum(wq)


def _check(sd, M):
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or not 1 <= int(M) <= M_CAP:
        raise UnsupportedOptionError("M", f"a rule's M is an integer in [1, {M_CAP}], got {M!r}")
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        raise UnsupportedOptionError("sd", f"sd must be a number in (0, 1], got {sd!r}") from None
    if not (math.isfinite(sd) and 0.0 < sd <= 1.0):
        raise UnsupportedOptionError("sd", f"sd must be finite and in (0, 1], got {sd!r}")
    return sd, int(M)


def _compute(sd: float, M: int) -> Rule:
    """The rule, uncached: values and sd-derivatives from one jvp of the jitted program."""
    sd, M = _check(sd, M)
    (z, w), (dz, dw) = jax.jvp(lambda x: _tn_nodes(x, M), (jnp.asarray(sd),), (jnp.ones(()),))
    z, w, dz, dw = (_ro(np.asarray(a)) for a in (z, w, dz, dw))
    s = _ro(1.0 + sd * z)
    return Rule("truncated_normal", sd, M, z, w, s, float(s[0]), _sha(z, w), dz, dw)


@functools.lru_cache(maxsize=None)
def _cached(sd: float, M: int) -> Rule:
    return _compute(sd, M)


def truncated_normal_rule(sd, M) -> Rule:
    """The M-node Gauss rule of the standard normal truncated to s = 1 + sd z > 0 (0 < sd <= 1, 1 <= M <= 192)."""
    sd, M = _check(sd, M)
    return _cached(sd, M)


def gauss_hermite_rule(v, M) -> Rule:
    """Spec 002's `normal_frac_sd` nodes, for the record only (SRK's _gh_nodes: probabilists' Gauss-Hermite, weights
    / sqrt(2 pi)). Its evaluation keeps the fused path, with its drop-and-renormalise step (register D-116)."""
    z, w = hermegauss(int(M))
    z, w = _ro(z), _ro(w / np.sqrt(2.0 * np.pi))
    s = _ro(1.0 + float(v) * z)
    return Rule("normal_frac_sd", float(v), int(M), z, w, s, float(s[0]), _sha(z, w))


def composite_rule(sd, panels: int, order: int = 8) -> Rule:
    """An independent rule for the reference's cross-check (research R9): Gauss-Legendre panels of `order` points on
    `panels` equal sub-intervals of the truncated z-range [max(-1/sd, -8), 8], weights by the normal density,
    normalised. It shares nothing with the Stieltjes recurrence; the caller refines `panels` until two refinements
    agree. No sd-derivatives."""
    sd, _ = _check(sd, 1)
    if isinstance(panels, bool) or not isinstance(panels, (int, np.integer)) or panels < 1:
        raise UnsupportedOptionError("panels", f"panels must be an integer >= 1, got {panels!r}")
    a = max(-1.0 / sd, -Z_HI)
    x, wx = leggauss(int(order))
    edges = np.linspace(a, Z_HI, int(panels) + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    z = (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel()
    w = (0.5 * (hi - lo) * wx * np.exp(-0.5 * (0.5 * (hi - lo) * x + 0.5 * (hi + lo)) ** 2)).ravel()
    z, w = _ro(z), _ro(w / w.sum())
    s = _ro(1.0 + sd * z)
    return Rule("composite", sd, int(z.size), z, w, s, float(s[0]), _sha(z, w))
