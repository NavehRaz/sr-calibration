"""SRParams: named, frozen, validated; dimensionless groups by name (data-model.md; T017, T019)."""
import dataclasses
import math

import pytest

from sraging.params import SRParams

GOOD = dict(eta=1e-3, beta=0.05, epsilon=4e-3, kappa=0.5)


def test_defaults_and_as_dict():
    p = SRParams(**GOOD)
    assert (p.xc, p.h_ext, p.time_unit) == (1.0, 0.0, "input units")
    assert p.as_dict() == {**GOOD, "xc": 1.0, "h_ext": 0.0, "time_unit": "input units"}


def test_keyword_only_and_no_iteration():
    with pytest.raises(TypeError):
        SRParams(1e-3, 0.05, 4e-3, 0.5)            # no positional constructor
    with pytest.raises(TypeError):
        list(SRParams(**GOOD))                      # no __iter__


@pytest.mark.parametrize("field,value", [
    ("eta", 0.0), ("eta", -1.0), ("eta", math.inf), ("eta", math.nan),
    ("beta", -1e-9), ("beta", math.nan),
    ("epsilon", 0.0), ("kappa", 0.0), ("kappa", -2.0),
    ("xc", 0.0), ("h_ext", -0.1), ("eta", "1"), ("eta", True),
])
def test_refusals_name_the_field(field, value):
    with pytest.raises(ValueError, match=f"SRParams.{field}"):
        SRParams(**{**GOOD, field: value})


def test_beta_zero_is_allowed():
    assert SRParams(**{**GOOD, "beta": 0.0}).beta == 0.0


def test_frozen():
    p = SRParams(**GOOD)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.eta = 2.0


def test_dimensionless_matches_fp_jax_groups_at_xc1():
    p = SRParams(**GOOD)
    L = 37.5
    d = p.dimensionless(L)
    assert set(d) == {"rho_eta", "rho_beta", "rho_epsilon", "kappa"}
    assert d == {"rho_eta": L ** 2 * p.eta, "rho_beta": L * p.beta, "rho_epsilon": L * p.epsilon,
                 "kappa": p.kappa}


@pytest.mark.parametrize("c", [0.5, 2.0, 7.0])
def test_dimensionless_is_invariant_under_the_exact_symmetry(c):
    p = SRParams(**GOOD, xc=1.3)
    q = SRParams(eta=c * p.eta, beta=c * p.beta, epsilon=c ** 2 * p.epsilon, kappa=c * p.kappa, xc=c * p.xc)
    a, b = p.dimensionless(10.0), q.dimensionless(10.0)
    for k in a:
        assert math.isclose(a[k], b[k], rel_tol=1e-12), k


@pytest.mark.parametrize("L", [0.0, -1.0, math.inf])
def test_dimensionless_refuses_bad_L(L):
    with pytest.raises(ValueError):
        SRParams(**GOOD).dimensionless(L)
