"""ParamLayout (research R1; data-model.md): names <-> the log vector, and the FR-008 refusals."""
import numpy as np
import pytest

from sraging.engine.layout import DEFAULT_FREE, ParamLayout
from sraging.errors import UnsupportedOptionError


def test_default_layout_is_v033_z_order():
    assert ParamLayout().names == ("eta", "beta", "epsilon", "kappa") == DEFAULT_FREE


def test_pack_unpack_grad_dict_round_trip():
    lay = ParamLayout(names=("eta", "beta", "epsilon", "kappa", "h_ext", "seg:rapa"))
    vals = {"eta": 1e-5, "beta": 0.03, "epsilon": 2e-3, "kappa": 1e-3, "h_ext": 1e-4, "seg:rapa": 1.5}
    z = lay.pack(vals)
    assert np.allclose(np.exp(z), [vals[n] for n in lay.names], rtol=1e-15, atol=0)
    assert {k: float(np.exp(v)) for k, v in lay.unpack(z).items()} == pytest.approx(vals, rel=1e-15)
    assert lay.grad_dict(np.arange(6.0)) == {n: float(i) for i, n in enumerate(lay.names)}


@pytest.mark.parametrize("name", ["entry", "delta", "window", "segment_edge", "segment_xc", "edge:rapa"])
def test_snapped_inputs_cannot_be_free(name):
    with pytest.raises(UnsupportedOptionError, match="snaps to the grid"):
        ParamLayout(names=("eta", name))


def test_unknown_and_duplicate_names_are_refused():
    with pytest.raises(UnsupportedOptionError, match="unknown parameter"):
        ParamLayout(names=("eta", "gamma"))
    with pytest.raises(UnsupportedOptionError, match="duplicate"):
        ParamLayout(names=("eta", "eta"))


def test_pack_refuses_missing_and_non_positive_values():
    lay = ParamLayout()
    with pytest.raises(UnsupportedOptionError, match="missing"):
        lay.pack({"eta": 1.0})
    with pytest.raises(ValueError, match="beta must be finite and > 0"):
        lay.pack({"eta": 1.0, "beta": 0.0, "epsilon": 1.0, "kappa": 1.0})
