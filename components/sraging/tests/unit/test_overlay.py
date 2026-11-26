"""The data-vs-fit overlay (spec 006 T044-T045; FR-026; data-model "kappa ends"): the cohort's KM with its band and the
FP survival curve at both kappa ends of a fit, each plotted array read back (1e-12) and a planted wrong curve failing
the read-back; the labels name the definition, kappa and a box-limited end; a result without kappa_ends is refused by
name. Synthetic theta, no calibration fit. Agg only."""
import dataclasses
import json
import pathlib

import numpy as np
import pytest

from sraging import SRParams
from sraging.cohort import Cohort
from sraging.errors import UnsupportedOptionError
from sraging.population import population
from sraging.result import FitResult
from sraging.survival import km
from test_plots import close, km_steps, lines, style_problems, xy

ROOT = pathlib.Path(__file__).resolve().parents[2]
C = Cohort.from_csv(ROOT / "tests/unit/data/small_cohort.csv", delta=1.0, time_unit="days")
THETA_LO = [1e-3, 0.1, 0.0200, 0.1]            # median lifetime near the cohort's ~50 days at both ends
THETA_HI = [1e-3, 0.1, 0.0155, 0.5]
DEFINITION = "archive points within 1.92 nats of ll_best"


@pytest.fixture(scope="module")
def plots():
    pytest.importorskip("matplotlib")
    import sraging.plots as PL
    return PL


def _result(lo=True, hi=False, ends=True, theta_lo=THETA_LO, theta_hi=THETA_HI):
    r = FitResult.from_dict(json.loads((ROOT / "examples/results/itp_c2016_het3_control_female_jax2.json").read_text()))
    d = dict(r.diagnostics)
    d.pop("kappa_ends", None)
    if ends:
        d["kappa_ends"] = {"definition": DEFINITION, "kappa_lo": theta_lo[-1], "kappa_hi": theta_hi[-1],
                           "theta_lo": list(theta_lo), "theta_hi": list(theta_hi),
                           "box_limited_lo": lo, "box_limited_hi": hi}
    return dataclasses.replace(r, diagnostics=d)


def _P(t):
    return SRParams(eta=t[0], beta=t[1], epsilon=t[2], kappa=t[3], h_ext=t[4] if len(t) > 4 else 0.0,
                    time_unit="days")


def _S(theta, ages):
    return np.asarray(population(_P(theta), ages).S, float)


def test_km_steps_and_band_are_the_cohorts(plots):
    fig, data = plots.overlay(_result(), C)
    L, k = lines(fig), km(C)
    want_x, want_y = km_steps(k)
    assert close(xy(L["km"])[0], want_x) and close(xy(L["km"])[1], want_y)
    assert "km_band" in {c.get_gid() for ax in fig.axes for c in ax.get_children()}
    assert close(data["arrays"]["km"]["y"], want_y)
    assert {"km_lo", "km_hi"} <= set(data["arrays"])


def test_each_fp_curve_is_the_engines_survival_at_its_end(plots):
    fig, data = plots.overlay(_result(), C)
    L = lines(fig)
    for gid, th in (("fp_kappa_lo", THETA_LO), ("fp_kappa_hi", THETA_HI)):
        x, y = xy(L[gid])
        assert x.min() <= C.ages.min() and x.max() >= C.ages.max()
        assert close(y, _S(th, x)) and close(data["arrays"][gid]["y"], y)
    assert not close(xy(L["fp_kappa_lo"])[1], xy(L["fp_kappa_hi"])[1], 1e-3)


def test_labels_name_definition_kappa_and_box_limit(plots):
    fig, data = plots.overlay(_result(lo=False, hi=True), C)
    L = lines(fig)
    lo, hi = L["fp_kappa_lo"].get_label(), L["fp_kappa_hi"].get_label()
    assert "κ = 0.1" in lo and "κ = 0.5" in hi
    assert "box-limited" in hi and "box-limited" not in lo
    leg = fig.legends[0]
    assert DEFINITION in leg.get_title().get_text()                 # the definition heads the legend of the two ends
    texts = [t.get_text() for t in leg.get_texts()]
    assert any("box-limited" in t for t in texts) and any("κ = 0.1" in t for t in texts)
    assert data["provenance"]["definition"] == DEFINITION
    assert data["provenance"]["calibrator"] == {"name": "profile_jax2", "version": "1.0.1"}
    assert data["provenance"]["theta_lo"] == THETA_LO and len(data["provenance"]["data_sha256"]) == 64


def test_a_planted_shifted_curve_fails_the_read_back(plots, monkeypatch):
    import importlib
    ov = importlib.import_module("sraging.plots.overlay")      # the package binds the function over the module
    real = ov.population
    fig0, _ = plots.overlay(_result(), C)
    x0, y0 = xy(lines(fig0)["fp_kappa_lo"])

    def shifted(p, ages, **kw):
        pop = real(p, ages, **kw)
        return dataclasses.replace(pop, S=np.asarray(pop.S) * 0.99)
    monkeypatch.setattr(ov, "population", shifted)
    fig, _ = plots.overlay(_result(), C)
    assert not close(xy(lines(fig)["fp_kappa_lo"])[1], _S(THETA_LO, x0))      # the read-back catches it
    assert close(y0, _S(THETA_LO, x0))
    other = xy(lines(plots.overlay(_result(theta_lo=THETA_HI), C)[0])["fp_kappa_lo"])[1]
    assert not close(other, _S(THETA_LO, x0))


def test_style_and_h_ext(plots):
    fig, _ = plots.overlay(_result(), C)
    assert style_problems(fig) == []
    hfig, _ = plots.overlay(_result(theta_lo=THETA_LO + [0.001], theta_hi=THETA_HI + [0.001]), C)
    assert close(xy(lines(hfig)["fp_kappa_hi"])[1],
                 np.asarray(population(_P(THETA_HI + [0.001]),
                                       xy(lines(hfig)["fp_kappa_hi"])[0]).S))


def test_a_result_without_kappa_ends_is_refused_by_name(plots):
    with pytest.raises(UnsupportedOptionError, match="kappa_ends") as e:
        plots.overlay(_result(ends=False), C)
    assert "spec 006" in str(e.value)
