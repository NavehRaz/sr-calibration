"""Spec 006's new FitResult diagnostics and warning codes (research R5, R6, R7, R8; data-model.md "Warnings"; tasks
T011): `window` and `kappa_ends` are sraging's own diagnostics, so the schema stays /4 and identity_view (the
reference comparison) does not see them; the four codes fill their templates from their data and refuse what is
missing. Planted: the same keys outside SRAGING_DIAGNOSTICS would change identity_view."""
import dataclasses
import math
import os

import pytest

from sraging import result as R
from sraging.warnings import WINDOW_WARNINGS, window_warning

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")
WINDOW = {"window_given": [365.0, "inf"], "hi_mapped_from": "inf", "hi_used": 1501.0, "window_eff": [365.0, 1501.0],
          "n_dropped_below_lo": 3, "n_deaths_censored": 0, "n_censored_at_hi_eff": 0}
KAPPA_ENDS = {"definition": "archive points within 1.92 nats of ll_best", "kappa_lo": 0.001, "kappa_hi": 300.0,
              "theta_lo": [3.6e-05, 0.032, 0.0024, 0.001], "theta_hi": [1.6e-04, 93.6, 0.0113, 300.0],
              "box_limited_lo": True, "box_limited_hi": True}


@pytest.fixture(scope="module")
def fit():
    return R.FitResult.from_json(EXAMPLE)


def test_the_schema_is_unchanged_and_the_keys_are_sragings_own():
    assert R.SCHEMA == "sraging.fitresult/4"
    assert {"window", "kappa_ends"} <= set(R.SRAGING_DIAGNOSTICS)
    assert {"code", "floor_report", "leverage", "notes"} <= set(R.SRAGING_DIAGNOSTICS)


def test_the_new_keys_leave_identity_view_unchanged(fit):
    with_keys = dataclasses.replace(fit, diagnostics={**fit.diagnostics, "window": WINDOW, "kappa_ends": KAPPA_ENDS})
    assert with_keys.identity_view() == fit.identity_view()
    planted = dataclasses.replace(fit, diagnostics={**fit.diagnostics, "window_planted": WINDOW})   # the twin
    assert planted.identity_view() != fit.identity_view()


def test_the_new_keys_round_trip_through_json(fit):
    f = dataclasses.replace(fit, diagnostics={**fit.diagnostics, "window": WINDOW, "kappa_ends": KAPPA_ENDS})
    back = R.FitResult.from_json(f.to_json())
    assert back.diagnostics["window"] == WINDOW and back.diagnostics["kappa_ends"] == KAPPA_ENDS
    assert back.to_dict()["schema"] == R.SCHEMA


DATA = {"WINDOW_S_LO_CLIPPED": {"S_lo_min": 3.2e-14, "lo": 365.0, "calibrator": "profile_jax2_w"},
        "ENGINE_S_LO_CLIPPED": {"S_lo_min": 1e-15, "numerics": "v1"},
        "W_COVERAGE": {"detail": "eta 0.89 (171/192), epsilon 0.90 (165/183)", "cut": 0.914},
        "W_LARGE_N": {"n": 25000, "n_min": 25000, "detail": "eta 28 of 32 (14 at the box, 13 widened)",
                      "below": "spec 006's A1: eta 56 of 67"}}


def test_the_four_codes_are_registered_and_fill_their_templates():
    assert set(WINDOW_WARNINGS) == {"WINDOW_S_LO_CLIPPED", "ENGINE_S_LO_CLIPPED", "W_COVERAGE", "W_LARGE_N"}
    for code, data in DATA.items():
        w = window_warning(code, **data)
        assert w.code == code and w.data == data and "{" not in w.message
    assert "3.2e-14" in window_warning("WINDOW_S_LO_CLIPPED", **DATA["WINDOW_S_LO_CLIPPED"]).message
    assert "1e-12" in WINDOW_WARNINGS["WINDOW_S_LO_CLIPPED"] and "1e-12" in WINDOW_WARNINGS["ENGINE_S_LO_CLIPPED"]


def test_an_unknown_code_or_a_missing_field_is_refused():
    with pytest.raises(ValueError, match="unknown window warning"):
        window_warning("WINDOW_TYPO", S_lo_min=1.0)
    with pytest.raises(ValueError, match="'lo'"):
        window_warning("WINDOW_S_LO_CLIPPED", S_lo_min=1e-14, calibrator="profile_jax2_w")
    assert math.isfinite(DATA["W_COVERAGE"]["cut"])
