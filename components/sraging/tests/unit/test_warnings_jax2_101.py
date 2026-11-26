"""Spec 003b T020 (FR-011; research R12; contracts/registry-and-cli.md): H_NOT_USABLE keyed on the version, the
HAZARD_SHARP_LIMIT note on every profile_jax2_h 1.0.1 result (a note, not a warning), planted cases with twins."""
import pytest

from sraging.calibrators import profile_jax2 as PJ
from sraging.result import SRAGING_DIAGNOSTICS
from sraging.warnings import HAZARD_SHARP_LIMIT, h_not_usable


def test_h_not_usable_fires_on_1_0_0_and_not_on_1_0_1():
    w = h_not_usable("1.0.0")
    assert w.code == "H_NOT_USABLE" and w.data == {"version": "1.0.0"}
    assert h_not_usable("1.0.1") is None                          # the twin
    assert h_not_usable().code == "H_NOT_USABLE"                  # spec 003f's call form is 1.0.0's


@pytest.mark.parametrize("name,version,warn,note", [
    ("profile_jax2_h", "1.0.0", True, False), ("profile_jax2_h", "1.0.1", False, True),
    ("profile_jax2", "1.0.0", False, False), ("profile_jax2", "1.0.1", False, False)])
def test_warnings_and_notes_by_name_and_version(name, version, warn, note):
    port = PJ.PORTS[version]
    cls = port.classes[name]
    assert [w.code for w in PJ.version_warnings(port, cls)] == (["H_NOT_USABLE"] if warn else [])
    assert [n["code"] for n in PJ.notes_for(port, cls)] == (["HAZARD_SHARP_LIMIT"] if note else [])


def test_the_note_names_the_defect_its_size_and_the_inferred_cohorts():
    m = HAZARD_SHARP_LIMIT["message"]
    assert HAZARD_SHARP_LIMIT["code"] == "HAZARD_SHARP_LIMIT"
    assert "[T-H-HAZARD-SHARP]" in m and "13.6 nats" in m and "REV 6" in m
    # spec 003g T046: the measured size is v5's only. T066 (the interim review): of the 69 INFERRED cohorts 36 are valid
    # on v6, and only the 34 human ones were scored on a reference grid (W_REF), which can see a grid error at theta-hat;
    # the fit's own grid (W4) cannot, since an error common to the truth and theta-hat cancels (score-w2)
    assert "library v5 data only" in m and "69 cohorts the register INFERRED" in m
    assert "36 are valid on v6" in m and "34 human" in m and "reference grid" in m and "none more than 1.92" in m
    assert "own grid" in m and "cannot see" in m
    d = HAZARD_SHARP_LIMIT["data"]
    assert d["v6_inferred_valid"] == 36 and d["v6_inferred_ref_grid"] == 34 and d["v6_inferred_own_grid_only"] == 35
    assert d["v6_inferred_over_1_92"] == 0 and -0.37 < d["v6_inferred_ref_grid_w_max"] < -0.36
    assert "v6_inferred_measured" not in d and "v6_inferred_w_max" not in d     # the old counts mixed in INVALID ones
    assert "profile_jax2_h 1.0.1" in m


def test_notes_are_sraging_s_own_diagnostics_never_the_reference_comparison():
    assert "notes" in SRAGING_DIAGNOSTICS
