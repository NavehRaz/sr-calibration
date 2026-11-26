"""Public API refusals: no hidden options, unknown calibrators named, runtime checked (T025)."""
import subprocess
import sys

import pytest

import sraging
from sraging.errors import UnknownCalibratorError

COHORT = sraging.Cohort.from_arrays([3.0, 5.0, 8.0], [1, 0, 1], delta=1.0)


@pytest.mark.parametrize("kw", [{"xc": 2.0}, {"window": (1, 2)}, {"h_ext": 0.1}, {"engine": "langevin"}])
def test_unsupported_arguments_are_named(kw):
    with pytest.raises(TypeError, match=list(kw)[0]):
        sraging.fit_cohort(COHORT, **kw)


@pytest.mark.parametrize("name,version", [("nope", None), ("profile_v033", "9.9.9")])
def test_unknown_calibrator(name, version):
    with pytest.raises(UnknownCalibratorError) as ei:
        sraging.fit_cohort(COHORT, calibrator=name, version=version)
    assert any(r.startswith("profile_v033@") for r in ei.value.registered)


def test_registry_holds_exactly_the_registered_calibrators():
    """spec 001: profile_v033; spec 003 adds profile_jax2 and profile_jax2_h; spec 003b adds their 1.0.1 (the
    default stays profile_v033); spec 006 adds profile_jax2_w at its own version, 1.0.1+w1 (FR-002)."""
    assert sraging.registry.names() == ["profile_jax2@1.0.0", "profile_jax2@1.0.1", "profile_jax2_h@1.0.0",
                                        "profile_jax2_h@1.0.1", "profile_jax2_w@1.0.1+w1", "profile_v033@0.3.3"]
    assert sraging.registry.DEFAULT == "profile_jax2"                       # spec 003g T052 (Naveh, 2026-10-28)
    spec = sraging.registry.get("profile_v033")
    assert spec.numerics == "v033" and spec.settings["kappa_max"] == 300.0
    with pytest.raises(TypeError):
        spec.settings["kappa_max"] = 3000.0                    # read-only


def test_x64_off_is_refused():
    code = ("import sraging, jax\n"
            "from sraging.compat.v033 import runtime\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "try:\n    runtime.check()\nexcept sraging.errors.RuntimeConfigError as e:\n"
            "    print('REFUSED', 'x64' in str(e))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert "REFUSED True" in out.stdout, out.stderr


# ------------------------------------------------------------------------------------- spec 006: windows (T013)
WINDOWED = sraging.Cohort.from_arrays([3.0, 5.0, 8.0, 9.0, 12.0, 13.0, 20.0], [1, 1, 1, 1, 1, 0, 1], delta=1.0,
                                      window=(4.0, 15.0))


@pytest.mark.parametrize("name", ["profile_jax2", "profile_jax2@1.0.0", "profile_jax2_h@1.0.0"])
def test_a_calibrator_that_does_not_fit_windows_refuses_one_naming_the_ones_that_do(name):
    """FR-009: refused before any fit, by name, naming W and H."""
    cal, _, version = name.partition("@")
    with pytest.raises(sraging.errors.UnsupportedOptionError) as ex:
        sraging.fit_cohort(WINDOWED, calibrator=cal, version=version or None)
    assert ex.value.option == "window"
    assert "profile_jax2_w" in str(ex.value) and "profile_jax2_h" in str(ex.value)


def test_profile_v033_refuses_a_window_as_before():
    with pytest.raises(sraging.errors.UnsupportedOptionError) as ex:
        sraging.fit_cohort(WINDOWED, calibrator="profile_v033")
    assert "window" in str(ex.value)


@pytest.mark.parametrize("name", ["profile_jax2", "profile_jax2_h", "profile_jax2_w", "profile_v033"])
@pytest.mark.parametrize("what,make", [
    ("entry", lambda: sraging.Cohort.from_arrays([3.0, 5.0, 8.0], [1, 0, 1], delta=1.0, entry=[0.0, 0.0, 1.0])),
    ("per-row delta", lambda: sraging.Cohort.from_arrays([3.0, 5.0, 8.0], [1, 0, 1], delta=[1.0, 1.0, 1.0])),
])
def test_entry_ages_and_a_per_row_delta_stay_refused_by_every_calibrator(name, what, make):
    with pytest.raises(sraging.errors.UnsupportedOptionError) as ex:
        sraging.fit_cohort(make(), calibrator=name)
    assert what.split()[0] in str(ex.value)

