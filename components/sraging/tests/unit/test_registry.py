"""The calibrator registry, the single config point (constitution X; spec 001 FR-010; spec 003 T009)."""
import pytest

from sraging import registry
from sraging.errors import UnknownCalibratorError


def test_the_default_is_profile_jax2_and_every_listed_module_registers():
    """Spec 003g T052 (Naveh's yes, 2026-10-28, after the gate): the default is profile_jax2 (1.0.1); the twin:
    profile_v033 stays registered and resolves by name."""
    assert registry.DEFAULT == "profile_jax2" and registry.get().name == "profile_jax2"
    assert registry.get().version == "1.0.1"
    names = registry.names()
    assert "profile_v033@0.3.3" in names and registry.get("profile_v033").version == "0.3.3"
    for mod in registry.CALIBRATOR_MODULES:
        assert mod.startswith("sraging.calibrators.")


def test_unknown_name_or_version_is_refused_and_names_the_registered_ones():
    with pytest.raises(UnknownCalibratorError) as ex:
        registry.get("no_such_calibrator")
    assert "profile_v033@0.3.3" in str(ex.value)
    with pytest.raises(UnknownCalibratorError):
        registry.get("profile_v033", "9.9.9")


def test_a_name_without_a_version_means_its_registered_version():
    assert registry.get("profile_v033").version == registry.get("profile_v033", "0.3.3").version == "0.3.3"


def test_a_different_calibrator_under_a_registered_name_and_version_is_refused():
    spec = registry.get("profile_v033")
    other = registry.CalibratorSpec(spec.name, spec.version, spec.numerics, {}, spec.fit)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(other)


# ---------------------------------------------------------------------------- spec 003b: several versions per name

@pytest.fixture
def scratch(monkeypatch):
    """A registry with a fake two-version name, restored afterwards (the global registry is never left changed)."""
    registry._ensure_loaded()
    monkeypatch.setattr(registry, "_REGISTRY", {k: dict(v) for k, v in registry._REGISTRY.items()})
    monkeypatch.setattr(registry, "_CURRENT", dict(registry._CURRENT))
    fit = registry.get("profile_v033").fit
    old = registry.CalibratorSpec("fake_cal", "1.0.0", "jax2", {}, fit)
    new = registry.CalibratorSpec("fake_cal", "1.0.1", "jax2", {}, fit)
    registry.register(old)
    registry.register(new, current=True)
    return old, new


def test_a_bare_name_means_its_current_version_and_an_exact_version_its_own(scratch):
    old, new = scratch
    assert registry.get("fake_cal") is new and registry.current("fake_cal") == "1.0.1"
    assert registry.get("fake_cal", "1.0.0") is old and registry.get("fake_cal", "1.0.1") is new
    assert {"fake_cal@1.0.0", "fake_cal@1.0.1"} <= set(registry.names())


def test_a_first_registration_is_current_and_a_later_one_only_when_asked(monkeypatch):
    registry._ensure_loaded()
    monkeypatch.setattr(registry, "_REGISTRY", {k: dict(v) for k, v in registry._REGISTRY.items()})
    monkeypatch.setattr(registry, "_CURRENT", dict(registry._CURRENT))
    fit = registry.get("profile_v033").fit
    registry.register(registry.CalibratorSpec("fake2", "1", "jax2", {}, fit))
    registry.register(registry.CalibratorSpec("fake2", "2", "jax2", {}, fit))           # the twin: not current
    assert registry.current("fake2") == "1"


def test_an_unknown_version_of_a_known_name_lists_every_name_and_version(scratch):
    with pytest.raises(UnknownCalibratorError) as ex:
        registry.get("fake_cal", "9.9.9")
    assert "fake_cal@1.0.0" in str(ex.value) and "fake_cal@1.0.1" in str(ex.value)


def test_the_jax2_names_mean_1_0_1_and_1_0_0_stays_selectable():
    assert registry.get("profile_jax2").version == "1.0.1"
    assert registry.get("profile_jax2_h").version == "1.0.1"
    assert registry.get("profile_jax2", "1.0.0").version == "1.0.0"
    assert registry.get("profile_jax2_h", "1.0.0").version == "1.0.0"
    assert {"profile_jax2@1.0.0", "profile_jax2@1.0.1", "profile_jax2_h@1.0.0",
            "profile_jax2_h@1.0.1"} <= set(registry.names())
    assert registry.DEFAULT == "profile_jax2"                       # spec 003b FR-015 held; spec 003g T052 switched


@pytest.mark.parametrize("name", ["mle_profile_jax2_kfix_k0p001", "mle_profile_jax2_h_kfix_k1",
                                  "profile_jax2_kfix_k0p5"])
def test_a_kappa_fixed_name_is_refused_with_its_reason(name):
    from sraging.errors import UnregisteredCalibratorError
    for version in (None, "1.0.1"):
        with pytest.raises(UnregisteredCalibratorError, match="registered in a later spec, after the kappa0 table"):
            registry.get(name, version)
    assert issubclass(UnregisteredCalibratorError, UnknownCalibratorError)


def test_an_ordinary_unknown_name_is_not_called_kappa_fixed():
    from sraging.errors import UnregisteredCalibratorError
    with pytest.raises(UnknownCalibratorError) as ex:                                  # the twin
        registry.get("profile_jax3")
    assert not isinstance(ex.value, UnregisteredCalibratorError)


def test_fit_cohort_takes_name_at_version_as_the_cli_does(monkeypatch):
    import sraging
    seen = []
    monkeypatch.setattr(registry, "get", lambda name, version=None: seen.append((name, version)) or
                        type("S", (), {"fit": staticmethod(lambda c: "fitted")})())
    from sraging import Cohort
    c = Cohort.from_arrays([1.0, 2.0, 3.0], [1, 1, 1], delta=1.0)
    assert sraging.fit_cohort(c, calibrator="profile_jax2@1.0.0") == "fitted"
    assert sraging.fit_cohort(c, calibrator="profile_jax2") == "fitted"
    assert seen == [("profile_jax2", "1.0.0"), ("profile_jax2", None)]
    with pytest.raises(ValueError, match="twice"):
        sraging.fit_cohort(c, calibrator="profile_jax2@1.0.0", version="1.0.1")

