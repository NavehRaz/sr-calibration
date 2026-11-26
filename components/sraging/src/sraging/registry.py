"""The calibrator registry: the single config point (FR-010; constitution X).

A calibrator is chosen by name (and optionally version) here and nowhere else. Each module listed in
`CALIBRATOR_MODULES` registers its calibrators when imported (spec 001: `profile_v033`; spec 003: `profile_jax2`
and `profile_jax2_h`). Since spec 003b a name may carry several versions; a name without a version means its
CURRENT version (the first registered, unless a later registration asks to be current): `profile_jax2` means
1.0.1, and `profile_jax2@1.0.0` stays selectable (Naveh, 2026-10-19). Names a frozen package ships but sraging
does not register (the six kappa-fixed jax2 names) are refused with their reason. The default is `profile_jax2`
(1.0.1) since spec 003g's gate; `profile_v033` stays registered and is selected by name.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

from sraging.errors import UnknownCalibratorError, UnregisteredCalibratorError

DEFAULT = "profile_jax2"                 # spec 003g T052: the switch, on Naveh's yes after the gate (2026-10-28)
CALIBRATOR_MODULES = ("sraging.calibrators.profile_v033", "sraging.calibrators.profile_jax2")


@dataclass(frozen=True)
class CalibratorSpec:
    """name, version, numerics version, explicit read-only settings, and the fit callable."""

    name: str
    version: str
    numerics: str
    settings: Mapping
    fit: Callable

    def __post_init__(self):
        object.__setattr__(self, "settings", MappingProxyType(dict(self.settings)))


_REGISTRY: dict[str, dict[str, CalibratorSpec]] = {}     # name -> version -> spec
_CURRENT: dict[str, str] = {}                             # name -> the version a bare name means
#: name -> why it is refused (spec 003b FR-007; filled by the calibrator modules that know such names)
UNREGISTERED: dict[str, str] = {}


def register(spec: CalibratorSpec, *, current: bool = False) -> None:
    """Register `spec` under (name, version). The first version of a name is its current one; `current=True` makes
    this one current. A different spec under a registered (name, version) is refused."""
    versions = _REGISTRY.setdefault(spec.name, {})
    if spec.version in versions and versions[spec.version] is not spec:
        raise ValueError(f"a different calibrator is already registered as {spec.name!r}@{spec.version}")
    versions[spec.version] = spec
    if current or spec.name not in _CURRENT:
        _CURRENT[spec.name] = spec.version


def _ensure_loaded() -> None:
    """Import every calibrator module (each registers itself). Idempotent."""
    import importlib
    for mod in CALIBRATOR_MODULES:
        importlib.import_module(mod)


def names() -> list[str]:
    _ensure_loaded()
    return sorted(f"{s.name}@{s.version}" for versions in _REGISTRY.values() for s in versions.values())


def current(name: str) -> str:
    """The version a bare `name` means."""
    _ensure_loaded()
    if name not in _CURRENT:
        raise UnknownCalibratorError(name, names())
    return _CURRENT[name]


def get(name: str = DEFAULT, version: str | None = None) -> CalibratorSpec:
    """The registered calibrator `name` at `version` (its current version if None); UnregisteredCalibratorError for
    a name a frozen package ships but sraging does not register; UnknownCalibratorError otherwise."""
    _ensure_loaded()
    wanted = name if version is None else f"{name}@{version}"
    if name in UNREGISTERED:
        raise UnregisteredCalibratorError(wanted, names(), UNREGISTERED[name])
    versions = _REGISTRY.get(name)
    if not versions:
        raise UnknownCalibratorError(wanted, names())
    spec = versions.get(_CURRENT[name] if version is None else version)
    if spec is None:
        raise UnknownCalibratorError(wanted, names())
    return spec
