"""The method interface, and the registry that finds implementations of it.

A method is three things:

    descriptor          -> MethodDescriptor   (metadata, including `version` and `bounds`)
    cost(spec)          -> CostDeclaration    (a hypothesis; the harness measures it)
    calibrate(obs)      -> ClaimSet           (the work)

Nothing here knows about the reference role. A method is scoreable as challenger and as reference
with no code difference -- if the reference were special here, it would stop being scored by the
same rules as its challengers, which is how an incumbent becomes unfalsifiable.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
from typing import Protocol, runtime_checkable

from .types import ClaimSet, ContractError, CostDeclaration, MethodDescriptor, Observation
from .validate import validate_cost_declaration, validate_descriptor


@runtime_checkable
class CalibrationMethod(Protocol):
    """What a method must provide. Structural, so no inheritance is imposed on method authors."""

    @property
    def descriptor(self) -> MethodDescriptor: ...

    def cost(self, spec: "CohortCharacteristics") -> CostDeclaration: ...

    def calibrate(self, observation: Observation) -> ClaimSet: ...


class CohortCharacteristics(dict):
    """What a method is told about a cohort when asked to PREDICT its cost.

    A plain dict subclass rather than a dataclass: methods vary in what they care about, and a
    fixed record would either omit what some method needs or force every method to accept fields
    it ignores. Keys are stable: `n`, `arm`, `delta`, `windowed`, `kappa`, `ratio`, `cv`.

    It carries no truth values -- a method could otherwise learn the answer by asking what a
    calibration would cost.
    """


def characteristics_from_spec(spec) -> CohortCharacteristics:
    """Build the cost-prediction record from a library cohort spec, truth stripped."""
    return CohortCharacteristics(
        n=int(spec.n),
        arm=str(spec.arm),
        delta=None if spec.delta is None else float(spec.delta),
        windowed=spec.window is not None,
        has_hazard=spec.external_hazard is not None,
        has_heterogeneity=spec.het_param is not None,
        censored=bool(spec.censor_frac),
    )


# --------------------------------------------------------------------------- registry

_REGISTRY: dict[str, CalibrationMethod] = {}


def register(method: CalibrationMethod) -> CalibrationMethod:
    """Validate and register. Validation happens HERE so a violation costs seconds, not a week
    of cluster time discovering it on result number four hundred."""
    d = method.descriptor
    validate_descriptor(d)
    if d.name in _REGISTRY and _REGISTRY[d.name] is not method:
        raise ContractError(
            f"two different methods both call themselves {d.name!r}; names must be unique")
    _REGISTRY[d.name] = method
    return method


def get(name: str) -> CalibrationMethod:
    if name not in _REGISTRY:
        discover()
    if name not in _REGISTRY:
        raise ContractError(
            f"no method named {name!r}. Registered: {sorted(_REGISTRY) or '(none)'}. "
            "A method is found by placing a package under `methods/` that calls "
            "`srbench.contract.method.register(...)` at import time."
        )
    return _REGISTRY[name]


def registered() -> dict[str, CalibrationMethod]:
    discover()
    return dict(_REGISTRY)


#: Methods that are REGISTERED but must not be EXECUTED, keyed by (name, version).
#:
#: A method package can register a name its own authors say is not usable. `mle_profile_jax2_h`
#: 1.0.0 is the case this exists for: it ships inside the frozen `mle_profile_jax2` package, cannot
#: be split out without editing bytes that are frozen by agreement, and is NOT validated on library
#: v5 -- but nothing stopped `srbench run --method mle_profile_jax2_h` from calibrating with it and
#: writing the results into the SHARED store. That store is content-addressed and merged across
#: sessions, so the records would have spread, and this project has already paid once for exactly
#: that: 565 `ModuleNotFoundError` records had to be hunted down and purged from the cluster store.
#:
#: KEYED ON (name, version), never on name alone. `mle_profile_jax2_h` 1.0.1 is validated and must
#: run the moment it lands; a name-only guard would block the fix along with the defect.
#:
#: This blocks EXECUTION only. Listing, `get()`, descriptor inspection, scoring stored records and
#: the gates all keep working -- a quarantined method is visible and inspectable, just not runnable.
QUARANTINED: dict[tuple[str, str], str] = {
    ("mle_profile_jax2_h", "1.0.0"):
        "registered by the frozen mle_profile_jax2 package but NOT validated on library v5, and it "
        "has no shared records (jax2 CONTEXT-SYNC 2026-10-14). Running it would write unvalidated "
        "records into the shared store. Use 1.0.1 when it lands.",
}


def ensure_runnable(method: "CalibrationMethod") -> None:
    """Refuse to EXECUTE a method its own authors have declared unusable.

    Called from `run.execute.run`, which is where calibrations happen and records are written, so
    the guard covers programmatic callers as well as the CLI.
    """
    d = method.descriptor
    why = QUARANTINED.get((d.name, d.version))
    if why is None:
        return
    raise ContractError(
        f"{d.name} {d.version} is QUARANTINED and must not be run: {why}\n"
        "It stays registered and inspectable on purpose; only execution is refused. If this is "
        "wrong, the entry to remove is in srbench/contract/method.py:QUARANTINED."
    )


def methods_dir() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "methods")


def discover(path: str | None = None) -> list[str]:
    """Import every package under `methods/`, so registration happens as a side effect.

    An adapter that fails to import is reported by name rather than silently skipped: a method
    that vanishes from the listing because of a typo looks exactly like a method nobody wrote.
    """
    root = path or methods_dir()
    if not os.path.isdir(root):
        return []
    failures = []
    for entry in sorted(pkgutil.iter_modules([root])):
        mod_path = os.path.join(root, entry.name, "__init__.py")
        if not os.path.exists(mod_path):
            mod_path = os.path.join(root, f"{entry.name}.py")
        try:
            spec = importlib.util.spec_from_file_location(f"srbench_methods.{entry.name}",
                                                          mod_path)
            if spec is None or spec.loader is None:
                failures.append((entry.name, "no import spec"))
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as ex:  # noqa: BLE001 - report, never swallow
            failures.append((entry.name, f"{type(ex).__name__}: {ex}"))
    if failures:
        lines = "\n".join(f"    {n}: {why}" for n, why in failures)
        raise ContractError(f"method adapters failed to load:\n{lines}")
    return sorted(_REGISTRY)


def check(method: CalibrationMethod, spec=None) -> None:
    """Full pre-run check of one method: descriptor, and its cost declaration if a spec is given."""
    validate_descriptor(method.descriptor)
    if spec is not None:
        c = method.cost(characteristics_from_spec(spec))
        if not isinstance(c, CostDeclaration):
            raise ContractError(
                f"method '{method.descriptor.name}': cost() must return a CostDeclaration, "
                f"got {type(c).__name__}")
        validate_cost_declaration(c, method.descriptor.name)
