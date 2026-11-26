"""The parity comparisons, in one place (used by tests/parity and tools/acceptance.py; research R6, R7).

Each function returns a list of differences (empty = identical). The planted-perturbation tests
(tests/parity/test_perturbation.py) run exactly these functions on perturbed inputs and require them to
report a difference (FR-020), so a check that cannot fail is caught.
"""
from __future__ import annotations

import math

import numpy as np

TIMING_DIAGNOSTICS = ("compile_s", "seed_s", "mle_s")
CLAIM_FIELDS = ("parameter", "shape", "lower", "upper", "point", "level", "n_components")


def bytes_diff(port, oracle, what="value") -> list[str]:
    """Bit-for-bit equality of two numeric results (arrays, scalars or tuples of them)."""
    if isinstance(port, tuple) or isinstance(oracle, tuple):
        if not (isinstance(port, tuple) and isinstance(oracle, tuple) and len(port) == len(oracle)):
            return [f"{what}: structure differs"]
        out = []
        for i, (a, b) in enumerate(zip(port, oracle)):
            out += bytes_diff(a, b, f"{what}[{i}]")
        return out
    a, b = np.asarray(port), np.asarray(oracle)
    if a.dtype != b.dtype or a.shape != b.shape:
        return [f"{what}: dtype/shape {a.dtype}{a.shape} != {b.dtype}{b.shape}"]
    if a.tobytes() != b.tobytes():
        return [f"{what}: bytes differ (port {a.ravel()[:4]!r} vs oracle {b.ravel()[:4]!r})"]
    return []


def arrays_diff(ages, events, ref_ages, ref_events, what="cohort") -> list[str]:
    out = bytes_diff(np.asarray(ages), np.asarray(ref_ages), f"{what}.ages")
    out += bytes_diff(np.asarray(events), np.asarray(ref_events), f"{what}.events")
    return out


def _canon(x):
    if isinstance(x, (np.floating,)):
        x = float(x)
    elif isinstance(x, np.integer):
        x = int(x)
    elif isinstance(x, np.bool_):
        x = bool(x)
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, tuple):
        x = list(x)
    return x


def tree_diff(port, oracle, path="") -> list[str]:
    """Recursive exact comparison; floats by float.hex (so -0.0 != 0.0 and NaN == NaN)."""
    a, b = _canon(port), _canon(oracle)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present only in {'oracle' if k not in a else 'port'}")
            else:
                out += tree_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += tree_diff(x, y, f"{path}[{i}]")
        return out
    if isinstance(a, bool) or isinstance(b, bool):
        return [] if (type(a) is type(b) and a == b) else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, float) and isinstance(b, float):
        same = (math.isnan(a) and math.isnan(b)) or a.hex() == b.hex()
        return [] if same else [f"{path}: {a!r} != {b!r}"]
    if isinstance(a, int) and isinstance(b, int):
        return [] if a == b else [f"{path}: {a!r} != {b!r}"]
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} {a!r} != {type(b).__name__} {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def oracle_identity_view(cs) -> dict:
    """The research-R7 identity set of a reference srbench ClaimSet (object or to_dict() form)."""
    if not isinstance(cs, dict):
        cs = {"status": cs.status, "claims": [{k: getattr(c, k) for k in CLAIM_FIELDS} for c in cs.claims],
              "diagnostics": dict(cs.diagnostics or {})}
    diag = {k: v for k, v in (cs.get("diagnostics") or {}).items() if k not in TIMING_DIAGNOSTICS + ("code",)}
    claims = [{k: c.get(k, {"lower": None, "upper": None, "point": None, "level": 0.95,
                            "n_components": 1}.get(k)) for k in CLAIM_FIELDS} for c in cs.get("claims", [])]
    return {"status": cs.get("status"), "claims": claims, "diagnostics": diag}


def identity_diff(port_view, oracle_view) -> list[str]:
    return tree_diff(port_view, oracle_view, "")
