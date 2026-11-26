"""ParamLayout: the named order of the log-parameter vector at a jit boundary (research R1; data-model.md).

An optimiser needs one flat vector, and the engine's jitted kernels take one (with every feature off it
must be exactly v0.3.3's z = log(eta, beta, epsilon, kappa), so the program is the same). Constitution
VIII forbids an order that is implicit: here the order is carried by this object, built from names, and
no public function accepts or returns a bare vector without its layout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from sraging.errors import UnsupportedOptionError

MODEL_NAMES = ("eta", "beta", "epsilon", "kappa", "xc", "h_ext", "het_v")
DEFAULT_FREE = ("eta", "beta", "epsilon", "kappa")
# Inputs that snap to the grid have no gradient (FR-008); they can never be free.
SNAPPED = ("entry", "window", "delta", "segment_edge", "segment_xc")


def _check_name(name: str) -> None:
    if name in MODEL_NAMES or (name.startswith("seg:") and len(name) > 4):
        return
    if name in SNAPPED or name.startswith("edge:"):
        raise UnsupportedOptionError(name,
            f"{name!r} snaps to the grid (segment edges, entry ages, delta, a segment's xc) and has no "
            f"gradient; it cannot be a free parameter (FR-008)")
    raise UnsupportedOptionError(name, f"unknown parameter name {name!r}; free parameters are {MODEL_NAMES} "
                                 f"or 'seg:<id>'")


@dataclass(frozen=True)
class ParamLayout:
    """`names`: the free parameters in vector order (each enters as its natural log). `fixed`: every
    other model input, by name, at its value (not logged)."""

    names: tuple[str, ...] = DEFAULT_FREE
    fixed: tuple[tuple[str, float], ...] = field(default=())

    def __post_init__(self):
        names = tuple(self.names)
        if len(set(names)) != len(names):
            raise UnsupportedOptionError("free", f"duplicate free parameter in {names}")
        for n in names:
            _check_name(n)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "fixed", tuple(sorted(dict(self.fixed).items())))

    @property
    def size(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def pack(self, values: dict) -> np.ndarray:
        """Named positive values -> the log vector, in layout order."""
        missing = [n for n in self.names if n not in values]
        if missing:
            raise UnsupportedOptionError(missing[0], f"values for {missing} are missing")
        out = []
        for n in self.names:
            v = float(values[n])
            if not (math.isfinite(v) and v > 0):
                raise ValueError(f"{n} must be finite and > 0 to enter the log vector, got {v!r}")
            out.append(math.log(v))
        return np.asarray(out, float)

    def unpack(self, z) -> dict:
        """Inside or outside a trace: name -> z[i] (indexing only, no other operation)."""
        return {n: z[i] for i, n in enumerate(self.names)}

    def grad_dict(self, g) -> dict:
        """The gradient vector (d loglik / d log value) -> name -> float."""
        g = np.asarray(g, float)
        return {n: float(g[i]) for i, n in enumerate(self.names)}
