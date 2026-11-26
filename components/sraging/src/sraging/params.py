"""SRParams: the SR model parameters, by name (data-model.md; constitution VIII).

Physical units; xc = 1 by default. No positional constructor and no iteration: parameters are always
addressed by name. Positional vectors exist only inside `sraging.compat`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(name, v):
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
        raise ValueError(f"SRParams.{name} must be a finite number, got {v!r}")
    return float(v)


@dataclass(frozen=True, kw_only=True)
class SRParams:
    """eta, beta, epsilon, kappa (physical units), xc (default 1), h_ext (default 0), time_unit (label)."""

    eta: float
    beta: float
    epsilon: float
    kappa: float
    xc: float = 1.0
    h_ext: float = 0.0
    time_unit: str = "input units"

    def __post_init__(self):
        for name, rule in (("eta", "> 0"), ("beta", ">= 0"), ("epsilon", "> 0"), ("kappa", "> 0"),
                           ("xc", "> 0"), ("h_ext", ">= 0")):
            v = _finite(name, getattr(self, name))
            if (rule == "> 0" and not v > 0) or (rule == ">= 0" and not v >= 0):
                raise ValueError(f"SRParams.{name} must be {rule}, got {v!r}")
            object.__setattr__(self, name, v)
        if not isinstance(self.time_unit, str):
            raise ValueError(f"SRParams.time_unit must be a string, got {self.time_unit!r}")

    def dimensionless(self, L: float) -> dict[str, float]:
        """The named dimensionless groups for an explicit time scale L.

        rho_eta = L^2 eta / xc, rho_beta = L beta / xc, rho_epsilon = L epsilon / xc^2, kappa / xc:
        invariant under the exact symmetry (xc, eta, beta, epsilon, kappa) -> (c xc, c eta, c beta,
        c^2 epsilon, c kappa). At xc = 1 they are the groups fp_jax forms (rh = L**2*eta, rb = L*beta,
        re = L*eps).
        """
        L = _finite("L", L)
        if not L > 0:
            raise ValueError(f"dimensionless(L) needs L > 0, got {L!r}")
        return {"rho_eta": L ** 2 * self.eta / self.xc, "rho_beta": L * self.beta / self.xc,
                "rho_epsilon": L * self.epsilon / self.xc ** 2, "kappa": self.kappa / self.xc}

    def as_dict(self) -> dict:
        return {"eta": self.eta, "beta": self.beta, "epsilon": self.epsilon, "kappa": self.kappa,
                "xc": self.xc, "h_ext": self.h_ext, "time_unit": self.time_unit}
