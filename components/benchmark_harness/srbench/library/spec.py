"""The frozen cohort specification: everything needed to regenerate one cohort and know its truth.

NAMING RULE, enforced by the field names here: a dimensionless quantity and its physical
counterpart never share a name. Dimensionless parameters carry the `rho_` prefix; physical ones
do not. `kappa` is already dimensionless and so carries no prefix -- that is the one exception and
it is why it is called out here rather than left to be rediscovered.

Mislabelling the two has cost this project a factor of L^2 once (an axis labelled `rho_*` that
held physical values) and a factor of L twice (a negligibility floor applied without its scale
factor, once 926x too strict and once L x too lenient).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

#: The observation-model groups. Scored SEPARATELY and never pooled: a coverage number measured
#: on `pure` does not transfer to a cohort with a background hazard, and 24 of 27 real fits carry
#: at least one feature.
ARMS = ("pure", "hazard", "window", "het", "censor", "realfull")


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """One cohort's identity, truth and observation model."""

    cohort_id: str
    position: int                      # index in the library's fixed order; defines every prefix
    arm: str

    # --- dimensionless truth, in the mean-lifetime-1 convention ---
    rho_eta: float
    rho_beta: float
    rho_eps: float
    kappa: float                       # already dimensionless; no rho_ prefix by definition

    # --- observation model ---
    n: int
    L_phys: float = 1.0                # physical mean lifetime; sets the time unit
    seed: int = 0
    delta: float | None = None         # recording resolution, in the ages' own units

    # data-level features: baked into the frozen cohort at build time
    external_hazard: float | None = None
    het_param: str | None = None
    het_var: float = 0.0
    censor_frac: float = 0.0

    # presentation-level feature: applied at run time by the observation transform (research D6),
    # because `materialize` upstream never reads it -- the shipped `var_trunc` variant produces a
    # cohort bit-identical to a pure one.
    window: tuple[float, float] | None = None

    #: Which BLOCK of the library this cohort belongs to. "main" is the tiered body -- quick,
    #: medium and full are prefixes of it. A cohort outside "main" is reachable only by asking for
    #: its group by name, so an optional block can be added without changing what any tier means.
    #: Defaults to "main" so a manifest written before groups existed still loads.
    group: str = "main"

    # --- bookkeeping ---
    stratum: str = ""                  # the cell used for per-stratum claims
    cluster_key: str = ""              # the parameter point; the unit of clustering
    real_name: str | None = None       # set when planted at a real fitted cohort's parameters
    notes: str = ""

    # ---------------------------------------------------------------- derived truth
    @property
    def ratio(self) -> float:
        """rho_beta / rho_eta -- the removal timescale in lifespans. One of the two manifold
        coordinates a survival curve actually determines."""
        return self.rho_beta / self.rho_eta if self.rho_eta else float("nan")

    @property
    def log_ratio(self) -> float:
        r = self.ratio
        return math.log10(r) if r > 0 else float("nan")

    def physical_truth(self) -> dict[str, float]:
        """The physical (eta, beta, epsilon, kappa) at xc = 1 and L = L_phys.

        THE SINGLE CONVERSION SITE. Every physical value in the harness comes through here, so
        that the dimensionless/physical confusion has exactly one place to be got wrong and one
        place to be tested.

            rho_eta = eta * L^2 / xc     rho_beta = beta * L / xc     rho_eps = eps * L / xc^2
        """
        L = float(self.L_phys)
        out = {
            "eta": self.rho_eta / (L * L),
            "beta": self.rho_beta / L,
            "epsilon": self.rho_eps / L,
            "kappa": float(self.kappa),
        }
        # THE EXTERNAL HAZARD, present only on cohorts that carry one.
        #
        # Conditional on purpose, and it does the work of a dozen guards: every scorer already
        # reads `truths.get(cid, {}).get(param)` and skips a cohort with no truth, so a pure
        # cohort contributes to no denominator, no decline count and no coverage for this quantity
        # -- automatically, with nothing to remember.
        #
        # SCORED AS THE RATE, not as the stored level. `external_hazard` holds ExtH, a log-scale
        # Makeham level in this cohort's own age units (`plan.hazard_in_age_units`), and ExtH can
        # be negative -- which `claims.log_error` cannot score, since it takes a log ratio. The
        # rate h = exp(-ExtH) is positive and log-comparable like the other four, so that is what
        # a method claims and what it is scored against. This is the single conversion site, which
        # is why the conversion belongs here and nowhere else.
        if self.external_hazard is not None:
            out["external_hazard"] = math.exp(-float(self.external_hazard))
        return out

    @property
    def features(self) -> tuple[str, ...]:
        """Which observation-model features this cohort actually carries."""
        out = []
        if self.external_hazard is not None:
            out.append("external_hazard")
        if self.window is not None:
            out.append("fit_window")
        if self.het_param is not None:
            out.append("heterogeneity")
        if self.censor_frac:
            out.append("censoring")
        return tuple(out)

    # ---------------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        d = asdict(self)
        d["window"] = list(self.window) if self.window is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CohortSpec":
        w = d.get("window")
        return cls(
            cohort_id=str(d["cohort_id"]),
            position=int(d["position"]),
            arm=str(d["arm"]),
            group=str(d.get("group", "main")),
            rho_eta=float(d["rho_eta"]),
            rho_beta=float(d["rho_beta"]),
            rho_eps=float(d["rho_eps"]),
            kappa=float(d["kappa"]),
            n=int(d["n"]),
            L_phys=float(d.get("L_phys", 1.0)),
            seed=int(d.get("seed", 0)),
            delta=None if d.get("delta") is None else float(d["delta"]),
            external_hazard=(None if d.get("external_hazard") is None
                             else float(d["external_hazard"])),
            het_param=d.get("het_param") or None,
            het_var=float(d.get("het_var", 0.0)),
            censor_frac=float(d.get("censor_frac", 0.0)),
            window=None if w is None else (float(w[0]), float(w[1])),
            stratum=str(d.get("stratum", "")),
            cluster_key=str(d.get("cluster_key", "")),
            real_name=d.get("real_name") or None,
            notes=str(d.get("notes", "")),
        )


def arm_for(external_hazard=None, window=None, het_param=None, censor_frac=0.0) -> str:
    """Which arm a feature combination belongs to.

    Used at build time to CHECK the arm label against the features actually present, because a
    mislabelled arm is exactly the upstream `var_trunc` defect: a cohort carrying a "truncated"
    label and no truncation.
    """
    present = [bool(external_hazard is not None), bool(window is not None),
               bool(het_param is not None), bool(censor_frac)]
    if not any(present):
        return "pure"
    if sum(present) > 1:
        return "realfull"
    return ("hazard", "window", "het", "censor")[present.index(True)]
