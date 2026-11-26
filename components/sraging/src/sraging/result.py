"""FitResult and its parts (data-model.md; FR-004, FR-007; research R7, R11; spec 003 schema /4: FR-026, research R6).

JSON is Python's dialect: floats via repr (exact float64 round trip), NaN/Infinity tokens allowed.
`identity_view()` is the projection compared with the reference: the claims and the reference's
diagnostics minus the timings and the code-provenance line.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from sraging.params import SRParams

SCHEMA = "sraging.fitresult/4"
READABLE = ("sraging.fitresult/1", "sraging.fitresult/2", "sraging.fitresult/3", SCHEMA)   # older files are upgraded
CHECK_CODES = ("BASIN_GAIN", "BASIN_OUTSIDE", "CENSOR_TIE", "DELTA_GRID", "BOX_EDGE", "FLOORED_DEATHS",
               "AGE0_DEATHS")
#: sraging's own; never the reference comparison. Spec 006 adds the window W and H fitted (research R5) and theta at
#: both kappa ends (R6), so the schema stays /4
SRAGING_DIAGNOSTICS = ("code", "floor_report", "leverage", "notes", "window", "kappa_ends")
TIMING_DIAGNOSTICS = ("compile_s", "seed_s", "mle_s")
CLAIM_FIELDS = ("parameter", "shape", "lower", "upper", "point", "level", "n_components")


@dataclass(frozen=True)
class Claim:
    """What the data support about one quantity; the reference's fields plus a label."""

    parameter: str
    shape: str
    lower: float | None
    upper: float | None
    point: float | None
    level: float
    n_components: int
    label: str = ""
    scored: bool = True                     # spec 003: False for anything not truth-scored
    label_info: dict | None = None          # {kind, edge, log10_cap_over_floor, regime} (data-model.md)
    widening: dict | None = None            # {claims_profile, lower_moved_dex, upper_moved_dex}
    decline_reason: str | None = None       # why a quantity has no claim (counted in the decline rate)

    def reference_dict(self) -> dict:
        return {k: getattr(self, k) for k in CLAIM_FIELDS}

    def to_dict(self) -> dict:
        return {**self.reference_dict(), "label": self.label, "scored": self.scored, "label_info": self.label_info,
                "widening": self.widening, "decline_reason": self.decline_reason}


class Claims(tuple):
    """The claims of a fit in report order, indexable by position or by quantity key (spec 003 FR-026):
    `claims[0]`, `claims["eta_over_eps"]`; iteration yields the Claim objects."""

    def __new__(cls, items=()):
        items = tuple(items)
        keys = [c.parameter for c in items]
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if dup:
            raise ValueError(f"duplicate claim quantities: {dup}")
        return super().__new__(cls, items)

    def __getitem__(self, k):
        if isinstance(k, str):
            for c in self:
                if c.parameter == k:
                    return c
            raise KeyError(k)
        return super().__getitem__(k)

    def keys(self) -> tuple:
        return tuple(c.parameter for c in self)

    def get(self, k, default=None):
        try:
            return self[k]
        except KeyError:
            return default


@dataclass(frozen=True)
class Warning:  # noqa: A001 - the domain name; never confused with the builtin inside sraging
    code: str
    message: str
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CheckRecord:
    code: str
    ran: bool
    fired: bool


def plain(x):
    """numpy scalars/arrays and tuples -> plain JSON-able Python values (bit-preserving for floats)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return [plain(v) for v in x.tolist()]
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


@dataclass(frozen=True)
class FitResult:
    """The outcome of one fit. See data-model.md for every field."""

    status: str
    cohort: dict
    theta_hat: SRParams | None
    theta_labels: dict
    claims: tuple
    loglik: float
    warnings: tuple
    checks: tuple
    diagnostics: dict
    timing: dict
    provenance: dict
    nuisance: dict | None = None            # spec 003: profile_jax2_h's external hazard (never claimed)
    mechanisms: dict | None = None          # spec 003: what fired in a jax2 fit
    gap_nats: float | None = None           # best evaluated point minus theta-hat (jax2)
    archive_points: int | None = None       # how many points the fit evaluated (the archive is not stored)
    combinations: dict | None = None        # {on, cost_s, order, order_source}
    schema: str = SCHEMA

    def __post_init__(self):
        if not isinstance(self.claims, Claims):
            object.__setattr__(self, "claims", Claims(self.claims))

    @property
    def calibrator(self) -> dict | None:
        """{name, version} of the calibrator that produced the fit (recorded in provenance)."""
        return (self.provenance or {}).get("calibrator")

    # ---- views ------------------------------------------------------------------------------
    @property
    def clean(self) -> bool:
        """True only if every check ran and none fired (constitution V: WARN or NOT RUN is not PASS), and the
        fit did not fail inside the reference (D-015: every MLE start failed, status still "ok")."""
        return (bool(self.checks) and all(c.ran and not c.fired for c in self.checks) and not self.warnings
                and not self.all_starts_failed and not self.leverage_not_run)

    @property
    def leverage_not_run(self) -> bool:
        """Spec 003f: a jax2 result with no leverage block (written before 003f) has not been checked for leverage
        (constitution X). profile_v033 runs its own check since spec 006; its earlier results, written under X's
        recorded deviation, are not marked here (their reading is unchanged)."""
        return (self.provenance or {}).get("numerics") == "jax2" and "leverage" not in (self.diagnostics or {})

    @property
    def all_starts_failed(self) -> bool:
        """D-015: v0.3.3 reports status ok with four none claims when every MLE start fails (FR-018)."""
        return (self.diagnostics or {}).get("error") == "all starts failed"

    def identity_view(self) -> dict:
        """The reference-comparable projection (research R7)."""
        diag = {k: v for k, v in self.diagnostics.items() if k not in TIMING_DIAGNOSTICS + SRAGING_DIAGNOSTICS}
        return {"status": self.status, "claims": [c.reference_dict() for c in self.claims],
                "diagnostics": plain(diag)}

    # ---- JSON ------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        th = None
        if self.theta_hat is not None:
            th = {**self.theta_hat.as_dict(), "xc_fixed": True}
        return plain({
            "schema": self.schema, "status": self.status, "cohort": self.cohort, "theta_hat": th,
            "theta_labels": self.theta_labels,
            "claims": {c.parameter: c.to_dict() for c in self.claims},
            "loglik": self.loglik,
            "warnings": [{"code": w.code, "message": w.message, "data": w.data} for w in self.warnings],
            "checks": [{"code": c.code, "ran": c.ran, "fired": c.fired} for c in self.checks],
            "diagnostics": self.diagnostics, "timing": self.timing, "provenance": self.provenance,
            "nuisance": self.nuisance, "mechanisms": self.mechanisms, "gap_nats": self.gap_nats,
            "archive_points": self.archive_points, "combinations": self.combinations,
        })

    def to_json(self, path=None) -> str:
        text = json.dumps(self.to_dict(), indent=1, allow_nan=True)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text + "\n")
        return text

    @classmethod
    def from_dict(cls, d: dict) -> "FitResult":
        if d.get("schema") not in READABLE:
            raise ValueError(f"unknown FitResult schema {d.get('schema')!r}; this sraging reads {READABLE}")
        checks = [CheckRecord(**c) for c in d["checks"]]
        if d["schema"] == "sraging.fitresult/1":           # read-only upgrade: the sixth check did not exist
            checks.append(CheckRecord("FLOORED_DEATHS", False, False))
        if d["schema"] in ("sraging.fitresult/1", "sraging.fitresult/2"):   # nor the seventh
            checks.append(CheckRecord("AGE0_DEATHS", False, False))
        th = d.get("theta_hat")
        theta = None
        if th is not None:
            theta = SRParams(eta=th["eta"], beta=th["beta"], epsilon=th["epsilon"], kappa=th["kappa"],
                             xc=th["xc"], h_ext=th["h_ext"], time_unit=th["time_unit"])
        raw = d["claims"]
        claims = Claims(Claim(**c) for c in (raw.values() if isinstance(raw, dict) else raw))   # /4 keyed; /1-/3 a list
        prov = d["provenance"] if d["schema"] == SCHEMA else {**d["provenance"], "upgraded_from": d["schema"]}
        return cls(
            status=d["status"], cohort=d["cohort"], theta_hat=theta, theta_labels=d["theta_labels"],
            claims=claims, loglik=d["loglik"],
            warnings=tuple(Warning(**w) for w in d["warnings"]),
            checks=tuple(checks),
            diagnostics=d["diagnostics"], timing=d["timing"], provenance=prov,
            nuisance=d.get("nuisance"), mechanisms=d.get("mechanisms"), gap_nats=d.get("gap_nats"),
            archive_points=d.get("archive_points"), combinations=d.get("combinations"), schema=SCHEMA)

    @classmethod
    def from_json(cls, text_or_path) -> "FitResult":
        s = str(text_or_path)
        if not s.lstrip().startswith("{") and os.path.exists(s):
            with open(s) as fh:
                s = fh.read()
        return cls.from_dict(json.loads(s))


def canon(x):
    """A comparison key that is exact for floats (float.hex: distinguishes -0.0 and NaN payload class)."""
    x = plain(x)
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return ("int", x)
    if isinstance(x, float):
        return ("float", "nan" if math.isnan(x) else x.hex())
    if isinstance(x, list):
        return [canon(v) for v in x]
    if isinstance(x, dict):
        return {k: canon(v) for k, v in sorted(x.items())}
    return ("repr", repr(x))
