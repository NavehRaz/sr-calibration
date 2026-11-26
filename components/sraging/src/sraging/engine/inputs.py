"""Named model inputs beyond SRParams: age segments and heterogeneity (data-model.md; spec FR-002, FR-003).

A Segment changes one parameter on an age window: a MULTIPLIER for eta, beta, epsilon, kappa, xc and the
DPU accumulation rate A (base 1; it speeds up production only), a RATE for the external hazard h_ext (overriding the base rate inside the
window, as the simulator assigns it). Windows are in the cohort's time unit; None means for life.
Refusals name the segments (constitution VIII): two segments on one parameter whose windows overlap
(register D-111; SRK's fp_piecewise multiplied them silently). A window that runs past the last observed
age closes where it says (D-110; fp_piecewise kept it open).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sraging.errors import UnsupportedOptionError

SEG_PARAMS = ("eta", "beta", "epsilon", "kappa", "xc", "A", "h_ext")
MULTIPLIERS = ("eta", "beta", "epsilon", "kappa", "xc", "A")


class SegmentConflictError(UnsupportedOptionError):
    """Two segments change one parameter at one age."""


@dataclass(frozen=True)
class Segment:
    id: str
    parameter: str
    value: float
    window: tuple | None = None

    def __init__(self, *, id, parameter, value, window=None):
        if not isinstance(id, str) or not id or ":" in id:
            raise ValueError(f"Segment.id must be a non-empty string without ':', got {id!r}")
        if parameter not in SEG_PARAMS:
            raise UnsupportedOptionError(parameter, f"a segment changes one of {SEG_PARAMS}, not {parameter!r}")
        v = float(value)
        if not math.isfinite(v) or (parameter in MULTIPLIERS and not v > 0) or (parameter == "h_ext" and v < 0):
            raise ValueError(f"segment {id!r}: {parameter} needs a finite value "
                             f"{'> 0 (a multiplier)' if parameter in MULTIPLIERS else '>= 0 (a rate)'}, got {value!r}")
        if window is not None:
            a, b = float(window[0]), float(window[1])
            if not (math.isfinite(a) and a >= 0 and b > a):
                raise ValueError(f"segment {id!r}: window needs 0 <= start < stop (stop may be inf), got {window!r}")
            window = (a, b)
        for k, x in (("id", id), ("parameter", parameter), ("value", v), ("window", window)):
            object.__setattr__(self, k, x)

    @property
    def start(self) -> float:
        return 0.0 if self.window is None else self.window[0]

    @property
    def stop(self) -> float:
        return math.inf if self.window is None else self.window[1]


@dataclass(frozen=True)
class AgeSegments:
    segments: tuple

    def __init__(self, *segments):
        segs = tuple(segments)
        ids = [s.id for s in segs]
        if len(set(ids)) != len(ids):
            raise UnsupportedOptionError("id", f"segment ids must be unique, got {ids}")
        for i, a in enumerate(segs):
            for b in segs[i + 1:]:
                if a.parameter == b.parameter and a.start < b.stop and b.start < a.stop:
                    raise SegmentConflictError(
                        a.parameter, f"segments {a.id!r} and {b.id!r} both change {a.parameter} on "
                        f"[{max(a.start, b.start)!r}, {min(a.stop, b.stop)!r}); give one segment per parameter per "
                        f"age (register D-111: SRK's fp_piecewise multiplied them silently)")
        object.__setattr__(self, "segments", segs)

    def __iter__(self):
        return iter(self.segments)

    def by_id(self, sid) -> Segment:
        for s in self.segments:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def _sel(self, hazard):
        """The march's segments (hazard=False: everything but h_ext) or the hazard's (hazard=True). The
        hazard acts outside the Fokker-Planck equation, so its edges never touch the time mesh."""
        return [s for s in self.segments if (s.parameter == "h_ext") == hazard]

    def edges(self, hazard=False) -> np.ndarray:
        """The finite window endpoints > 0, sorted and unique (span boundaries, in physical time)."""
        e = {x for s in self._sel(hazard) for x in (s.start, s.stop) if math.isfinite(x) and x > 0}
        return np.array(sorted(e), float)

    def spans(self, hazard=False):
        """[(start, stop)] covering [0, inf) on which the march's parameters (or the hazard) are constant."""
        e = [0.0, *self.edges(hazard).tolist(), math.inf]
        return list(zip(e[:-1], e[1:]))

    def active(self, span, hazard=False):
        """{parameter: segment} active on the span (each span lies wholly inside or outside a window)."""
        lo, hi = span
        mid = lo + 1.0 if not math.isfinite(hi) else 0.5 * (lo + hi)
        return {s.parameter: s for s in self._sel(hazard) if s.start <= mid < s.stop}


HET_PARAMS = ("eta", "beta", "epsilon", "kappa", "xc")
DEFINITIONS = ("truncated_normal", "normal_frac_sd")
M_CAP = 192                         # the ladder's cap (sraging.engine.mixture.CAP; research R3)
M_CHECKED_MAX = 128                 # a checked fixed M is compared with the smallest rung >= ceil(1.5 M) <= 192
MEASURED_SD = (0.1, 0.5)            # the sd range spec 004's panel measured (HET_OUTSIDE_MEASURED_RANGE above it)
_UNSET = object()


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


@dataclass(frozen=True)
class Heterogeneity:
    """One heterogeneous parameter, by its definition's name (spec 004 FR-001, FR-002; data-model "Heterogeneity";
    research R2):

    - `truncated_normal` (the default): the parameter is P s with s = 1 + sd Z, Z standard normal truncated to s > 0
      (gen-1's law); 0 < sd <= 1 (`Heterogeneity.zero(parameter)`, sd = 0 and s = 1 exactly, is a member of a
      Langevin tuple only: research R27's exact reduction; alone it is refused where it is used); M = "auto" (the
      checked ladder, research R3) or a fixed int in [1, 192], checked against the smallest rung >= ceil(1.5 M)
      unless check=False (then never clean);
    - `normal_frac_sd` (spec 002's, by name only): P (1 + v Z) on M Gauss-Hermite nodes, the non-positive ones dropped
      (registers D-116, D-117); v >= 0 and an explicit M; its fused path is unchanged byte for byte.

    A spread given as `v` without naming the definition is refused (OD-8: `v` never silently changes meaning). One
    parameter only: several are a tuple of Heterogeneity objects, which only the Langevin engine accepts."""

    parameter: str
    definition: str = "truncated_normal"
    sd: float | None = None
    v: float | None = None
    M: int | str = "auto"
    check: bool = True

    def __init__(self, *, parameter, sd=None, v=None, definition=_UNSET, M="auto", check=True):
        if isinstance(parameter, (list, tuple)):
            raise UnsupportedOptionError("parameter", "one Heterogeneity is one parameter; several are a tuple of "
                                         "Heterogeneity objects, which only the Langevin engine accepts (spec 004 "
                                         "FR-023); the FP engine refuses them")
        if parameter not in HET_PARAMS:
            raise UnsupportedOptionError("parameter", f"heterogeneity is in one of {HET_PARAMS}, not {parameter!r}")
        named = definition is not _UNSET
        definition = "truncated_normal" if not named else definition
        if definition not in DEFINITIONS:
            raise UnsupportedOptionError("definition", f"unknown heterogeneity definition {definition!r}; the "
                                         f"definitions are 'truncated_normal' (sd, the default) and 'normal_frac_sd' "
                                         f"(spec 002's v)")
        if sd is not None and v is not None:
            raise UnsupportedOptionError("sd", "give one spread: sd (truncated_normal) or v (normal_frac_sd), not both")
        if v is not None and not named:
            raise UnsupportedOptionError("v", "name the definition: v is spec 002's spread of 'normal_frac_sd' "
                                         "(P(1 + vZ), Gauss-Hermite with dropped nodes); the default is "
                                         "'truncated_normal' with sd. Pass definition='normal_frac_sd' with v, or sd= "
                                         "for the truncated normal (OD-8)")
        if not isinstance(check, bool):
            raise UnsupportedOptionError("check", f"check is True or False, got {check!r}")
        if definition == "truncated_normal":
            sd, M = self._truncated_normal(sd, v, M, check)
        else:
            v, M = self._normal_frac_sd(sd, v, M, check)
        for k, x in (("parameter", parameter), ("definition", definition), ("sd", sd), ("v", v), ("M", M),
                     ("check", check)):
            object.__setattr__(self, k, x)

    @classmethod
    def zero(cls, parameter) -> "Heterogeneity":
        """A zero-spread member, sd = 0 (s = 1.0 exactly, no draw), for a Langevin heterogeneity tuple only (research
        R27: a tuple whose other spreads are 0 equals the single-parameter draw bit for bit). The constructor keeps
        refusing sd = 0 (R2: omit the heterogeneity), and every single-parameter entry point refuses this member."""
        h = cls(parameter=parameter, sd=1.0)
        object.__setattr__(h, "sd", 0.0)
        return h

    @property
    def zero_spread(self) -> bool:
        return self.definition == "truncated_normal" and self.sd == 0.0

    @staticmethod
    def _truncated_normal(sd, v, M, check):
        if v is not None:
            raise UnsupportedOptionError("v", "'truncated_normal' takes sd, not v (v is 'normal_frac_sd''s spread)")
        if sd is None:
            raise UnsupportedOptionError("sd", "'truncated_normal' needs its spread sd (0 < sd <= 1)")
        try:
            sd = float(sd)
        except (TypeError, ValueError):
            raise UnsupportedOptionError("sd", f"sd must be a number in (0, 1], got {sd!r}") from None
        if not (math.isfinite(sd) and 0.0 < sd <= 1.0):
            raise UnsupportedOptionError("sd", f"sd must be finite and in (0, 1], got {sd!r} (sd = 0: omit the "
                                         f"heterogeneity)")
        if M == "auto":
            if not check:
                raise UnsupportedOptionError("check", "check=False needs a fixed integer M: M='auto' is the checked "
                                             "ladder (research R3)")
        elif not _is_int(M) or not 1 <= M <= M_CAP:
            raise UnsupportedOptionError("M", f"M is 'auto' or an integer in [1, {M_CAP}] (the cap), got {M!r}")
        elif check and M > M_CHECKED_MAX:
            raise UnsupportedOptionError("M", f"a checked fixed M is compared with the smallest rung >= ceil(1.5 M), "
                                         f"which for M = {M} lies above the cap {M_CAP}; a checked M is at most "
                                         f"{M_CHECKED_MAX}, or pass check=False")
        return sd, M

    @staticmethod
    def _normal_frac_sd(sd, v, M, check):
        if sd is not None:
            raise UnsupportedOptionError("sd", "'normal_frac_sd' takes v, not sd (sd is 'truncated_normal''s spread)")
        if v is None:
            raise UnsupportedOptionError("v", "'normal_frac_sd' needs its spread v")
        v = float(v)
        if not (math.isfinite(v) and v >= 0):
            raise ValueError(f"heterogeneity v must be finite and >= 0, got {v!r}")
        if not (_is_int(M) and M >= 1):
            raise UnsupportedOptionError("M", f"'normal_frac_sd' needs an explicit integer M >= 1, got {M!r}: "
                                         f"automatic M is not ported for it (SRK's choose_M); it is the "
                                         f"truncated normal's")
        if not check:
            raise UnsupportedOptionError("check", "'normal_frac_sd''s value is never checked (spec 002's fused path); "
                                         "Evaluator.m_report's whole block reports its change with M")
        return v, M

    @property
    def spread(self) -> float:
        return self.sd if self.definition == "truncated_normal" else self.v

    @property
    def warnings(self) -> tuple:
        """HET_OUTSIDE_MEASURED_RANGE above the panel's sd range (data-model): such a value is never clean."""
        if self.definition == "truncated_normal" and self.sd > MEASURED_SD[1]:
            from sraging.warnings import sampler_warning
            return (sampler_warning("HET_OUTSIDE_MEASURED_RANGE", sd=self.sd, lo=MEASURED_SD[0], hi=MEASURED_SD[1]),)
        return ()


def refuse_zero_spread(het, where: str) -> None:
    """One heterogeneous parameter at sd = 0 is refused where it is used (R2: omit the heterogeneity); a zero spread
    is a Langevin tuple member's only (research R27: s = 1.0 exactly, the exact reduction)."""
    if isinstance(het, Heterogeneity) and het.zero_spread:
        raise UnsupportedOptionError("sd", f"{where}: sd must be in (0, 1] for one heterogeneous parameter, got 0 "
                                     f"(sd = 0: omit the heterogeneity; Heterogeneity.zero is a member of a Langevin "
                                     f"heterogeneity tuple only)")
