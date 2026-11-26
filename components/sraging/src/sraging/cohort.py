"""Cohort: one group's death and censoring data, immutable (data-model.md).

`from_arrays` is the core constructor; `from_csv` (FR-002, FR-003) reads the reference's two-column CSV
with its header rule but REFUSES every row the reference reader would silently skip or recode (research
R9; register D-101). Arrays are built exactly as the reference builds them -- `np.asarray(.., float)` /
`np.asarray(.., int)` -- so a valid file loads byte-identically (FR-023).
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field

import numpy as np

from sraging.errors import CohortDataError, CohortSpecError

_DELTA_WHY = ("the recording resolution delta changes how every death is scored (with delta, a death is "
              "scored as happening somewhere inside its recording interval; without it, at exactly that "
              "age). Give delta in the ages' units, or state explicitly that the ages are continuous times.")


def _spec(delta, continuous):
    if continuous and delta is not None:
        raise CohortSpecError(f"give either delta or continuous=True, not both: {_DELTA_WHY}")
    if not continuous and delta is None:
        raise CohortSpecError(f"no recording resolution given: {_DELTA_WHY}")
    if delta is not None:
        if np.ndim(delta) == 1 and not isinstance(delta, (str, bytes)):
            d = np.asarray(delta, float)                 # per-row resolution (SRK's sampling_dt array)
            if d.size < 1 or not np.all(np.isfinite(d)) or not np.all(d > 0):
                raise CohortSpecError("a per-row delta must be finite and > 0 in every row")
            return d, bool(continuous)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) \
                or not delta > 0:
            raise CohortSpecError(f"delta must be a finite number > 0, got {delta!r}")
        delta = float(delta)
    return delta, bool(continuous)


def _readonly(a):
    a.setflags(write=False)
    return a


def _age_reason(a, e, delta):
    """Why a row's age is refused, or None. With delta, a death at age 0 is a death found at the first check:
    it is scored over the first recording interval [0, delta) (Naveh 2026-10-03; register D-105). The
    reference reader drops every age <= 0 row silently (D-101)."""
    if a < 0:
        return f"age {a!r} is negative"
    if a == 0:
        if delta is None:
            return ("age 0 with continuous times: a death or censoring at exactly 0 cannot be scored (with "
                    "--delta, a death at age 0 means one in the first interval [0, delta))")
        if e == 0:
            return "censored at age 0: it carries no information; remove the row"
    return None


@dataclass(frozen=True, eq=False)
class Cohort:
    """ages (float64, read-only), events (int64 in {0, 1}, read-only), delta or continuous, name."""

    ages: np.ndarray
    events: np.ndarray
    delta: float | np.ndarray | None
    continuous: bool
    name: str | None = field(default=None)
    entry: np.ndarray | None = field(default=None)       # per-row entry ages (left truncation; spec 002)
    window: tuple | None = field(default=None)           # common fit window (t_lo, t_hi) (spec 002)
    time_unit: str = field(default="input units")        # must equal SRParams.time_unit in the engine

    @classmethod
    def from_arrays(cls, ages, events, *, delta=None, continuous=False, name=None, entry=None,
                    window=None, time_unit="input units") -> "Cohort":
        delta, continuous = _spec(delta, continuous)
        try:
            a = np.asarray(ages, float)
            ev_raw = np.asarray(events, float)
        except (TypeError, ValueError) as ex:
            raise CohortDataError(f"ages and events must be numeric: {ex}") from None
        if a.ndim != 1 or a.size < 1:
            raise CohortDataError(f"ages must be a 1-d sequence with n >= 1, got shape {a.shape}")
        if ev_raw.shape != a.shape:
            raise CohortDataError(f"events must have the same length as ages ({a.size}), got {ev_raw.shape}")
        if np.ndim(delta) == 1 and delta.shape != a.shape:
            raise CohortSpecError(f"a per-row delta must have one value per row ({a.size}), got {delta.shape}")
        rows = []
        for i in np.flatnonzero(~np.isfinite(a)):
            rows.append((int(i) + 1, f"age {a[i]!r} is not finite"))
        for i in np.flatnonzero(np.isfinite(a)):
            why = _age_reason(float(a[i]), float(ev_raw[i]), delta if np.ndim(delta) == 0 else delta[i])
            if why:
                rows.append((int(i) + 1, why))
        for i in np.flatnonzero(~np.isin(ev_raw, (0.0, 1.0))):
            rows.append((int(i) + 1, f"event {ev_raw[i]!r} is not 0 or 1"))
        if rows:
            raise CohortDataError("cohort refused", sorted(rows))
        e = np.asarray(events, int)                     # validated: exactly the reference's construction
        if not np.any(e == 1):
            raise CohortDataError("cohort refused: no deaths (every event is 0)")
        if a is ages:                                  # never alias (and freeze) the caller's array
            a = a.copy()
        if e is events:
            e = e.copy()
        if np.ndim(delta) == 1:
            delta = _readonly(np.array(delta, float))
        if not isinstance(time_unit, str):
            raise CohortSpecError(f"time_unit must be a string, got {time_unit!r}")
        ent = _entry(entry, window, a, e, delta)
        win = _window(window, entry, delta)
        return cls(_readonly(a), _readonly(e), delta, continuous, name, ent, win, time_unit)

    @classmethod
    def from_csv(cls, path, *, delta=None, continuous=False, name=None, entry=None, window=None,
                 time_unit="input units") -> "Cohort":
        delta, continuous = _spec(delta, continuous)
        if np.ndim(delta) == 1:
            raise CohortSpecError("from_csv takes one delta for the file; build a per-row delta with "
                                  "Cohort.from_arrays")
        ages, events = read_csv(path, delta=delta)
        return cls.from_arrays(ages, events, delta=delta, continuous=continuous,
                               name=name if name is not None else os.path.basename(str(path)),
                               entry=entry, window=window, time_unit=time_unit)

    @property
    def n(self) -> int:
        return int(self.ages.size)

    @property
    def n_deaths(self) -> int:
        return int(np.sum(self.events == 1))

    @property
    def n_censored(self) -> int:
        return int(np.sum(self.events == 0))

    @property
    def has_features(self) -> bool:
        """True when the cohort uses anything spec 001's calibrator does not support."""
        return self.entry is not None or self.window is not None or np.ndim(self.delta) == 1

    @property
    def n_excluded_by_window(self) -> int:
        """Rows recorded before the window opens: excluded from a windowed likelihood, and counted."""
        return 0 if self.window is None else int(np.sum(self.ages < self.window[0]))

    @property
    def n_at_risk(self) -> int:
        return self.n - self.n_excluded_by_window


_GRID_TOL = 1e-9


def _on_grid(x, delta) -> bool:
    k = round(x / delta)
    return abs(x - k * delta) <= _GRID_TOL * max(1.0, abs(x))


def _entry(entry, window, a, e, delta):
    """Per-row entry ages (left truncation), validated row by row (research R8)."""
    if entry is None:
        return None
    if window is not None:
        raise CohortSpecError("give either per-row entry ages or a common window, not both (a window is "
                              "entry at t_lo for every row)")
    en = np.asarray(entry, float)
    if en.shape != a.shape:
        raise CohortSpecError(f"entry must have one age per row ({a.size}), got shape {en.shape}")
    rows = []
    for i in range(a.size):
        x = float(en[i])
        d = None if delta is None else float(delta if np.ndim(delta) == 0 else delta[i])
        if not math.isfinite(x) or x < 0:
            rows.append((i + 1, f"entry age {x!r} must be finite and >= 0"))
        elif e[i] == 1 and d is not None and not x < a[i] + d:
            rows.append((i + 1, f"entry {x!r} is at or after the end of the death's recording interval "
                                f"[{a[i]!r}, {a[i] + d!r})"))
        elif (e[i] == 0 or d is None) and x > a[i]:
            rows.append((i + 1, f"entry {x!r} is after the recorded age {a[i]!r}"))
    if rows:
        raise CohortDataError("entry ages refused", rows)
    return _readonly(en.copy())


def _window(window, entry, delta):
    """A common fit window (t_lo, t_hi): 0 <= t_lo < t_hi, both on the recording grid when delta is one
    value (research R8; the engine does not snap silently)."""
    if window is None:
        return None
    try:
        lo, hi = (float(window[0]), float(window[1]))
    except (TypeError, ValueError, IndexError):
        raise CohortSpecError(f"window must be (t_lo, t_hi), got {window!r}") from None
    if not (math.isfinite(lo) and lo >= 0 and hi > lo):
        raise CohortSpecError(f"window needs 0 <= t_lo < t_hi (t_hi may be inf), got {window!r}")
    if np.ndim(delta) == 1:
        raise CohortSpecError("a window needs one recording resolution delta, not a per-row delta")
    if delta is not None:
        for name, x in (("t_lo", lo), ("t_hi", hi)):
            if math.isfinite(x) and not _on_grid(x, delta):
                k = math.floor(x / delta)
                raise CohortSpecError(f"window {name} = {x!r} is not on the recording grid (multiples of "
                                      f"delta = {delta!r}); the nearest grid ages are {k * delta!r} and "
                                      f"{(k + 1) * delta!r}. The engine does not snap a window silently.")
    return (lo, hi)


def read_csv(path, *, delta=None):
    """(ages, events) from a two-column CSV, by the reference's header rule; refuses instead of skipping.

    The column detection is a verbatim copy of `SRK/fit_cohort.py:read_csv` (tests/oracle/v033, lines
    52-80). Where the reference `continue`s past a row (short, non-numeric, non-finite, age <= 0) or recodes
    it (any nonzero event -> 1), this records (row number, reason) and raises one CohortDataError listing
    every such row (FR-003). Entirely blank lines are ignored, as in the reference. Row numbers are 1-based
    lines of the file. `delta` decides the age-0 rule (`_age_reason`); None means continuous times.
    """
    ages, events, bad = [], [], []
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd, None)
        icol, ecol = 0, 1
        first_row_is_data = False
        if head:
            low = [h.strip().lower() for h in head]
            # FIRST match wins, and the two columns may not collide (verbatim rule of the reference).
            icol = ecol = None
            for i, h in enumerate(low):
                if icol is None and ("death" in h or "time" in h or "age" in h):
                    icol = i
            for i, h in enumerate(low):
                if ecol is None and i != icol and ("event" in h or "status" in h
                                                   or "censor" in h):
                    ecol = i
            icol = 0 if icol is None else icol
            ecol = (1 if icol != 1 else 0) if ecol is None else ecol
            if icol == ecol:
                raise CohortDataError(f"{path}: could not tell the age column from the event column "
                                      f"(header {head}). Rename them, e.g. 'death times,events'.")
            # a header row that is actually data (no letters) must not be discarded
            if all(not any(c.isalpha() for c in h) for h in head):
                first_row_is_data = True
        rows = ([(1, head)] if first_row_is_data else []) + [(k + 2, r) for k, r in enumerate(rd)]
        for lineno, row in rows:
            if not row:
                continue                                         # an entirely blank line: ignored
            if len(row) <= max(icol, ecol):
                bad.append((lineno, f"short row {row!r}: needs an age and an event"))
                continue
            try:
                a = float(row[icol])
                e = float(row[ecol])
            except ValueError:
                bad.append((lineno, f"non-numeric cell in {row!r}"))
                continue
            if not math.isfinite(a):
                bad.append((lineno, f"age {row[icol]!r} is not finite"))
                continue
            why = _age_reason(a, e, delta)
            if why:
                bad.append((lineno, f"{why} (the reference reader drops age <= 0 rows silently)"))
                continue
            if not math.isfinite(e) or e not in (0.0, 1.0):
                bad.append((lineno, f"event {row[ecol]!r} is not 0 or 1 (the reference counts any nonzero "
                                    f"value as a death)"))
                continue
            ages.append(a)
            events.append(1 if e else 0)
    if bad:
        raise CohortDataError(f"{path}: {len(bad)} row(s) refused (fix the file; nothing was skipped)", bad)
    if not ages:
        raise CohortDataError(f"{path}: no usable rows. Expected a header plus two columns, "
                              f"'death times,events'.")
    return np.asarray(ages, float), np.asarray(events, int)
