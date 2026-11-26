"""Cohort sets and their reader (spec 005 US2; FR-004 to FR-006; data-model "CohortSet", "Reader"; register D-505).

    s, refused = read_cohorts("cages.csv", delta=1.0, time_unit="days", cohort_col="cohort", cage_col="cage_id")
    s.select(sex="F", dose=1)          # exact column names, typed values: 1 matches 1 and 1.0, never "1"
    s.by("cohort")                     # {cohort: CohortSet of its cages}

A CohortSet holds named, unchanged `Cohort`s and a design table (one row per member; free columns; values keep their
types). Selection never matches a substring of a column name or of a value: SRK's `DatasetCollection.getKeys` selected
`cage:_1` when asked for `age` = 1 and matched string values only (D-505).
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import NamedTuple

import numpy as np

from sraging.cohort import Cohort, _age_reason, _spec
from sraging.errors import CohortDataError, CohortSpecError, UnsupportedOptionError

#: the header names the reader recognises (data-model "Reader"), compared stripped and lower-cased; `censor` is never
#: guessed (its 0/1 means the opposite of an event)
TIME_HEADERS = ("death times", "death_times", "death time", "age")
EVENT_HEADERS = ("events", "event")
_INT = re.compile(r"^[+-]?\d+$")


def _scalar(v):
    """A design value as int, float, str, bool or None (numpy scalars to their Python type); anything else refused."""
    if isinstance(v, np.generic):
        v = v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    raise CohortSpecError(f"a design value must be an int, float, str, bool or None, got {v!r}")


def _matches(have, want) -> bool:
    """FR-004: exact and typed. Numbers compare as numbers (an int matches an equal int or float), bools only bools,
    strings only equal strings, None only None; NaN matches nothing."""
    if isinstance(have, bool) or isinstance(want, bool):
        return isinstance(have, bool) and isinstance(want, bool) and have == want
    if isinstance(have, (int, float)) and isinstance(want, (int, float)):
        return have == want
    if isinstance(have, str) and isinstance(want, str):
        return have == want
    return have is None and want is None


@dataclass(frozen=True, eq=False)
class CohortSet:
    """Named cohorts plus a design table (data-model "CohortSet"). `cohorts`: key -> Cohort, unique string keys;
    `design`: key -> {column: value}, the same columns in every row; `time_unit`: every member's. `selection` is the
    criteria that produced the set (None when built); `notes` states what the reader left out of the design;
    `columns` are the design's (given for an empty set, so that it still refuses an unknown column)."""

    cohorts: Mapping
    design: Mapping = field(kw_only=True)
    time_unit: str = field(kw_only=True)
    selection: Mapping | None = field(default=None, kw_only=True)
    notes: tuple = field(default=(), kw_only=True)
    columns: tuple | None = field(default=None, kw_only=True)

    def __post_init__(self):
        if not isinstance(self.cohorts, Mapping) or not isinstance(self.design, Mapping):
            raise CohortSpecError("a CohortSet takes a mapping key -> Cohort and a design mapping key -> row")
        if not isinstance(self.time_unit, str):
            raise CohortSpecError(f"time_unit must be a string, got {self.time_unit!r}")
        bad = [k for k in self.cohorts if not isinstance(k, str)]
        if bad:
            raise CohortSpecError(f"keys must be strings; got {bad!r}")
        bad = [k for k, c in self.cohorts.items() if not isinstance(c, Cohort)]
        if bad:
            raise CohortSpecError(f"members must be sraging.Cohort; not: {bad!r}")
        no_member = [k for k in self.design if k not in self.cohorts]
        no_row = [k for k in self.cohorts if k not in self.design]
        if no_member or no_row:
            raise CohortSpecError("the design table has one row per member: "
                                  + "; ".join(([f"design rows without a member: {no_member!r}"] if no_member else [])
                                              + ([f"members without a design row: {no_row!r}"] if no_row else [])))
        rows = {}
        cols = None if self.columns is None or self.design else tuple(self.columns)
        for k in self.cohorts:
            row = self.design[k]
            if not isinstance(row, Mapping) or not all(isinstance(c, str) for c in row):
                raise CohortSpecError(f"design row {k!r} must map column names (strings) to values")
            if cols is None:
                cols = tuple(row)
            elif set(row) != set(cols):
                raise CohortSpecError(f"every design row has the same columns: {k!r} has {sorted(row)}, the first "
                                      f"row {sorted(cols)}")
            rows[k] = MappingProxyType({c: _scalar(row[c]) for c in cols})
        mixed = {k: c.time_unit for k, c in self.cohorts.items() if c.time_unit != self.time_unit}
        if mixed:
            raise CohortSpecError(f"every member's time_unit must be the set's ({self.time_unit!r}); "
                                  f"{', '.join(f'{k!r} is in {u!r}' for k, u in mixed.items())}. Convert them first "
                                  f"with sraging.units.convert")
        object.__setattr__(self, "cohorts", MappingProxyType(dict(self.cohorts)))
        object.__setattr__(self, "design", MappingProxyType(rows))
        if self.selection is not None:
            object.__setattr__(self, "selection", MappingProxyType(dict(self.selection)))
        object.__setattr__(self, "notes", tuple(self.notes))
        object.__setattr__(self, "columns", tuple(cols or ()))

    @classmethod
    def from_members(cls, members, *, time_unit, notes=()) -> "CohortSet":
        """From (key, cohort, design row) triples; duplicate keys are refused, naming them (FR-006)."""
        members = list(members)
        seen, dup = set(), []
        for k, _, _ in members:
            if k in seen and k not in dup:
                dup.append(k)
            seen.add(k)
        if dup:
            raise CohortSpecError(f"duplicate keys: {', '.join(repr(k) for k in dup)}")
        return cls({k: c for k, c, _ in members}, design={k: r for k, _, r in members}, time_unit=time_unit,
                   notes=notes)

    @property
    def count(self) -> int:
        return len(self.cohorts)

    def __len__(self):
        return len(self.cohorts)

    def __iter__(self):
        return iter(self.cohorts)

    def __contains__(self, key):
        return key in self.cohorts

    def __getitem__(self, key) -> Cohort:
        return self.cohorts[key]

    def __repr__(self):
        sel = "" if self.selection is None else f", selected by {dict(self.selection)!r}"
        return f"CohortSet({len(self)} cohorts, time_unit={self.time_unit!r}{sel})"

    def _subset(self, keys, selection) -> "CohortSet":
        return CohortSet({k: self.cohorts[k] for k in keys}, design={k: self.design[k] for k in keys},
                         time_unit=self.time_unit, selection=selection, notes=self.notes, columns=self.columns)

    def _column(self, column, what):
        if column not in self.columns:
            raise UnsupportedOptionError(column, f"{what}: no column {column!r} in the design table; its columns are "
                                                 f"{list(self.columns)}")

    def select(self, **exact) -> "CohortSet":
        """The members whose design row has exactly each given value in each named column (typed: 1 matches 1 and
        1.0, never "1"). An unknown column is refused, naming it; no match is an empty set (its count is 0)."""
        want = {}
        for col, v in exact.items():
            self._column(col, "select")
            if isinstance(v, (list, tuple, set, frozenset, dict, np.ndarray)):
                raise UnsupportedOptionError(col, f"select takes one value per column, got {col}={v!r}; select "
                                                  f"each value, or group with by({col!r})")
            want[col] = _scalar(v)
        keys = [k for k, row in self.design.items() if all(_matches(row[c], v) for c, v in want.items())]
        return self._subset(keys, want)

    def by(self, column) -> dict:
        """{value: CohortSet of the members with that value in `column`}, in order of first appearance."""
        self._column(column, "by")
        groups: dict = {}
        for k, row in self.design.items():
            groups.setdefault(row[column], []).append(k)
        return {v: self._subset(ks, {column: v}) for v, ks in groups.items()}


# ------------------------------------------------------------------------------------------------- the reader
class RefusedRow(NamedTuple):
    """A row the reader did not use: its file, its 1-based line in that file, and why (FR-005)."""
    file: str
    row: int
    reason: str


def _typed(cell: str):
    """A design cell as int (an integer literal), float (another number) or str (data-model "values keep their
    types")."""
    s = cell.strip()
    if _INT.match(s):
        return int(s)
    try:
        x = float(s)
    except ValueError:
        return s
    return x if math.isfinite(x) else s


def _header(path, head, *, cohort_col, cage_col, delta_col, entry_col) -> dict:
    """{role: column index} for one file's header, each role found exactly once; refused naming the header."""
    low = [h.strip().lower() for h in head]
    names = [h.strip() for h in head]
    t = [i for i, h in enumerate(low) if h in TIME_HEADERS]
    e = [i for i, h in enumerate(low) if h in EVENT_HEADERS]
    if not t:
        raise CohortDataError(f"{path}: no age column in the header {names}; expected one of "
                              f"{', '.join(repr(h) for h in TIME_HEADERS)}")
    if not e:
        hint = (" (a censor column is never read as events: its 0/1 means the opposite; give an events column)"
                if any("censor" in h for h in low) else "")
        raise CohortDataError(f"{path}: no event column in the header {names}; expected one of "
                              f"{', '.join(repr(h) for h in EVENT_HEADERS)}{hint}")
    if len(t) > 1 or len(e) > 1:
        two = [names[i] for i in (t if len(t) > 1 else e)]
        raise CohortDataError(f"{path}: two columns could be the {'age' if len(t) > 1 else 'event'}: {two}")
    roles = {"age": t[0], "event": e[0]}
    for role, col in (("cohort", cohort_col), ("cage", cage_col), ("delta", delta_col), ("entry", entry_col)):
        if col is None:
            continue
        if col not in names:
            raise CohortDataError(f"{path}: no column {col!r} ({role}_col) in the header {names}")
        roles[role] = names.index(col)
    taken = list(roles.values())
    if len(set(taken)) != len(taken):
        raise CohortSpecError(f"{path}: one column given two roles: {roles}")
    return roles


def _float(cell, what):
    try:
        x = float(cell)
    except ValueError:
        return None, f"non-numeric {what} {cell!r}"
    if not math.isfinite(x):
        return None, f"{what} {cell!r} is not finite"
    return x, None


def _parse_row(cells, roles, ncols, delta, continuous):
    """(fields, None) for a usable row, or (None, reason). Nothing is recoded or skipped silently."""
    if len(cells) != ncols:
        return None, f"{len(cells)} cells where the header has {ncols}: {cells!r}"
    age, why = _float(cells[roles["age"]], "age")
    if why:
        return None, why
    ev, why = _float(cells[roles["event"]], "event")
    if why:
        return None, why
    if ev not in (0.0, 1.0):
        return None, f"event {cells[roles['event']]!r} is not 0 or 1 (no value is recoded)"
    out = {"age": age, "event": int(ev)}
    for role in ("cohort", "cage"):
        if role in roles:
            v = cells[roles[role]].strip()
            if not v:
                return None, f"no {role} in this row"
            out[role] = v
    d = None if continuous else delta
    if "delta" in roles:
        d, why = _float(cells[roles["delta"]], "delta")
        if why:
            return None, why
        if not d > 0:
            return None, f"delta {d!r} is not > 0"
        out["delta"] = d
    if "entry" in roles:
        x, why = _float(cells[roles["entry"]], "entry")
        if why:
            return None, why
        if x < 0:
            return None, f"entry age {x!r} is negative"
        out["entry"] = x
    why = _age_reason(age, ev, d)                       # Cohort's own age rule (D-101, D-105)
    return (None, why) if why else (out, None)


def _files(path):
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(".csv") and os.path.isfile(os.path.join(path, f)))
        if not files:
            raise CohortDataError(f"{path}: no .csv files in this directory")
        return files
    if os.path.isfile(path):
        return [path]
    raise CohortDataError(f"{path}: no such file or directory")


def read_cohorts(path, *, delta=None, continuous=False, time_unit, cohort_col=None, cage_col=None, delta_col=None,
                 entry_col=None, window=None) -> tuple:
    """(CohortSet, refused rows) from one long CSV or a directory of CSVs (FR-005; data-model "Reader").

    - The age and event columns are found by their header names (TIME_HEADERS, EVENT_HEADERS); a header with neither,
      or with two candidates, is refused, naming it. `censor` is never guessed.
    - Members: one per cohort, or per (cohort, cage) with `cage_col`. The cohort is `cohort_col`'s value, or the file's
      stem when there is no cohort column. Keys are "cohort" or "cohort/cage"; the same key from two files, or two
      pairs that spell the same key, is refused, naming it.
    - The design table has columns `cohort`, `cage` (with `cage_col`) and every other column whose value is the same in
      all of a member's rows, typed (int, float, str). A column that varies within a member is left out, and `notes`
      says so.
    - `delta` (one value) or `continuous`, or `delta_col` (a per-row delta, kept per row: a calibrator that takes one
      delta refuses it); `entry_col` gives per-row entry ages; `window` (lo, hi) is every member's common fit window
      (spec 006).
    - A row that cannot be used (short, non-numeric, non-finite, an event other than 0/1, no cohort or cage, an age
      Cohort refuses) is listed as (file, line, reason) and left out; so are the rows of a member that
      `Cohort.from_arrays` refuses (for example, one with no deaths). Each member is built by `Cohort.from_arrays`.
    """
    if delta_col is not None:
        if delta is not None or continuous:
            raise CohortSpecError("give a delta column (delta_col) or one delta / continuous=True, not both")
    else:
        delta, continuous = _spec(delta, continuous)
    if np.ndim(delta) == 1:
        raise CohortSpecError("read_cohorts takes one delta for the files; give a per-row delta as a column "
                              "(delta_col)")
    if not isinstance(time_unit, str):
        raise CohortSpecError(f"time_unit must be a string, got {time_unit!r}")
    refused: list[RefusedRow] = []
    groups: dict = {}                     # (file, cohort, cage) -> {"rows": [...], "lines": [...]}
    extras: dict = {}                     # file -> [(index, name)] of the design candidates
    for f in _files(str(path)):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            if not head:
                raise CohortDataError(f"{f}: empty file (expected a header)")
            roles = _header(f, head, cohort_col=cohort_col, cage_col=cage_col, delta_col=delta_col,
                            entry_col=entry_col)
            used = set(roles.values())
            extras[f] = [(i, h.strip()) for i, h in enumerate(head) if i not in used]
            stem = os.path.splitext(os.path.basename(f))[0]
            for lineno, cells in enumerate(rd, start=2):
                if not cells or all(not c.strip() for c in cells):
                    continue
                row, why = _parse_row(cells, roles, len(head), delta, continuous)
                if why:
                    refused.append(RefusedRow(f, lineno, why))
                    continue
                cohort = row.get("cohort", stem)
                g = groups.setdefault((f, cohort, row.get("cage")), {"rows": [], "lines": [], "cells": []})
                g["rows"].append(row)
                g["lines"].append(lineno)
                g["cells"].append(cells)
    keyed: dict = {}
    for (f, cohort, cage), g in groups.items():
        key = cohort if cage is None else f"{cohort}/{cage}"
        keyed.setdefault(key, []).append((f, cohort, cage, g))
    dup = {k: [(v[0], v[1], v[2]) for v in vs] for k, vs in keyed.items() if len(vs) > 1}
    if dup:
        raise CohortDataError("duplicate (cohort, cage) keys: " + "; ".join(
            f"{k!r} from {', '.join(f'{os.path.basename(f)} ({co!r}, {cg!r})' for f, co, cg in v)}"
            for k, v in dup.items()))
    common = None
    for f, ex in extras.items():
        names = [n for _, n in ex]
        common = names if common is None else [n for n in common if n in names]
    common = common or []
    partial = sorted({n for ex in extras.values() for _, n in ex} - set(common))
    clash = [n for n in common if n in ("cohort", "cage")]
    if clash:
        raise CohortSpecError(f"column(s) {clash} would collide with the design table's cohort/cage columns; "
                              f"pass them as cohort_col / cage_col, or rename them")
    members, varying = [], []
    for key, [(f, cohort, cage, g)] in keyed.items():
        lines = list(g["lines"])
        cohort_value = _typed(cohort) if cohort_col is not None else cohort
        row = {"cohort": cohort_value, **({} if cage is None else {"cage": _typed(cage)})}
        idx = dict((n, i) for i, n in extras[f])
        for n in common:
            vals = {_typed(c[idx[n]]) for c in g["cells"]}
            if len(vals) == 1:
                row[n] = vals.pop()
            elif n not in varying:
                varying.append(n)
        cohort_obj, bad = _member(key, g["rows"], lines, f, delta, continuous, delta_col, entry_col, time_unit,
                                  window=window)
        refused += bad
        if cohort_obj is not None:
            members.append((key, cohort_obj, row))
    for n in varying:                                   # a column must be in every row of the design table
        for _, _, row in members:
            row.pop(n, None)
    notes = tuple([f"column {n!r} is not in every file: not in the design table" for n in partial]
                  + [f"column {n!r} varies within a member: not in the design table" for n in varying])
    refused.sort(key=lambda r: (r.file, r.row))
    return CohortSet.from_members(members, time_unit=time_unit, notes=notes), tuple(refused)


def _member(key, rows, lines, f, delta, continuous, delta_col, entry_col, time_unit, retry=True, window=None):
    """(Cohort or None, refused rows): `Cohort.from_arrays` on the member's rows. Rows it refuses by number are listed
    and the member is rebuilt once without them; a member it refuses as a whole lists all its rows."""
    try:
        return Cohort.from_arrays([r["age"] for r in rows], [r["event"] for r in rows],
                                  delta=np.array([r["delta"] for r in rows]) if delta_col is not None else delta,
                                  continuous=continuous, name=key,
                                  entry=[r["entry"] for r in rows] if entry_col is not None else None,
                                  window=window, time_unit=time_unit), []
    except CohortDataError as ex:
        if ex.rows and retry:
            drop = {r - 1 for r, _ in ex.rows}
            bad = [RefusedRow(f, lines[r - 1], f"{key}: {why}") for r, why in sorted(ex.rows)]
            keep = [i for i in range(len(rows)) if i not in drop]
            if not keep:
                return None, bad
            c, more = _member(key, [rows[i] for i in keep], [lines[i] for i in keep], f, delta, continuous,
                              delta_col, entry_col, time_unit, retry=False, window=window)
            return c, bad + more
        msg = str(ex).splitlines()[0]
        return None, [RefusedRow(f, ln, f"{key}: {msg}") for ln in lines]
