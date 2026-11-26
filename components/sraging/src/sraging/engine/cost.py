"""The stated cost of a heterogeneous evaluation (spec 004 FR-008; research R8; data-model "Cost statement").

    stated_cpu = M x t_hom x f_mode

- **t_hom** is the CPU of one homogeneous value through the evaluator's own per-node programs (one node at s = 1 and
  the assembly): the median of three timed runs after a warm-up, in the same process (`mixture.time_hom`).
- **f_mode** is a constant per mode, in units of t_hom per node: `value` (pass 1), `value_and_grad` (both passes),
  `pass2` (their difference, a gradient's second pass). The factors are frozen from pilot P0 at freeze-H: read from
  `tests/acceptance/panel_004.toml` [cost] (f_value, f_grad, and the P0 result file with its sha256, checked against
  the file's bytes) once `naveh_ok_h` is set. Before that they are research R4's measurement, marked "unfrozen".
- **Compilation** is neither stated nor counted in the measured CPU; it is recorded apart (`compile_cpu_s`).
- **At W workers** the stated wall time is ceil(M / W) x t_hom x f_mode; the stated CPU does not depend on W.

A statement is made before its work and gets its measured CPU (parent plus workers) and wall time after; HR-5 compares
the two over the panel's cells. `blind` is HR-5's plant: a predictor that ignores M (it states M_BLIND nodes).
"""
from __future__ import annotations

import hashlib
import math
import os
import tomllib
from dataclasses import asdict, dataclass, replace

from sraging.errors import UnsupportedOptionError

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PANEL = os.path.join(_REPO, "tests", "acceptance", "panel_004.toml")
MODES = ("value", "value_and_grad", "pass2", "hom")
M_BLIND = 11                      # HR-5's plant: the node count in use before spec 004, stated whatever M is
#: research R4/R8 [M] (specs/004-samplers/evidence/cost_probe.json, 2 v6 cohorts at M = 31): 0.85-0.90 and 3.1-3.3
#: homogeneous values per node; used, marked "unfrozen", until P0's factors are frozen at freeze-H
UNFROZEN = {"value": 0.90, "value_and_grad": 3.30,
            "note": "unfrozen: research R4's cost probe (specs/004-samplers/evidence/cost_probe.json), before P0's "
                    "factors are frozen in panel_004.toml [cost] at freeze-H"}


@dataclass(frozen=True)
class CostStatement:
    mode: str
    M: int
    t_hom_cpu_s: float
    factor: float
    factor_source: dict
    stated_cpu_s: float
    stated_wall_s: float
    workers: int = 1
    phase: str = ""
    measured_cpu_s: float | None = None
    measured_wall_s: float | None = None

    @property
    def ratio(self) -> float | None:
        return None if self.measured_cpu_s is None else self.measured_cpu_s / self.stated_cpu_s

    def measured(self, cpu_s, wall_s) -> "CostStatement":
        return replace(self, measured_cpu_s=float(cpu_s), measured_wall_s=float(wall_s))

    def as_dict(self) -> dict:
        return {**asdict(self), "ratio": self.ratio}


def factors(panel_path: str = PANEL, root: str = _REPO) -> dict:
    """{value, value_and_grad, source}: P0's frozen factors once freeze-H's naveh_ok_h is set and [cost] names its
    source file with a matching sha256; the unfrozen ones before. A [cost] whose sha256 does not match its file is
    refused."""
    try:
        pan = tomllib.load(open(panel_path, "rb"))
    except OSError:
        pan = None
    cost = (pan or {}).get("cost") or {}
    if not (pan and str(pan.get("naveh_ok_h", "")).strip() and {"f_value", "f_grad", "source", "sha256"} <= set(cost)):
        return {"value": UNFROZEN["value"], "value_and_grad": UNFROZEN["value_and_grad"],
                "source": {"status": "unfrozen", "note": UNFROZEN["note"]}}
    path = cost["source"] if os.path.isabs(cost["source"]) else os.path.join(root, cost["source"])
    got = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if got != cost["sha256"]:
        raise ValueError(f"panel_004.toml [cost]: {cost['source']} has sha256 {got[:12]}..., the freeze pins "
                         f"{cost['sha256'][:12]}...")
    return {"value": float(cost["f_value"]), "value_and_grad": float(cost["f_grad"]),
            "source": {"status": "frozen", "file": cost["source"], "sha256": got, "panel": panel_path}}


_factors = factors                       # `state`'s keyword shadows the name


def factor_of(mode: str, f: dict) -> float:
    if mode == "value":
        return f["value"]
    if mode == "value_and_grad":
        return f["value_and_grad"]
    if mode == "pass2":
        return f["value_and_grad"] - f["value"]
    if mode == "hom":
        return 1.0
    raise UnsupportedOptionError("mode", f"a cost mode is one of {MODES}, got {mode!r}")


def state(mode, M, t_hom_cpu_s, *, factors=None, workers=1, phase="") -> CostStatement:
    """The statement before the work: M x t_hom x f_mode CPU, and ceil(M / workers) x t_hom x f_mode wall."""
    f = factors if factors is not None else _factors()
    k = factor_of(mode, f)
    cpu = M * t_hom_cpu_s * k
    return CostStatement(mode=mode, M=int(M), t_hom_cpu_s=float(t_hom_cpu_s), factor=k, factor_source=f["source"],
                         stated_cpu_s=cpu, stated_wall_s=math.ceil(M / max(1, int(workers))) * t_hom_cpu_s * k,
                         workers=int(workers), phase=phase)


def blind(statement: dict) -> dict:
    """HR-5's plant: the same statement made by a predictor that ignores M (it states M_BLIND nodes)."""
    return {**statement, "M": M_BLIND, "stated_cpu_s": M_BLIND * statement["t_hom_cpu_s"] * statement["factor"]}
