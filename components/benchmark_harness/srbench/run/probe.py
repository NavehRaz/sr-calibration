"""Measuring what a calibration actually costs.

A DECLARED COST IS A HYPOTHESIS. This project's recurring failure is believing numbers that were
asserted rather than measured, so the harness probes real calibrations before committing to a run
and reports any divergence from what the method claimed.

THE DECLARATION AND THE MEASUREMENT ARE KEPT AS SEPARATE RECORDS. The natural design overwrites
the declaration with the measurement; then there is nothing left to compare and the divergence --
which is the interesting part -- has been destroyed by the act of measuring.

Both wall-clock and CPU time are recorded. Wall-clock is what a user waits and what a budget is
spent in; on a shared, preemptable cluster the two diverge, and having both is what turns a
surprising forecast into a diagnosis instead of a guess.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ..contract.method import CalibrationMethod, characteristics_from_spec
from ..contract.types import CostDeclaration, CostMeasurement

#: Probes spread across the selection, not taken from its head: cost varies across the library,
#: and probing only the first few would measure one corner of it.
DEFAULT_PROBES = 5


@dataclass(frozen=True, slots=True)
class CostReport:
    declared: CostDeclaration | None
    measured: CostMeasurement
    diverges: bool
    method: str
    method_version: str

    @property
    def per_cohort_seconds(self) -> float:
        """What every forecast uses: the MEASURED median, never the declared one."""
        return self.measured.median_seconds

    def to_dict(self) -> dict:
        return {"method": self.method, "method_version": self.method_version,
                "declared": self.declared.to_dict() if self.declared else None,
                "measured": self.measured.to_dict(), "diverges": self.diverges}

    def format(self) -> str:
        lines = [f"method            {self.method}  v{self.method_version}"]
        if self.declared:
            lines.append(f"declared          median {self.declared.median_seconds:.4g} s   "
                         f"range [{self.declared.min_seconds:.4g}, "
                         f"{self.declared.max_seconds:.4g}]")
        m = self.measured
        lines.append(f"MEASURED          median {m.median_seconds:.4g} s   "
                     f"range [{m.min_seconds:.4g}, {m.max_seconds:.4g}]     "
                     f"n_probes = {m.n_probes}")
        if self.diverges:
            lines.append("                  ** divergence: the measured median is outside the "
                         "declared range **")
            lines.append("                     forecasts below use the MEASURED value")
        return "\n".join(lines)


def probe_indices(n_total: int, n_probes: int) -> list[int]:
    """Evenly spread positions, so the probe samples the selection rather than its head."""
    if n_total <= n_probes:
        return list(range(n_total))
    step = n_total / n_probes
    return [min(int(i * step), n_total - 1) for i in range(n_probes)]


from .execute import _reveal  # one definition, three call sites


def measure(method: CalibrationMethod, library, cohort_ids, store=None,
            n_probes: int = DEFAULT_PROBES, settings: dict | None = None) -> CostReport:
    """Run real calibrations and measure. Reuses stored results where they exist."""
    from ..contract.types import CONTRACT_VERSION
    from ..store.key import make_key

    d = method.descriptor
    ids = list(cohort_ids)
    picks = [ids[i] for i in probe_indices(len(ids), n_probes)]

    declared = None
    if picks:
        try:
            declared = method.cost(characteristics_from_spec(library.spec(picks[0])))
        except Exception:  # noqa: BLE001 - a method that cannot declare is still measurable
            declared = None

    walls, cpus = [], []
    for cid in picks:
        obs = library.observation(cid, reveal=_reveal(method))
        rec = None
        if store is not None:
            key = make_key(method_name=d.name, method_version=d.version, cohort_id=cid,
                           library_version=library.version, contract_version=CONTRACT_VERSION,
                           observation=obs, settings=settings)
            rec = store.get(key)
        if rec is not None and rec.wall_clock_seconds:
            walls.append(rec.wall_clock_seconds)
            cpus.append(rec.cpu_seconds or rec.wall_clock_seconds)
            continue
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            method.calibrate(obs)
        except Exception:  # noqa: BLE001 - a failure still took time, and that time is real
            pass
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)

    w = sorted(walls)
    med = w[len(w) // 2] if len(w) % 2 else 0.5 * (w[len(w) // 2 - 1] + w[len(w) // 2])
    measured = CostMeasurement(median_seconds=med, min_seconds=w[0], max_seconds=w[-1],
                               n_probes=len(w), wall_clock_seconds=tuple(walls),
                               cpu_seconds=tuple(cpus), cohort_ids=tuple(picks))
    diverges = bool(declared and not declared.contains(med))
    return CostReport(declared=declared, measured=measured, diverges=diverges,
                      method=d.name, method_version=d.version)


def forecast(per_cohort_seconds: float, tiers: dict) -> dict[str, float]:
    """Projected CPU-hours per tier, from the MEASURED cost."""
    return {name: n * per_cohort_seconds / 3600.0 for name, n in tiers.items()}


def format_forecast(per_cohort_seconds: float, tiers: dict) -> str:
    f = forecast(per_cohort_seconds, tiers)
    rows = [f"forecast          {name:<8} ({n} cohorts)  {f[name]:>10.1f} CPU-h"
            for name, n in tiers.items()]
    return "\n".join(rows)
