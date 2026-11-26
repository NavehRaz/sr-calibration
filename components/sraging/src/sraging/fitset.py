"""Fit every cohort of a set, one spawned process per fit, W at a time (spec 005 US6; FR-021; research R8; HR-6).

    run = fit_set(cohort_set, workers=4, out_dir="fits/")    # fits/<cohort>.json, fits/summary.csv, fits/run.json

- Each fit runs in a fresh spawned process (`multiprocessing` "spawn", one task per child), which sets the runtime
  `sraging.compat.v033.runtime.configure()` sets (PJRT_NPROC=1, float64) before any JAX computation, then calls the
  calibrator's fit on the cohort rebuilt by `Cohort.from_arrays`. A registered calibrator reaches the worker by name and
  version and is resolved there when its fit does not pickle; any other as its fit. A result therefore does not depend
  on W, or on which cohorts shared a pool; whether a spawned fit is bit-identical to an in-process one is pilot P1's
  question (HR-6).
- Each result is written as `sraging fit-cohort --json` writes it; `summary.csv` lists every cohort (status, theta-hat,
  loglik, warnings, error). A cohort whose fit raises or fails is recorded with its error and the others complete.
- W and the pids are provenance only (`run.json`, FitSetRun.provenance), never in a result or the summary.
"""
from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import json
import math
import os
import platform
import socket
from dataclasses import dataclass

import numpy as np

from sraging.errors import UnsupportedOptionError

SUMMARY_FIELDS = ("cohort", "status", "eta", "beta", "epsilon", "kappa", "xc", "h_ext", "time_unit", "loglik",
                  "warnings", "error", "json",
                  "window_lo_eff", "window_hi_eff", "n_edge_censored")     # spec 006: the window used, the edge tie


@dataclass(frozen=True)
class CohortFit:
    """One cohort's outcome: its name, "ok" or "error", the result JSON's path (None when nothing was written), the
    error text."""

    name: str
    status: str
    path: str | None
    error: str | None


@dataclass(frozen=True)
class FitSetRun:
    """Every cohort's outcome, the summary's path and the run's provenance (W, host, versions, start, end, pids)."""

    fits: tuple
    summary_path: str
    provenance: dict

    @property
    def status(self) -> str:
        return "ok" if all(f.status == "ok" for f in self.fits) else "error"


def _init_worker():
    """In each spawned worker, before any JAX computation: the runtime the calibrators require."""
    os.environ["PJRT_NPROC"] = "1"
    from sraging.compat.v033 import runtime
    runtime.configure()


def _payload(c) -> dict:
    """The cohort's constructor arguments, so the worker rebuilds it with Cohort.from_arrays (read-only arrays)."""
    return {"ages": np.asarray(c.ages), "events": np.asarray(c.events),
            "delta": c.delta if c.delta is None or np.ndim(c.delta) == 0 else np.asarray(c.delta),
            "continuous": c.continuous, "name": c.name,
            "entry": None if c.entry is None else np.asarray(c.entry), "window": c.window, "time_unit": c.time_unit}


def _fit_one(key, payload, carried, out_path) -> dict:
    """One fit in a worker: its row for the summary, the pid, and the JSON written as fit-cohort --json writes it.
    `carried` is what `_carried` sent: (name, version) of a registered calibrator, resolved here, or a fit callable."""
    from sraging.cohort import Cohort
    row = {"cohort": key, "pid": os.getpid()}
    try:
        fit = _fit_of(carried)
        cohort = Cohort.from_arrays(payload["ages"], payload["events"], delta=payload["delta"],
                                    continuous=payload["continuous"], name=payload["name"], entry=payload["entry"],
                                    window=payload["window"], time_unit=payload["time_unit"])
        r = fit(cohort)
        if r.theta_hat is not None and r.theta_hat.time_unit != cohort.time_unit:     # as fit-cohort's --time-unit
            r = dataclasses.replace(r, theta_hat=dataclasses.replace(r.theta_hat, time_unit=cohort.time_unit))
        r.to_json(out_path)
    except Exception as ex:                                     # noqa: BLE001 -- recorded as this cohort's error
        return {**row, "status": "error", "error": f"{type(ex).__name__}: {ex}", "json": None}
    ok = r.status == "ok" and not r.all_starts_failed
    th = r.theta_hat.as_dict() if r.theta_hat is not None else {}
    win = (r.diagnostics or {}).get("window") or {}
    eff = win.get("window_eff") or (None, None)
    return {**row, "status": "ok" if ok else "error",
            "window_lo_eff": eff[0], "window_hi_eff": eff[1], "n_edge_censored": win.get("n_deaths_censored"),
            "error": None if ok else (f"{r.status}: {(r.diagnostics or {}).get('exception', '')}" if r.status != "ok"
                                      else "every optimiser start failed (D-015)"),
            "json": out_path, "loglik": r.loglik, "warnings": ";".join(w.code for w in r.warnings),
            **{k: th.get(k) for k in ("eta", "beta", "epsilon", "kappa", "xc", "h_ext", "time_unit")}}


def _resolve(calibrator):
    """(name, version, what a worker receives for it) from None (the default), "name[@version]" or a
    CalibratorSpec."""
    from sraging import registry
    if calibrator is None:
        spec = registry.get(registry.DEFAULT)
    elif isinstance(calibrator, registry.CalibratorSpec):
        spec = calibrator
    elif isinstance(calibrator, str):
        name, _, version = calibrator.partition("@")
        spec = registry.get(name, version or None)
    else:
        raise UnsupportedOptionError("calibrator", f"calibrator is None, 'name[@version]' or a CalibratorSpec, got "
                                                   f"{calibrator!r}")
    return spec.name, spec.version, _carried(spec)


def _carried(spec):
    """What a spawned worker receives for the calibrator: its fit when the fit pickles (a top-level function); else
    (name, version) of a registered spec, which the worker resolves from its own registry -- the registered fits are
    closures (`_make_fit.<locals>.fit`), which do not pickle (pilot P1, 2026-11-08: every spawned fit failed). An
    unregistered spec whose fit does not pickle is refused here, before any process starts."""
    import pickle
    from sraging import registry
    from sraging.errors import UnknownCalibratorError
    try:
        pickle.dumps(spec.fit)
        return spec.fit
    except Exception as err:                                    # noqa: BLE001 -- any pickling failure falls through
        why = err
    try:
        if registry.get(spec.name, spec.version) is spec:
            return (spec.name, spec.version)
    except UnknownCalibratorError:
        pass
    raise UnsupportedOptionError("calibrator", f"an unregistered calibrator's fit must pickle to reach a spawned "
                                               f"worker (define it at a module's top level): {why}")


def _fit_of(carried):
    """The fit callable in the worker, from what `_carried` sent."""
    if isinstance(carried, tuple):
        from sraging import registry
        return registry.get(*carried).fit
    return carried


def _file_names(keys) -> dict:
    names = {k: str(k).replace(os.sep, "__").replace(" ", "_") + ".json" for k in keys}
    if len(set(names.values())) != len(names):
        raise UnsupportedOptionError("cohort_set", "two cohort names give the same result file name; rename them")
    return names


def fit_set(cohort_set, *, workers=1, calibrator=None, out_dir) -> FitSetRun:
    """Fit each member of `cohort_set` in its own spawned process, `workers` at a time; write out_dir/<name>.json,
    summary.csv and run.json (FR-021)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from sraging.cohorts import CohortSet
    import sraging
    if not isinstance(cohort_set, CohortSet):
        raise UnsupportedOptionError("cohort_set", f"fit_set takes a CohortSet, got {type(cohort_set).__name__}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise UnsupportedOptionError("workers", f"workers must be an integer >= 1, got {workers!r}")
    name, version, carried = _resolve(calibrator)
    files = _file_names(cohort_set)
    os.makedirs(out_dir, exist_ok=True)
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    rows = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_init_worker,
                             max_tasks_per_child=1) as ex:
        futures = {key: ex.submit(_fit_one, key, _payload(cohort_set[key]), carried,
                                  os.path.join(out_dir, files[key])) for key in cohort_set}
        for key, fut in futures.items():
            try:
                rows[key] = fut.result()
            except Exception as err:                            # noqa: BLE001 -- a worker crash is this cohort's error
                rows[key] = {"cohort": key, "pid": None, "status": "error",
                             "error": f"the worker failed: {type(err).__name__}: {err}", "json": None}
    summary = os.path.join(out_dir, "summary.csv")
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for key in cohort_set:
            w.writerow({k: ("" if v is None or (isinstance(v, float) and math.isnan(v)) else v)
                        for k, v in rows[key].items()})
    prov = {"workers": workers, "pids": {k: rows[k]["pid"] for k in cohort_set},
            "calibrator": {"name": name, "version": version}, "host": socket.gethostname(),
            "platform": platform.platform(), "sraging": sraging.__version__, "started": started,
            "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "n_cohorts": len(cohort_set)}
    with open(os.path.join(out_dir, "run.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    fits = tuple(CohortFit(name=k, status=rows[k]["status"], path=rows[k]["json"], error=rows[k]["error"])
                 for k in cohort_set)
    return FitSetRun(fits=fits, summary_path=summary, provenance=prov)
