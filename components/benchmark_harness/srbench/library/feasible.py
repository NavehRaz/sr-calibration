"""STAGE 0 of the library build: what part of the manifold can each κ actually reach?

**This runs before any parameter point is chosen**, because the answer cannot be assumed. A regular
grid in `(ρβ, ρε)` does not produce a regular grid on the observable `(log₁₀ ratio, cv)` plane, and
the predecessor benchmark was bitten by assuming it does: its map holds 6 points at κ=0.05 against
21 at κ=120, so selecting a κ in its explorer and reading "regime dependence" mostly reads *which
manifold positions that κ happens to contain*.

INFEASIBILITY IS A RESULT, NOT AN ERROR. Once `ρε` is large enough that diffusion alone crosses the
threshold in less than mean time 1, no production rate can stretch the mean back out, and
`solve_rho_eta_for_mean_one` returns nothing. Those cells are recorded and reported. A missing cell
that reads as a covered one is how a gap becomes an invisible assumption.

**This module is part of the BUILD PATH and therefore imports the system under test.** Nothing
downstream does: selection, the frozen library and every run read its stored output.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

#: The κ ladder. Five log-spaced across the decade every real fit occupies (measured: the 27 fitted
#: anchors run 0.0102 to 0.977, ALL below 1), plus two probes above it -- κ=2 just past the
#: transition where the removal term changes character, κ=10 well into the linear regime.
KAPPAS = (0.01, 0.0316, 0.1, 0.316, 1.0, 2.0, 10.0)

#: The knob sweep. 14 x 14 per κ, against the predecessor design's effective 3 x 6.
RHO_BETA = tuple(float(x) for x in np.logspace(-1.0, 3.5, 14))
RHO_EPS = tuple(float(x) for x in np.logspace(-3.0, 0.9, 14))

#: Coarse cells on the observable plane. Used ONLY to define the common core that seeds selection --
#: points themselves are continuous. Edges match the predecessor's so the two are comparable.
RATIO_EDGES = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.7)
CV_EDGES = (0.0, 0.05, 0.10, 0.15, 0.25, 1.0)


def cell_of(log_ratio: float, cv: float) -> tuple[int, int] | None:
    """Which coarse cell a manifold position falls in, or None if outside the edges."""
    if not (math.isfinite(log_ratio) and math.isfinite(cv)):
        return None
    i = j = None
    for a in range(len(RATIO_EDGES) - 1):
        if RATIO_EDGES[a] <= log_ratio < RATIO_EDGES[a + 1]:
            i = a
            break
    for b in range(len(CV_EDGES) - 1):
        if CV_EDGES[b] <= cv < CV_EDGES[b + 1]:
            j = b
            break
    return None if (i is None or j is None) else (i, j)


N_CELLS = (len(RATIO_EDGES) - 1) * (len(CV_EDGES) - 1)


def sweep(kappas=KAPPAS, rho_beta=RHO_BETA, rho_eps=RHO_EPS, jobs: int = 8,
          verbose: bool = True) -> list[dict]:
    """Run the feasibility sweep. Returns one row per (κ, ρβ, ρε) combination.

    Delegates the mean-one solve to the system under test rather than reimplementing it: that
    solve is validated numerics this project depends on, and a second implementation would be a
    new source of error with nothing to gain.
    """
    from ..provenance import resolve_sut
    resolve_sut()  # refuses on the stale mount; applies the numpy shim
    from SRK.benchmark import kappa_feasible as _KF  # type: ignore

    rows = _KF.run(kappas=tuple(kappas), rho_beta=tuple(rho_beta), rho_eps=tuple(rho_eps),
                   jobs=jobs, verbose=verbose)
    return list(rows)


def summarise(rows: list[dict], kappas=None) -> dict:
    """Per-κ reach, and the common core that makes a cross-κ comparison paired."""
    kappas = list(kappas or sorted({r["kappa"] for r in rows}))
    per = {}
    for k in kappas:
        rk = [r for r in rows if abs(r["kappa"] - k) < 1e-12]
        ok = [r for r in rk if r.get("ok")]
        cells = {cell_of(r["log_ratio"], r["cv"]) for r in ok}
        cells.discard(None)
        per[k] = {
            "n_total": len(rk),
            "n_feasible": len(ok),
            "cells": sorted(cells),
            "n_cells": len(cells),
            "log_ratio_min": min((r["log_ratio"] for r in ok), default=float("nan")),
            "log_ratio_max": max((r["log_ratio"] for r in ok), default=float("nan")),
            "cv_min": min((r["cv"] for r in ok), default=float("nan")),
            "cv_max": max((r["cv"] for r in ok), default=float("nan")),
        }
    core = set.intersection(*[set(map(tuple, per[k]["cells"])) for k in kappas]) if kappas else set()
    return {"kappas": kappas, "per_kappa": per,
            "common_core": sorted(core), "n_common_core": len(core), "n_cells_total": N_CELLS}


def format_report(summary: dict) -> str:
    lines = ["PER-KAPPA FEASIBILITY OF THE (log10 ratio, cv) MANIFOLD", "=" * 78,
             f"{'kappa':>8} {'feasible':>10} {'cells':>8}   {'ratio span (dex)':>22}   {'cv span':>18}",
             "-" * 78]
    for k in summary["kappas"]:
        p = summary["per_kappa"][k]
        lines.append(
            f"{k:>8g} {p['n_feasible']:>5}/{p['n_total']:<4} {p['n_cells']:>3}/{summary['n_cells_total']:<4}   "
            f"{p['log_ratio_min']:>8.2f} .. {p['log_ratio_max']:<8.2f}   "
            f"{p['cv_min']:>7.3f} .. {p['cv_max']:<7.3f}")
    lines += ["",
              f"COMMON CORE reachable at ALL {len(summary['kappas'])} kappa: "
              f"{summary['n_common_core']} of {summary['n_cells_total']} cells.",
              "That core is what makes a cross-kappa comparison PAIRED -- outside it, two kappa",
              "values are compared at different manifold positions, which is the confound this",
              "stratification exists to remove."]
    core = {tuple(c) for c in summary["common_core"]}
    for k in summary["kappas"]:
        extra = len({tuple(c) for c in summary["per_kappa"][k]["cells"]} - core)
        if extra:
            lines.append(f"  kappa={k:<8g} reaches {extra} cell(s) outside the common core")
    return "\n".join(lines)


# --------------------------------------------------------------------------- stored measurement

def default_path() -> str:
    """The feasibility measurement shipped with the repository."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "data", "library", "feasibility.json.gz")


def load(path: str | None = None) -> dict:
    """Read the stored sweep. Imports NOTHING from the system under test.

    Everything after stage 0 reads this file rather than re-running the sweep: the measurement is
    an input to the design, and re-measuring it at build time would let the design drift silently
    when the solver changes underneath it.
    """
    import gzip
    p = path or default_path()
    opener = gzip.open if p.endswith(".gz") else open
    with opener(p, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="where to write the feasibility JSON")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args(argv)

    rows = sweep(jobs=a.jobs)
    summary = summarise(rows, KAPPAS)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"kappas": list(KAPPAS), "rows": rows, "summary": summary}, fh)
    print(format_report(summary))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
