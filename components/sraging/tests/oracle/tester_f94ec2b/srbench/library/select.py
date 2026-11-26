"""STAGE 1 of the library build: choose the parameter points, reproducibly.

Pure functions over the stored feasibility sweep. **Imports nothing from the system under test** --
selection is arithmetic on a measurement that has already been made.

THE PROCEDURE, and the order matters:

  1. SEED FROM THE COMMON CORE. One point in every cell reachable at *every* κ, placed first. This
     is what makes a cross-κ comparison PAIRED: outside the core, two κ values are being compared
     at different manifold positions, which is the confound the stratification exists to remove.

  2. FILL BY FARTHEST-POINT SAMPLING in the normalised (log₁₀ ratio, cv) plane, which spreads the
     remainder over whatever else that κ can reach.

MEASURED, and it set the density: the common core for this κ ladder is **exactly 20 cells**, and
gate G14's floor is 20 points per κ. At 20 the entire budget would go to the core and nothing would
probe each κ's own extension -- the 11 cells only κ=10 reaches would never be sampled. So the target
is 24: 20 core-seeded, 4 exploratory. See `specs/.../library-feasibility.md`.

Determinism is a requirement, not a convenience (FR-023): the same feasibility input must always
produce the same points, or a rebuilt library is a different library.
"""
from __future__ import annotations

import math

N_PER_KAPPA = 24


def _normalise(rows: list[dict]) -> list[tuple[float, float]]:
    """Map (log_ratio, cv) into [0,1]^2 so farthest-point distance is not dominated by units.

    Without this the log-ratio axis (spanning ~3 dex) would swamp cv (spanning ~0.8), and
    "farthest" would mean "farthest in ratio" -- coverage of one axis masquerading as coverage of
    the plane.
    """
    lr = [r["log_ratio"] for r in rows]
    cv = [r["cv"] for r in rows]
    lo_r, hi_r = min(lr), max(lr)
    lo_c, hi_c = min(cv), max(cv)
    span_r = (hi_r - lo_r) or 1.0
    span_c = (hi_c - lo_c) or 1.0
    return [((r["log_ratio"] - lo_r) / span_r, (r["cv"] - lo_c) / span_c) for r in rows]


def farthest_point(rows: list[dict], k: int, seed_idx: tuple[int, ...] = ()) -> list[int]:
    """Indices of `k` rows spread across the plane, starting from `seed_idx`.

    Greedy and deterministic: each step takes the candidate whose minimum distance to everything
    already chosen is largest, with the lowest index winning ties.
    """
    if not rows or k <= 0:
        return []
    pts = _normalise(rows)
    chosen = list(seed_idx)
    if not chosen:
        # Deterministic start: the point closest to the centroid. Starting from an extreme would
        # bias the whole chain outward.
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        chosen = [min(range(len(pts)), key=lambda i: (pts[i][0] - cx) ** 2 + (pts[i][1] - cy) ** 2)]

    best = [min(math.dist(pts[i], pts[c]) for c in chosen) for i in range(len(pts))]
    while len(chosen) < min(k, len(rows)):
        nxt = max(range(len(pts)), key=lambda i: (best[i], -i))
        if nxt in chosen:
            break
        chosen.append(nxt)
        for i in range(len(pts)):
            d = math.dist(pts[i], pts[nxt])
            if d < best[i]:
                best[i] = d
    return chosen[:k]


def select_for_kappa(feasible_rows: list[dict], core_cells: set[tuple[int, int]],
                     n_points: int = N_PER_KAPPA) -> dict:
    """Choose `n_points` for one κ. Core cells first, then farthest-point fill.

    Returns the chosen rows plus an account of how many came from each stage -- the account is
    reported rather than kept internal, because "how paired is this comparison?" is answered by
    the core count, not by the total.
    """
    from .feasible import cell_of

    ok = [r for r in feasible_rows if r.get("ok")]
    if not ok:
        return {"points": [], "n_core": 0, "n_explore": 0, "reduced": True,
                "reason": "no feasible knob combination"}

    # Stage 1: one point per core cell. Within a cell, take the row closest to the cell's centre
    # in the normalised plane, so the choice does not depend on sweep ordering.
    by_cell: dict[tuple[int, int], list[int]] = {}
    for i, r in enumerate(ok):
        c = cell_of(r["log_ratio"], r["cv"])
        if c is not None and c in core_cells:
            by_cell.setdefault(c, []).append(i)

    pts = _normalise(ok)
    seed_idx: list[int] = []
    for c in sorted(by_cell):
        members = by_cell[c]
        mx = sum(pts[i][0] for i in members) / len(members)
        my = sum(pts[i][1] for i in members) / len(members)
        seed_idx.append(min(members, key=lambda i: ((pts[i][0] - mx) ** 2 + (pts[i][1] - my) ** 2, i)))

    n_core = len(seed_idx)

    # Stage 2: spread the remainder over whatever else this κ reaches.
    chosen = farthest_point(ok, n_points, tuple(seed_idx))
    reduced = len(chosen) < n_points
    return {
        "points": [ok[i] for i in chosen],
        "n_core": n_core,
        "n_explore": max(0, len(chosen) - n_core),
        "reduced": reduced,
        "reason": (f"only {len(ok)} feasible combinations available" if reduced else ""),
    }


def select_all(summary: dict, rows: list[dict], n_points: int = N_PER_KAPPA) -> dict:
    """Choose points at every κ. `summary` comes from `feasible.summarise`."""
    core = {tuple(c) for c in summary["common_core"]}
    out = {}
    for k in summary["kappas"]:
        rk = [r for r in rows if abs(r["kappa"] - k) < 1e-12]
        out[k] = select_for_kappa(rk, core, n_points)
    return {"n_per_kappa": n_points, "n_common_core": len(core), "by_kappa": out}


def format_report(selection: dict) -> str:
    lines = [f"POINT SELECTION -- target {selection['n_per_kappa']} per kappa "
             f"({selection['n_common_core']} common-core cells)", "=" * 78,
             f"{'kappa':>8} {'points':>8} {'core':>6} {'explore':>9}  note", "-" * 78]
    for k, s in selection["by_kappa"].items():
        note = ""
        if s["reduced"]:
            note = f"REDUCED: {s['reason']}"
        lines.append(f"{k:>8g} {len(s['points']):>8} {s['n_core']:>6} {s['n_explore']:>9}  {note}")
    total = sum(len(s["points"]) for s in selection["by_kappa"].values())
    lines += ["", f"total map points: {total}",
              "Core points are placed FIRST at every kappa, so a cross-kappa comparison is paired",
              "by construction -- and so is any prefix of the library."]
    return "\n".join(lines)
