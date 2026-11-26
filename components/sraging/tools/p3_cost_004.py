"""Spec 004's pilot P3, FP against Langevin cost for `engine="auto"` (tasks T080; research R19, R21; C10). The phase
`p3-cost` in tools/acceptance_004.py runs it; its result is the `auto` boundary T130 records.

At n in NS, for two heterogeneous configurations at sd 0.2 on one preset:
- **out of gen-1's table** (eta heterogeneity): gen-1 cannot draw it (only xc), so the rows record its refusal; the
  Langevin draws it, with its error statement;
- **in the table** (xc heterogeneity): gen-1 draws it (materialize, with its computed grid error) while n <= HET_N_CAP,
  and the Langevin draws it too.
Each row records the CPU of each engine, gen-1's grid-error caveats and the Langevin's KS and step bias. The preset,
its recording delta and the key are P3's configuration (tools/presets_004.py ROLES["p3"]: the human preset, delta
1 yr, key 3003; Naveh 2026-10-30); the Langevin's dt is P1's step for the preset's heterogeneous clamp cell.
"""
from __future__ import annotations

import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NS = (2_000, 20_000, 100_000)
SD = 0.2
CONFIGS = (("out-of-table", "eta"), ("in-table", "xc"))


def cohort_spec(params, parameter, n, key, delta, L):
    """gen-1's CohortSpec for (params, xc heterogeneity at SD), or (None, why) outside its domain (only xc; a valid
    spec). The tester's conversion at xc = 1: rho_eta = eta L^2, rho_beta = beta L, rho_eps = epsilon L; the external
    hazard as ExtH = -ln(h)."""
    from sraging import generator as GEN
    from sraging.errors import UnsupportedOptionError
    if parameter != "xc":
        return None, f"gen-1 draws heterogeneity in xc only, not {parameter}"
    if params.xc != 1.0:
        return None, "gen-1's specs are at xc = 1"
    kw = dict(name=f"p3-{parameter}-{n}", rho_eta=params.eta * L * L, rho_beta=params.beta * L, rho_eps=params.epsilon * L,
              kappa=params.kappa, n=int(n), L_phys=float(L), seed=int(key), delta=float(delta), het_param="xc",
              het_var=SD)
    if params.h_ext > 0:
        kw["external_hazard"] = -math.log(params.h_ext)
    try:
        return GEN.CohortSpec(**kw), None
    except UnsupportedOptionError as ex:
        return None, str(ex)


def measure(params, parameter, n, key, delta, dt) -> dict:
    """One (configuration, n) row: gen-1 (in its domain) and the Langevin, each with its CPU."""
    from sraging import generator as GEN
    from sraging.engine.curves import model_L
    from sraging.engine.inputs import Heterogeneity
    from sraging.langevin import simulate
    row = {"config": next(c for c, p in CONFIGS if p == parameter), "parameter": parameter, "sd": SD, "n": int(n)}
    spec, why = cohort_spec(params, parameter, n, key, delta, model_L(params))
    if spec is not None and n > GEN.HET_N_CAP:
        spec, why = None, f"n = {n} > gen-1's HET_N_CAP = {GEN.HET_N_CAP} for a heterogeneous cohort"
    if spec is None:
        row["gen1"] = {"in_domain": False, "reason": why}
    else:
        c0 = time.process_time()
        g = GEN.materialize(spec)
        row["gen1"] = {"in_domain": True, "cpu_s": time.process_time() - c0, "grid_error": g.grid_error,
                       "caveats": list(g.caveats)}
    c0 = time.process_time()
    s = simulate(params, n=int(n), key=int(key), dt=float(dt), heterogeneity=Heterogeneity(parameter=parameter, sd=SD),
                 delta=float(delta))
    row["langevin"] = {"cpu_s": time.process_time() - c0, "ks_D": s.error.ks_D, "crit_95": s.error.crit_95,
                       "verdict": s.error.verdict, "step_bias": s.error.step_bias, "clean": s.clean,
                       "warnings": [w.code for w in s.warnings], "t_max": s.provenance["t_max"]}
    return row


def boundary(rows) -> dict:
    """What `auto` records (T130): per configuration and n, the engine it takes and why. Inside gen-1's domain gen-1
    (its law is exact up to its stated grid error); outside it the Langevin (Dg), with both CPUs where both ran."""
    out = []
    for r in rows:
        g = r["gen1"]
        out.append({"config": r["config"], "n": r["n"], "engine": "gen-1" if g["in_domain"] else "langevin",
                    "reason": "inside gen-1's domain" if g["in_domain"] else g["reason"],
                    "cpu_gen1_s": g.get("cpu_s"), "cpu_langevin_s": r["langevin"]["cpu_s"]})
    return {"rule": "gen-1 inside its domain (xc or no heterogeneity, n <= HET_N_CAP, a valid CohortSpec); the "
                    "Langevin outside it (Dg), with ENGINE_AUTO_LANGEVIN naming the reason", "rows": out}
