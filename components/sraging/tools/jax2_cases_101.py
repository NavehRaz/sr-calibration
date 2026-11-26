"""Spec 003b's value/gradient bytes on jax2 1.0.1 (tasks T013; research R8; 003e close item 5): the first-interval
mass below 1e-16 that fix B (the first-interval mass from F, not S(0) - S(delta)) exists for, with its twin on
v0.3.3's numerics, which cancels there and scores the -700 floor.

The objective is built by tools/jax2_cases.py's `objective` (fit's own statements, over whichever modules it is
given) from the 1.0.1 port's modules or the 1.0.1 oracle's (tools/oracle_jax2_101.py; one oracle per process). The
per-row terms are fp_jax's own (`make_loglik(with_terms=True)`) on the same inputs.

    python tools/jax2_cases_101.py scan     # the deterministic theta scan: prints PANEL [parity.bytes] entries
    python tools/jax2_cases_101.py check    # the panel's entries: bytes and twin, one JSON line per entry
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (HERE, os.path.join(REPO, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

import jax2_cases as C  # noqa: E402

PARAMS = C.PARAMS
MASS_CUT = 1e-16          # 003e close item 5: the first-interval mass below 1e-16
FLOOR = -700.0            # v0.3.3's (and fp_jax's) log floor
#: the scan: epsilon x 10^(-k * 0.01), k = 0..600, from jax2's own 1.0.1 theta-hat of the cohort (its S3 record).
#: Fine steps, because v0.3.3's twin is the floor on only part of the range: measured 2026-10-20 over the scan, where
#: the 1.0.1 term lies between the floor and log(1e-16), v0.3.3 scored exactly -700 on 34 of 127 theta
#: (hazard-k0.01-p04-c1-s1) and 56 of 164 (hazard-k0.01-p04-c0-s0), and S(0) - S(delta)'s round-off (-29 to -37
#: nats) on the rest; both are the cancellation fix B removes.
SCAN_STEP_DEX, SCAN_K = 0.01, 600


def port_modules() -> dict:
    from sraging.compat.jax2_101 import claims, estimator, fp_jax, observation_model, settings
    return {"om": observation_model, "est": estimator, "fj": fp_jax, "claims": claims,
            "classes": {"jax2": settings.MleProfileJax2, "jax2_h": settings.MleProfileJax2H}, "side": "port"}


def oracle_modules() -> dict:
    import oracle_jax2_101 as OJ1
    o = OJ1.load()
    return {"om": o.adapter._sibling("observation_model"), "est": o.estimator, "fj": o.fp_jax, "claims": o.adapter,
            "classes": o.classes, "side": "oracle"}


def objective(mods, obs, name):
    return C.objective(mods, obs, name)


def terms_fn(mods, obs, name):
    """(z -> per-row terms, prepared ages, prepared events): fp_jax's `with_terms` on fit's own inputs."""
    cls = mods["classes"][C.SHORT[name]]
    prep = mods["om"].prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=cls.WINDOW,
                              relabel_on=cls.RELABEL)
    t, e = np.asarray(prep.ages, float), np.asarray(prep.events, int)
    delta = obs.delta if obs.delta else None
    L, kw, _c, _f = mods["est"]._grid(t, e, delta, floor_early=cls.FLOOR_EARLY, n_rule=cls.N_RULE,
                                      dt_rule=cls.DT_RULE, **({"early_all": True} if cls.EARLY_ALL else {}))
    tf = mods["fj"].make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                                sampling_dt=kw.get("sampling_dt"), hazard=cls.HAZARD, window=prep.window_eff,
                                with_terms=True, exact_from_F=bool(cls.EXACT_F), early=kw.get("early"))[2]
    return (lambda z: np.asarray(tf(np.asarray(z, float))["term"], float)), t, e


def terms(mods, obs, name, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(per-row terms at z, prepared ages, prepared events)."""
    f, t, e = terms_fn(mods, obs, name)
    return f(z), t, e


def age0(t, e) -> np.ndarray:
    """The rows of deaths recorded at age 0: deaths in the first interval [0, delta)."""
    return np.flatnonzero((np.asarray(t) == 0.0) & (np.asarray(e) == 1))


def v033_fn(obs):
    """(theta -> v0.3.3's per-row terms on the raw cohort, its age-0 death rows): the old numerics, no hazard."""
    from sraging import Cohort
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.params import SRParams
    c = Cohort.from_arrays(np.asarray(obs.ages, float), np.asarray(obs.events, int), delta=obs.delta)
    ev = Evaluator(c, numerics="v033", grid=GridPolicy("v033"))
    return (lambda th: np.asarray(ev.terms(SRParams(**{p: th[p] for p in PARAMS}))[0], float)), age0(c.ages, c.events)


def compare(po, oo, z) -> list[str]:
    from parity_checks import bytes_diff
    out = bytes_diff(np.asarray(po.f(z)), np.asarray(oo.f(z)), "value")
    (va, ga), (vb, gb) = po.vg(z), oo.vg(z)
    return out + bytes_diff(np.asarray(va), np.asarray(vb), "vg.value") + bytes_diff(np.asarray(ga), np.asarray(gb),
                                                                                    "vg.grad")


def first_bin(port, obs, theta: dict, fns=None) -> dict:
    """The J objective's age-0 terms on the port at theta, their mass, and v0.3.3's twin terms. `fns`: the compiled
    (terms_fn, v033_fn) of this cohort, to reuse across theta."""
    (tf, t, e), (f33, rows33) = fns or (terms_fn(port, obs, "jax2"), v033_fn(obs))
    tt = tf(C.z_of(theta))
    rows = age0(t, e)
    t33 = f33(theta)
    term = float(np.max(tt[rows])) if rows.size else None
    return {"n_age0": int(rows.size), "term": term, "mass": None if term is None else math.exp(term),
            "v033_terms": sorted({float(x) for x in t33[rows33]}), "n_age0_v033": int(rows33.size)}


def scan_one(port, obs, theta0: dict) -> dict:
    """The first k with the port's first-interval mass below MASS_CUT, its term above the floor, and v0.3.3's twin at
    exactly the floor: theta = theta0 with epsilon x 10^(-k * SCAN_STEP_DEX)."""
    fns = (terms_fn(port, obs, "jax2"), v033_fn(obs))
    for k in range(SCAN_K + 1):
        th = dict(theta0, epsilon=theta0["epsilon"] * 10 ** (-k * SCAN_STEP_DEX))
        fb = first_bin(port, obs, th, fns)
        if fb["term"] is None:
            return {"found": False, "reason": "no age-0 death"}
        if fb["mass"] < MASS_CUT and fb["term"] > FLOOR + 10 and fb["v033_terms"] == [FLOOR]:
            return {"found": True, "k": k, "theta": th, **fb}
    return {"found": False, "reason": f"no k <= {SCAN_K}"}


def scan(entries: list[dict]) -> list[dict]:
    """entries: [{cid, library, name, theta0 (dict), theta0_source}] -> the [parity.bytes] entries found."""
    import oracle_jax2_101 as OJ1
    port = port_modules()
    out = []
    for ent in entries:
        obs = OJ1.observation(ent["cid"], ent["library"])
        r = scan_one(port, obs, ent["theta0"])
        out.append({**{k: ent[k] for k in ("cid", "library", "name", "theta0_source")}, "theta0": ent["theta0"], **r})
    return out


def check(entry: dict) -> dict:
    """One [parity.bytes] entry, side by side: the port's and the oracle's value and value+gradient bytes at theta (the
    J objective; for a `_h` entry also the hazard objective at the box floor and hL = 0.1), the port's first-interval
    mass and v0.3.3's twin."""
    import oracle_jax2_101 as OJ1
    port, orc = port_modules(), oracle_modules()
    obs = OJ1.observation(entry["cid"], entry["library"])
    th = {p: float(entry["theta"][p]) for p in PARAMS}
    diffs = {}
    po, pi = objective(port, obs, "jax2")
    oo, oi = objective(orc, obs, "jax2")
    diffs["jax2"] = ([] if pi == oi else ["grid info differs"]) + compare(po, oo, C.z_of(th))
    if C.SHORT[entry["name"]] == "jax2_h":
        po, pi = objective(port, obs, "jax2_h")
        oo, oi = objective(orc, obs, "jax2_h")
        for hl, h in C.h_points(pi):
            diffs[f"jax2_h@{hl}"] = ([] if pi == oi else ["grid info differs"]) + compare(po, oo, C.z_of(th, h))
    fb = first_bin(port, obs, th)
    ofb_t, ot, oe = terms(orc, obs, "jax2", C.z_of(th))
    orc_term = float(np.max(ofb_t[age0(ot, oe)]))
    return {"cid": entry["cid"], "name": entry["name"], "diffs": diffs, "bytes_equal": not any(diffs.values()),
            **fb, "oracle_term": orc_term, "term_bytes_equal": float(fb["term"]).hex() == orc_term.hex()}


def main(argv=None):
    import tomllib
    argv = sys.argv[1:] if argv is None else argv
    panel = tomllib.load(open(os.path.join(REPO, "tests", "acceptance", "panel_003b.toml"), "rb"))
    if argv[:1] == ["check"]:
        import oracle_jax2_101 as OJ1
        problems = OJ1.preflight()
        if problems:
            print(json.dumps({"not_applicable": problems}), flush=True)
            return
        for ent in (panel.get("parity") or {}).get("bytes") or []:
            print(json.dumps(check(ent)), flush=True)
        return
    if argv[:1] == ["scan"]:
        rel = os.environ.get("SRAGING_J2_REL_REPORTS")
        if not rel:
            sys.exit("scan: set SRAGING_J2_REL_REPORTS to jax2's REL reports/rel101 (read-only)")
        entries = []
        for p in panel["parity"]["pairs"]:
            if p.get("record") and p["cid"].startswith("hazard-"):
                rec = json.load(open(os.path.join(rel, p["record"])))
                th = [float(x) for x in rec["result"]["theta"]]
                entries.append({"cid": p["cid"], "library": p["library"], "name": p["name"],
                                "theta0": dict(zip(PARAMS, th[:4])), "theta0_source": p["record"]})
        for r in scan(entries):
            print(json.dumps(r), flush=True)
        return
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
