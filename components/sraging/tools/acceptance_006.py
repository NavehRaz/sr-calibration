"""Spec 006's acceptance phases (specs/006-calibrator-capabilities/contracts/acceptance.md; research R11-R17).

    .venv/bin/python tools/acceptance_006.py <phase> [--dev] [--dry-run] [--resume JOBID --job-head SHA]

Phases: parity (R-7, R-8; fit-free), identity (R-5, R-6; Mac, side by side), stored (R-1, R-2 on stored records; 0
fits), pilot, a1, a2, realfull (WEXAC, in waves), v033-leverage (R-9; WEXAC), unchanged (FR-030; the MAIN checkout).

Every phase runs through spec 003's `acceptance_003.Run` (suffix "006"): a clean committed tree, its price printed
first, its result in tests/acceptance/results/<phase>_006.json with the repo head, the host and its cost. A phase that
reads a rule needs tests/acceptance/rules_006.toml committed with `frozen = true` (Naveh's OK, T058) for a non-dev run;
`parity` and `identity` read no rule (byte identity, and a tolerance the phase measures itself) and may run non-dev
before the freeze. `--dev` runs a pilot, labelled "development", which never counts. CPU is this process's plus its
children's plus a WEXAC job's (FR-020); a phase that reaches its cap (CAPS, the same as rules_006.toml's [caps]) stops,
keeps what finished and reads `capped`, never PASS. a1 and a2 have no cap until the freeze re-prices them with the
pilot's r, and refuse to run before it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tomllib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

import acceptance_003 as A3  # noqa: E402
import acceptance_005 as A5  # noqa: E402
from acceptance_005 import Budget, counts_as_pass, file_sha256, phase_status, run_capped  # noqa: E402,F401

RULES_REL = "tests/acceptance/rules_006.toml"
RULES = os.path.join(REPO, RULES_REL)

PHASES = ("parity", "identity", "stored", "pilot", "a1", "a2", "realfull", "v033-leverage", "unchanged")
#: contracts/acceptance.md and tasks T002, in CPU-h; rules_006.toml's [caps] repeats them (tests/unit/
#: test_acceptance_006.py checks it). a1 and a2 are None until the freeze sets 1.25 x the cell re-priced with r (T058).
CAPS = {"parity": 0.5, "identity": 5.0, "stored": 0.1, "pilot": 12.0, "a1": None, "a2": None, "realfull": 24.0,
        "v033-leverage": 8.0, "unchanged": 1.0}
#: the phases that read no rule, so they may run non-dev before the freeze
RULE_FREE = ("parity", "identity")
#: where each phase's CPU is spent (FR-020 counts WEXAC and the Mac separately)
WHERE = {"parity": "mac", "identity": "mac", "stored": "mac", "pilot": "wexac", "a1": "wexac", "a2": "wexac",
         "realfull": "wexac", "v033-leverage": "wexac", "unchanged": "mac"}
PRICE = {"parity": "< 0.1 CPU-h, Mac, fit-free [I]", "identity": "~3 CPU-h, Mac, side by side [I]",
         "stored": "seconds, Mac, 0 fits [I]", "pilot": "8-10 CPU-h, WEXAC, 6 cohorts [I, research R14]",
         "a1": "153 CPU-h, WEXAC, 196 cohorts (54 with the lead pair) in waves [T057f, from the pilot's r]",
         "a2": "105 CPU-h, WEXAC, 44 cohorts (11 with the lead pair) in waves [T057f, from the pilot's r]",
         "realfull": "18.6 CPU-h, WEXAC, 20 cohorts [T057f]",
         "v033-leverage": "P3 4.3 CPU-h on WEXAC, R-9's candidates up to ~2.7 CPU-h on the Mac [T054, T057e]",
         "unchanged": "~0.5 CPU-h, Mac, the MAIN checkout [I]"}


def read_rules() -> dict:
    return tomllib.load(open(RULES, "rb"))


# --------------------------------------------------------------------------------------------- the envelope
def start(phase, *, dev) -> "A3.Run":
    """Spec 003's Run (suffix 006): refuses a dirty tree or an uncommitted manifest unless dev; a non-dev run of a
    phase that reads a rule also needs rules_006.toml frozen (T058); a1 and a2 need their cap, set at the freeze."""
    rules = read_rules()
    if CAPS[phase] is None and (rules.get("caps") or {}).get(phase) is None:
        sys.exit(f"[{phase}] REFUSED: no cap yet; the freeze sets it at 1.25 x the cell re-priced with the pilot's r "
                 f"(tasks T058). No gate fit runs before the freeze")
    if not dev and phase not in RULE_FREE and not rules.get("frozen"):
        sys.exit(f"[{phase}] REFUSED: {RULES_REL} is not frozen (frozen = false): no non-dev run before the freeze "
                 f"with Naveh's OK (tasks T058); pass --dev for a pilot that never counts")
    return A3.Run(phase, manifests=[RULES_REL], dev=dev, cost=PRICE[phase], suffix="006")


def cap_for(phase) -> float:
    """The phase's cap in CPU-h: CAPS, or for a1 and a2 the frozen rules' [caps] entry."""
    return float(CAPS[phase] if CAPS[phase] is not None else read_rules()["caps"][phase])


def finish(run, payload: dict, budget: Budget | None = None):
    out = dict(payload)
    out["manifests_sha256"] = {m: file_sha256(os.path.join(REPO, m)) for m in run.manifests
                               if os.path.exists(os.path.join(REPO, m))}
    if budget is not None:
        out["cap"] = {**budget.record(), "where": WHERE[run.phase]}
        out["status"] = phase_status(out.get("status"), budget)
    return run.write(out)


# ------------------------------------------------------------------------ parity (R-7, R-8, FR-005, FR-016; T037)
#: the parity cases: the example ITP control on [365, 1100] days, and v6's 46.7 % edge-tie window cohort on its window
PARITY_CASES = (("examples/data/itp_c2016_het3_control_female.csv", (365.0, 1100.0)), ("window-k2-p01-c1-s1", None))
#: log-space shifts of theta from each case's base (the example fit's theta-hat; v6's truth), where S(lo) is resolved;
#: the tolerance is measured on these. Between them and the clip lies a band (1e-12 < S(lo) << 1) where the meshes'
#: relative error in S(lo) is multiplied by n (measured: 9.6e-12 against 7.8e-12, 11.5 nats on n = 52), so no point is
#: judged there; the underflow point (gate M2's) is reached by eta x 10^k until the engine's S(lo) < UNDERFLOW_S_LO,
#: where every implementation clips at 1e-12 and must agree
PARITY_SHIFTS = ((0.0, 0.0, 0.0, 0.0), (0.5, -0.3, 0.2, 2.0), (-1.0, 1.0, -0.5, 6.0))
UNDERFLOW_S_LO = 1e-15
PLANTS = ("cond", "edge", "clip")


def parity_cohort(case):
    """(the cohort as a user gives it, its base theta as SRParams)."""
    import json
    from sraging import Cohort, SRParams
    path, window = case
    if path.endswith(".csv"):
        c = Cohort.from_csv(os.path.join(REPO, path), delta=1.0, time_unit="days", window=window)
        th = json.load(open(os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")))
        b = th["theta_hat"]
        return c, SRParams(eta=b["eta"], beta=b["beta"], epsilon=b["epsilon"], kappa=b["kappa"], time_unit="days")
    import oracle_jax2_101 as OJ1
    lib = OJ1.library("v6")
    ages, events = lib.data(path, verify=True)
    spec = lib.spec(path)
    t = lib.truths([path])[path]
    return (Cohort.from_arrays(ages, events, delta=spec.delta, window=tuple(spec.window), name=path),
            SRParams(eta=t["eta"], beta=t["beta"], epsilon=t["epsilon"], kappa=t["kappa"]))


def underflow_theta(cohort, base):
    """base with eta x 10^k for the smallest k = 1..12 at which the engine's S(lo) is below UNDERFLOW_S_LO."""
    from sraging import SRParams
    from sraging.engine.evaluator import Evaluator
    ev = Evaluator(cohort, numerics="v1")
    for k in range(1, 13):
        p = SRParams(**{**base.as_dict(), "eta": base.eta * 10.0 ** k})
        if ev.window_report(p)["S_lo_min"] < UNDERFLOW_S_LO:
            return p
    raise RuntimeError("no eta x 10^k (k <= 12) drives S(lo) below the underflow threshold")


def parity_case(cohort, base, plant=None) -> dict:
    """R-7 and FR-005 for one cohort, fit-free. The data are what W fits (window_path, then the frozen snap); on them
    W's frozen likelihood (window on and off), the engine at numerics "v1" and "v033" (window on and off, the same
    rows) are evaluated at base x exp(shift). The no-window difference W - engine is the tolerance; the window's own
    term (on minus off) must agree within it. `plant` (the unit tests) breaks one thing: W's conditioning term x0.99
    ("cond"), the engine fed the rows before the snap's edge tie ("edge"), or the engine's v1 clip undone ("clip")."""
    import math
    import numpy as np
    from sraging import Cohort, SRParams
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.engine import observation as OBS
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    from sraging.compat.tester_f94ec2b.observation import apply_window
    wc, rec, _rows = PJ.window_path(cohort, PJ.MleProfileJax2W, port=PJ.PORT_W)
    inp = PJ.fit_inputs(wc, "profile_jax2_w", PJ.PORT_W)
    kw, fj = inp["kw"], PJ.PORT_W.fp_jax

    def mk(window):
        return fj.make_loglik(inp["t"], inp["e"], inp["L"], N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"],
                              Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"), hazard=False, window=window,
                              with_terms=True, exact_from_F=bool(kw.get("exact_from_F")), early=kw.get("early"))
    (fw, _a, tw), (fj_, _b, tj) = mk(inp["window"]), mk(None)
    win = tuple(float(x) for x in inp["window"])
    if plant == "edge":                       # the rows as apply_window leaves them: the edge tie not applied
        ages, events = apply_window(cohort.ages, cohort.events, (win[0], win[1]))
    else:
        ages, events = inp["t"], inp["e"]
    grid = GridPolicy("fixed", N=int(kw["N"]), dt=float(kw["dt"]), tau_max=float(kw["tau_max"]), L=float(inp["L"]))
    ev = {(n, w): Evaluator(Cohort.from_arrays(ages, events, delta=cohort.delta, window=win if w else None,
                                               time_unit=cohort.time_unit), numerics=n, grid=grid)
          for n in ("v1", "v033") for w in (True, False)}
    z0 = np.log([base.eta, base.beta, base.epsilon, base.kappa])
    under = underflow_theta(wc, base)
    rows = []
    for shift in PARITY_SHIFTS + ("underflow",):
        if shift == "underflow":
            z = np.log([under.eta, under.beta, under.epsilon, under.kappa])
        else:
            z = z0 + np.asarray(shift)
        p = SRParams(**dict(zip(("eta", "beta", "epsilon", "kappa"), map(float, np.exp(z)))), time_unit=base.time_unit)
        dw, dj = tw(z), tj(z)
        cond = float(np.asarray(dw["cond"]))
        w_on = float(fw(z)) if plant != "cond" else float(fj_(z)) + 0.99 * cond
        row = {"shift": shift if shift == "underflow" else list(shift), "S_lo": float(np.asarray(dw["S_lo"])),
               "W_on": w_on, "W_off": float(fj_(z)),
               "decomposition_terms_equal": bool(np.array_equal(np.asarray(dw["term"]), np.asarray(dj["term"]))),
               "decomposition_value_ulps": abs(float(fw(z)) - (float(fj_(z)) + cond)) / math.ulp(abs(float(fw(z))))}
        for n in ("v1", "v033"):
            e_on, e_off = ev[(n, True)].value(p), ev[(n, False)].value(p)
            if plant == "clip" and n == "v1":         # spec 002-005's v1: every row on the log+ floor, not the clip
                _t, aux = ev[(n, True)].terms(p)
                e_on += float(len(ages)) * (float(aux["log_S_lo"]) - float(OBS._logp(aux["S_lo"])))
            row[n] = {"on": float(e_on), "off": float(e_off),
                      "S_lo": float(ev[(n, True)].window_report(p)["S_lo_min"])}
        rows.append(row)
    out = {}
    # the tolerance comes from the points where the window's term is informative: S(lo) resolved (not clipped) and
    # below 1 (at S(lo) = 1 the term is 0, so such a point tests nothing and its no-window difference -- mesh error at
    # an extreme theta, 20 nats measured on v6's edge-tie cohort -- would only loosen the tolerance)
    informative = [i for i, r in enumerate(rows) if r["shift"] != "underflow" and 1e-12 <= r["S_lo"] < 1.0 - 1e-9]
    for n in ("v1", "v033"):
        d_off = [r["W_off"] - r[n]["off"] for r in rows]
        d_term = [(r["W_on"] - r["W_off"]) - (r[n]["on"] - r[n]["off"]) for r in rows]
        tol = max(abs(d_off[i]) for i in informative) if informative else 0.0
        out[n] = {"tolerance": tol, "tolerance_points": informative, "no_window_diffs": d_off,
                  "window_term_diffs": d_term, "passes": bool(informative) and all(abs(x) <= tol for x in d_term)}
    dec = all(r["decomposition_terms_equal"] and r["decomposition_value_ulps"] <= 4 for r in rows)
    u = rows[-1]
    under = u["S_lo"] < 1e-12 and all(u[n]["S_lo"] < 1e-12 for n in ("v1", "v033"))
    return {"rows": rows, "R-7": out, "FR-005": {"passes": dec}, "underflow_point": under,
            "passes": all(v["passes"] for v in out.values()) and dec and under, "plant": plant,
            "window_eff": list(win), "n": int(len(inp["t"])), "edge_tie_deaths": int(rec["n_deaths_censored"])}


#: R-8: the value at hi = inf equals a large finite hi's exactly; the gradient to round-off -- the two are different
#: compiled programs (no S_hi operations at inf), so XLA may order the gradient's sums differently (measured up to
#: 8.5e-14 relative on the parity cases; exact on tests/unit/engine/test_engine_window_006.py's cohort)
R8_GRAD_RTOL = 1e-12


def parity_grid(cohort):
    """The engine's fixed grid on W's inputs (N, dt, tau_max, L of the frozen fit's grid): one mesh for every
    evaluation compared (the engine's automatic grid depends on the window's end, so hi = inf and a large finite hi
    would otherwise be compared on different meshes)."""
    from sraging.calibrators import profile_jax2 as PJ
    from sraging.engine.grid import GridPolicy
    wc, _rec, _rows = PJ.window_path(cohort, PJ.MleProfileJax2W, port=PJ.PORT_W)
    inp = PJ.fit_inputs(wc, "profile_jax2_w", PJ.PORT_W)
    kw = inp["kw"]
    return GridPolicy("fixed", N=int(kw["N"]), dt=float(kw["dt"]), tau_max=float(kw["tau_max"]), L=float(inp["L"]))


def parity_edges(cohort, base) -> dict:
    """R-8: the engine at hi = inf against a large finite on-grid hi (value and gradient, h off and on, on one fixed
    grid), and at the underflow point the clip reported and no spurious maximum."""
    import math
    from sraging import Cohort, SRParams
    from sraging.engine.evaluator import Evaluator
    lo = float(cohort.window[0])
    grid = parity_grid(cohort)
    out = {}
    for label, h in (("no_h", None), ("h", 1e-4)):
        free = ("eta", "beta", "epsilon", "kappa") + (("h_ext",) if h else ())
        p = SRParams(**{**base.as_dict(), "h_ext": h or 0.0})
        vals = {}
        for hi in (math.inf, 1.0e7):
            c = Cohort.from_arrays(cohort.ages, cohort.events, delta=cohort.delta, window=(lo, hi),
                                   time_unit=cohort.time_unit)
            v, g = Evaluator(c, numerics="v1", free=free, grid=grid).value_and_grad(p)
            vals[hi] = (float(v), {k: float(x) for k, x in dict(g).items()})
        (vi, gi), (vb, gb) = vals[math.inf], vals[1.0e7]
        rel = max(abs(gi[k] - gb[k]) / max(abs(gb[k]), 1e-300) for k in gi)
        out[label] = {"finite": math.isfinite(vi) and all(math.isfinite(x) for x in gi.values()),
                      "equal": vi == vb and rel <= R8_GRAD_RTOL, "value_equal": vi == vb, "grad_max_rel": rel,
                      "value": vi}
    under = underflow_theta(cohort, base)
    e = Evaluator(cohort, numerics="v1", grid=grid)
    rep = e.window_report(under)
    out["underflow"] = {"lo_clipped": rep["lo_clipped"], "S_lo_min": rep["S_lo_min"], "value": e.value(under),
                        "value_at_base": e.value(base)}
    out["passes"] = (all(out[k]["finite"] and out[k]["equal"] for k in ("no_h", "h")) and rep["lo_clipped"]
                     and out["underflow"]["value"] <= 0.0)
    return out


def phase_parity(args):
    """R-7, R-8 and FR-005 (FR-016, C4), fit-free, before any fit. Reads no rule: its tolerance is measured here."""
    import oracle_jax2_101 as OJ1
    run = start("parity", dev=args.dev)
    budget = Budget("parity", cap_for("parity"))
    problems = OJ1.preflight()
    if problems:
        finish(run, {"status": "not_applicable", "reasons": problems}, budget)
        sys.exit(3)
    cases = {}
    for case in PARITY_CASES:
        cohort, base = parity_cohort(case)
        main = parity_case(cohort, base)
        # a plant counts where it can act: the edge plant only where the snap censored deaths at the edge
        plants = [pl for pl in PLANTS if pl != "edge" or main["edge_tie_deaths"] > 0]
        cases[case[0]] = {"R-7_FR-005": main, "R-8": parity_edges(cohort, base),
                          "plants_fail": {pl: not parity_case(cohort, base, plant=pl)["passes"] for pl in plants},
                          "plants_not_applicable": {pl: "no death censored at the window's edge in this case"
                                                    for pl in PLANTS if pl not in plants}}
        r = cases[case[0]]
        print(f"[parity] {case[0]}: R-7 {r['R-7_FR-005']['R-7']['v1']['passes']}/"
              f"{r['R-7_FR-005']['R-7']['v033']['passes']} (v1/v033), FR-005 {r['R-7_FR-005']['FR-005']['passes']}, "
              f"R-8 {r['R-8']['passes']}, plants fail {r['plants_fail']}", flush=True)
    ok = all(r["R-7_FR-005"]["passes"] and r["R-8"]["passes"] and all(r["plants_fail"].values())
             for r in cases.values())
    finish(run, {"status": "pass" if ok else "fail", "cases": cases}, budget)
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------------------------- identity (R-5, R-6; T020)
#: rules_006.toml's [panel.identity], repeated here so the phase reads no rule (tests/unit/test_acceptance_006.py
#: checks they agree): R-5's three unwindowed cohorts (a repo CSV, in days with delta 1 as 003f read it, or a v6 id)
#: and R-6's two windowed v6 cohorts, the pilot's cheapest
IDENTITY_UNWINDOWED = ("tests/data/itp_plus_age0.csv", "pure-k0.01-p01-c0-s0", "pure-k1-p01-c0-s0")
IDENTITY_WINDOWED = ("window-k0.01-p01-c0-s0", "window-k0.316-p03-c1-s0")
#: R-6's names, each against the oracle's run of the same switches: W (J's with the window on) and H
R6_ORACLE = {"profile_jax2_w": ("jax2", {"window": True}), "profile_jax2_h": ("jax2_h", {})}
#: R-5: what differs between W and J off-window by design (FR-003); every other path must be equal, floats by hex
R5_EXCLUDED = ("/diagnostics/switches/window", "/diagnostics/observation_model/window_on")


def identity_pairs() -> list[dict]:
    return ([{"cid": c, "name": "profile_jax2_w", "mode": "r5"} for c in IDENTITY_UNWINDOWED]
            + [{"cid": c, "name": n, "mode": "r6"} for c in IDENTITY_WINDOWED for n in R6_ORACLE])


def r5_view(r) -> dict:
    """What R-5 compares: theta-hat, the log-likelihood, the warning codes and identity_view (the claims and the
    method's diagnostics), each float by its hex."""
    return {"theta_hat": None if r.theta_hat is None else r.theta_hat.as_dict(), "loglik": r.loglik,
            "warnings": [w.code for w in r.warnings], **r.identity_view()}


def r5_diff(rw, rj) -> list[str]:
    """W's result against J's on an unwindowed cohort: the differing paths, R5_EXCLUDED left out."""
    from parity_checks import tree_diff
    return [p for p in tree_diff(r5_view(rw), r5_view(rj)) if not any(p.startswith(x) for x in R5_EXCLUDED)]


def identity_cohort(cid):
    """R-5's cohort: a repo CSV (days, delta 1) or a v6 cohort's raw arrays (the tester's data, read through the 1.0.1
    oracle's library loader, checksum-verified), with no window."""
    from sraging import Cohort
    if cid.endswith(".csv"):
        return Cohort.from_csv(os.path.join(REPO, cid), delta=1.0, time_unit="days")
    import oracle_jax2_101 as OJ1
    lib = OJ1.library("v6")
    ages, events = lib.data(cid, verify=True)
    return Cohort.from_arrays(ages, events, delta=lib.spec(cid).delta, name=cid)


def windowed_cohort(cid):
    """R-6's cohort as a user would give it: the v6 cohort's raw arrays with its window, before the tester's rule."""
    from sraging import Cohort
    import oracle_jax2_101 as OJ1
    lib = OJ1.library("v6")
    ages, events = lib.data(cid, verify=True)
    spec = lib.spec(cid)
    return Cohort.from_arrays(ages, events, delta=spec.delta, window=tuple(spec.window), name=cid)


def identity_worker(cid: str, name: str, mode: str, cited: dict, library: str = "v6") -> dict:
    """One identity pair in this (spawned) process. r5: W and J fit the unwindowed cohort through the public API.
    r6: W's or H's public window path (window_path, then the port's run) against the 1.0.1 oracle's run_fit of the
    same switches on the tester's own observation, at run_fit level; the two observations are compared first."""
    import time
    import traceback
    try:
        import sraging.compat.jax2_101  # noqa: F401  (configures the runtime before the oracle's fp_jax)
        import oracle_jax2_101 as OJ1
        from sraging import fit_cohort
        from sraging.calibrators import profile_jax2 as PJ
        o = OJ1.load()
        if mode == "r5":
            cohort = identity_cohort(cid)
            c0 = time.process_time()
            rw = fit_cohort(cohort, calibrator="profile_jax2_w", combinations=False)
            wc = time.process_time() - c0
            rj = fit_cohort(cohort, calibrator="profile_jax2", combinations=False)
            d = r5_diff(rw, rj)
            return {"cohort_id": cid, "rows": [{"cohort_id": cid, "name": name, "rule": "R-5", "level": "fitresult",
                                                "equal": not d, "diff_paths": d[:40], "n_diffs": len(d),
                                                "excluded": list(R5_EXCLUDED), "w_cpu_s": wc,
                                                "j_cpu_s": time.process_time() - c0 - wc}]}
        import acceptance_003b as B
        port = PJ.PORT_W if name == "profile_jax2_w" else PJ.PORTS["1.0.1"]
        wcoh, record, _rows = PJ.window_path(windowed_cohort(cid), port.classes[name], port)
        pobs = PJ.observation(wcoh)
        oobs = OJ1.observation(cid, library)
        same_obs = (pobs.ages == tuple(float(a) for a in oobs.ages) and pobs.events == tuple(int(e) for e in oobs.events)
                    and pobs.delta == oobs.delta and pobs.window == tuple(oobs.window))
        c0 = time.process_time()
        out = PJ.run_port(pobs, name, capture=True, port=port)
        pc = time.process_time() - c0
        short, extra = R6_ORACLE[name]
        c0 = time.process_time()
        ores = o.run_fit(oobs, **{**OJ1.fit_kwargs(short), **extra})
        oclaims, osc = o.final_claim_kwargs(ores, o.classes[short].SELF_CONSISTENT)
        oc = time.process_time() - c0
        lv = B.run_fit_diff((out["result"], out["claims"], out["sc_block"]), (ores, oclaims, osc), cited)
        return {"cohort_id": cid, "rows": [{"cohort_id": cid, "name": name, "rule": "R-6", "level": "run_fit",
                                            "observation_equal": same_obs, "equal": lv["equal"] and same_obs,
                                            "diff_paths": lv["diff_paths"][:40], "n_diffs": len(lv["diff_paths"]),
                                            "srk": lv["srk"], "window": record, "oracle_switches": {
                                                **OJ1.fit_kwargs(short), **extra},
                                            "port_cpu_s": pc, "oracle_cpu_s": oc}]}
    except Exception as ex:  # noqa: BLE001 -- a crashed pair is a row, never a silent pass
        return {"cohort_id": cid, "name": name, "mode": mode, "error": f"{type(ex).__name__}: {ex}",
                "trace": traceback.format_exc()[-3000:]}


def identity_verdict(results, n_pairs, stopped) -> dict:
    rows = [row for r in results for row in r.get("rows", [])]
    errors = [r for r in results if "error" in r]
    differ = [row for row in rows if row.get("equal") is not True]
    ok = not errors and not differ and not stopped and len(rows) == n_pairs
    by = {rule: sum(1 for row in rows if row["rule"] == rule and row["equal"]) for rule in ("R-5", "R-6")}
    return {"status": "pass" if ok else "fail", "rows": rows, "errors": errors, "stopped": stopped,
            "equal_by_rule": by, "differences": [{k: row.get(k) for k in ("cohort_id", "name", "rule", "diff_paths",
                                                                          "observation_equal", "srk")}
                                                 for row in differ]}


def phase_identity(args):
    """R-5 and R-6 (FR-003, FR-004), side by side on this host, one pair per spawned process (one oracle per
    process), one at a time. Reads no rule: byte identity."""
    import platform
    import acceptance_003b as B
    import oracle_jax2_101 as OJ1
    run = start("identity", dev=args.dev)
    budget = Budget("identity", cap_for("identity"))
    problems = OJ1.preflight()
    if problems:
        finish(run, {"status": "not_applicable", "reasons": problems}, budget)
        sys.exit(3)
    pairs = identity_pairs()
    results, stopped = [], False
    for p in pairs:                              # the cap is checked between pairs (FR-020)
        if budget.reached():
            budget.capped = True
            break
        r, stop = B.run_pairs([p], 1, B.cited_srk(), worker=identity_worker)
        results += r
        if stop:
            stopped = True
            break
    out = identity_verdict(results, len(pairs), stopped)
    print(f"[identity] {out['status']}: equal {out['equal_by_rule']} of R-5 {len(IDENTITY_UNWINDOWED)}, R-6 "
          f"{len(IDENTITY_WINDOWED) * len(R6_ORACLE)}", flush=True)
    finish(run, {**out, "pairs": len(pairs), "host": platform.node(), "cpu": B.cpu_model()}, budget)
    if out["status"] != "pass":
        sys.exit(1)


# --------------------------------------------------------------------------------------- stored (FR-018; T026)
PINS_REL = "tests/acceptance/pins_003g.json"
WINDOW_GROUPS = ("main", "bign")


def window_arm(lib) -> list:
    """v6's gated window arm (research R14): arm window, group main or bign (the sharp 52 not covered)."""
    return [s.cohort_id for s in lib.specs if s.arm == "window" and s.group in WINDOW_GROUPS]


def connect_v6():
    """(the pinned tester modules, Pins, the v6 Library): 003g's clone 0805d36, every record read checked against
    pins_003g.json; the tester's data read-only from its live checkout."""
    from sraging.adapters import srbench as ADP
    mods = ADP.connect_pinned(clone=ADP.TESTER_V6_PINNED, head=ADP.TESTER_V6_HEAD)
    pins = ADP.load_pins(os.path.join(REPO, PINS_REL), ADP.DEFAULT_TESTER)
    lib = mods["library.load"].Library(root=ADP.library_root("v6", ADP.DEFAULT_TESTER))
    return mods, pins, lib


def pinned_sets(mods, pins, run, ids):
    """A pinned run's ClaimSets for `ids`, each record read once and checked against its pin (003g's _sets)."""
    import json
    from sraging.adapters import srbench as ADP
    man = json.loads(pins.read_file(os.path.join("runs", f"{run}.json")))
    keep = [c for c in ids if c in man["keys"]]
    return ADP.load_pinned_sets(mods, man, os.path.join(ADP.DEFAULT_TESTER, "data", "store"), pins=pins, run_id=run,
                                cohort_ids=keep)


def _r1_text(q, v) -> str:
    cov = "none closed" if v["coverage"] is None else f"{v['coverage']:.3f}"
    cut = "" if v.get("cut") is None else f", cut {v['cut']:.3f}"
    return f"{q} {cov} ({v['hits']}/{v['claimed']}{cut})"


def pure_arm(lib) -> list:
    """v6's pure arm in the window arm's groups (main, bign): J's 688 records whose coverage set D (research R12)."""
    return [s.cohort_id for s in lib.specs if s.arm == "pure" and s.group in WINDOW_GROUPS]


def phase_stored(args):
    """FR-018: R-1's cut and R-2 on J's, H's and the null's stored window-arm records, at 0 fits. PASS when the rules
    discriminate as designed: J fails R-1, H passes; and (sr-naveh F3, 2026-11-11) J's pure-arm records, an equal case
    near 0.95, pass at their N, while H's closed claims shrunk to 0.88 coverage fail at theirs. J's window design
    effect and its clusters with 2 or more misses are reported (F4)."""
    import gate_006 as GATE
    run = start("stored", dev=args.dev)
    budget = Budget("stored", cap_for("stored"))
    rules = read_rules()
    mods, pins, lib = connect_v6()
    ids = window_arm(lib)
    truths, clusters = lib.truths(ids), lib.cluster_map(ids)
    CL = mods["score.claims"]
    counts, bounds = {}, {}
    for label in ("J", "H", "null"):
        sets = pinned_sets(mods, pins, rules["pins"][label], ids)
        counts[label] = GATE.coverage_counts(CL, sets, truths)
        bounds[label] = {q: GATE.closed_bounds(sets, q) for q in GATE.GATED + GATE.REPORTED}
    out = GATE.stored_summary(counts, bounds, clusters, rules["rules"])
    R1 = rules["rules"]["R-1"]
    pure = pure_arm(lib)
    out["equal_case"] = {"what": "J's pure-arm records (main, bign): must pass R-1 at their N",
                         "n_cohorts": len(pure),
                         **GATE.r1_table(GATE.coverage_counts(CL, pinned_sets(mods, pins, rules["pins"]["J"], pure),
                                                              lib.truths(pure)), R1)}
    out["planted_088"] = {"what": "H's window-arm closed claims shrunk in log space to 0.88 coverage: must fail R-1",
                          **{q: GATE.planted_shrink(bounds["H"][q], truths, q, R1) for q in GATE.GATED}}
    out["J_window_design_effect"] = {q: GATE.design_effect(bounds["J"][q], truths, clusters, q) for q in GATE.GATED}
    planted_fail = all(not out["planted_088"][q]["passes"] for q in GATE.GATED)
    ok = out["discriminates"] and out["equal_case"]["passes"] and planted_fail
    for label, r in out["runs"].items():
        if "R-1" in r:
            print(f"[stored] {label}: " + ", ".join(_r1_text(q, v) for q, v in r["R-1"]["gated"].items()), flush=True)
    print("[stored] equal case, J pure arm: " + ", ".join(_r1_text(q, v) for q, v in
                                                         out["equal_case"]["gated"].items()), flush=True)
    print("[stored] planted 0.88: " + ", ".join(_r1_text(q, out["planted_088"][q]) for q in GATE.GATED), flush=True)
    print("[stored] J's window D: " + ", ".join(f"{q} {v['D']:.2f}" for q, v in out["J_window_design_effect"].items()
                                                  if v["D"] is not None), flush=True)
    finish(run, {"status": "pass" if ok else "fail", "n_cohorts": len(ids),
                 "n_clusters": len(set(clusters.values())), **out}, budget)
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------------- the WEXAC cells (FR-019, FR-020; T027)
#: WEXAC (plan "Target Platform"): ~/.ssh/config's alias and the pinned venv (checked against uv.lock at each submit);
#: everything remote lives under ~/sraging_006/<head7>/, the v6 arrays under ~/sraging_006/v6 (no git there)
WEXAC = {"host": "wexac", "python": "~/sraging_cov003/venv/bin/python", "root": "sraging_006"}
#: ssh's own timeouts on every call: a dead link fails in about a minute, as a counted failed poll (2026-11-12: the VPN
#: dropped and the poller's ssh, with no timeout, hung on the dead connection for hours)
SSH_OPTS = ("-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")


def _remote(cmd, *, capture=True):
    """005's _remote with SSH_OPTS on the ssh, or on rsync's ssh."""
    if cmd[0] == "ssh":
        cmd = ["ssh", *SSH_OPTS, *cmd[1:]]
    elif cmd[0] == "rsync":
        cmd = ["rsync", "-e", " ".join(("ssh",) + SSH_OPTS), *cmd[1:]]
    return A5._remote(cmd, capture=capture)


#: spec 003's array settings (tools/wexac/coverage_003.sh): one CPU family, one core, the long queue
LSF = {"queue": "long", "wlimit": 10080, "select": "select[model==AMD_EPYC]", "mem_mb": 4096}
THREAD_ENV = ("PYTHONDONTWRITEBYTECODE=1 PJRT_NPROC=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
              "NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 "
              "XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'")
#: the pilot's expected CPU per cohort as a multiple of J's stored CPU, until it measures r: research R14 priced the
#: six at 8-10 CPU-h for 8,178 J CPU-s, the lead pair and leverage included [I]
PILOT_FACTOR = 4.4
#: v033's expected CPU per P3 cohort as a multiple of its stored CPU, until the ITP pilot measures the check's share [I]
V033_FACTOR = 2.0
#: the gate's cells: no fit of theirs before the freeze, --dev or not (tasks, hard rules); v033's P3 likewise
GATE_PHASES = ("a1", "a2", "realfull")
CELL_PHASES = ("pilot",) + GATE_PHASES + ("v033-leverage",)
POLL_S = 120
POLL_RETRIES = 30
RESULTS_GATE = os.path.join(REPO, "tests", "acceptance", "results", "gate_006")


V033_PANEL = "tests/acceptance/panel_003f.toml"
V033_STRATA = ("human_no_basin_fix", "human_basin_fix", "main_pure_n1000", "main_censor", "v6_single_age0")
ITP_AGE0 = "tests/data/itp_plus_age0.csv"


def v033_ids(rules) -> list:
    """R-9's null panel: rules_006.toml [panel.v033] ids once the freeze sets them (P3 or the fallback), else 003f's P3
    (panel_003f.toml [p3], 26 v6 cohorts)."""
    ids = ((rules.get("panel") or {}).get("v033") or {}).get("ids")
    if ids:
        return list(ids)
    p3 = tomllib.load(open(os.path.join(REPO, V033_PANEL), "rb"))["p3"]
    return [c for s in V033_STRATA for c in p3[s]]


def cell_ids(phase, lib, rules) -> list:
    """The cell's cohorts (research R14; rules_006.toml [panel])."""
    pan = rules["panel"]
    if phase == "v033-leverage":
        return v033_ids(rules)
    if phase == "pilot":
        return list(pan["pilot"])
    if phase in ("a1", "a2"):
        lim = pan["a1"]["n_below"]
        return [s.cohort_id for s in lib.specs if s.arm == "window" and s.group in pan[phase]["groups"]
                and ((s.n < lim) if phase == "a1" else (s.n >= pan["a2"]["n_at_least"]))]
    return [s.cohort_id for s in lib.specs if s.arm == "realfull" and s.window is not None
            and s.external_hazard is None and not s.het_var]


def cell_panel(phase, ids, lib, head, lead=(), nodes=None) -> dict:
    """What the elements read: each cohort's delta, window, truth (the tester's Library.truths), digest (the
    manifest's checksum), n, cluster and true kappa; whether it walks the lead pair (the pilot's cohorts, and Dg's in
    A1 and A2) and H's lowest-h stored node (R-4, exact)."""
    truths, clusters = lib.truths(ids), lib.cluster_map(ids)
    out = {}
    for c in ids:
        s = lib.spec(c)
        out[c] = {"delta": s.delta, "window": None if s.window is None else list(s.window), "truth": truths[c],
                  "digest": lib.manifest["checksums"][c], "n": s.n, "cluster": clusters[c], "kappa": s.kappa,
                  "lead_pair": phase == "pilot" or c in set(lead), "h_node": (nodes or {}).get(c)}
    return {"cell": phase, "head": head, "lead_pair": phase == "pilot", "cohorts": out,
            "calibrator": "profile_v033" if phase == "v033-leverage" else "profile_jax2_w"}


def dg_ids(phase, lib, rules, ids) -> list:
    """Dg's cohorts in a cell (Naveh, 2026-11-11): one per cluster among the cell's cohorts, drawn by the frozen seed
    from the cluster's cohorts in id order (never by price, which would select small n); none outside A1 and A2."""
    import numpy as np
    if phase not in ("a1", "a2"):
        return []
    by = {}
    for c, k in lib.cluster_map(ids).items():
        by.setdefault(k, []).append(c)
    rng = np.random.default_rng(int(rules["panel"]["dg"]["seed"]))
    return [sorted(by[k])[int(rng.integers(len(by[k])))] for k in sorted(by)]


def h_nodes(mods, pins, rules, ids) -> dict:
    """{cohort: {value, loglik}} of H's lowest-h stored profile node (R-4, exact): the element evaluates H at
    (theta-hat_W, value) and R-4 compares it with loglik."""
    out = {}
    for cs in pinned_sets(mods, pins, rules["pins"]["H"], ids):
        prof = ((cs.diagnostics or {}).get("hazard") or {}).get("profile") or []
        if prof:
            node = min(prof, key=lambda n: n["value"])
            out[cs.cohort_id] = {"value": float(node["value"]), "loglik": float(node["loglik"]),
                                 "run": rules["pins"]["H"]}
    return out


def j_prices(mods, pins, rules, ids, run=None) -> dict:
    """{cohort: J's stored CPU seconds} from J's pinned records (what priced the cells); `run` another pinned run (v033's,
    for R-9's panel)."""
    return {cs.cohort_id: float(cs.cpu_seconds) for cs in pinned_sets(mods, pins, run or rules["pins"]["J"], ids)}


V033_RUN = "jax033-v6-full-bign"                 # v0.3.3's pinned v6 run (pins_003g.json): R-9's prices


def pack_waves(prices: dict, factor, limit_s: float) -> list[list]:
    """The cell's cohorts, cheapest first, in waves whose expected CPU (price x factor, the factor one number or one
    per cohort) stays within limit_s; a cohort above the limit alone is its own wave (FR-020: at most 1/4 of the
    cell)."""
    waves, cur, tot = [], [], 0.0
    for c in sorted(prices, key=lambda k: (prices[k], k)):
        e = prices[c] * (factor[c] if isinstance(factor, dict) else factor)
        if cur and tot + e > limit_s:
            waves.append(cur)
            cur, tot = [], 0.0
        cur.append(c)
        tot += e
    if cur:
        waves.append(cur)
    return waves


def wave_allowed(spent_h: float, expected_h: float, cap_h: float) -> bool:
    """FR-020: the next wave goes only if what the cell has spent plus the wave's expected CPU stays within its cap."""
    return spent_h + expected_h <= cap_h


def spec_spent_wexac_h() -> float:
    """The spec's WEXAC CPU so far: every cell's ledger (FR-020: pilots, failed and superseded attempts count)."""
    return sum(ledger_spent_h(ph, "wexac") for ph in CELL_PHASES)


def spec_allows(expected_h: float, rules) -> bool:
    """The spec's WEXAC bound (Naveh, 2026-11-11: 300 CPU-h with Dg(a)): the next wave goes only if every cell's
    spend so far plus the wave's expected CPU stays within it. The cells' own caps sum past it."""
    bound = (rules.get("budget") or {}).get("wexac_cpu_h")
    return bound is None or spec_spent_wexac_h() + expected_h <= float(bound)


def factor_for(phase, rules, lead=False) -> float:
    """The expected CPU per cohort as a multiple of J's: the pilot's assumed factor, or the frozen r (r_lead for a
    cohort that walks the lead pair: T055, the pilot's fit, leverage and lead pair)."""
    if phase == "pilot":
        return PILOT_FACTOR
    if phase == "v033-leverage":                 # the fit plus its check, as a multiple of v033's stored CPU
        return float((rules.get("pricing") or {}).get("v033_factor", V033_FACTOR))
    return float(rules["pricing"]["r_lead" if lead else "r"])


def cell_paths(short, phase) -> dict:
    root = f"{WEXAC['root']}/{short}"
    return {"repo": f"{root}/repo", "run": f"{root}/run/{phase}", "out": f"{root}/out/{phase}",
            "logs": f"{root}/logs/{phase}", "v6": f"{WEXAC['root']}/v6"}


def job_script(paths) -> str:
    """The array element's script: the single-thread environment, then one gate_006 element per $LSB_JOBINDEX."""
    return ("#!/bin/bash\n"
            f"export {THREAD_ENV}\n"
            f"cd $HOME/{paths['repo']}\n"
            f"exec {WEXAC['python']} tools/gate_006.py element --panel $HOME/{paths['run']}/panel.json "
            f"--wave $HOME/{paths['run']}/$1 --v6 $HOME/{paths['v6']} --out $HOME/{paths['out']}\n")


def submit_command(paths, short, phase, k, n) -> list:
    import shlex
    name = f"g006_{phase}_{short}_w{k}"
    res = f"{LSF['select']} rusage[mem={LSF['mem_mb']}] span[hosts=1]"
    return ["ssh", WEXAC["host"], f"cd ~/{paths['run']} && bsub -q {LSF['queue']} -W {LSF['wlimit']} -n 1 -R "
                                  f"{shlex.quote(res)} -J {shlex.quote(name + f'[1-{n}]')} -o ~/{paths['logs']}/%J_%I.out "
                                  f"-e ~/{paths['logs']}/%J_%I.err ./job.sh wave_{k}.txt"]


#: LSF's job states; anything else bjobs prints (`Job <id> is not found`) is not a state
LSF_STATES = ("PEND", "RUN", "DONE", "EXIT", "PSUSP", "USUSP", "SSUSP", "WAIT", "ZOMBI", "UNKWN")


def array_states(stat_text) -> dict:
    """{state: count} of `bjobs -noheader -a -o stat` lines."""
    out = {}
    for ln in stat_text.split():
        if ln in LSF_STATES:
            out[ln] = out.get(ln, 0) + 1
    return out


def bhist_elements(text) -> dict:
    """{index: (state, cpu_s)} of the finished elements in `bhist -l` of one array: LSF's record once bjobs has purged
    the array after its clean period. bhist wraps lines at 80 columns, continuing after 21 spaces. (2026-11-11: the
    network was down past the clean period, and a poller reading bjobs alone waited on `Job <id> is not found`.)"""
    import re
    flat = re.sub(r"\n {21}", "", text)
    out = {}
    for block in re.split(r"(?=Job <\d+\[\d+\]>)", flat):
        m = re.match(r"Job <\d+\[(\d+)\]>", block)
        end = re.search(r"(Done successfully|Exited[^.]*)\. The CPU time used is ([\d.]+) seconds", block) if m else None
        if end:
            out[int(m.group(1))] = ("DONE" if end.group(1).startswith("Done") else "EXIT", float(end.group(2)))
    return out


def array_done(states: dict, n: int) -> bool:
    return sum(states.get(s, 0) for s in ("DONE", "EXIT")) >= n


def poll_array(job_id, n, *, remote=None, wait_s=None, retries=POLL_RETRIES) -> dict:
    """bjobs on the array until all n elements are DONE or EXIT; consecutive failed polls are waited out. bjobs drops
    each element an hour after it ends (LSF's clean period), so when it lists fewer than n and none still live, the
    finished ones are read from LSF's history (2026-11-12: wave 4's poller waited an hour on 1 listed element of 6)."""
    remote = remote or _remote
    wait_s = POLL_S if wait_s is None else wait_s
    failed = 0
    while True:
        try:
            states = array_states(remote(["ssh", WEXAC["host"], f"bjobs -noheader -a -o stat {job_id}"]))
            live = any(s not in ("DONE", "EXIT") for s in states)
            if not live and sum(states.values()) < n:       # some or all purged by bjobs: LSF's history has them
                hist = array_states(" ".join(st for st, _ in bhist_elements(
                    remote(["ssh", WEXAC["host"], f"bhist -l {job_id} 2>/dev/null"])).values()))
                states = {s: max(states.get(s, 0), hist.get(s, 0)) for s in ("DONE", "EXIT")
                          if states.get(s, 0) or hist.get(s, 0)}
            failed = 0
            if array_done(states, n):
                return states
        except RuntimeError as err:
            failed += 1
            print(f"[gate] poll failed ({failed}/{retries}): {str(err)[:200]}", flush=True)
            if failed > retries:
                raise
        time.sleep(wait_s)


LIVE_STATES = ("PEND", "RUN", "PSUSP", "USUSP", "SSUSP")


def running_job(name, *, remote=None):
    """The id of a live array named `name` (a wave already submitted from this head), else None."""
    remote = remote or _remote
    try:
        text = remote(["ssh", WEXAC["host"], f"bjobs -noheader -o 'jobid stat' -J {name} 2>/dev/null || true"])
    except RuntimeError:
        return None
    ids = {ln.split()[0] for ln in text.splitlines() if len(ln.split()) >= 2 and ln.split()[1] in LIVE_STATES}
    return min(ids) if ids else None


def lsf_cpu(job_id, *, remote=None) -> dict:
    """{index: CPU seconds} from `bjobs -o "jobindex cpu_used"` (the cross-check of the elements' own counts), or from
    `bhist -l` once bjobs has purged the array."""
    remote = remote or _remote
    out = {}
    for ln in remote(["ssh", WEXAC["host"], f"bjobs -noheader -a -o 'jobindex cpu_used' {job_id}"]).splitlines():
        parts = ln.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                out[int(parts[0])] = float(parts[1])
            except ValueError:
                pass
    if not out:
        out = {i: cpu for i, (_st, cpu) in
               bhist_elements(remote(["ssh", WEXAC["host"], f"bhist -l {job_id} 2>/dev/null"])).items()}
    return out


def read_rows(local_dir, ids, head) -> dict:
    """{cohort: row} of the fetched rows written at `head` (a row from another head is not this run's)."""
    import json
    out = {}
    for c in ids:
        p = os.path.join(local_dir, f"{c}.json")
        if os.path.exists(p):
            r = json.load(open(p))
            if r.get("head") == head:
                out[c] = r
    return out


def ledger_path(phase):
    return os.path.join(RESULTS_GATE, f"{phase}_ledger.json")


def ledger_spent_h(phase, where=None) -> float:
    """The cell's CPU so far, every wave of every attempt and head (FR-020: superseded and failed runs count); `where`
    keeps one host's entries ("wexac" or "mac": a Mac step's spend is entered with where = "mac", a wave has none)."""
    import json
    p = ledger_path(phase)
    if not os.path.exists(p):
        return 0.0
    return sum(w["element_cpu_s"] for w in json.load(open(p))["waves"]
               if where is None or w.get("where", "wexac") == where) / 3600.0


def unrecorded_jobs(phase, rows: dict) -> dict:
    """{job id: [cohort ids]} of returned rows whose LSF job the ledger has not counted: a poller that died after the
    job ran (FR-020: every wave's CPU counts)."""
    import json
    p = ledger_path(phase)
    known = {str(w["job_id"]) for w in json.load(open(p))["waves"]} if os.path.exists(p) else set()
    out = {}
    for c, r in rows.items():
        jid = str(r.get("lsf_jobid") or "")
        if jid and jid not in known:
            out.setdefault(jid, []).append(c)
    return out


def wave_record(k, job_id, head, todo, got, states, lsf) -> dict:
    spent = sum(float(r.get("element_cpu_s") or 0.0) for r in got.values())
    return {"wave": k, "job_id": job_id, "head": head, "n": len(todo), "states": states,
            "element_cpu_s": max(spent, sum(lsf.values())), "elements_cpu_s": spent, "lsf_cpu_s": sum(lsf.values()),
            "missing": sorted(set(todo) - set(got))}


def ledger_add(phase, wave: dict):
    import json
    p = ledger_path(phase)
    d = json.load(open(p)) if os.path.exists(p) else {"phase": phase, "waves": []}
    d["waves"].append(wave)
    os.makedirs(RESULTS_GATE, exist_ok=True)
    json.dump(d, open(p, "w"), indent=1, sort_keys=True)


def phase_cell(args):
    """pilot, a1, a2, realfull (FR-019, FR-020): the cell's cohorts fitted with W on WEXAC, in waves of at most 1/4 of
    its price, the cap checked before each wave with every wave's CPU counted (the ledger); rows fetched into
    tests/acceptance/results/gate_006/<phase>/. A wave whose rows are there at this head is skipped; at the cap the
    cell stops and asks Naveh. Then the cell is scored on the Mac (score_cell)."""
    import json
    import subprocess
    import tempfile
    phase = args.phase
    rules = read_rules()
    if phase in GATE_PHASES and not rules.get("frozen"):
        sys.exit(f"[{phase}] REFUSED: a gate cell fits only after the freeze (tasks T058), --dev or not")
    if phase == "v033-leverage" and args.dev and not getattr(args, "resume", None):     # --resume collects, never
        return phase_v033_pilot(args)                                                    # the pilot (2026-11-14)
    if phase == "v033-leverage" and not rules.get("frozen"):
        sys.exit(f"[{phase}] REFUSED: R-9's null panel fits only after the freeze (tasks T058); --dev runs the "
                 f"one-fit ITP pilot")
    head = A3.git("rev-parse", "HEAD")
    resume = bool(getattr(args, "resume", None))
    if resume:                                       # collect a cell submitted from --job-head; never submit
        head = A3.git("rev-parse", args.job_head)
        if head != A3.git("rev-parse", "HEAD") and not args.dev:
            sys.exit(f"[{phase}] REFUSED: --job-head {args.job_head} is not HEAD, and rows of another commit never "
                     f"count (pass --dev to collect and score them as a development result)")
    short = head[:7]
    mods, pins, lib = connect_v6()
    ids = cell_ids(phase, lib, rules)
    prices = j_prices(mods, pins, rules, ids, run=V033_RUN if phase == "v033-leverage" else None)
    lead = dg_ids(phase, lib, rules, ids)
    factor = {c: factor_for(phase, rules, lead=c in lead) for c in ids}
    cap_h = cap_for(phase)
    cell_price_s = sum(prices[c] * factor[c] for c in ids)
    waves = pack_waves(prices, factor, cell_price_s / 4.0)
    paths = cell_paths(short, phase)
    local = os.path.join(RESULTS_GATE, phase)
    if args.dry_run:
        print(f"[{phase}] {len(ids)} cohorts ({len(lead)} with the lead pair), J {sum(prices.values()) / 3600:.1f} "
              f"CPU-h x {sorted(set(factor.values()))} = "
              f"{cell_price_s / 3600:.1f} expected, cap {cap_h}; {len(waves)} waves of "
              f"{[len(w) for w in waves]}")
        for k, w in enumerate(waves, 1):
            print(f"[{phase}] wave {k}: {' '.join(submit_command(paths, short, phase, k, len(w)))}")
        sys.exit(0)
    run = start(phase, dev=args.dev)
    budget = Budget(phase, cap_h)
    os.makedirs(local, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"sraging006_{phase}_") as tmp:
        exp, v6 = os.path.join(tmp, "repo"), os.path.join(tmp, "v6")
        os.makedirs(exp)
        os.makedirs(os.path.join(v6, "cohorts"))
        subprocess.run(["git", "-C", REPO, "archive", "--format=tar", "-o", os.path.join(tmp, "head.tar"), head],
                       check=True)
        subprocess.run(["tar", "-x", "-C", exp, "-f", os.path.join(tmp, "head.tar")], check=True)
        import shutil
        for c in ids:
            shutil.copy2(os.path.join(lib.root, "cohorts", f"{c}.npz"), os.path.join(v6, "cohorts", f"{c}.npz"))
        rundir = os.path.join(tmp, "run")
        os.makedirs(rundir)
        nodes = h_nodes(mods, pins, rules, ids) if phase in ("pilot",) + GATE_PHASES else {}
        json.dump(cell_panel(phase, ids, lib, head, lead=lead, nodes=nodes),
                  open(os.path.join(rundir, "panel.json"), "w"), indent=1)
        open(os.path.join(rundir, "job.sh"), "w").write(job_script(paths))
        os.chmod(os.path.join(rundir, "job.sh"), 0o755)
        h = WEXAC["host"]
        _remote(["ssh", h, f"mkdir -p ~/{paths['repo']} ~/{paths['run']} ~/{paths['out']} ~/{paths['logs']} "
                              f"~/{paths['v6']}/cohorts"])
        _remote(["rsync", "-az", "--delete", f"{exp}/", f"{h}:{paths['repo']}/"])
        _remote(["rsync", "-az", f"{v6}/", f"{h}:{paths['v6']}/"])
        _remote(["rsync", "-az", f"{rundir}/", f"{h}:{paths['run']}/"])
        remote = {}                                  # WEXAC's copy against the head, by explicit sha256 (005's rule)
        for ln in _remote(["ssh", h, f"cd ~/{paths['repo']} && find . -type f -not -path './.cache/*' -not -path "
                                        f"'*/__pycache__/*' -print0 | xargs -0 sha256sum"]).splitlines():
            if ln.strip():
                sha, p = ln.split(None, 1)
                remote[p.strip().removeprefix("./")] = sha
        probs = A5.tree_problems(A5.tree_sha256(exp), remote)
        if probs:
            sys.exit(f"[{phase}] REFUSED: WEXAC's copy differs from {short}: {probs[:5]}")
        freeze = _remote(["ssh", h, f"{WEXAC['python']} -c {A5._q(A5.FREEZE_PY)}"])
        py = _remote(["ssh", h, f"{WEXAC['python']} -c {A5._q('import platform; print(platform.python_version())')}"]
                        ).strip()
        rp = A5.runtime_problems(freeze, py, A5.runtime_pins())
        if rp:
            finish(run, {"status": "not_applicable", "reasons": [f"runtime mismatch with uv.lock: {p}" for p in rp],
                         "head": head}, budget)
            sys.exit(3)
        budget.remote_cpu_s = ledger_spent_h(phase) * 3600.0
        jobs = []
        for k, wave in enumerate(waves, 1):
            _remote(["rsync", "-az", f"{h}:{paths['out']}/", f"{local}/"])
            done = read_rows(local, wave, head)
            for jid, cs in unrecorded_jobs(phase, done).items():     # its poller died after the job ran
                rec = wave_record(k, jid, head, cs, {c: done[c] for c in cs}, poll_array(jid, len(cs)), lsf_cpu(jid))
                print(f"[{phase}] wave {k}: job {jid} ran without being counted; counted now "
                      f"({rec['element_cpu_s'] / 3600:.2f} CPU-h)", flush=True)
                ledger_add(phase, rec)
                jobs.append(rec)
                budget.remote_cpu_s = ledger_spent_h(phase) * 3600.0
            todo = [c for c in wave if c not in done]
            if not todo:
                continue
            exp_h = sum(prices[c] * factor[c] for c in todo) / 3600.0
            if not wave_allowed(budget.spent_cpu_h, exp_h, cap_h):
                budget.capped = True
                print(f"[{phase}] STOP: spent {budget.spent_cpu_h:.2f} + wave {k}'s expected {exp_h:.2f} CPU-h > cap "
                      f"{cap_h}: ask Naveh (FR-020)", flush=True)
                break
            if not spec_allows(exp_h, rules):
                budget.capped = True
                print(f"[{phase}] STOP: the spec's WEXAC spend {spec_spent_wexac_h():.2f} + wave {k}'s expected "
                      f"{exp_h:.2f} CPU-h > its bound {rules['budget']['wexac_cpu_h']}: ask Naveh (FR-020)", flush=True)
                break
            job_id = running_job(f"g006_{phase}_{short}_w{k}")
            if job_id:                                   # a re-run after the poller died: collect, never resubmit
                print(f"[{phase}] wave {k}: job {job_id} is still running; polling it", flush=True)
            elif resume:
                print(f"[{phase}] wave {k}: {len(todo)} cohort(s) have no row at {short} and no live job; --resume "
                      f"never submits: {todo}", flush=True)
                continue
            else:
                wf = os.path.join(rundir, f"wave_{k}.txt")
                open(wf, "w").write("\n".join(todo) + "\n")
                _remote(["rsync", "-az", wf, f"{h}:{paths['run']}/"])
                sub = _remote(submit_command(paths, short, phase, k, len(todo)))
                import re
                m = re.search(r"Job <(\d+)>", sub)
                if not m:
                    raise RuntimeError(f"bsub gave no job id: {sub!r}")
                job_id = m.group(1)
                print(f"[{phase}] wave {k}: {len(todo)} cohorts, job {job_id}, expected {exp_h:.2f} CPU-h", flush=True)
            states = poll_array(job_id, len(todo))
            _remote(["rsync", "-az", f"{h}:{paths['out']}/", f"{local}/"])
            got = read_rows(local, todo, head)
            wave_rec = wave_record(k, job_id, head, todo, got, states, lsf_cpu(job_id))
            ledger_add(phase, wave_rec)
            jobs.append(wave_rec)
            budget.remote_cpu_s = ledger_spent_h(phase) * 3600.0
            if budget.reached():
                budget.capped = True
                print(f"[{phase}] STOP: the cell reached its cap ({budget.spent_cpu_h:.2f} of {cap_h} CPU-h): ask "
                      f"Naveh (FR-020)", flush=True)
                break
    rows = read_rows(local, ids, head)
    if resume and str(args.resume) not in {str(w["job_id"]) for w in json.load(open(ledger_path(phase)))["waves"]}:
        print(f"[{phase}] WARNING: --resume {args.resume} is not a job of this cell's ledger", flush=True)
    scored = (score_v033(rows, ids, prices, r9_control(budget), rules["rules"]["R-9"]) if phase == "v033-leverage"
              else score_cell(phase, rows, ids, mods, pins, lib, rules, prices))
    finish(run, {**scored, "head": head, "n_cohorts": len(ids), "n_rows": len(rows), "waves": jobs,
                 "factor": factor, "lead_pair_ids": lead, "price_cpu_h": cell_price_s / 3600.0,
                 "ledger": os.path.relpath(
                     ledger_path(phase), REPO)}, budget)


def itp_control(budget=None) -> dict:
    """R-9's positive control on this Mac: profile_v033 on the ITP example (delta 1, days, as 003f read it), whose
    age-0 death must be flagged."""
    import time as _t
    from sraging import Cohort, fit_cohort
    c0 = _t.process_time()
    r = fit_cohort(Cohort.from_csv(os.path.join(REPO, ITP_AGE0), delta=1.0, time_unit="days"), calibrator="profile_v033")
    lev = r.diagnostics.get("leverage") or {}
    d = lev.get("dropped") or {}
    return {"label": lev.get("label"), "dropped": d, "flagged_age0": lev.get("label") == "flag" and d.get("age") == 0.0,
            "fit_cpu_s": (r.timing or {}).get("cpu_s"), "leverage_cpu_s": (lev.get("cost") or {}).get("cpu_s"),
            "cpu_s": _t.process_time() - c0, "warnings": [w.code for w in r.warnings]}


#: R-9's positive-control candidates under option C (Naveh, 2026-11-11; sr-naveh_cuts_006.md Q4), in this order. Each
#: counts only if J's own check flags its planted death: measured in 003b's AC11 rows (ac11_003b.json) for the
#: planted two, by one J fit first for the ITP one. The late plant is 003b's late_rule (1.5 x the oldest age).
R9_CANDIDATES = ("itp-late", "planted-k0.1-p03-n300-late", "planted-k1-p03-n300-late")
R9_ASSESSED = ("flag", "no_flag", "flat_valley_slide", "failed")       # not_assessed: theta-hat not stationary
AC11_RESULT = "tests/acceptance/results/ac11_003b.json"


def r9_candidate(cid):
    """(Cohort, the planted death's 1-based row) of one R-9 candidate: the ITP example without its age-0 death plus one
    late death, or a registered AC11 planted cohort (acceptance_003b.planted_cohort, byte for byte as 003b fitted it)."""
    import numpy as np
    from sraging import Cohort
    if cid == "itp-late":
        c = Cohort.from_csv(os.path.join(REPO, ITP_AGE0), delta=1.0, time_unit="days")
        keep = np.asarray(c.ages, float) > 0.0
        ages, events = np.asarray(c.ages, float)[keep], np.asarray(c.events, int)[keep]
        ages, events = np.append(ages, round(float(ages.max()) * 1.5)), np.append(events, 1)
        return Cohort.from_arrays(ages, events, delta=1.0, time_unit="days", name=cid), ages.size
    import acceptance_003b as A3B
    return A3B.planted_cohort(A3B.ac11_registration(), cid)


def leverage_of(cohort, calibrator, planted_row) -> dict:
    """One fit's leverage label, and whether it flags with the planted death dropped (rows 1-based on both sides, as
    acceptance_003b.ac11_worker reads them)."""
    import time as _t
    from sraging import fit_cohort
    c0 = _t.process_time()
    r = fit_cohort(cohort, calibrator=calibrator)
    lev = r.diagnostics.get("leverage") or {}
    d = lev.get("dropped") or {}
    return {"calibrator": calibrator, "label": lev.get("label"), "reason": lev.get("reason"), "dropped": d,
            "flagged_planted": lev.get("label") == "flag" and d.get("row") == planted_row,
            "warnings": [w.code for w in r.warnings], "cpu_s": _t.process_time() - c0}


def j_flag_measured(cid) -> bool | None:
    """J's AC11 reading of a planted id (ac11_003b.json): True when it flagged with the planted death dropped."""
    import json
    for x in json.load(open(os.path.join(REPO, AC11_RESULT)))["rows"]:
        if x["cohort_id"] == cid:
            return x.get("label") == "flag" and bool(x.get("planted_dropped"))
    return None


#: R-9's control on the Mac, priced for the cap check: the ITP fit and up to three candidates [T054, T057e]
R9_CONTROL_PRICE_H = 2.7


def r9_not_run(reason) -> dict:
    """R-9's control when the cap stops it: not run, so R-9's positive half fails."""
    return {"not_run": reason, "itp_age0": None, "itp_refuses": False, "candidates": [], "control": None,
            "option": None, "control_flagged": False}


def r9_control(budget=None) -> dict:
    """R-9's positive control under option C, on this Mac: ITP's age-0 control must be refused (not_assessed, with
    LEVERAGE); then the candidates in order, each valid only where J flags its planted death; the first that v033's
    check assesses is the control, and it must flag the planted death. None assessed: option B. FR-020 holds on the
    Mac too: the control starts only if the cell's spend plus its price stays within the cap, and no fit starts once
    the cap is reached (2026-11-11: it ran 3.04 Mac CPU-h past the cell's 4.93 on WEXAC, unchecked, to the 8 cap)."""
    if budget is not None and not wave_allowed(budget.spent_cpu_h, R9_CONTROL_PRICE_H, budget.cap_cpu_h):
        budget.capped = True
        return r9_not_run(f"cell capped: spent {budget.spent_cpu_h:.2f} + the control's {R9_CONTROL_PRICE_H} CPU-h > "
                          f"cap {budget.cap_cpu_h}")
    itp = itp_control(budget)
    tried, control = [], None
    for cid in R9_CANDIDATES:
        if budget is not None and budget.reached():
            budget.capped = True
            return {**r9_not_run(f"cell capped after {len(tried)} candidate(s): spent {budget.spent_cpu_h:.2f} of "
                                 f"{budget.cap_cpu_h}"), "itp_age0": itp, "candidates": tried}
        coh, row = r9_candidate(cid)
        rec = {"cid": cid, "planted_row": row}
        if cid == "itp-late":
            rec["J"] = leverage_of(coh, "profile_jax2", row)
            rec["j_valid"] = rec["J"]["flagged_planted"]
        else:
            rec["j_valid"] = bool(j_flag_measured(cid))
        if rec["j_valid"]:
            rec["v033"] = leverage_of(coh, "profile_v033", row)
        tried.append(rec)
        print(f"[v033-leverage] R-9 candidate {cid}: J valid {rec['j_valid']}; v033 "
              f"{(rec.get('v033') or {}).get('label')}", flush=True)
        if rec["j_valid"] and rec["v033"]["label"] in R9_ASSESSED:
            control = rec
            break
    return {"itp_age0": itp, "itp_refuses": itp["label"] == "not_assessed" and "LEVERAGE" in itp["warnings"],
            "candidates": tried, "control": None if control is None else control["cid"],
            "option": "C" if control else "B",
            "control_flagged": bool(control and control["v033"]["flagged_planted"])}


def phase_v033_pilot(args):
    """R-9's one-fit pilot (--dev; tasks T054): the ITP positive control, and P3's price from v033's stored CPU with
    the check's measured share."""
    run = start("v033-leverage", dev=True)
    budget = Budget("v033-leverage", cap_for("v033-leverage"))
    ctl = itp_control()
    rules = read_rules()
    mods, pins, lib = connect_v6()
    ids = v033_ids(rules)
    prices = j_prices(mods, pins, rules, ids, run=V033_RUN)
    share = (ctl["leverage_cpu_s"] or 0.0) / ctl["fit_cpu_s"] if ctl.get("fit_cpu_s") else None
    factor = 1.0 + (share or 0.0)
    p3 = tomllib.load(open(os.path.join(REPO, V033_PANEL), "rb"))["p3"]
    human = set(p3["human_no_basin_fix"]) | set(p3["human_basin_fix"])          # P3's 13 human cohorts
    price = {"factor_measured": factor, "share_of_fit": share, "p3_cpu_h": sum(prices.values()) * factor / 3600.0,
             "p3_non_human_cpu_h": sum(v for c, v in prices.items() if c not in human) * factor / 3600.0,
             "n": len(ids), "n_human": len(human), "cap_cpu_h": cap_for("v033-leverage"),
             "stored_v033_cpu_s": prices}
    print(f"[v033-leverage] ITP age-0 flagged: {ctl['flagged_age0']} ({ctl['label']}); P3 {len(ids)} cohorts at "
          f"x{factor:.2f}: {price['p3_cpu_h']:.1f} CPU-h (non-human {price['p3_non_human_cpu_h']:.1f}) against the "
          f"{price['cap_cpu_h']} cap", flush=True)
    finish(run, {"status": "pass" if ctl["flagged_age0"] else "fail", "positive_control": ctl, "price": price}, budget)


def score_v033(rows, ids, prices, control, rule) -> dict:
    """R-9 (Naveh, 2026-11-11, option C then B): ITP's age-0 control refused; the first candidate v033 assesses flagged
    (no candidate assessed: option B, R-9 fails and the deviation is retargeted); on the null panel at most
    rule["null_max_flags"] flags among the assessed fits, with at least rule["null_min_assessed"] assessed (else
    inconclusive, to Naveh)."""
    ok = {c: r for c, r in rows.items() if r.get("status") == "ok"}
    flags = sorted(c for c, r in ok.items() if r.get("leverage_flag") == "flag")
    assessed = sum(1 for r in ok.values() if r.get("leverage_flag") in R9_ASSESSED)
    labels = {lab: sum(1 for r in ok.values() if r.get("leverage_flag") == lab)
              for lab in sorted({str(r.get("leverage_flag")) for r in ok.values()})}
    complete = len(ok) == len(ids)
    null = {"flags": flags, "assessed": assessed, "labels": labels, "n": len(ok),
            "max_flags": rule["null_max_flags"], "min_assessed": rule["null_min_assessed"],
            "conclusive": assessed >= rule["null_min_assessed"], "passes": len(flags) <= rule["null_max_flags"]}
    positive = control["itp_refuses"] and control["option"] == "C" and control["control_flagged"]
    status = ("fail" if not (complete and positive and null["passes"]) else
              "pass" if null["conclusive"] else "inconclusive")
    return {"status": status, "R-9": {"positive_control": control, "null": null, "option": control["option"]},
            "errors": {c: r.get("error") for c, r in rows.items() if r.get("status") != "ok"},
            "missing": sorted(set(ids) - set(rows))}


def w_sets(mods, rows) -> list:
    """W's rows as tester ClaimSets (the four parameter claims), for the tester's own scorer."""
    from sraging.adapters import srbench as ADP
    T = mods["contract.types"]
    return [ADP.claimset_from_kwargs(T, c, "sraging_profile_jax2_w", r["claims"], {}, None, r.get("fit_cpu_s"))
            if r.get("status") == "ok" else T.ClaimSet(cohort_id=c, method_name="sraging_profile_jax2_w",
                                                         status=f"error:{r.get('error', '')[:80]}")
            for c, r in rows.items()]


def score_cell(phase, rows, ids, mods, pins, lib, rules, prices) -> dict:
    """The cell under the table of 2026-11-11 (spec.md Clarifications; gate_006):
    - a1 gates: R-1 (0.95 against 0.88) on A1's closed claims, W's data-determined one-sided claims under the collapse
      rule, R-2 against H, R-3 and R-4;
    - a2 gates: R-1 under the collapse rule and the truncated-above rate (either trips W's warning at n >= 25,000),
      the one-sided claims, R-3 and R-4; R-2 is reported;
    - realfull and the pilot are reported;
    - in every cell a failure of R-3 or R-4 stops W (to Naveh), FR-022 above its rate needs a new acceptance, and the
      lead pair (Dg), where it ran, is judged under the collapse rule. Closed and one-sided coverage, closure and cap
      rates, truncation, and misses by side and per cluster are reported. The pilot also reports r and the lead pair's
      cost."""
    import statistics
    import gate_006 as GATE
    R = rules["rules"]
    ok = {c: r for c, r in rows.items() if r.get("status") == "ok"}
    rws = list(ok.values())
    truths, clusters = lib.truths(list(ok)), lib.cluster_map(list(ok))
    sets = w_sets(mods, ok)
    counts = GATE.coverage_counts(mods["score.claims"], sets, truths)
    hsets = pinned_sets(mods, pins, rules["pins"]["H"], list(ok))
    nsets = pinned_sets(mods, pins, rules["pins"]["null"], list(ok))
    hb = {q: GATE.closed_bounds(hsets, q) for q in GATE.GATED}
    r1_rule = GATE.collapse(R["R-1"]) if phase == "a2" else R["R-1"]
    out = {"R-1": {**GATE.r1_table(counts, r1_rule), "rule": "collapse" if phase == "a2" else "0.95 against 0.88"},
           "one_sided": GATE.one_sided_gate(rws, truths, R["R-1"]),
           "R-2": {q: {"vs_H": GATE.r2_vs_h(GATE.w_closed_bounds(rws, q, hb[q]), hb[q], R["R-2"]["max_dex"]),
                       "vs_null": GATE.paired_log10_ratio(GATE.closed_bounds(sets, q), GATE.closed_bounds(nsets, q))}
                   for q in GATE.GATED},
           "R-3": GATE.r3(rws, R["R-3"]["max_gap_nats"]), "R-4": GATE.r4(rws, R["R-4"]["tolerance_nats"]),
           "FR-022": GATE.fr022(rws, R["FR-022"]["max_rate"]),
           "shapes": GATE.shape_report(rws, truths), "truncation": GATE.truncation(rws, hb),
           "misses": GATE.misses(rws, truths, clusters),
           "errors": {c: r.get("error") for c, r in rows.items() if r.get("status") != "ok"},
           "missing": sorted(set(ids) - set(rows))}
    if any((r.get("combinations") or {}).get("on") for r in rws):
        out["Dg"] = GATE.lead_pair_report(rws, truths, R["R-1"])
    el = sum(float(r.get("element_cpu_s") or 0) for r in rws)
    out["r"] = {"element_over_j": el / sum(prices[c] for c in ok) if ok else None,
                "fit_over_j": (sum(float(r.get("fit_cpu_s") or 0) for r in rws) / sum(prices[c] for c in ok)
                               if ok else None)}
    complete = len(ok) == len(ids)
    trunc_ok = all((v["rate"] or 0.0) <= R["R-1"]["a2_truncated_max"] for v in out["truncation"].values())
    gates = {"a1": {"R-1": out["R-1"]["passes"], "one_sided": out["one_sided"]["passes"],
                    "R-2": all(v["vs_H"]["passes"] for v in out["R-2"].values())},
             "a2": {"R-1 (collapse)": out["R-1"]["passes"], "truncated_above": trunc_ok,
                    "one_sided": out["one_sided"]["passes"]}}.get(phase, {})
    gates.update({"R-3": out["R-3"]["passes"], "R-4": out["R-4"]["passes"]})
    out["gates"] = gates
    out["consequence"] = {
        "stops_W": not (out["R-3"]["passes"] and out["R-4"]["passes"]),
        "W_COVERAGE_warning": phase == "a1" and not all(v for k, v in gates.items() if k not in ("R-3", "R-4")),
        "warning_n_ge_25000": phase == "a2" and not (out["R-1"]["passes"] and trunc_ok and out["one_sided"]["passes"]),
        "needs_new_acceptance_FR022": not out["FR-022"]["passes"],
        "lead_pair_warning": "Dg" in out and not out["Dg"]["passes"]}
    if phase == "pilot":
        lead = [r["combinations"].get("ratio_to_fit") for r in rws if (r.get("combinations") or {}).get("on")]
        lead = [x for x in lead if x is not None]
        out["Dg_price"] = {"lead_pair_over_fit": lead, "median": statistics.median(lead) if lead else None,
                           "max": max(lead) if lead else None, "threshold": 2.6}
        out["status"] = "pass" if complete else "fail"
    elif phase in ("a1", "a2"):
        out["status"] = "pass" if complete and all(gates.values()) else "fail"
    else:                                                     # realfull: reported; R-3 and R-4 still stop W
        out["status"] = "pass" if complete and not out["consequence"]["stops_W"] else "fail"
    return out


# ------------------------------------------------------------------------------------------- unchanged (FR-030)
CLAIMS_RESULT = "tests/acceptance/results/claims-001_003.json"


def unchanged_groups() -> dict:
    """The suites `unchanged` runs, each by pytest in its own process: the unit and parity suites, every earlier
    spec's results test (005's bounded at d604a97), and 006's closure-edit and records checks."""
    import glob
    results = sorted(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, "tests", "acceptance",
                                                                               "test_results*.py"))
                     if not p.endswith("test_results_006.py"))
    return {"unit": ["tests/unit"], "parity": ["tests/parity"], "results": results,
            "closure": ["tests/acceptance/test_results_006.py::test_closure_edits_are_listed",
                        "tests/acceptance/test_results_006.py::test_the_listed_engine_files_are_the_engine_files_edited"]}


def claims_reading(path=None) -> dict:
    """FR-030 in place of 005's "no closure change": spec 001's claims re-run (spec 003's `unchanged`, at the close) is
    40/40 on claims identity, theta-hat, value bytes and gradient bytes, ran on a clean committed tree, and no file of
    the claims closure changed since its head; and spec 003's untouched check finds no violation (the engine files
    006 edits are spec-exempt, SPEC_ENGINE_CHANGES["006"])."""
    import json
    p = os.path.join(REPO, path or CLAIMS_RESULT)
    d = json.load(open(p))
    head = d["repo_head"]
    later = A3.git("diff", "--name-only", head, "HEAD", "--", *A3.import_closure()).split()
    doc_only, exempt = [], []
    violations = A3.untouched_problems(doc_only, exempt, A3.merge_base())
    ok_checks = A3.claims_exempt_ok(d.get("checks"))
    return {"claims_head": head, "claims_status": d.get("status"), "claims_acceptance": d.get("status_for_acceptance"),
            "checks_40_of_40": ok_checks, "closure_changed_since": later, "violations": violations,
            "spec_exempt": exempt, "docstring_only": doc_only,
            "ok": (ok_checks and d.get("status") == "pass" and d.get("status_for_acceptance") == "acceptance"
                   and not d.get("tree_dirty") and not later and not violations)}


def phase_unchanged(args):
    """FR-030 (T077), in the MAIN checkout: the unit suite, the parity suite, every earlier results test and 006's
    closure checks, each by pytest in its own process (003e's rule: 0 failures and 0 errors, every skip on its lists),
    and spec 001's claims re-run 40/40 with no closure change since its head (claims_reading)."""
    import tempfile
    import acceptance_003e as E
    import acceptance_004 as A4
    run = start("unchanged", dev=args.dev)
    budget = Budget("unchanged", cap_for("unchanged"))
    reading = claims_reading()
    groups_def = unchanged_groups()
    with tempfile.TemporaryDirectory(prefix="sraging006_unchanged_") as tmp:
        rows = run_capped(budget, list(groups_def),
                          lambda g: {"group": g, **A4.unchanged_group(g, groups_def[g], tmp)})
    groups = {r.pop("group"): r for r in rows}
    groups_ok, unlisted = E.unchanged_verdict(groups) if len(groups) == len(groups_def) else (False, [])
    for g in groups.values():
        g.pop("cases", None)
    status = "pass" if groups_ok and reading["ok"] else "fail"
    print(f"[unchanged] groups ok {groups_ok}; claims re-run {reading['checks_40_of_40']} at "
          f"{reading['claims_head'][:7]}, closure changed since {reading['closure_changed_since']}", flush=True)
    finish(run, {"status": status, "groups": groups, "groups_ok": groups_ok, "skips_unlisted": unlisted,
                 "claims": reading}, budget)


def _todo(name, task):
    def phase(args):
        sys.exit(f"{name}: not implemented yet (specs/006-calibrator-capabilities/tasks.md {task})")
    phase.__name__ = f"phase_{name.replace('-', '_')}"
    return phase


PHASE_FUNCS = {"parity": phase_parity, "identity": phase_identity,
               "stored": phase_stored, "pilot": phase_cell, "a1": phase_cell, "a2": phase_cell,
               "realfull": phase_cell,
               "v033-leverage": phase_cell, "unchanged": phase_unchanged}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--dev", action="store_true", help="development run (a pilot): never counts")
    ap.add_argument("--dry-run", action="store_true", help="the WEXAC phases: print the exact commands; contact nothing")
    ap.add_argument("--resume", metavar="JOBID", help="a WEXAC phase: collect this already-submitted LSF job and the "
                                                      "cell's other rows at --job-head, counting any uncounted job; "
                                                      "never submits")
    ap.add_argument("--job-head", metavar="SHA", help="--resume: the commit the job was submitted from (not HEAD: "
                                                      "--dev only)")
    args = ap.parse_args(argv)
    if (args.dry_run or args.resume) and WHERE[args.phase] != "wexac":
        ap.error("--dry-run and --resume are the WEXAC phases'")
    if bool(args.resume) != bool(args.job_head):
        ap.error("--resume and --job-head go together")
    PHASE_FUNCS[args.phase](args)


if __name__ == "__main__":
    main()
