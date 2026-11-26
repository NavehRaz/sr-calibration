"""The leverage check on the jax2 calibrators (spec 003f; constitution X; algorithms §13).

After a jax2 fit, on the fit's own likelihood and grid (the evaluation `floors.report_jax2` makes):
- gap/sd (§13 eq. lev-gap) and the worst individual's absolute share of the log-likelihood (eq. lev-share);
- each individual's influence: the one-step shift Δz_i = -I⁻¹ g_i of z = log θ when i is dropped (I the information
  on the coordinates not at a box bound, g_i = ∂ℓ_i/∂z), carried to the flag quantities.
The most influential individual is dropped and the cohort without it is refitted twice: the calibrator's own MLE stage
(`estimator.fit(stages=("mle",))` called exactly as the port's `run_fit` calls it) and the estimator's L-BFGS-B
started at the reported θ̂ on the same reduced objective (the warm start; nothing under `compat/` changes). Spec 003b:
every step runs on the fit's own port (`profile_jax2.JaxPort`; 1.0.0 when none is given), and the block records g0,
the warm start's gain at θ̂ on the full data (a measurement no label reads yet).
The better of the two, by the value march, is compared with the reported θ̂: R_j over the flag quantities and Δ_j in
total nats on the reduced objective (§13 eqs. lev-ratio, lev-itp). The rule in force (`sraging.leverage_rule`) labels
the result; a refit that ends below θ̂ on its own objective is `failed`, never a measurement. The check never changes
the fit: θ̂, claims and the likelihood stay as they were; its result goes to `diagnostics["leverage"]`, outside the
reference comparison."""
from __future__ import annotations

import math
import platform
import subprocess
import time

import numpy as np

VERSION = "003f/2"
PARAMS = ("eta", "beta", "epsilon", "kappa")
QUANTITIES = ("eta_over_eps", "rbar_over_eta", "eta", "epsilon")
QUANTITY_SETS = {"all4": QUANTITIES, "lead_pair": ("eta_over_eps", "rbar_over_eta"), "raw": ("eta", "epsilon")}
GAP_CUT, N_GATE = 2.0, 20          # §13 eq. lev-gap
REL_CUT = 1e-6                     # information eigenvalues at or below REL_CUT × the largest are cut: the FD Hessian
                                   # agrees with forward-over-forward to 9e-8 relative (P1), so smaller ones are noise
BOUND_TOL = 1e-6                   # a coordinate this close (in log) to its box bound is fixed
FD_STEP = 1e-4                     # the Hessian: central differences of the port's analytic gradient, in z
WARM_GAIN_TOL = 1e-3               # nats: the warm start replaces the MLE stage only when better by more than the
                                   # estimator's own optimiser noise (its _POLISH_GAIN_TOL)


# ---------------------------------------------------------------------------------------------- free statistics
def gap_sd(ages, events, gap_cut=GAP_CUT, n_gate=N_GATE) -> dict:
    """§13 eq. lev-gap on the death times (events = 1, ddof 1): flag when g > gap_cut and at least n_gate deaths."""
    d = np.sort(np.asarray(ages, float)[np.asarray(events, int) == 1])
    n = int(d.size)
    if n < n_gate:
        return {"g": None, "n_deaths": n, "applicable": False, "flagged": False}
    sd = float(np.std(d, ddof=1))
    g = float((d[-1] - d[-2]) / sd) if sd > 0 else math.inf
    return {"g": g, "n_deaths": n, "applicable": True, "flagged": bool(g > gap_cut)}


def share(terms, ages, events) -> dict:
    """§13 eq. lev-share, the absolute form, for the worst-fitting individual (argmin ℓ_i). `row` is the 1-based data
    row, as in the floor report."""
    terms = np.asarray(terms, float)
    j = int(np.argmin(terms))
    return {"row": j + 1, "age": float(np.asarray(ages, float)[j]), "event": int(np.asarray(events, int)[j]),
            "term": float(terms[j]), "share": float(abs(terms[j]) / np.sum(np.abs(terms)))}


# --------------------------------------------------------------------------------------------- the quantities
def quantities(z) -> dict:
    """The flag quantities in natural log: raw η and ε, η/ε, r̄/η with r̄ = β(1 − κ ln(1 + 1/κ))."""
    eta, beta, eps, kap = (math.exp(float(v)) for v in z[:4])
    rbar = beta * (1.0 - kap * math.log1p(1.0 / kap))
    return {"eta": math.log(eta), "epsilon": math.log(eps), "eta_over_eps": math.log(eta / eps),
            "rbar_over_eta": math.log(rbar / eta)}


def quantity_gradients(z, step=1e-6) -> dict:
    """∂q/∂z by central differences (closed forms); zeros on any component past the four SR parameters (h)."""
    z = np.asarray(z, float)
    out = {q: np.zeros(z.size) for q in QUANTITIES}
    for k in range(4):
        zp, zm = z.copy(), z.copy()
        zp[k] += step
        zm[k] -= step
        qp, qm = quantities(zp), quantities(zm)
        for q in QUANTITIES:
            out[q][k] = (qp[q] - qm[q]) / (2 * step)
    return out


# --------------------------------------------------------------------------------------------------- influence
def influence(G, H, z, lower, upper, rel_cut=REL_CUT) -> dict:
    """The one-step shifts Δz_i = -I⁻¹ g_i on the free coordinates (I = -H restricted to them; directions with an
    eigenvalue at or below rel_cut × the largest -- including non-positive ones -- are cut and counted)."""
    G, H, z = np.asarray(G, float), np.asarray(H, float), np.asarray(z, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    free = [not (abs(z[k] - lower[k]) < BOUND_TOL or abs(z[k] - upper[k]) < BOUND_TOL) for k in range(z.size)]
    fi = [k for k in range(z.size) if free[k]]
    DZ = np.zeros((G.shape[0], z.size))
    rec = {"free": free, "active": [k for k in range(z.size) if not free[k]], "eigenvalues": [],
           "min_eigenvalue": None, "min_relative": None, "n_negative": 0, "condition": None, "n_cut": 0,
           "n_kept": 0, "rel_cut": rel_cut}
    if not fi:
        return dict(rec, DZ=DZ)
    info = -H[np.ix_(fi, fi)]
    info = 0.5 * (info + info.T)
    w, V = np.linalg.eigh(info)
    top = float(np.max(w))
    keep = w > rel_cut * top if top > 0 else np.zeros(w.size, bool)
    if np.any(keep):
        inv = V[:, keep] @ np.diag(1.0 / w[keep]) @ V[:, keep].T
        DZ[:, fi] = -(G[:, fi] @ inv)
    rec.update(eigenvalues=[float(x) for x in w], min_eigenvalue=float(np.min(w)),
               min_relative=float(np.min(w) / top) if top > 0 else None, n_negative=int(np.sum(w < 0)),
               n_cut=int(np.sum(~keep)), n_kept=int(np.sum(keep)),
               condition=float(top / np.min(w[keep])) if np.any(keep) else None)
    return dict(rec, DZ=DZ)


def shifts(DZ, z) -> dict:
    """Each individual's one-step shift of every flag quantity (natural log)."""
    g = quantity_gradients(z)
    return {q: np.asarray(DZ, float) @ g[q] for q in QUANTITIES}


def ranking(sh, set_name) -> dict:
    """The influence score under a quantity set (max |shift| over it), its top and its median individual. A tie for the
    top keeps the lowest row (the stable order) and is counted in `n_tied` (1 = no tie)."""
    score = np.max(np.abs(np.vstack([sh[q] for q in QUANTITY_SETS[set_name]])), axis=0)
    order = np.argsort(-score, kind="stable")
    top = int(order[0])
    return {"score": score, "top": top, "median": int(np.argsort(score, kind="stable")[score.size // 2]),
            "n_tied": int(np.sum(score == score[top]))}


def label(R, delta_total, *, ratio, delta_total_nats, fail_nats) -> str:
    """The rule (Naveh, 2026-10-19): `failed` when R or Δ is not finite or the refit ends more than `fail_nats` below θ̂
    on its own objective; `flag` when R > ratio and Δ > delta_total_nats (total nats); `flat_valley_slide` when R > ratio
    at a smaller cost; `no_flag` otherwise."""
    if R is None or delta_total is None or not (math.isfinite(R) and math.isfinite(delta_total)):
        return "failed"
    if delta_total < -fail_nats:
        return "failed"
    if R > ratio:
        return "flag" if delta_total > delta_total_nats else "flat_valley_slide"
    return "no_flag"


# -------------------------------------------------------------------------------- the fit's own likelihood and grid
def _z(theta_hat, hazard) -> np.ndarray:
    th = theta_hat.as_dict() if hasattr(theta_hat, "as_dict") else dict(theta_hat)
    z = [math.log(th[p]) for p in PARAMS]
    if hazard:
        z.append(math.log(th["h_ext"]))
    return np.asarray(z, float)


def _bounds(box, hazard):
    names = PARAMS + (("h",) if hazard else ())
    return (np.asarray([math.log(box[p][0]) for p in names]), np.asarray([math.log(box[p][1]) for p in names]))


def _port(port=None):
    """The port a step runs on: the fit's own; 1.0.0's when none is given (spec 003f's calls)."""
    from sraging.calibrators import profile_jax2 as PJ
    return PJ.V100 if port is None else port


def _loglik(inp, with_terms=False, port=None):
    fj = _port(port).fp_jax
    kw = inp["kw"]
    return fj.make_loglik(inp["t"], inp["e"], inp["L"], N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"],
                          Gtol=kw["Gtol"], sampling_dt=kw.get("sampling_dt"), hazard=inp["hazard"],
                          window=inp["window"], with_terms=with_terms, exact_from_F=bool(kw.get("exact_from_F")),
                          early=kw.get("early"))


def terms_and_derivatives(cohort, name, z, port=None) -> dict:
    """ℓ_i at z on the fit's own prepared data and grid, G = ∂ℓ_i/∂z (forward mode through the value march) and the
    total's Hessian (central differences of the port's analytic gradient, `vg`)."""
    import jax
    import jax.numpy as jnp
    from sraging.calibrators import profile_jax2 as PJ
    inp = PJ.fit_inputs(cohort, name, _port(port))
    f, vg, tf = _loglik(inp, with_terms=True, port=port)
    zz = jnp.asarray(z)
    d = tf(zz)
    terms = np.asarray(d["term"], float)
    G = np.asarray(jax.jacfwd(lambda x: tf(x)["term"])(zz), float)
    H = np.zeros((z.size, z.size))
    for k in range(z.size):
        zp, zm = np.array(z, float), np.array(z, float)
        zp[k] += FD_STEP
        zm[k] -= FD_STEP
        H[:, k] = (np.asarray(vg(jnp.asarray(zp))[1]) - np.asarray(vg(jnp.asarray(zm))[1])) / (2 * FD_STEP)
    H = 0.5 * (H + H.T)
    grad = np.asarray(vg(jnp.asarray(np.asarray(z, float)))[1], float)
    return {"terms": terms, "cond": float(np.asarray(d["cond"])), "G": G, "H": H, "gradient": grad, "inputs": inp,
            "hessian_source": "finite_difference"}


def hessian_forward(cohort, name, z, port=None) -> np.ndarray:
    """The total's Hessian by forward-over-forward differentiation of the value march (pilot P1's cross-check)."""
    import jax
    import jax.numpy as jnp
    from sraging.calibrators import profile_jax2 as PJ
    _f, _vg, tf = _loglik(PJ.fit_inputs(cohort, name, _port(port)), with_terms=True, port=port)

    def total(x):
        d = tf(x)
        return jnp.sum(d["term"]) + d["cond"]
    return np.asarray(jax.jacfwd(jax.jacfwd(total))(jnp.asarray(z)), float)


# ------------------------------------------------------------------------------------------------------ the refit
def _without(cohort, drop_row):
    from sraging.calibrators import profile_jax2 as PJ
    obs = PJ.observation(cohort)
    if drop_row is None:
        return obs
    keep = [i for i in range(len(obs.ages)) if i != drop_row]
    return PJ.Observation(cohort_id=f"{obs.cohort_id} without row {drop_row}",
                          ages=tuple(obs.ages[i] for i in keep), events=tuple(obs.events[i] for i in keep),
                          delta=obs.delta, window=obs.window)


def refit_mle(cohort, name, drop_row, *, raise_errors=False, port=None) -> dict:
    """The calibrator's own MLE stage on the cohort without `drop_row` (None: on the whole cohort): the two steps of
    the port's `run_fit`, with its exact keywords (1.0.1's own included), plus `stages=("mle",)`."""
    from sraging.calibrators import profile_jax2 as PJ
    P = _port(port)
    jc, est, fj, om = P.claims, P.estimator, P.fp_jax, P.observation_model
    k = PJ.run_kwargs(P.classes[name])
    obs = _without(cohort, drop_row)
    c0 = time.process_time()
    try:
        prep = om.prepare(obs.ages, obs.events, obs.delta, obs.window, window_on=k["window"], relabel_on=k["relabel"])
        res = est.fit(fj, prep.ages, prep.events, delta=obs.delta, levels=(jc.LEVEL,), kappa_max=jc.KAPPA_MAX,
                      hazard=k["hazard"], window=prep.window_eff,
                      **({"shadow": True} if k["shadow"] else {}),
                      **({"exact_from_F": True} if k["exact_from_F"] else {}),
                      **({"self_consistent": True} if k["self_consistent"] else {}),
                      **({"floor_early": True} if k["floor_early"] else {}),
                      **({"n_rule": True} if k["n_rule"] else {}),
                      **({"dt_rule": True} if k["dt_rule"] else {}),
                      **({"early_all": True} if k["early_all"] else {}),
                      **({"inner_polish": True} if k.get("inner_polish_j") else {}),        # 1.0.1 (spec 003b)
                      **({"bracket_bisect": True} if k.get("bracket_bisect") else {}),
                      **({"extent_z": True} if k.get("extent_z") else {}),
                      **({"box_flag": True} if k.get("box_flag") else {}),
                      **({"kappa_fixed": float(k["kappa_fixed"])} if k.get("kappa_fixed") is not None else {}),
                      **({"combos": tuple(k["combos"])} if k.get("combos") else {}),
                      stages=("mle",))
    except Exception as ex:  # noqa: BLE001 -- a failed refit is a result, never a silent "no flag"
        if raise_errors:
            raise
        return {"status": "failed", "reason": f"{type(ex).__name__}: {str(ex)[:300]}", "theta": None,
                "cpu_s": time.process_time() - c0}
    dg = res.get("diagnostics") or {}
    ct = prep.censor_tie or {}
    out = {"status": "failed" if dg.get("error") else "ok", "reason": dg.get("error"),
           "theta": [float(v) for v in res["theta"]], "loglik_stage": res.get("loglik"),
           "relabelled": int(ct.get("n_relabelled", 0) or 0), "cpu_s": time.process_time() - c0}
    return out


def refit_warm(cohort, name, drop_row, z0, port=None) -> dict:
    """The estimator's own L-BFGS-B (`_lbfgs`, the MLE stage's options and fallbacks) on the objective of the cohort
    without `drop_row` (its own prepared data, grid and box: `profile_jax2.fit_inputs`), started at z0 = the reported θ̂.
    `_lbfgs` never returns a point worse than its start. `drop_row` None: the whole cohort (g0)."""
    from sraging.calibrators import profile_jax2 as PJ
    P = _port(port)
    est, fj = P.estimator, P.fp_jax
    c0 = time.process_time()
    try:
        inp = PJ.fit_inputs(_without(cohort, drop_row), name, P)
        obj = est._Objective(fj, inp["t"], inp["e"], inp["L"], inp["kw"], inp["box"], hazard=inp["hazard"],
                             window=inp["window"])
        bounds = [(math.log(inp["box"][p][0]), math.log(inp["box"][p][1])) for p in obj.params]
        v, z, plateau = est._lbfgs(obj, obj.nll_grad, obj.nll, np.asarray(z0, float), bounds, est._LBFGS_MLE,
                                   dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
    except Exception as ex:  # noqa: BLE001 -- recorded; the other start may still stand
        return {"status": "failed", "reason": f"{type(ex).__name__}: {str(ex)[:300]}", "theta": None,
                "cpu_s": time.process_time() - c0}
    ok = bool(np.isfinite(v))
    return {"status": "ok" if ok else "failed", "reason": None if ok else "non-finite objective",
            "theta": [float(x) for x in np.exp(z)], "loglik_stage": float(-v), "plateau": bool(plateau),
            "n_grad": int(obj.n_g), "exc_counts": dict(obj.exc_counts), "cpu_s": time.process_time() - c0}


def reduced_loglik(cohort, name, drop_row, zs, port=None) -> list:
    """The objective of the cohort without `drop_row` (its own prepared data and grid), by the value march, at each z:
    the stored L-BFGS-B value is not used ([T-LBFGS-STORED-VALUE])."""
    import jax.numpy as jnp
    from sraging.calibrators import profile_jax2 as PJ
    f, _vg = _loglik(PJ.fit_inputs(_without(cohort, drop_row), name, _port(port)), port=port)[:2]
    return [float(f(jnp.asarray(np.asarray(z, float)))) for z in zs]


def g0(cohort, name, z, port=None) -> dict:
    """The warm start's gain at θ̂ on the FULL data (spec 003b FR-010.1): the estimator's L-BFGS-B from the reported θ̂
    (`refit_warm`, no row dropped), the gain by the value march as Δ_j is measured. `_lbfgs` never returns a point
    worse than its start, so g0 >= 0 up to the march's round-off; a negative is recorded as it is (`below_start`),
    never clipped. A measurement only: no label reads it until Naveh's post-pilot decision."""
    kw = {} if port is None else {"port": port}
    w = refit_warm(cohort, name, None, z, **kw)
    if w["status"] != "ok":
        return {"status": "failed", "reason": w["reason"], "cpu_s": w.get("cpu_s")}
    ll0, llw = reduced_loglik(cohort, name, None, [z, np.log(np.asarray(w["theta"], float))], **kw)
    gain = llw - ll0
    return {"status": "ok", "g0": gain, "below_start": bool(gain < 0), "theta": w["theta"], "ll_ref": ll0,
            "ll_warm": llw, "cpu_s": w.get("cpu_s")}


def warm_termination(cohort, name, z, port=None) -> dict:
    """How the estimator's L-BFGS-B (its objective, `_LBFGS_MLE`) ends when started at theta-hat on the full data:
    scipy's status, iterations, evaluations and message, which `_lbfgs` discards (spec 003g FR-011; measured the same
    way as specs/003b-jax2-101-port/review/evidence/pilot_lnsrch.json). A diagnostic only: its point is never used."""
    from scipy.optimize import minimize
    from sraging.calibrators import profile_jax2 as PJ
    P = _port(port)
    est = P.estimator
    c0 = time.process_time()
    try:
        inp = PJ.fit_inputs(_without(cohort, None), name, P)
        obj = est._Objective(P.fp_jax, inp["t"], inp["e"], inp["L"], inp["kw"], inp["box"], hazard=inp["hazard"],
                             window=inp["window"])
        bounds = [(math.log(inp["box"][p][0]), math.log(inp["box"][p][1])) for p in obj.params]
        z0 = np.clip(np.asarray(z, float), [b[0] for b in bounds], [b[1] for b in bounds])
        r = minimize(obj.nll_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds, options=est._LBFGS_MLE)
    except Exception as ex:  # noqa: BLE001 -- recorded; the label does not read it
        return {"status": "failed", "reason": f"{type(ex).__name__}: {str(ex)[:300]}",
                "cpu_s": time.process_time() - c0}
    return {"status": "ok", "scipy_status": int(r.status), "message": str(r.message), "nit": int(r.nit),
            "nfev": int(r.nfev),
            "moved": float(np.max(np.abs(np.asarray(r.x, float) - z0))), "cpu_s": time.process_time() - c0}


def not_assessed(label, g0_block, free_gradient_norm, n, g_crit_per_ind=None) -> bool:
    """Spec 003g FR-011 (DV-3): the check could not see. The warm start on the full data did not move from theta-hat
    (g0 == 0 exactly: `_lbfgs` returned its start) at a free-gradient norm above G_CRIT_PER_IND per individual (n, all
    individuals), so no refit without one row can be compared with it. Whether theta-hat is non-stationary or the
    objective is kinked there, the evidence does not say. Only a `no_flag` or `flat_valley_slide` is replaced; a
    `flag` or `failed` stands."""
    from sraging.leverage_rule import G_CRIT_PER_IND
    g = G_CRIT_PER_IND if g_crit_per_ind is None else g_crit_per_ind
    return (label in ("no_flag", "flat_valley_slide") and (g0_block or {}).get("status") == "ok"
            and g0_block.get("g0") == 0.0 and free_gradient_norm is not None and bool(n)
            and free_gradient_norm / n > g)


# ------------------------------------------------------------------------------------------------------ the check
def _cpu_model() -> str:
    try:
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def _provenance(name, library, port=None) -> dict:
    """The fit's calibrator and ITS version (the port's: since spec 003b a bare name means 1.0.1, so the registry's
    current version is not the fit's)."""
    return {"calibrator": name, "version": _port(port).version, "host": platform.node(),
            "machine": platform.machine(), "cpu": _cpu_model(), "library": library}


def _compare(z_ref, z_new) -> dict:
    """Per quantity: the ratio new/ref and its log; R per quantity set (§13 eq. lev-ratio)."""
    qa, qb = quantities(z_ref), quantities(z_new)
    ratios = {q: {"log": qb[q] - qa[q], "ratio": math.exp(qb[q] - qa[q])} for q in QUANTITIES}
    R = {s: math.exp(max(abs(ratios[q]["log"]) for q in qs)) for s, qs in QUANTITY_SETS.items()}
    return {"ratios": ratios, "R": R}


def _drop(cohort, name, j, z, ages, events, rule, **pk) -> dict:
    """One drop: both refits of the cohort without data row j (0-based), the better one by the value march (the MLE
    stage unless the warm start beats it by more than WARM_GAIN_TOL), compared with the reported θ̂ (z). `pk`: the
    port, when not 1.0.0's."""
    d = {"row": int(j) + 1, "age": float(ages[j]), "event": int(events[j])}
    refits = {"mle_stage": refit_mle(cohort, name, j, **pk), "warm": refit_warm(cohort, name, j, z, **pk)}
    d["refits"] = refits
    ok = {k: np.log(np.asarray(r["theta"], float)) for k, r in refits.items() if r["status"] == "ok"}
    if not ok:
        d["refit"] = dict(refits["mle_stage"], start="mle_stage")
        return d
    lls = reduced_loglik(cohort, name, j, [z] + list(ok.values()), **pk)
    ll_ref, by = lls[0], dict(zip(ok, lls[1:]))
    fin = {k: v for k, v in by.items() if math.isfinite(v)}
    if "mle_stage" in fin and not ("warm" in fin and fin["warm"] > fin["mle_stage"] + WARM_GAIN_TOL):
        start = "mle_stage"
    elif fin:
        start = max(fin, key=fin.get)
    else:
        start = "mle_stage" if "mle_stage" in ok else "warm"
    d["refit"] = dict(refits[start], start=start)
    d.update(_compare(z, ok[start]))
    d.update(ll_ref=ll_ref, ll_new=by[start], ll_by_start=by)
    d["delta_total"] = by[start] - ll_ref
    d["delta_nats"] = d["delta_total"] / (ages.size - 1)
    d["labels"] = {s: label(d["R"][s], d["delta_total"], ratio=rule["ratio"],
                            delta_total_nats=rule["delta_total_nats"], fail_nats=rule["fail_nats"])
                   for s in QUANTITY_SETS}
    return d


def _failed_reason(d) -> str:
    if d.get("delta_total") is None:
        return "; ".join(f"{k}: {r.get('reason')}" for k, r in (d.get("refits") or {}).items())
    if not (math.isfinite(d["delta_total"]) and all(math.isfinite(v) for v in d["R"].values())):
        return "a non-finite ratio or log-likelihood"
    return (f"the best refit without row {d['row']} ended {-d['delta_total']:.3g} nats below theta-hat on that data's "
            f"own objective: the optimizer stopped short, so the move is not a measurement")


def check(cohort, name, theta_hat, diagnostics, *, rule=None, library=None, fit_cpu_s=None,
          drop_rows=None, port=None) -> dict:
    """The leverage result of data-model.md for one jax2 fit. Never raises: a failure is a labelled result.
    `drop_rows` (pilot P3 only) refits several individuals: {"top_<set>", "median_<set>"} -> 0-based index. Every
    row the result reports is the 1-based data row, as in the floor report. `port`: the fit's (spec 003b; 1.0.0's
    when None); every step runs on it, and g0 is recorded when the drop was measured."""
    from sraging.leverage_rule import RULE
    rule = dict(RULE if rule is None else rule)
    P = _port(port)
    pk = {} if P.version == "1.0.0" else {"port": P}       # 1.0.0: spec 003f's calls, unchanged
    c0 = time.process_time()
    out = {"version": VERSION, "label": "not_run", "reason": None, "rule": rule,
           "provenance": _provenance(name, library, P)}
    try:
        cls = P.classes[name]
        hazard = bool(cls.HAZARD)
        ages, events = np.asarray(cohort.ages, float), np.asarray(cohort.events, int)
        out["gap_sd"] = gap_sd(ages, events)
        z = _z(theta_hat, hazard)
        td = terms_and_derivatives(cohort, name, z, **pk)
        inp = td["inputs"]
        if td["terms"].size != ages.size or not np.array_equal(np.asarray(inp["t"], float), ages):
            raise RuntimeError("the prepared data do not keep the cohort's rows in order")
        out["share"] = share(td["terms"], ages, events)
        lo, hi = _bounds(inp["box"], hazard)
        inf = influence(td["G"], td["H"], z, lo, hi)
        sh = shifts(inf["DZ"], z)
        ranks = {s: ranking(sh, s) for s in QUANTITY_SETS}

        def who(j):                                 # rows are 1-based data rows, as in the floor report
            return {"row": int(j) + 1, "age": float(ages[j]), "event": int(events[j]),
                    "shift": {q: float(sh[q][j]) for q in QUANTITIES}}
        out["influence"] = {s: {"top": who(r["top"]), "median": who(r["median"]), "n_tied_top": r["n_tied"]}
                            for s, r in ranks.items()}
        out["influence"]["hessian"] = {k: inf[k] for k in ("free", "active", "eigenvalues", "min_eigenvalue",
                                                           "min_relative", "n_negative", "condition", "n_cut",
                                                           "n_kept", "rel_cut")}
        out["influence"]["hessian"]["source"] = td["hessian_source"]
        names = PARAMS + (("h",) if hazard else ())
        out["at_theta_hat"] = {
            "log_distance_to_bounds": {p: [float(z[k] - lo[k]), float(hi[k] - z[k])] for k, p in enumerate(names)},
            "gradient": [float(x) for x in td["gradient"]],
            "free_gradient_norm": float(np.linalg.norm(td["gradient"][[k for k in range(z.size) if inf["free"][k]]]))}
        out["at_theta_hat"]["free_gradient_per_individual"] = out["at_theta_hat"]["free_gradient_norm"] / cohort.n
        bf = (diagnostics or {}).get("basin_fix") or {}
        out["basin_fix_fired"] = int(bf.get("n_refit", 0) or 0) > 0
        if inf["n_kept"] == 0:
            out.update(label="not_assessable", reason="no information direction survives the cut: the influence "
                                                      "ranking is empty, so no individual can be named")
            return out
        rows = drop_rows or {"top": ranks[rule["quantities"]]["top"]}
        drops = {key: _drop(cohort, name, j, z, ages, events, rule, **pk) for key, j in rows.items()}
        main = drops.get("top") or drops.get(f"top_{rule['quantities']}")
        if drop_rows is not None:
            out["drops"] = drops
        if main is None:
            out.update(label="not_run", reason="no top-influence drop under the rule's quantities")
            return out
        out.update(dropped={k: main[k] for k in ("row", "age", "event")}, refit=main["refit"],
                   refits=main["refits"], n_tied_top=ranks[rule["quantities"]]["n_tied"])
        if "labels" not in main:
            out.update(label="failed", reason=f"both refits failed: {_failed_reason(main)}")
            return out
        out.update({k: main[k] for k in ("ratios", "R", "ll_ref", "ll_new", "ll_by_start", "delta_total",
                                         "delta_nats", "labels")})
        out["label"] = main["labels"][rule["quantities"]]
        if out["label"] == "failed":
            out["reason"] = _failed_reason(main)
        out["g0"] = g0(cohort, name, z, **pk)                # spec 003b: recorded
        if (out["g0"] or {}).get("status") == "ok" and out["g0"].get("g0") == 0.0:
            out["warm_termination"] = warm_termination(cohort, name, z, **pk)
        gn = out["at_theta_hat"]["free_gradient_norm"]
        if not_assessed(out["label"], out["g0"], gn, cohort.n):        # spec 003g FR-011 (Naveh Da; DV-3)
            from sraging.leverage_rule import G_CRIT_PER_IND
            out.update(label="not_assessed",
                       reason=f"the optimiser could not move from theta-hat: its warm start on the full data stopped "
                              f"where it began (g0 = 0) at a free-gradient norm of {gn:.3g} over {cohort.n} "
                              f"individuals ({gn / cohort.n:.3g} per individual > {G_CRIT_PER_IND:g}), so a refit "
                              f"without one row cannot be compared with it")
    except Exception as ex:  # noqa: BLE001 -- recorded; the fit is untouched
        out.update(label="not_run", reason=f"{type(ex).__name__}: {str(ex)[:300]}")
    finally:
        cpu = time.process_time() - c0
        out["cost"] = {"cpu_s": cpu, "share_of_fit": (cpu / fit_cpu_s) if fit_cpu_s else None}
    return out


# ------------------------------------------------------------------------------------ profile_v033 (spec 006, T040)
#: Spec 006 (FR-024; research R17): profile_v033's own drop-one check, 003f's rule unchanged. The terms, G and the
#: Hessian come from the engine at numerics "v033" on v0.3.3's grid (GridPolicy("v033"), equal to v0.3.3's objective
#: bit for bit, as floors.report uses it); the refits are v0.3.3's MLE stage (the replay sraging.v033_mle, bit-verified
#: in spec 003g) and its own L-BFGS-B warm start from theta-hat; every Delta by v0.3.3's objective. compat/v033 is read,
#: never edited. A separate path from the jax2 check above, which it leaves byte for byte as it was.
V033 = "profile_v033"


def _v033_rows(cohort, drop_row=None):
    t, e = np.asarray(cohort.ages, float), np.asarray(cohort.events, int)
    if drop_row is None:
        return t, e
    keep = np.arange(t.size) != drop_row
    return t[keep], e[keep]


def v033_objective(cohort, drop_row=None):
    """(v0.3.3's _Objective on the cohort without `drop_row`, on its own grid and box, the box)."""
    from sraging.calibrators.profile_v033 import KAPPA_MAX
    from sraging.compat.v033 import estimator as E
    from sraging.compat.v033 import fp_jax as fj
    t, e = _v033_rows(cohort, drop_row)
    L, kw, _ceiling = E.solver_grid(t, e, cohort.delta if cohort.delta else None)
    box = E.box_for(KAPPA_MAX)
    return E._Objective(fj, t, e, L, kw, box), box


def terms_and_derivatives_v033(cohort, z) -> dict:
    """terms_and_derivatives for profile_v033: the engine at numerics "v033" on v0.3.3's grid."""
    import jax
    import jax.numpy as jnp
    from sraging.calibrators.profile_v033 import KAPPA_MAX
    from sraging.compat.v033 import estimator as E
    from sraging.engine.evaluator import Evaluator
    from sraging.engine.grid import GridPolicy
    ev = Evaluator(cohort, numerics="v033", grid=GridPolicy("v033"))
    zz = jnp.asarray(z)
    terms = np.asarray(ev._terms(zz)[0], float)
    G = np.asarray(jax.jacfwd(lambda x: ev._terms(x)[0])(zz), float)
    H = np.zeros((z.size, z.size))
    for k in range(z.size):
        zp, zm = np.array(z, float), np.array(z, float)
        zp[k] += FD_STEP
        zm[k] -= FD_STEP
        H[:, k] = (np.asarray(ev._vg(jnp.asarray(zp))[1]) - np.asarray(ev._vg(jnp.asarray(zm))[1])) / (2 * FD_STEP)
    H = 0.5 * (H + H.T)
    grad = np.asarray(ev._vg(jnp.asarray(np.asarray(z, float)))[1], float)
    return {"terms": terms, "G": G, "H": H, "gradient": grad, "hessian_source": "finite_difference",
            "inputs": {"t": np.asarray(cohort.ages, float), "box": E.box_for(KAPPA_MAX)}}


def refit_mle_v033(cohort, drop_row) -> dict:
    """v0.3.3's MLE stage (sraging.v033_mle.mle_stage) on the cohort without `drop_row`."""
    from sraging.v033_mle import mle_stage
    c0 = time.process_time()
    t, e = _v033_rows(cohort, drop_row)
    try:
        r = mle_stage(t, e, cohort.delta if cohort.delta else None)
    except Exception as ex:  # noqa: BLE001 -- a failed refit is a result, never a silent "no flag"
        return {"status": "failed", "reason": f"{type(ex).__name__}: {str(ex)[:300]}", "theta": None,
                "cpu_s": time.process_time() - c0}
    ok = r["status"] == "ok"
    return {"status": "ok" if ok else "failed", "reason": None if ok else r["status"],
            "theta": r["theta"] if ok else None, "loglik_stage": r.get("loglik"), "cpu_s": time.process_time() - c0}


def refit_warm_v033(cohort, drop_row, z0) -> dict:
    """v0.3.3's own L-BFGS-B (`_lbfgs`, the MLE stage's options) on its objective without `drop_row`, from z0."""
    from sraging.compat.v033 import estimator as E
    c0 = time.process_time()
    try:
        obj, box = v033_objective(cohort, drop_row)
        bounds = [(math.log(box[p][0]), math.log(box[p][1])) for p in E.PARAMS]
        v, z, plateau = E._lbfgs(obj, obj.nll_grad, obj.nll, np.asarray(z0, float), bounds, E._LBFGS_MLE,
                                 dict(maxiter=1200, xatol=1e-5, fatol=1e-4))
    except Exception as ex:  # noqa: BLE001 -- recorded; the other start may still stand
        return {"status": "failed", "reason": f"{type(ex).__name__}: {str(ex)[:300]}", "theta": None,
                "cpu_s": time.process_time() - c0}
    ok = bool(np.isfinite(v))
    return {"status": "ok" if ok else "failed", "reason": None if ok else "non-finite objective",
            "theta": [float(x) for x in np.exp(z)], "loglik_stage": float(-v), "plateau": bool(plateau),
            "cpu_s": time.process_time() - c0}


def reduced_loglik_v033(cohort, drop_row, zs) -> list:
    """v0.3.3's objective without `drop_row` at each z (its value, not a stored optimiser value)."""
    obj, _box = v033_objective(cohort, drop_row)
    return [-float(obj.nll(np.asarray(z, float))) for z in zs]


def g0_v033(cohort, z) -> dict:
    """The warm start's gain at theta-hat on the full data, as g0 for jax2."""
    w = refit_warm_v033(cohort, None, z)
    if w["status"] != "ok":
        return {"status": "failed", "reason": w["reason"], "cpu_s": w.get("cpu_s")}
    ll0, llw = reduced_loglik_v033(cohort, None, [z, np.log(np.asarray(w["theta"], float))])
    return {"status": "ok", "g0": llw - ll0, "below_start": bool(llw - ll0 < 0), "theta": w["theta"], "ll_ref": ll0,
            "ll_warm": llw, "cpu_s": w.get("cpu_s")}


def _drop_v033(cohort, j, z, ages, events, rule) -> dict:
    """One drop for profile_v033: _drop's rule on v0.3.3's refits and objective."""
    d = {"row": int(j) + 1, "age": float(ages[j]), "event": int(events[j])}
    refits = {"mle_stage": refit_mle_v033(cohort, j), "warm": refit_warm_v033(cohort, j, z)}
    d["refits"] = refits
    ok = {k: np.log(np.asarray(r["theta"], float)) for k, r in refits.items() if r["status"] == "ok"}
    if not ok:
        d["refit"] = dict(refits["mle_stage"], start="mle_stage")
        return d
    lls = reduced_loglik_v033(cohort, j, [z] + list(ok.values()))
    ll_ref, by = lls[0], dict(zip(ok, lls[1:]))
    fin = {k: v for k, v in by.items() if math.isfinite(v)}
    if "mle_stage" in fin and not ("warm" in fin and fin["warm"] > fin["mle_stage"] + WARM_GAIN_TOL):
        start = "mle_stage"
    elif fin:
        start = max(fin, key=fin.get)
    else:
        start = "mle_stage" if "mle_stage" in ok else "warm"
    d["refit"] = dict(refits[start], start=start)
    d.update(_compare(z, ok[start]))
    d.update(ll_ref=ll_ref, ll_new=by[start], ll_by_start=by)
    d["delta_total"] = by[start] - ll_ref
    d["delta_nats"] = d["delta_total"] / (ages.size - 1)
    d["labels"] = {s: label(d["R"][s], d["delta_total"], ratio=rule["ratio"],
                            delta_total_nats=rule["delta_total_nats"], fail_nats=rule["fail_nats"])
                   for s in QUANTITY_SETS}
    return d


def check_v033(cohort, theta_hat, diagnostics, *, rule=None, fit_cpu_s=None) -> dict:
    """The leverage result of data-model.md (spec 003f) for one profile_v033 fit, by 003f's rule: the influence ranking
    from the engine's v033 terms, then the top-influence row dropped and refitted. Never raises: a failure is a
    labelled result."""
    from sraging.calibrators.profile_v033 import VERSION
    from sraging.leverage_rule import RULE
    rule = dict(RULE if rule is None else rule)
    c0 = time.process_time()
    out = {"version": VERSION, "label": "not_run", "reason": None, "rule": rule,
           "provenance": {"calibrator": V033, "version": VERSION, "host": platform.node(),
                          "machine": platform.machine(), "cpu": _cpu_model(), "library": None,
                          "terms": "the engine at numerics v033 on v0.3.3's grid",
                          "refits": "v0.3.3's MLE stage (sraging.v033_mle) and its L-BFGS-B warm start"}}
    try:
        if theta_hat is None:
            out.update(label="not_run", reason="no theta-hat to check")
            return out
        ages, events = np.asarray(cohort.ages, float), np.asarray(cohort.events, int)
        out["gap_sd"] = gap_sd(ages, events)
        z = _z(theta_hat, False)
        td = terms_and_derivatives_v033(cohort, z)
        if td["terms"].size != ages.size:
            raise RuntimeError("the engine's terms do not keep the cohort's rows")
        out["share"] = share(td["terms"], ages, events)
        lo, hi = _bounds(td["inputs"]["box"], False)
        inf = influence(td["G"], td["H"], z, lo, hi)
        sh = shifts(inf["DZ"], z)
        ranks = {s: ranking(sh, s) for s in QUANTITY_SETS}

        def who(j):
            return {"row": int(j) + 1, "age": float(ages[j]), "event": int(events[j]),
                    "shift": {q: float(sh[q][j]) for q in QUANTITIES}}
        out["influence"] = {s: {"top": who(r["top"]), "median": who(r["median"]), "n_tied_top": r["n_tied"]}
                            for s, r in ranks.items()}
        out["influence"]["hessian"] = {k: inf[k] for k in ("free", "active", "eigenvalues", "min_eigenvalue",
                                                           "min_relative", "n_negative", "condition", "n_cut",
                                                           "n_kept", "rel_cut")}
        out["influence"]["hessian"]["source"] = td["hessian_source"]
        out["at_theta_hat"] = {
            "log_distance_to_bounds": {p: [float(z[k] - lo[k]), float(hi[k] - z[k])] for k, p in enumerate(PARAMS)},
            "gradient": [float(x) for x in td["gradient"]],
            "free_gradient_norm": float(np.linalg.norm(td["gradient"][[k for k in range(z.size) if inf["free"][k]]]))}
        out["at_theta_hat"]["free_gradient_per_individual"] = out["at_theta_hat"]["free_gradient_norm"] / cohort.n
        if inf["n_kept"] == 0:
            out.update(label="not_assessable", reason="no information direction survives the cut: the influence "
                                                      "ranking is empty, so no individual can be named")
            return out
        j = ranks[rule["quantities"]]["top"]
        main = _drop_v033(cohort, j, z, ages, events, rule)
        out.update(dropped={k: main[k] for k in ("row", "age", "event")}, refit=main["refit"],
                   refits=main["refits"], n_tied_top=ranks[rule["quantities"]]["n_tied"])
        if "labels" not in main:
            out.update(label="failed", reason=f"both refits failed: {_failed_reason(main)}")
            return out
        out.update({k: main[k] for k in ("ratios", "R", "ll_ref", "ll_new", "ll_by_start", "delta_total",
                                         "delta_nats", "labels")})
        out["label"] = main["labels"][rule["quantities"]]
        if out["label"] == "failed":
            out["reason"] = _failed_reason(main)
        out["g0"] = g0_v033(cohort, z)
        gn = out["at_theta_hat"]["free_gradient_norm"]
        if not_assessed(out["label"], out["g0"], gn, cohort.n):
            from sraging.leverage_rule import G_CRIT_PER_IND
            out.update(label="not_assessed",
                       reason=f"the optimiser could not move from theta-hat: its warm start on the full data stopped "
                              f"where it began (g0 = 0) at a free-gradient norm of {gn:.3g} over {cohort.n} "
                              f"individuals ({gn / cohort.n:.3g} per individual > {G_CRIT_PER_IND:g}), so a refit "
                              f"without one row cannot be compared with it")
    except Exception as ex:  # noqa: BLE001 -- recorded; the fit is untouched
        out.update(label="not_run", reason=f"{type(ex).__name__}: {str(ex)[:300]}")
    finally:
        cpu = time.process_time() - c0
        out["cost"] = {"cpu_s": cpu, "share_of_fit": (cpu / fit_cpu_s) if fit_cpu_s else None}
    return out

