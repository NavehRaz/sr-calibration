"""Profile the Drosophila cohort's log-likelihood (the first README example, now review evidence in
specs/001-reference-fit/review/evidence/drosophila/) in beta, past the prior box's edge (beta <= 1e3), to show
which ends of its intervals are set by the box rather than the data (register D-018).

At each fixed beta, log(eta), log(epsilon) and log(kappa) are maximised without bounds by L-BFGS-B with
the port's own value-and-gradient (same grid and delta as the fit), starting from theta-hat moved along
the ridge (eta and epsilon scaled with beta). Writes specs/001-reference-fit/review/evidence/drosophila/<name>.ridge.json.

    .venv/bin/python tools/example_ridge.py
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
NAME = "drosophila_simons2024_25C_2pct_batch1"
EVIDENCE = os.path.join("specs", "001-reference-fit", "review", "evidence", "drosophila")
BETAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)


def main():
    from sraging import Cohort
    from sraging.compat.v033 import estimator as E
    from sraging.compat.v033 import fp_jax as fj
    res = json.load(open(os.path.join(REPO, EVIDENCE, f"{NAME}.json")))
    delta = res["cohort"]["delta"]
    c = Cohort.from_csv(os.path.join(REPO, EVIDENCE, f"{NAME}.csv"), delta=delta)
    t, e = np.asarray(c.ages, float), np.asarray(c.events, int)
    L, kw, _ = E.solver_grid(t, e, delta)
    _, vg = fj.make_loglik(t, e, L, N=kw["N"], dt=kw["dt"], tau_max=kw["tau_max"], Gtol=kw["Gtol"],
                           sampling_dt=kw.get("sampling_dt"))
    th = [res["theta_hat"][p] for p in ("eta", "beta", "epsilon", "kappa")]
    ll_hat = float(vg(np.log(th))[0])

    def nll(y, lb):
        val, grad = vg(np.array([y[0], lb, y[1], y[2]]))
        return -float(val), -np.asarray(grad)[[0, 2, 3]]

    points = []
    for beta in BETAS:
        s = beta / th[1]
        r = minimize(nll, [math.log(th[0] * s), math.log(th[2] * s), math.log(th[3])], args=(math.log(beta),),
                     jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        eta, eps, kappa = np.exp(r.x)
        points.append({"beta": beta, "loglik": -float(r.fun), "eta": float(eta), "epsilon": float(eps),
                       "kappa": float(kappa), "converged": bool(r.success), "nit": int(r.nit)})
        print(f"beta {beta:>8g}: loglik {-r.fun:.6f}  eta {eta:.4g}  epsilon {eps:.4g}  kappa {kappa:.3g}"
              f"  {'ok' if r.success else r.message}", flush=True)
    inside = [p for p in points if p["beta"] <= 1e3]
    beyond = [p for p in points if p["beta"] > 1e3]
    out = {"cohort": NAME, "box_beta_max": 1e3, "loglik_theta_hat": ll_hat, "points": points,
           "max_gain_beyond_box": max(p["loglik"] for p in beyond) - ll_hat,
           "span_30_to_30000": (next(p["loglik"] for p in points if p["beta"] == 30000.0)
                                - next(p["loglik"] for p in points if p["beta"] == 30.0)),
           "eta_at_box_edge": next(p["eta"] for p in inside if p["beta"] == 1e3),
           "method": "fixed beta; L-BFGS-B over log(eta, epsilon, kappa), unbounded; port value_and_grad"}
    path = os.path.join(REPO, EVIDENCE, f"{NAME}.ridge.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"-> {os.path.relpath(path, REPO)}: gain beyond the box {out['max_gain_beyond_box']:.4f} nats; "
          f"loglik span beta 30 -> 30000 {out['span_30_to_30000']:.4f} nats")


if __name__ == "__main__":
    main()
