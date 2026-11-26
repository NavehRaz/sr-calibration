"""Run: .venv/bin/python tools/age0_parity.py (about 10 CPU-min). Age-0 deaths (first-interval, D-105): the port's fit equals the oracle estimator's on the same arrays,
and the age-0 terms are finite (not the -700 floor). Side by side, one process."""
import os, sys, json
REPO = os.path.expanduser("~/code/sraging")
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
sys.path[:0] = [os.path.join(REPO, "tools"), os.path.join(REPO, "src")]
import numpy as np
import oracle_v033 as O
o = O.load()
from sraging import Cohort
from sraging.calibrators import profile_v033
from parity_checks import tree_diff
import acceptance as A

src = Cohort.from_csv(os.path.join(REPO, "tests/unit/data/small_cohort.csv"), delta=1.0)
ages = np.array(src.ages, float); ev = np.array(src.events, int)
first = np.argsort(ages)[:5]                     # the 5 earliest rows become deaths at age 0 (first interval)
ages[first] = 0.0; ev[first] = 1
c = Cohort.from_arrays(ages, ev, delta=1.0, name="small_cohort with 5 first-interval deaths")
port = profile_v033.fit(c)
ref = o.estimator.fit(o.fp_jax, list(ages), list(ev), delta=1.0, levels=(0.95,), kappa_max=300.0)
from sraging.compat.v033 import claims as vclaims
ref_claims, ref_diag = vclaims.build(ref)
d = tree_diff([x.reference_dict() for x in port.claims], ref_claims)
L, kw, _ = o.estimator.solver_grid(ages, ev, 1.0)
th = [port.theta_hat.eta, port.theta_hat.beta, port.theta_hat.epsilon, port.theta_hat.kappa]
terms = np.asarray(A.per_individual_terms(o.fp_jax, ages, ev, L, kw["N"], kw["dt"], kw["tau_max"], kw["Gtol"],
                                          sampling_dt=kw.get("sampling_dt"))(np.log(th)))
out = {"status": port.status, "claims_identical_to_oracle_estimator": d == [], "diff": d[:5],
       "age0_terms": [float(x) for x in terms[first]], "min_other_term": float(np.min(np.delete(terms, first))),
       "loglik_port": port.loglik, "loglik_oracle": ref.get("loglik"),
       "checks": [(k.code, k.ran, k.fired) for k in port.checks], "cpu_s": port.timing.get("cpu_s")}
json.dump(out, open(os.path.join(REPO, "specs", "001-reference-fit", "review", "evidence", "age0_parity.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
