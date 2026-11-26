"""The README's engine example (spec 002): the ITP cohort's log-likelihood at the example fit's theta-hat, with
a fit window, an external hazard, an age segment and heterogeneity, and the model's curves.

    .venv/bin/python examples/engine_example.py

It prints what the README shows and writes examples/results/engine_example.json; tools/verify_readme.py
checks the README's numbers against that file.
"""
import json
import os
import platform
import subprocess

import numpy as np

from sraging import Cohort, SRParams
from sraging.engine import AgeSegments, Heterogeneity, Segment, curves, evaluator

HERE = os.path.dirname(os.path.abspath(__file__))
fit = json.load(open(os.path.join(HERE, "results", "itp_c2016_het3_control_female.json")))
theta = {k: fit["theta_hat"][k] for k in ("eta", "beta", "epsilon", "kappa")}
p = SRParams(**theta, time_unit="days")
csv = os.path.join(HERE, "data", "itp_c2016_het3_control_female.csv")
c = Cohort.from_csv(csv, delta=1.0, time_unit="days")

out = {}
ev = evaluator(c)                                           # numerics "v1", grid "v033", scheme "bdf2"
ll, g = ev.value_and_grad(p)                                # g: d loglik / d log(parameter), by name
old = evaluator(c, numerics="v033").value(p)                # v0.3.3's arithmetic: the fit's own value
out["plain"] = {"loglik": ll, "loglik_v033": old, "grad": g, "grid": ev.grid.summary()}
print(f"plain:       loglik {ll:.4f} (v0.3.3 arithmetic {old:.4f}); gradient "
      + ", ".join(f"{k} {v:+.3g}" for k, v in g.items()))

cw = Cohort.from_csv(csv, delta=1.0, time_unit="days", window=(400.0, 1400.0))
v = evaluator(cw).value(p)
out["window"] = {"loglik": v, "window": [400.0, 1400.0], "n_at_risk": cw.n_at_risk}
print(f"window:      loglik {v:.4f} on {cw.n_at_risk} mice at risk in [400, 1400) days")

ph = SRParams(**theta, h_ext=1e-4, time_unit="days")
v = evaluator(c, fixed={"h_ext": 1e-4}).value(ph)
out["hazard"] = {"loglik": v, "h_ext": 1e-4}
print(f"hazard:      loglik {v:.4f} with an external hazard of 1e-4 per day")

seg = AgeSegments(Segment(id="rapa", parameter="beta", value=1.5, window=(600.0, 900.0)))
es = evaluator(c, segments=seg, free=("eta", "beta", "epsilon", "kappa", "seg:rapa"))
v, gs = es.value_and_grad(p)
out["segment"] = {"loglik": v, "grad_seg_rapa": gs["seg:rapa"]}
print(f"segment:     loglik {v:.4f} with beta x 1.5 on [600, 900) days; d loglik / d log(rapa) {gs['seg:rapa']:+.4g}")

v = evaluator(c, heterogeneity=Heterogeneity(parameter="xc", definition="normal_frac_sd", v=0.2, M=7)).value(p)
out["heterogeneity"] = {"loglik": v, "parameter": "xc", "v": 0.2, "M": 7}
print(f"mixture:     loglik {v:.4f} with x_c heterogeneous (v = 0.2, 7 nodes)")

ages = np.array([600.0, 800.0, 1000.0, 1200.0])
cv = curves(p, ages)
out["curves"] = {"ages": ages.tolist(), "S": cv.S.tolist(), "hazard": cv.hazard.tolist(),
                 "grid_L": cv.grid.L}
print("curves:      S " + ", ".join(f"{a:.0f} d {s:.3f}" for a, s in zip(ages, cv.S)))

head = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
out["provenance"] = {"git_sha": head, "python": platform.python_version()}
with open(os.path.join(HERE, "results", "engine_example.json"), "w") as fh:
    json.dump(out, fh, indent=1)
    fh.write("\n")
