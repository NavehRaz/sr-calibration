"""Model curves of the worked example: the exact FP survival and hazard at the STORED theta-hat of
~/code/sraging/examples/results/itp_c2016_het3_control_female_jax2.json (no fit; one engine evaluation).

Needs sraging, so it runs in sraging's own venv:
  ~/code/sraging/.venv/bin/python -B tables/example_curve.py   ->  tables/example_curve.csv
The header line records the sraging commit and the sha256 of the stored fit.
"""
import hashlib, os, subprocess, sys

import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paper_paths as PP  # noqa: E402  (where the inputs are: here, or in the release)

import sraging
from sraging.params import SRParams
from sraging.population import population

HERE = os.path.dirname(os.path.abspath(__file__))
SR = "~/code/sraging"
FIT = PP.path(SR + "/examples/results/itp_c2016_het3_control_female_jax2.json")
CSV = PP.path(SR + "/examples/data/itp_c2016_het3_control_female.csv")

raw = open(FIT, "rb").read()
fit = json.loads(raw)
th = fit["theta_hat"]
params = SRParams(eta=th["eta"], beta=th["beta"], epsilon=th["epsilon"], kappa=th["kappa"], h_ext=th["h_ext"],
                  time_unit=th["time_unit"])
last = max(float(line.split(",")[0]) for line in open(CSV).read().splitlines()[1:] if line.strip())
ages = np.linspace(0.0, 1.05 * last, 400)
pop = population(params, ages)
head = (f"{sraging.__version__} (release)" if PP.RELEASE else
        subprocess.run(["git", "-C", PP.local(SR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())

out = os.path.join(HERE, "example_curve.csv")
with open(out, "w") as f:
    f.write(f"# sraging {head}; fit sha256 {hashlib.sha256(raw).hexdigest()}; engine {pop.engine_used} ({pop.kind})\n")
    f.write("age_days,S,hazard_per_day\n")
    hz = pop.hazard if pop.hazard is not None else np.full_like(ages, np.nan)
    for a, s, h in zip(ages, np.asarray(pop.S, float), np.asarray(hz, float)):
        f.write(f"{a:.6g},{s:.10g},{h:.10g}\n")
print("wrote", out, "engine", pop.engine_used, pop.kind)
