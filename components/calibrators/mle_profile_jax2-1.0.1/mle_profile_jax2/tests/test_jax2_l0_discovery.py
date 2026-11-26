"""L0: discovery, registration, lazy imports, sibling keys, environment guards, switches. Every test that
imports the adapter does so in a SUBPROCESS (see _jax2_testlib)."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import _jax2_testlib as T

# 1.0.1.dev2: the two base names and the six kfix names (one package, one lock); sorted, as `registered()` lists them
NAMES = ("mle_profile_jax2", "mle_profile_jax2_h", "mle_profile_jax2_h_kfix_k0p001", "mle_profile_jax2_h_kfix_k0p5",
         "mle_profile_jax2_h_kfix_k1", "mle_profile_jax2_kfix_k0p001", "mle_profile_jax2_kfix_k0p5",
         "mle_profile_jax2_kfix_k1")
TWIN = "mle_profile_jax2_combotwin"           # a SEPARATE validation-only package (its own lock and tests)
#: 1.0.1 (the freeze): the twin is NOT shipped -- tester main has methods/mle_profile_jax2 without it (measured in the
#: freeze rehearsal, reports/freeze101/REHEARSAL.txt: with the twin required, 2 of these tests failed there). It is
#: checked wherever its directory exists (the release branch) and is not required elsewhere.
TWINS = (TWIN,) if os.path.isdir(os.path.join(T.METHODS, TWIN)) else ()
ALL_METHODS = ("decliner", "fast_stub", "mle_profile", "mle_profile_jax", "null_prior") + NAMES + TWINS

# Block the heavy dependencies, then discover: a lazy-import violation is an ImportError at discovery.
_BLOCKED = """
import sys, json
for m in ("jax", "jaxlib", "SRK", "scipy"):
    sys.modules[m] = None
"""


def test_discovery_without_jax_scipy_or_srk():
    out = T.run_py(_BLOCKED + """
from srbench.contract import method as M
reg = M.registered()
d = {n: reg[n].descriptor.to_dict() for n in __NAMES__}
print(json.dumps(dict(names=sorted(reg), d=d,
    heavy=sorted(k for k in ("numpy", "jax", "scipy", "SRK") if sys.modules.get(k) is not None),
    siblings=sorted(k for k in sys.modules if k.startswith("srbench_methods.mle_profile_jax2.")))))
""".replace("__NAMES__", repr(NAMES)))
    assert out["_rc"] == 0, out["_stderr"]
    assert set(NAMES) <= set(out["names"])
    assert out["heavy"] == [], f"discovery imported {out['heavy']}"
    assert out["siblings"] == [], "discovery must not load fp_jax / estimator / observation_model"
    for n in NAMES:
        assert out["d"][n]["name"] == n
        assert out["d"][n]["parameters_returned"] == ["eta", "beta", "epsilon", "kappa"]


def test_registers_only_under_the_discovery_key():
    """Loaded under any other name (as pytest's package collection loads it) the adapter registers nothing,
    so a later discover() still works; loaded by discovery it registers both names, once."""
    out = T.run_py(_BLOCKED + """
import importlib.util, os
from srbench.contract import method as M
p = os.path.join("methods", "mle_profile_jax2", "__init__.py")
spec = importlib.util.spec_from_file_location("mle_profile_jax2", p)
m = importlib.util.module_from_spec(spec); sys.modules["mle_profile_jax2"] = m; spec.loader.exec_module(m)
before = sorted(n for n in M._REGISTRY if n.startswith("mle_profile_jax2"))
after = sorted(n for n in M.registered() if n.startswith("mle_profile_jax2"))
print(json.dumps(dict(before=before, after=after, key=m._KEY)))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["before"] == [] and [n for n in out["after"] if n != TWIN] == list(NAMES)
    assert out["key"] == "srbench_methods.mle_profile_jax2"


def test_cli_methods_lists_every_method():
    r = subprocess.run([sys.executable, "-m", "srbench.cli", "methods"], cwd=T.ROOT, env=T.child_env(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    listed = {ln.split()[0] for ln in r.stdout.splitlines()[2:] if ln.strip()}
    assert set(ALL_METHODS) <= listed, (listed, r.stdout)
    assert "INVALID" not in r.stdout


def test_descriptors_versions_and_capabilities():
    from version_lock import versions
    out = T.run_py(_BLOCKED + """
from srbench.contract import method as M
reg = M.registered()
ad = type(reg["mle_profile_jax2"]).calibrate.__globals__
res = {}
for n in __NAMES__:
    m = reg[n]; c = type(m)
    res[n] = dict(version=m.descriptor.version, caps=list(m.descriptor.capabilities),
                  sw=[c.HAZARD, c.WINDOW, c.RELABEL], base=[b.__name__ for b in c.__mro__][1],
                  derived=list(ad["capabilities_for"](c.HAZARD, c.WINDOW, c.RELABEL)),
                  init_takes_args=c.__init__ is not object.__init__, kappa0=c.KAPPA0,
                  same_module=type(m).calibrate.__globals__ is ad)
print(json.dumps(res))
""".replace("__NAMES__", repr(NAMES)))
    assert out["_rc"] == 0, out["_stderr"]
    vers = versions()
    for n in NAMES:
        r = out[n]
        assert r["version"] == vers[n]
        assert r["base"] == "_MleProfileJax2Base"            # one base class for every name
        assert r["caps"] == r["derived"]                       # declared == derived from the constants
        assert not r["init_takes_args"], "no runtime switch: behaviour is fixed per registered name"
        assert r["same_module"]                                # one package: one adapter module
        if "_kfix_k" in n:                                     # 1.0.1.dev2: the kfix names = J / H + KAPPA0
            base = "mle_profile_jax2_h" if n.startswith("mle_profile_jax2_h_") else "mle_profile_jax2"
            assert r["sw"] == out[base]["sw"] and r["caps"] == out[base]["caps"]
            assert r["kappa0"] == {"0p001": 1e-3, "0p5": 0.5, "1": 1.0}[n.rsplit("_k", 1)[1]]
        else:
            assert r["kappa0"] is None
    # jax2: relabel ON (Phase A confirms), no hazard / window -> ('censoring',)
    assert out["mle_profile_jax2"]["sw"] == [False, False, True]
    assert out["mle_profile_jax2"]["caps"] == ["censoring"]
    # _h (B2): hazard + window + relabel -> the design's three capabilities, in the design's order
    assert out["mle_profile_jax2_h"]["sw"] == [True, True, True]
    assert out["mle_profile_jax2_h"]["caps"] == ["external_hazard", "fit_window", "censoring"]


def test_capability_derivation_is_the_design_order():
    out = T.run_py(_BLOCKED + """
from srbench.contract import method as M
f = type(M.registered()["mle_profile_jax2"]).calibrate.__globals__["capabilities_for"]
print(json.dumps(dict(all=list(f(True, True, True)), none=list(f(False, False, False)))))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["all"] == ["external_hazard", "fit_window", "censoring"]     # design: _h 1.0.0
    assert out["none"] == []


def test_sibling_key_is_directory_derived_and_file_checked():
    out = T.run_py(_BLOCKED + """
import os, types
from srbench.contract import method as M
g = type(M.registered()["mle_profile_jax2"]).calibrate.__globals__
key = g["_KEY"]; here = g["_HERE"]
om = g["_sibling"]("observation_model")          # stdlib-only sibling: loads with jax / SRK blocked
ok = dict(key=key, here_base=os.path.basename(here), om_key=om.__name__,
          om_file_ok=os.path.realpath(om.__file__) == os.path.realpath(os.path.join(here, "observation_model.py")),
          cached=g["_sibling"]("observation_model") is om)
# poison the key with a module from another file: the next call must refuse, not run it
fake = types.ModuleType(key + ".observation_model"); fake.__file__ = "/elsewhere/observation_model.py"
sys.modules[key + ".observation_model"] = fake
try:
    g["_sibling"]("observation_model"); ok["poison"] = "returned"
except SystemExit as ex:
    ok["poison"] = type(ex).__name__
# a sibling that fails to import leaves no half-initialised module under its key
try:
    g["_sibling"]("fp_jax"); ok["fp_jax"] = "imported?!"
except ImportError:
    ok["fp_jax"] = "ImportError"
ok["fp_jax_left"] = (key + ".fp_jax") in sys.modules
print(json.dumps(ok))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["key"] == "srbench_methods." + out["here_base"] == "srbench_methods.mle_profile_jax2"
    assert out["om_key"] == out["key"] + ".observation_model"
    assert out["om_file_ok"] and out["cached"]
    assert out["poison"] == "RefusedEnvironment"
    assert out["fp_jax"] == "ImportError" and out["fp_jax_left"] is False


def test_env_guard_refuses_dt_ceiling_before_any_work():
    """SRK_DT_CEILING set: calibrate aborts (SystemExit) before importing anything heavy, so no record is
    stored; the estimator refuses on its own too."""
    out = T.run_py(_BLOCKED + """
from srbench.contract import method as M
from srbench.contract.types import Observation
reg = M.registered()
obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
res = {}
for n in __NAMES__:
    try:
        reg[n].calibrate(obs); res[n] = "returned"
    except SystemExit as ex:
        res[n] = [type(ex).__name__, "SRK_DT_CEILING" in str(ex), isinstance(ex, Exception)]
print(json.dumps(res))
""".replace("__NAMES__", repr(NAMES + TWINS)), SRK_DT_CEILING="1e-3")
    assert out["_rc"] == 0, out["_stderr"]
    for n in NAMES + TWINS:
        assert out[n] == ["RefusedEnvironment", True, False], out[n]


@pytest.mark.sut
def test_estimator_refusals_and_bad_inputs():
    """SRK_DT_CEILING refuses in estimator.fit; a NotImplementedError from the fit path refuses
    (RefusedEnvironment, nothing stored) instead of becoming an error record; the B2 likelihood rejects a
    window whose ages it cannot honour and an unknown stage. (B1's "switch ON raises NotImplementedError"
    is gone: B2 built both switches.)"""
    out = T.run_py("""
import sys, json, os
from srbench.contract import method as M
from srbench.contract.types import Observation
reg = M.registered()
g = type(reg["mle_profile_jax2"]).calibrate.__globals__
est = g["_sibling"]("estimator"); fj = g["_sibling"]("fp_jax")
res = {}
def trap(k, f):
    try:
        f(); res[k] = "returned"
    except BaseException as ex:
        res[k] = type(ex).__name__
os.environ["SRK_DT_CEILING"] = "1e-3"
trap("fit_ceiling", lambda: est.fit(fj, [1.0, 2.0], [1, 1]))
trap("fit_ceiling_hazard", lambda: est.fit(fj, [1.0, 2.0], [1, 1], hazard=True))
del os.environ["SRK_DT_CEILING"]
trap("fit_bad_stage", lambda: est.fit(fj, [1.0, 2.0], [1, 1], stages=("mle", "bogus")))
kw = dict(N=20, dt=0.1, tau_max=2.0)
trap("ll_window_age_below_lo", lambda: fj.make_loglik([1.0, 2.0], [1, 1], 1.5, window=(1.5, 2.0), **kw))
trap("ll_window_age_above_hi", lambda: fj.make_loglik([1.0, 2.0], [1, 1], 1.5, window=(0.5, 1.5), **kw))
trap("ll_window_empty", lambda: fj.make_loglik([1.0, 2.0], [1, 1], 1.5, window=(2.0, 2.0), **kw))
trap("ll_window_ok", lambda: fj.make_loglik([1.0, 2.0], [1, 1], 1.5, window=(0.5, 2.0), hazard=True, **kw))
trap("ll_bad_dl", lambda: fj.make_loglik([1.0, 2.0], [1, 1], 1.5, sampling_dt=[0.1, -0.1], hazard=True, **kw))
obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
def unbuilt(*a, **k):
    raise NotImplementedError("synthetic unbuilt path")
g["run_fit"] = unbuilt
trap("calibrate_not_implemented", lambda: reg["mle_profile_jax2_h"].calibrate(obs))
print(json.dumps(res))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["fit_ceiling"] == "RuntimeError" and out["fit_ceiling_hazard"] == "RuntimeError"
    assert out["fit_bad_stage"] == "ValueError"
    for k in ("ll_window_age_below_lo", "ll_window_age_above_hi", "ll_window_empty", "ll_bad_dl"):
        assert out[k] == "ValueError", (k, out[k])
    assert out["ll_window_ok"] == "returned"
    assert out["calibrate_not_implemented"] == "RefusedEnvironment"


@pytest.mark.sut
def test_old_and_new_adapters_in_one_process():
    """The frozen v0.3.3 adapter and this package load side by side: distinct sys.modules keys, distinct
    module objects, each from its own directory, each evaluating its own likelihood."""
    out = T.run_py("""
import sys, json, os
import numpy as np
from srbench.contract import method as M
reg = M.registered()
old = type(reg["mle_profile_jax"]).calibrate.__globals__
new = type(reg["mle_profile_jax2"]).calibrate.__globals__
newh = type(reg["mle_profile_jax2_h"]).calibrate.__globals__
of, oe = old["_sibling"]("fp_jax"), old["_sibling"]("estimator")
nf, ne = new["_sibling"]("fp_jax"), new["_sibling"]("estimator")
t = np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.7]); e = np.ones(t.size, int)
z = np.log([1.0, 2.0, 0.3, 0.5])
kw = dict(N=40, dt=0.02, tau_max=3.0, Gtol=1e-11)
vo = float(of.make_loglik(t, e, 1.1, **kw)[0](z)); vn = float(nf.make_loglik(t, e, 1.1, **kw)[0](z))
vod = float(of.make_loglik(t, e, 1.1, sampling_dt=0.1, **kw)[0](z))
vnd = float(nf.make_loglik(t, e, 1.1, sampling_dt=0.1, **kw)[0](z))
print(json.dumps(dict(
    keys=[of.__name__, oe.__name__, nf.__name__, ne.__name__],
    distinct=len({id(of), id(oe), id(nf), id(ne)}) == 4,
    files=[os.path.relpath(m.__file__, os.getcwd()) for m in (of, oe, nf, ne)],
    same_adapter_module_for_both_names=new["_sibling"] is newh["_sibling"],
    old_has_floor=hasattr(oe, "floor_info"), new_has_floor=hasattr(ne, "floor_info"),
    new_has_params_global=hasattr(ne, "PARAMS"), old_has_params_global=hasattr(oe, "PARAMS"),
    v=[vo, vn, vod, vnd])))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["keys"] == ["srbench_methods.mle_profile_jax.fp_jax", "srbench_methods.mle_profile_jax.estimator",
                           "srbench_methods.mle_profile_jax2.fp_jax", "srbench_methods.mle_profile_jax2.estimator"]
    assert out["distinct"]
    assert out["files"] == [os.path.join("methods", "mle_profile_jax", "fp_jax.py"),
                            os.path.join("methods", "mle_profile_jax", "estimator.py"),
                            os.path.join("methods", "mle_profile_jax2", "fp_jax.py"),
                            os.path.join("methods", "mle_profile_jax2", "estimator.py")]
    assert out["same_adapter_module_for_both_names"]
    assert out["new_has_floor"] and not out["old_has_floor"]
    assert out["old_has_params_global"] and not out["new_has_params_global"]
    vo, vn, vod, vnd = out["v"]
    import math
    assert all(math.isfinite(v) for v in out["v"])
    # exact-time deaths: fix B does not touch that path, so the two likelihoods agree to round-off
    assert abs(vo - vn) <= 1e-9 * max(1.0, abs(vo)), (vo, vn)
    # interval deaths: fix B's mass equals S(t) - S(t+delta) wherever that is resolved (here: all of it)
    assert abs(vod - vnd) <= 1e-6 * max(1.0, abs(vod)), (vod, vnd)
