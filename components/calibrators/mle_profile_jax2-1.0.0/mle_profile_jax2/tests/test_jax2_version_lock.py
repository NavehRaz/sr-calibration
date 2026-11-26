"""L0: VERSION_LOCK.json -- the code may not change under an unchanged version (tests/version_lock.py)."""
from __future__ import annotations

import copy
import json

import version_lock as VL


def test_lock_is_consistent_with_code_and_versions():
    assert VL.problems() == []


def test_lock_covers_every_runtime_file():
    lock = json.load(open(VL.LOCK))
    assert lock["files"] == VL.runtime_files()
    assert {"__init__.py", "estimator.py", "fp_jax.py", "observation_model.py"} <= set(lock["files"])
    for ent in lock["names"].values():
        assert sorted(ent["sha256"]) == lock["files"]


def test_rule_fails_on_code_change_under_same_version():
    """Positive control: a lock whose sha differs while the version is the same is a violation."""
    cur = VL.current()
    lock = copy.deepcopy(cur)
    for ent in lock["names"].values():
        ent["sha256"]["estimator.py"] = "0" * 64
    p = VL.problems(lock=lock, cur=cur)
    assert len(p) == len(cur["names"]) and all(x.startswith("CODE CHANGED") for x in p), p


def test_rule_flags_a_stale_lock_after_a_bump():
    cur = VL.current()
    lock = copy.deepcopy(cur)
    name = sorted(lock["names"])[0]
    lock["names"][name]["version"] = "0.0.0-old"
    p = VL.problems(lock=lock, cur=cur)
    assert p and p[0].startswith("lock is stale"), p


def test_rule_flags_a_new_runtime_file():
    cur = VL.current()
    lock = copy.deepcopy(cur)
    for ent in lock["names"].values():
        ent["sha256"].pop("observation_model.py")
    p = VL.problems(lock=lock, cur=cur)
    assert p and all("observation_model.py" in x for x in p), p


def test_versions_are_read_without_importing_the_adapter():
    v = VL.versions()
    assert set(v) == {"mle_profile_jax2", "mle_profile_jax2_h"}
    assert all(isinstance(x, str) and x for x in v.values())


# ---------------------------------------------------------------- B5: the same rule, enforced at RUN time

def _pkg_copy(tmp_path):
    """A copy of the package's runtime files + lock in a directory of the same name (so the sibling key is the
    real one); loaded in a subprocess under a NON-discovery key, so nothing is registered."""
    import os
    import shutil
    d = tmp_path / "mle_profile_jax2"
    d.mkdir()
    for f in VL.runtime_files() + ["VERSION_LOCK.json"]:
        shutil.copy2(os.path.join(VL.PKG, f), d / f)
    return d


_LOAD_COPY = """
import sys, json, importlib.util, os
d = {d!r}
spec = importlib.util.spec_from_file_location("jax2_vl_copy", os.path.join(d, "__init__.py"))
m = importlib.util.module_from_spec(spec); sys.modules["jax2_vl_copy"] = m; spec.loader.exec_module(m)
from srbench.contract.types import Observation
obs = Observation(cohort_id="x", ages=(1.0, 2.0), events=(1, 1), delta=None)
def refusal(f):
    try:
        f(); return "returned"
    except SystemExit as ex:
        return [type(ex).__name__, str(ex)[:600]]
"""


def test_runtime_guard_passes_on_the_package_and_an_untouched_copy(tmp_path):
    import _jax2_testlib as T
    d = _pkg_copy(tmp_path)
    out = T.run_py(_LOAD_COPY.format(d=str(d)) + """
spec2 = importlib.util.spec_from_file_location("jax2_vl_real", os.path.join({real!r}, "__init__.py"))
r = importlib.util.module_from_spec(spec2); sys.modules["jax2_vl_real"] = r; spec2.loader.exec_module(r)
print(json.dumps(dict(copy=m.version_lock_problems(), real=r.version_lock_problems(),
                      guard=refusal(m.version_lock_guard))))
""".replace("{real!r}", repr(VL.PKG)))
    assert out["_rc"] == 0, out["_stderr"]
    assert out["copy"] == [] and out["real"] == []
    assert out["guard"] == "returned"


def test_runtime_guard_refuses_a_code_change_under_the_same_version(tmp_path):
    """The must-fix of the B4 review: calibrate AND run_fit refuse (RefusedEnvironment, a SystemExit, so
    srbench stores nothing) when a runtime file differs from VERSION_LOCK.json while the version is the locked
    one -- before any package code beyond __init__ is loaded."""
    import _jax2_testlib as T
    d = _pkg_copy(tmp_path)
    with open(d / "estimator.py", "a") as fh:
        fh.write("\n# a behaviour change nobody versioned\n")
    out = T.run_py(_LOAD_COPY.format(d=str(d)) + """
res = dict(problems=m.version_lock_problems(),
           calibrate_h=refusal(lambda: m.MleProfileJax2H().calibrate(obs)),
           calibrate=refusal(lambda: m.MleProfileJax2().calibrate(obs)),
           run_fit=refusal(lambda: m.run_fit(obs, relabel=True)),
           siblings_loaded=sorted(k for k in sys.modules if k.startswith("srbench_methods.mle_profile_jax2.")))
print(json.dumps(res))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert len(out["problems"]) == 2 and all(p.startswith("CODE CHANGED BUT VERSION DID NOT") for p in out["problems"])
    assert all("estimator.py" in p for p in out["problems"])
    for k in ("calibrate_h", "calibrate", "run_fit"):
        assert out[k][0] == "RefusedEnvironment" and "CODE CHANGED BUT VERSION DID NOT" in out[k][1], out[k]
    assert out["siblings_loaded"] == []


def test_runtime_guard_refuses_a_stale_or_missing_lock(tmp_path):
    import _jax2_testlib as T
    d = _pkg_copy(tmp_path)
    src = (d / "__init__.py").read_text()
    v = VL.versions()["mle_profile_jax2_h"]
    assert src.count(f'"{v}"') == 2
    (d / "__init__.py").write_text(src.replace(f'"mle_profile_jax2_h": "{v}"', '"mle_profile_jax2_h": "9.9.9.dev9"'))
    out = T.run_py(_LOAD_COPY.format(d=str(d)) + """
res = dict(stale=m.version_lock_problems(), calibrate=refusal(lambda: m.MleProfileJax2H().calibrate(obs)))
os.remove(os.path.join(d, "VERSION_LOCK.json"))
res["missing"] = m.version_lock_problems()
res["run_fit_missing"] = refusal(lambda: m.run_fit(obs, relabel=True))
print(json.dumps(res))
""")
    assert out["_rc"] == 0, out["_stderr"]
    # the bumped name is stale; the other name's version is unchanged but its __init__.py sha moved
    assert any(p.startswith("VERSION_LOCK.json is stale: mle_profile_jax2_h") for p in out["stale"]), out["stale"]
    assert any(p.startswith("CODE CHANGED BUT VERSION DID NOT: mle_profile_jax2 ") and "__init__.py" in p
               for p in out["stale"]), out["stale"]
    assert out["calibrate"][0] == "RefusedEnvironment"
    assert len(out["missing"]) == 1 and out["missing"][0].startswith("cannot read")
    assert out["run_fit_missing"][0] == "RefusedEnvironment"


def test_runtime_guard_refuses_a_file_changed_after_it_was_loaded(tmp_path):
    """A sibling edited on disk after this process loaded it: the code running is not the file the lock
    matches, so the guard refuses even though disk and lock agree."""
    import _jax2_testlib as T
    d = _pkg_copy(tmp_path)
    out = T.run_py(_LOAD_COPY.format(d=str(d)) + """
om = m._sibling("observation_model")                 # stdlib-only sibling
res = dict(after_load=m.version_lock_problems(), loaded=sorted(m._LOADED_SHA))
res["fake_old_load"] = m.version_lock_problems(loaded=dict(observation_model="0" * 64))
print(json.dumps(res))
""")
    assert out["_rc"] == 0, out["_stderr"]
    assert out["after_load"] == [] and out["loaded"] == ["__init__", "observation_model"]
    assert out["fake_old_load"] == ["observation_model.py changed on disk after it was loaded in this process"]
