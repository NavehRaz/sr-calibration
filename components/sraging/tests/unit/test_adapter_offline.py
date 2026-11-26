"""Tester adapter, offline parts (T051; contracts/srbench-adapter.md; research R10).

These need no calibration and never import srbench from the live tester: they check the descriptor,
the window passed to the cohort (spec 006), the no-discovery rule, and the tester-tree fingerprint.
"""
import os
import time
import types

import pytest

from sraging.adapters import srbench as A
from sraging.adapters.observation import cohort_from_observation
from sraging.compat.v033 import claims as vclaims
from sraging.errors import UnsupportedOptionError


def obs(window=None, delta=1.0):
    return types.SimpleNamespace(cohort_id="x", ages=(3.0, 5.0, 8.0), events=(1, 0, 1), delta=delta,
                                 window=window)


def test_descriptor_fields_match_the_reference_but_the_name_does_not():
    d = A.descriptor_fields()
    assert d["name"] == "sraging_profile_v033" != "mle_profile_jax"
    assert d["version"].endswith("+v033")
    assert d["bounds"] == vclaims.BOUNDS
    assert d["capabilities"] == ()
    assert A.REFERENCE_COST == {"median_seconds": 590.0, "min_seconds": 140.0, "max_seconds": 70000.0}


def test_a_window_reaches_the_cohort():
    """Spec 006 (D-104 revised): the loader no longer refuses a window; the calibrator fits it or refuses it by name
    (tests/unit/test_adapter_006.py)."""
    assert cohort_from_observation(obs(window=(3.0, 8.0))).window == (3.0, 8.0)


def test_falsy_delta_means_continuous():
    c = cohort_from_observation(obs(delta=None))
    assert c.continuous and c.delta is None


def test_adapter_never_discovers(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(discover=lambda *a, **k: calls.append("discover"),
                                 get=lambda *a, **k: calls.append("get"),
                                 register=lambda m: calls.append(("register", m.descriptor.name)))
    A.register_with(fake, A.make_method(fake_types()))
    assert calls == [("register", "sraging_profile_v033")]


def fake_types():
    class MD:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    return types.SimpleNamespace(MethodDescriptor=MD, CostDeclaration=MD, Claim=MD, ClaimSet=MD)


def test_fingerprint_detects_changes(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.txt").write_text("one")
    fp0 = A.fingerprint(str(root))
    assert A.fingerprint(str(root)) == fp0
    (root / "b.txt").write_text("new")                       # an added file
    assert A.fingerprint(str(root)) != fp0
    os.remove(root / "b.txt")
    assert A.fingerprint(str(root)) == fp0
    (root / "a.txt").write_text("one!")                      # a changed size
    assert A.fingerprint(str(root)) != fp0
    (root / "a.txt").write_text("one")
    t = time.time() + 10
    os.utime(root / "a.txt", (t, t))                         # a changed mtime
    assert A.fingerprint(str(root)) != fp0


# ---- spec 003 (T040): the method table ---------------------------------------------------------------------

def test_the_method_table_lists_three_names_with_their_testers():
    m = A.methods()
    assert set(m) == {"sraging_profile_v033", "sraging_profile_jax2", "sraging_profile_jax2_h",          # spec 006:
                      "sraging_profile_jax2_101", "sraging_profile_jax2_h_101", "sraging_profile_jax2_w"}   # 1.0.1, W
    assert m["sraging_profile_v033"]["tester_head"] == A.TESTER_HEAD
    assert m["sraging_profile_jax2"]["tester_head"] == m["sraging_profile_jax2_h"]["tester_head"] == A.J2_HEAD
    d = A.descriptor_fields("sraging_profile_jax2")
    assert d["name"] == "sraging_profile_jax2" and d["version"].endswith("+jax2-1.0.0")
    assert A.descriptor_fields() == A.descriptor_fields("sraging_profile_v033")


def test_a_wrong_jax2_head_is_refused(tmp_path):
    import subprocess
    repo = tmp_path / "j2"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "x").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "x"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
                   check=True)
    with pytest.raises(UnsupportedOptionError, match="not the recorded revision"):
        A.connect(str(repo), A.J2_HEAD)


def test_a_jax2_claimset_carries_only_the_four_parameters():
    T = fake_types()
    cs = A.claimset_from_kwargs(T, "c", "sraging_profile_jax2",
                                [{"parameter": "eta", "shape": "closed", "lower": 1.0, "upper": 2.0, "point": 1.5,
                                  "level": 0.95}, {"parameter": "eta_over_eps", "shape": "closed", "lower": 1.0}],
                                {"x": 1})
    assert [c.parameter for c in cs.claims] == ["eta"] and cs.method_name == "sraging_profile_jax2"


def _git_repo(tmp_path):
    import subprocess
    repo = tmp_path / "j2"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    (repo / "x").write_text("x")
    subprocess.run(["git", "-C", str(repo), "add", "x"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x"],
                   check=True)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return repo, head


def test_the_pinned_tester_is_refused_when_dirty_or_moved(tmp_path):
    repo, head = _git_repo(tmp_path)
    assert A.pinned_problems(str(repo), head) == []
    assert any("not the recorded revision" in p for p in A.pinned_problems(str(repo), A.J2_HEAD))
    (repo / "x").write_text("changed")
    assert any("dirty" in p for p in A.pinned_problems(str(repo), head))
    with pytest.raises(UnsupportedOptionError, match="dirty"):
        A.connect(str(repo), head, require_clean=True)
    assert any("does not exist" in p for p in A.pinned_problems(str(tmp_path / "nope"), head))
