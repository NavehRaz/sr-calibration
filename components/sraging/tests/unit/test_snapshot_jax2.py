"""The jax2 1.0.0 oracle-snapshot builder (spec 003, T001; research R2; contracts/oracle-snapshot.md).

Each refusal is shown on a fake J2 repository built in a temporary directory (constitution V: the builder
must refuse each condition it guards against); the real snapshot is built once by T002.
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
import snapshot_jax2 as S  # noqa: E402

PY = ["__init__.py", "estimator.py", "fp_jax.py", "observation_model.py"]
SRK = ["__init__.py", "fp_likelihood.py"]
SRB = ["srbench/__init__.py", "srbench/contract/method.py"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git(repo, *a):
    subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True)


@pytest.fixture
def world(tmp_path):
    j2, v033 = str(tmp_path / "j2"), str(tmp_path / "v033")
    meth = os.path.join(j2, S.METHOD)
    os.makedirs(meth)
    for f in PY:
        open(os.path.join(meth, f), "w").write(f"# {f}\nx = 1\n")
    lock = {"names": {n: {"version": "1.0.0", "sha256": {f: sha(os.path.join(meth, f)) for f in PY}}
                      for n in ("mle_profile_jax2", "mle_profile_jax2_h")}}
    json.dump(lock, open(os.path.join(meth, "VERSION_LOCK.json"), "w"))
    for p in SRB:
        for root in (j2, os.path.join(v033, "sr_calibration_tester")):
            os.makedirs(os.path.dirname(os.path.join(root, p)), exist_ok=True)
            open(os.path.join(root, p), "w").write(f"# {p}\n")
    srk = os.path.join(v033, "SRtools_kappa", "SRK")
    os.makedirs(srk)
    for f in SRK:
        open(os.path.join(srk, f), "w").write(f"# SRK {f}\n")
    ident = os.path.join(j2, S.IDENTITY)
    os.makedirs(os.path.dirname(ident))
    json.dump({"result": {"diagnostics": {"srk_sha": {"files": {f: sha(os.path.join(srk, f)) for f in SRK}}}}},
              open(ident, "w"))
    json.dump({"files": [{"repo": "sr_calibration_tester", "path": p} for p in SRB]},
              open(os.path.join(v033, "MANIFEST.json"), "w"))
    git(j2, "init", "-q")
    git(j2, "add", "-A")
    git(j2, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "frozen")
    frozen = subprocess.run(["git", "-C", j2, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return dict(j2=j2, v033=v033, snap=str(tmp_path / "snap"), frozen=frozen, meth=meth, srk=srk)


def build(w, **kw):
    return S.build(j2=w["j2"], snap=w["snap"], v033=w["v033"], frozen=w["frozen"], **kw)


def test_a_clean_build_copies_the_files_and_cites_the_shared_ones(world):
    m = build(world)
    for f in PY + ["VERSION_LOCK.json"]:
        assert open(os.path.join(world["snap"], "mle_profile_jax2", f), "rb").read() == \
            open(os.path.join(world["meth"], f), "rb").read()
    assert {c["path"] for c in m["cited"]} == {f"SRK/{f}" for f in SRK} | set(SRB)
    assert m["source"]["frozen_commit"] == world["frozen"] and m["source"]["diff_empty"] is True
    assert m["version_lock_check"] == {f: sha(os.path.join(world["meth"], f)) for f in PY}
    assert os.path.exists(os.path.join(world["snap"], "MANIFEST.json"))
    assert os.path.exists(os.path.join(world["snap"], "README.md"))


def test_it_refuses_an_existing_snapshot(world):
    os.makedirs(world["snap"])
    with pytest.raises(S.Refused, match="exists"):
        build(world)


def test_it_refuses_a_dirty_method_directory(world):
    open(os.path.join(world["meth"], "fp_jax.py"), "a").write("y = 2\n")
    with pytest.raises(S.Refused, match="dirty"):
        build(world)


def test_it_refuses_a_head_whose_method_differs_from_the_frozen_commit(world):
    open(os.path.join(world["meth"], "estimator.py"), "a").write("z = 3\n")
    git(world["j2"], "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "later")
    with pytest.raises(S.Refused, match="differs from the frozen commit"):
        build(world)


def test_it_refuses_files_that_do_not_match_the_version_lock(world):
    lock = json.load(open(os.path.join(world["meth"], "VERSION_LOCK.json")))
    lock["names"]["mle_profile_jax2"]["sha256"]["fp_jax.py"] = "0" * 64
    json.dump(lock, open(os.path.join(world["meth"], "VERSION_LOCK.json"), "w"))
    git(world["j2"], "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "lock")
    with pytest.raises(S.Refused, match="VERSION_LOCK"):
        S.build(j2=world["j2"], snap=world["snap"], v033=world["v033"],
                frozen=subprocess.run(["git", "-C", world["j2"], "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip())


def test_it_refuses_an_srk_file_that_differs_from_v033(world):
    open(os.path.join(world["srk"], "fp_likelihood.py"), "a").write("# edited\n")
    with pytest.raises(S.Refused, match="SRK/fp_likelihood.py"):
        build(world)


def test_it_refuses_an_srbench_file_that_differs_from_v033(world):
    p = os.path.join(world["v033"], "sr_calibration_tester", SRB[1])
    open(p, "a").write("# edited\n")
    with pytest.raises(S.Refused, match="srbench/contract/method.py"):
        build(world)


# ------------------------------------------------------------------------------------------------- spec 003b: git
# `build_git` (spec 003b, T003-T004): the 1.0.1 snapshot is read from git OBJECTS at the tag's commit, never from a
# working tree, and every byte is checked against the handoff's sha256 before anything is written.

def _listing(rows):
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()


@pytest.fixture
def gworld(tmp_path):
    repo = str(tmp_path / "j2f")
    meth = os.path.join(repo, S.METHOD)
    os.makedirs(os.path.join(meth, "tests"))
    for f in PY:
        open(os.path.join(meth, f), "w").write(f"# {f} 1.0.1\nx = 1\n")
    open(os.path.join(meth, "tests", "test_x.py"), "w").write("def test(): pass\n")
    shas = {f: sha(os.path.join(meth, f)) for f in PY}
    json.dump({"files": PY, "names": {n: {"version": "1.0.1", "sha256": shas} for n in ("mle_profile_jax2",
                                                                                      "mle_profile_jax2_h")}},
              open(os.path.join(meth, "VERSION_LOCK.json"), "w"))
    v033 = str(tmp_path / "v033")
    srk = os.path.join(v033, "SRtools_kappa", "SRK")
    os.makedirs(srk)
    for f in SRK:
        open(os.path.join(srk, f), "w").write(f"# SRK {f}\n")
    record = str(tmp_path / "record.json")
    json.dump({"result": {"diagnostics": {"srk_sha": {"files": {f: sha(os.path.join(srk, f)) for f in SRK}}}}},
              open(record, "w"))
    srbsnap = str(tmp_path / "tester_snap")
    os.makedirs(srbsnap)
    json.dump({"listing_digest": "d" * 64, "tree": "t" * 40}, open(os.path.join(srbsnap, "MANIFEST.json"), "w"))
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "freeze")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", "v101", "-m", "frozen")
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    rows = []
    for rel in sorted(["VERSION_LOCK.json", *PY, "tests/test_x.py"]):
        rows.append(f"{sha(os.path.join(meth, rel))}  {S.METHOD}/{rel}")
    expect = {"runtime": shas, "version_lock": sha(os.path.join(meth, "VERSION_LOCK.json")), "listing": _listing(rows)}
    return dict(repo=repo, meth=meth, commit=commit, tag="v101", expect=expect, v033=v033, record=record,
                srbsnap=srbsnap, snap=str(tmp_path / "jax2_new"))


def gbuild(w, **over):
    kw = dict(repo=w["repo"], commit=w["commit"], tag=w["tag"], snap=w["snap"], expect=w["expect"], v033=w["v033"],
              srk_record=w["record"], srbench_snapshot=w["srbsnap"])
    kw.update(over)
    return S.build_git(**kw)


def test_build_git_copies_the_commits_bytes_and_records_the_digests(gworld):
    m = gbuild(gworld)
    for f in PY + ["VERSION_LOCK.json"]:
        assert open(os.path.join(gworld["snap"], "mle_profile_jax2", f), "rb").read() == \
            open(os.path.join(gworld["meth"], f), "rb").read()
    assert m["source"]["commit"] == gworld["commit"] and m["source"]["read_from"] == "git objects"
    assert m["listing_digest"] == gworld["expect"]["listing"] and len(m["listing"]) == 6
    assert {c["path"] for c in m["cited"] if c["repo"] == "SRtools_kappa"} == {f"SRK/{f}" for f in SRK}
    assert m["cited_srbench"]["listing_digest"] == "d" * 64
    assert os.path.exists(os.path.join(gworld["snap"], "README.md"))


def test_build_git_reads_objects_not_the_working_tree(gworld):
    open(os.path.join(gworld["meth"], "fp_jax.py"), "a").write("# a working-tree edit\n")     # dirty, uncommitted
    gbuild(gworld)                                                                          # twin: still the commit's
    assert open(os.path.join(gworld["snap"], "mle_profile_jax2", "fp_jax.py")).read().endswith("x = 1\n")


def test_build_git_refuses_an_existing_destination(gworld):
    os.makedirs(gworld["snap"])
    with pytest.raises(S.Refused, match="exists"):
        gbuild(gworld)


def test_build_git_refuses_a_tag_on_another_commit(gworld):
    open(os.path.join(gworld["meth"], "estimator.py"), "a").write("y = 2\n")
    git(gworld["repo"], "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "later")
    later = subprocess.run(["git", "-C", gworld["repo"], "rev-parse", "HEAD"], capture_output=True,
                           text=True).stdout.strip()
    with pytest.raises(S.Refused, match="tag"):
        gbuild(gworld, commit=later)


def test_build_git_refuses_bytes_that_differ_from_the_expected_sha256(gworld):
    exp = dict(gworld["expect"], runtime=dict(gworld["expect"]["runtime"], **{"fp_jax.py": "0" * 64}))
    with pytest.raises(S.Refused, match="fp_jax.py"):
        gbuild(gworld, expect=exp)
    assert not os.path.exists(gworld["snap"])                                               # nothing written


def test_build_git_refuses_a_lock_that_does_not_pin_the_files(gworld):
    lock = json.load(open(os.path.join(gworld["meth"], "VERSION_LOCK.json")))
    lock["names"]["mle_profile_jax2_h"]["sha256"]["estimator.py"] = "0" * 64
    json.dump(lock, open(os.path.join(gworld["meth"], "VERSION_LOCK.json"), "w"))
    git(gworld["repo"], "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-am", "bad lock")
    git(gworld["repo"], "tag", "-f", "-a", "v101", "-m", "moved")
    head = subprocess.run(["git", "-C", gworld["repo"], "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    exp = dict(gworld["expect"], version_lock=sha(os.path.join(gworld["meth"], "VERSION_LOCK.json")))
    with pytest.raises(S.Refused, match="VERSION_LOCK"):
        gbuild(gworld, commit=head, expect=exp)


def test_build_git_refuses_a_missing_file(gworld):
    exp = dict(gworld["expect"], runtime=dict(gworld["expect"]["runtime"], **{"extra.py": "0" * 64}))
    with pytest.raises(S.Refused, match="missing"):
        gbuild(gworld, expect=exp)


def test_build_git_refuses_a_wrong_listing_digest(gworld):
    with pytest.raises(S.Refused, match="listing"):
        gbuild(gworld, expect=dict(gworld["expect"], listing="0" * 64))


def test_build_git_refuses_an_srk_file_that_differs_from_v033(gworld):
    open(os.path.join(gworld["v033"], "SRtools_kappa", "SRK", "fp_likelihood.py"), "a").write("# edited\n")
    with pytest.raises(S.Refused, match="SRK/fp_likelihood.py"):
        gbuild(gworld)
