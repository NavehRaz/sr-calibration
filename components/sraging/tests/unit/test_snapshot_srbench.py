"""The srbench tree snapshot builder (spec 003b, T003-T004; contracts/oracle-snapshots.md): the 46 files 1.0.1 runs
against, read from git objects at the tag's commit and checked against jax2's SRBENCH_FILES.sha256. Each refusal is
shown on a fake repository, with a twin that builds (constitution V)."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
import snapshot_srbench as B  # noqa: E402

FILES = ["srbench/__init__.py", "srbench/contract/method.py", "srbench/library/load.py"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(tmp_path):
    repo = str(tmp_path / "j2f")
    for p in FILES:
        os.makedirs(os.path.dirname(os.path.join(repo, p)), exist_ok=True)
        open(os.path.join(repo, p), "w").write(f"# {p}\n")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c")
    commit = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", f"{commit}:srbench")
    rows = [f"{sha(open(os.path.join(repo, p), 'rb').read())}  {p}" for p in sorted(FILES)]
    sums = str(tmp_path / "SRBENCH_FILES.sha256")
    open(sums, "w").write("# header\n" + "\n".join(rows) + "\n")
    listing = sha(("\n".join(rows) + "\n").encode())
    return dict(repo=repo, commit=commit, tree=tree, sums=sums, sums_sha=sha(open(sums, "rb").read()),
                listing=listing, snap=str(tmp_path / "tester_new"))


def build(w, **over):
    kw = dict(repo=w["repo"], commit=w["commit"], sums=w["sums"], snap=w["snap"], expect_tree=w["tree"],
              expect_sums_sha256=w["sums_sha"], expect_listing=w["listing"])
    kw.update(over)
    return B.build(**kw)


def test_a_clean_build_writes_every_file_and_its_manifest(world):
    m = build(world)
    for p in FILES:
        assert open(os.path.join(world["snap"], p), "rb").read() == open(os.path.join(world["repo"], p), "rb").read()
    assert m["tree"] == world["tree"] and m["listing_digest"] == world["listing"] and len(m["files"]) == 3
    assert json.load(open(os.path.join(world["snap"], "MANIFEST.json")))["read_from"] == "git objects"


def test_it_reads_objects_not_the_working_tree(world):
    open(os.path.join(world["repo"], FILES[1]), "a").write("# dirty\n")
    build(world)
    assert open(os.path.join(world["snap"], FILES[1])).read() == f"# {FILES[1]}\n"


def test_it_refuses_an_existing_destination(world):
    os.makedirs(world["snap"])
    with pytest.raises(B.Refused, match="exists"):
        build(world)


def test_it_refuses_a_sums_file_with_another_sha256(world):
    with pytest.raises(B.Refused, match="SRBENCH_FILES"):
        build(world, expect_sums_sha256="0" * 64)


def test_it_refuses_another_tree(world):
    with pytest.raises(B.Refused, match="tree"):
        build(world, expect_tree="0" * 40)


def test_it_refuses_a_wrong_listing_digest(world):
    with pytest.raises(B.Refused, match="listing"):
        build(world, expect_listing="0" * 64)


def test_it_refuses_a_blob_whose_bytes_differ_from_its_line(world):
    lines = open(world["sums"]).read().replace(sha(open(os.path.join(world["repo"], FILES[2]), "rb").read()), "0" * 64)
    open(world["sums"], "w").write(lines)
    body = [ln for ln in lines.splitlines() if ln and not ln.startswith("#")]
    with pytest.raises(B.Refused, match="library/load.py"):
        build(world, expect_sums_sha256=sha(lines.encode()), expect_listing=sha(("\n".join(body) + "\n").encode()))
    assert not os.path.exists(world["snap"])


def test_it_refuses_a_tree_with_a_file_the_sums_do_not_list(world):
    open(os.path.join(world["repo"], "srbench", "extra.py"), "w").write("x = 1\n")
    git(world["repo"], "add", "-A")
    git(world["repo"], "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "extra")
    head = git(world["repo"], "rev-parse", "HEAD")
    with pytest.raises(B.Refused, match="extra.py"):
        build(world, commit=head, expect_tree=git(world["repo"], "rev-parse", f"{head}:srbench"))
