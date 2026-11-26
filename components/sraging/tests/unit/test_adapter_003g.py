"""How 003g reaches the tester through 003e's adapter, unchanged (research R4; tasks T009, T011): the pinned clone per
window follows the head jax2 names; an unpinned run, record or witness is refused; a run labelled with another library
is refused. Each refusal beside its twin."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_003g as ACC  # noqa: E402
from sraging.adapters import srbench as ADP  # noqa: E402
from sraging.errors import UnsupportedOptionError  # noqa: E402


def test_the_clone_follows_the_pinned_head():
    head = "c855d9a" + "0" * 33
    assert ACC.tester_clone(head).endswith(os.path.join(".cache", "tester_c855d9a"))
    with pytest.raises(UnsupportedOptionError, match="pinned tester clone"):     # no such clone: refused
        ADP.connect_pinned(clone=ACC.tester_clone("f" * 40), head="f" * 40)


def test_an_unpinned_run_file_is_refused_and_the_pinned_library_is_read(tmp_path):
    lib = tmp_path / "data" / "library" / "v6"
    lib.mkdir(parents=True)
    (lib / "manifest.json").write_bytes(b"{}")
    import hashlib
    data = {"libraries": {"v6": {"path": "data/library/v6/manifest.json",
                                 "sha256": hashlib.sha256(b"{}").hexdigest()}}, "runs": {}, "records": {}}
    pins = ADP.Pins(data, str(tmp_path))
    assert pins.read_file("data/library/v6/manifest.json") == b"{}"                # twin: pinned
    with pytest.raises(UnsupportedOptionError, match="not pinned"):
        pins.read_file("runs/jax2-v6-full-bign.json")
    with pytest.raises(UnsupportedOptionError, match="not pinned"):
        pins.read_record("jax2-v6-full-bign", "pure-k0.01-p01-c0-s0", "k" * 64, store=None)


def test_a_run_labelled_with_another_library_is_refused_on_v6():
    ADP.require_library("v6", method={"run_id": "a", "library_version": "v6"})       # twin
    with pytest.raises(UnsupportedOptionError, match="not the library named"):
        ADP.require_library("v6", method={"run_id": "jax2-v5-full-bign", "library_version": "v5"})


def test_the_003g_pins_start_with_the_v6_manifest_only():
    """Before pin-w1 the pins held the v6 manifest and nothing else; that pin is unchanged since. (Read through history
    since 2026-10-28: as first written it read the working file, and failed from pin-w1 on.)"""
    import json
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-C", ACC.REPO, *a], capture_output=True, text=True).stdout
    pin = git("log", "--reverse", "--format=%H", "-S", '"jax2-v6-full-bign"', "--", ACC.PINS_PATH).split()[0]
    before = json.loads(git("show", f"{pin}^:{ACC.PINS_PATH}"))
    assert before["libraries"]["v6"]["sha256"].startswith("399c9c55") and not before["runs"] and not before["records"]
    assert ACC.pins_data()["libraries"]["v6"] == before["libraries"]["v6"]
