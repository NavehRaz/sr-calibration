"""Spec 003e (T003): the adapter's library setting, the tester pin, content pins, the `_h` refusal, and the no-peek and
no-build scans. Each refusal is tested with a planted defect and its unplanted twin (constitution V). Fast; the one
test that imports the tester's code from the pinned clone runs in a subprocess, so it cannot meet another srbench."""
import hashlib
import inspect
import json
import os
import subprocess
import sys

import pytest

from sraging.adapters import srbench as ADP
from sraging.errors import UnsupportedOptionError

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------------------------------- the library setting
def test_the_library_is_required_and_named():
    with pytest.raises(UnsupportedOptionError, match="library"):
        ADP.library_root(None, "/t")
    with pytest.raises(UnsupportedOptionError, match="v7"):
        ADP.library_root("v7", "/t")
    assert ADP.library_root("v5", "/t") == os.path.join("/t", "data", "library", "v5")
    assert ADP.LIBRARIES == ("v1", "v2", "v3", "v4", "v5", "v6")


def test_a_run_labelled_with_another_library_is_refused_naming_both():
    v5, v2 = {"library_version": "v5", "run_id": "r5"}, {"library_version": "v2", "run_id": "r2"}
    ADP.require_library("v5", method=v5, reference=v5, null=v5)                 # the twin
    ADP.require_library("v5", method=v5, reference=v5, null=None)               # no null is allowed
    with pytest.raises(UnsupportedOptionError, match=r"(?s)method.*v2.*v5"):
        ADP.require_library("v5", method=v2, reference=v5)
    with pytest.raises(UnsupportedOptionError, match=r"(?s)reference.*v5.*v6"):  # both agree, the setting does not
        ADP.require_library("v6", method=dict(v5, library_version="v6"), reference=v5)


# ----------------------------------------------------------------------------------------------- the tester pin
def _git(cwd, *a):
    return subprocess.run(["git", "-C", cwd, *a], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    d = str(tmp_path / "t")
    os.makedirs(d)
    _git(d, "init", "-q")
    _git(d, "config", "user.email", "x@y")
    _git(d, "config", "user.name", "x")
    open(os.path.join(d, "f.txt"), "w").write("a\n")
    _git(d, "add", "f.txt")
    _git(d, "commit", "-q", "-m", "one")
    return d, _git(d, "rev-parse", "HEAD")


def test_a_clone_at_another_head_or_dirty_is_refused(repo):
    d, head = repo
    assert ADP.pinned_problems(d, head) == []                                   # the twin: clean, at the head
    assert ADP.pinned_problems(d, "0" * 40)                                     # another head
    open(os.path.join(d, "f.txt"), "a").write("b\n")
    assert any("dirty" in p for p in ADP.pinned_problems(d, head))             # dirty
    with pytest.raises(UnsupportedOptionError, match="tester"):
        ADP.connect_pinned(clone=d, head=head)


def test_the_pinned_clone_gives_the_comparison_module():
    if not os.path.isdir(ADP.TESTER_PINNED):
        pytest.skip("no pinned clone here (spec 003e T001)")
    code = ("from sraging.adapters import srbench as A; m = A.connect_pinned(); "
            "print(sorted(m)); import srbench; print(srbench.__file__)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert "report.comparison" in out.stdout and "run.execute" in out.stdout
    assert os.path.realpath(ADP.TESTER_PINNED) in os.path.realpath(out.stdout.split("\n")[1])


# ----------------------------------------------------------------------------------------------- content pins
def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_a_file_that_differs_from_its_pin_is_refused_by_name(tmp_path):
    root = tmp_path
    (root / "runs").mkdir()
    f = root / "runs" / "r.json"
    f.write_bytes(b'{"a": 1}')
    pins = ADP.Pins({"runs": {"r": {"path": "runs/r.json", "sha256": _sha(b'{"a": 1}')}}}, str(root))
    pins.check_file("runs/r.json")                                              # the twin
    f.write_bytes(b'{"a": 2}')
    with pytest.raises(UnsupportedOptionError, match="runs/r.json"):
        pins.check_file("runs/r.json")
    with pytest.raises(UnsupportedOptionError, match="not pinned"):
        pins.check_file("runs/other.json")


class _Store:
    def __init__(self, root):
        self.root = root

    def _path(self, digest):
        return os.path.join(self.root, digest[:2], digest[2:4], f"{digest}.json")


def test_a_record_that_differs_from_its_pin_is_refused_by_name(tmp_path):
    st = _Store(str(tmp_path))
    dg = "ab" * 32
    p = st._path(dg)
    os.makedirs(os.path.dirname(p))
    open(p, "wb").write(b"record")
    pins = ADP.Pins({"records": {"run": {"c1": {"key": dg, "sha256": _sha(b"record")}}}}, str(tmp_path))
    pins.check_record("run", "c1", st)                                          # the twin
    open(p, "wb").write(b"rewritten")
    with pytest.raises(UnsupportedOptionError, match="c1"):
        pins.check_record("run", "c1", st)
    with pytest.raises(UnsupportedOptionError, match="not pinned"):
        pins.check_record("run", "c2", st)


def test_pins_load_from_a_file(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"runs": {}}))
    assert isinstance(ADP.load_pins(str(p), str(tmp_path)), ADP.Pins)


# ------------------------------------------------------------------------------- compare_v5 with preloaded sets
class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _fake_mods(seen):
    def build_comparison(lib, a, b, ma, mb, null_sets=None):
        seen.update(method=a, reference=b, ref_manifest=mb)
        return {"by_parameter": {}, "cohort_set": {"paired": 0}}

    class Record:
        @staticmethod
        def from_dict(d):
            return _NS(claim_set=_NS(cohort_id=d["cs"]))
    return {"reference.pointer": _NS(require_comparable=lambda a, b: None),
            "library.load": _NS(Library=lambda root: root),
            "report.comparison": _NS(build_comparison=build_comparison, __file__=os.path.join(REPO, "a", "b", "c.py")),
            "store.store": _NS(ResultStore=_Store, CalibrationRecord=Record),
            "run.execute": _NS(load_results=lambda m, st: [_NS(cohort_id=c) for c in m["cohort_ids"]])}


@pytest.fixture
def tester(tmp_path):
    """A tiny tester data root: libraries v5 and v2 (cohort b's arrays differ), a pinned run `ref` with two records."""
    root = str(tmp_path)
    files = {"data/library/v5/manifest.json": json.dumps({"checksums": {"a": "x", "b": "y"}}).encode(),
             "data/library/v2/manifest.json": json.dumps({"checksums": {"a": "x", "b": "OTHER"}}).encode()}
    keys = {"a": "aa" * 32, "b": "bb" * 32}
    run = {"method": "ref_m", "method_version": "1", "library_version": "v5", "contract_version": "1.0",
           "run_id": "ref", "cohort_ids": ["a", "b"], "keys": keys}
    files["runs/ref.json"] = json.dumps(run).encode()
    st = _Store(os.path.join(root, "store"))
    recs = {}
    for c, k in keys.items():
        files[os.path.relpath(st._path(k), root)] = json.dumps({"cs": c}).encode()
        recs[c] = {"key": k, "sha256": _sha(json.dumps({"cs": c}).encode())}
    for rel, b in files.items():
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        open(os.path.join(root, rel), "wb").write(b)
    pins = ADP.Pins({"libraries": {"v5": {"path": "data/library/v5/manifest.json",
                                          "sha256": _sha(files["data/library/v5/manifest.json"])}},
                     "runs": {"ref": {"path": "runs/ref.json", "sha256": _sha(files["runs/ref.json"])}},
                     "records": {"ref": recs}}, root)
    return root, pins, run, st


LAB = {"method": "sraging_profile_jax2", "library_version": "v5", "contract_version": "1.0"}
MINE = [_NS(cohort_id="a"), _NS(cohort_id="b")]


def _cmp(tester, method=None, reference=None, library="v5", cohort_ids=None, seen=None):
    root, pins, run, st = tester
    method = method or {"manifest": LAB, "sets": MINE, "run_id": None, "computed_on": "v5"}
    reference = reference or {"run_id": "ref", "store_root": st.root}
    return ADP.compare_v5(_fake_mods(seen if seen is not None else {}), method, reference, library=library, pins=pins,
                          tester_root=root, cohort_ids=cohort_ids)


def test_preloaded_sets_are_sragings_own_claims_only(tester):
    seen = {}
    _cmp(tester, cohort_ids=["a"], seen=seen)                                   # the twin: restricted to the ids
    assert [cs.cohort_id for cs in seen["method"]] == ["a"] and [cs.cohort_id for cs in seen["reference"]] == ["a"]
    with pytest.raises(UnsupportedOptionError, match="pinned store"):
        _cmp(tester, method={"manifest": LAB, "sets": MINE, "run_id": "ref", "computed_on": "v5"})


def test_compare_v5_refuses_the_h_port_off_v2(tester):
    _cmp(tester)                                                                 # the twin: the J port at v5
    with pytest.raises(UnsupportedOptionError, match="sraging_profile_jax2_h"):
        _cmp(tester, method={"manifest": dict(LAB, method="sraging_profile_jax2_h"), "sets": MINE, "run_id": None,
                             "computed_on": "v5"})


def test_a_tester_run_is_read_from_its_pinned_file(tester):
    root, pins, run, st = tester
    seen = {}
    _cmp(tester, reference={"run_id": "ref", "store_root": st.root, "manifest": dict(run)}, seen=seen)  # the twin
    assert seen["ref_manifest"] == run
    with pytest.raises(UnsupportedOptionError, match="ref"):                     # a manifest that is not the file
        _cmp(tester, reference={"run_id": "ref", "store_root": st.root, "manifest": dict(run, method_version="2")})
    open(os.path.join(root, "runs", "ref.json"), "ab").write(b" ")
    with pytest.raises(UnsupportedOptionError, match="runs/ref.json"):           # the run file changed
        _cmp(tester)


def test_a_manifest_key_other_than_the_pinned_one_is_refused(tester):
    root, pins, run, st = tester
    other = dict(run, keys={"a": run["keys"]["b"], "b": run["keys"]["b"]})       # a's key points at b's record
    with pytest.raises(UnsupportedOptionError, match="(?s)ref/a.*key"):
        ADP.load_pinned_sets(_fake_mods({}), other, st.root, pins=pins, run_id="ref")
    got = ADP.load_pinned_sets(_fake_mods({}), run, st.root, pins=pins, run_id="ref")     # the twin
    assert [cs.cohort_id for cs in got] == ["a", "b"]
    open(st._path(run["keys"]["a"]), "ab").write(b" ")
    with pytest.raises(UnsupportedOptionError, match="ref/a"):                   # the record changed
        ADP.load_pinned_sets(_fake_mods({}), run, st.root, pins=pins, run_id="ref")


def test_sets_must_name_the_library_they_were_computed_on(tester):
    with pytest.raises(UnsupportedOptionError, match="computed_on"):
        _cmp(tester, method={"manifest": LAB, "sets": MINE, "run_id": None})
    _cmp(tester, method={"manifest": LAB, "sets": MINE, "run_id": None, "computed_on": "v2"},
         cohort_ids=["a"])                                                       # the twin: a's arrays are unchanged
    with pytest.raises(UnsupportedOptionError, match="(?s)v2.*v5.*'b'"):
        _cmp(tester, method={"manifest": LAB, "sets": MINE, "run_id": None, "computed_on": "v2"})


# ------------------------------------------------------------------------------------------- _h off its v2 pin
class _T:
    """Just enough of srbench.contract.types for make_jax2_method's class body."""
    class MethodDescriptor:
        def __init__(self, **kw):
            self.__dict__.update(kw)


def test_the_h_port_is_refused_off_its_v2_pin():
    with pytest.raises(UnsupportedOptionError, match="sraging_profile_jax2_h"):
        ADP.make_jax2_method(_T, "sraging_profile_jax2_h", library="v5")
    ADP.make_jax2_method(_T, "sraging_profile_jax2_h", library="v2")           # the twin
    ADP.make_jax2_method(_T, "sraging_profile_jax2_h")                           # history default: v2
    ADP.make_jax2_method(_T, "sraging_profile_jax2", library="v5")              # J is not refused


def test_run_records_refuses_the_h_port_off_v2():
    m = type("M", (), {"descriptor": type("D", (), {"name": "sraging_profile_jax2_h"})()})()
    with pytest.raises(UnsupportedOptionError, match="sraging_profile_jax2_h"):
        ADP.run_records({}, m, [], library="v5")


# ---------------------------------------------------------------------------------------------- history intact
def test_the_old_functions_keep_their_signatures():
    assert list(inspect.signature(ADP.compare).parameters) == ["mods", "port_manifest", "cohort_ids", "tester_root",
                                                               "port_store_root"]
    sig = inspect.signature(ADP.run_records)
    assert list(sig.parameters)[:6] == ["mods", "method", "cohort_ids", "tester_root", "store_root", "run_id"]
    assert sig.parameters["library"].default is None                           # additive; None keeps v1 (history)
    assert ADP.METHODS["sraging_profile_v033"]["library"] == "v1"
    assert ADP.METHODS["sraging_profile_jax2"]["library"] == "v2" == ADP.METHODS["sraging_profile_jax2_h"]["library"]


# ----------------------------------------------------------------------------------- no-peek and no-build scans
def _scanned():
    out = []
    for d, _, files in os.walk(os.path.join(REPO, "src")):
        out += [os.path.join(d, f) for f in files if f.endswith(".py")]
    t = os.path.join(REPO, "tools", "acceptance_003e.py")
    if os.path.exists(t):
        out.append(t)
    return out


def test_no_003e_path_reads_the_sealed_v6_run():
    sealed = "jax033" + "-v6"                       # spelled in two parts so this file does not match itself
    assert not [p for p in _scanned() if sealed in open(p, encoding="utf-8").read()]


def test_no_003e_path_calls_the_testers_library_build():
    bad = ("library.build", "library import build")
    assert not [p for p in _scanned() if any(b in open(p, encoding="utf-8").read() for b in bad)]
