"""The ports are verbatim: snapshot AST + the declared edits == port AST (spec 001 research R2; FR-011, FR-020;
spec 003 research R1, T005).

Each port directory's `VERBATIM.toml` declares, per ported module, its source in a frozen snapshot and the only
edits allowed. This test applies exactly those edits to the snapshot's AST and requires `ast.dump` equality with
the port, name by name (subset), module by module (whole), or statement by statement (logic_verbatim). The
planted-change tests prove the check can fail. Ports: `compat/v033` (v0.3.3), `compat/jax2` (jax2 1.0.0),
`compat/jax2_101` (jax2 1.0.1, spec 003b) and `compat/tester_f94ec2b` (the tester's `apply_window`, spec 006 R3, whose
snapshot manifest pins sha256s, not md5s). A last test proves the frozen ports and oracles never change.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import tomllib

import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMPAT = os.path.join(REPO, "src", "sraging", "compat")
PORTS = ("v033", "jax2", "jax2_101", "tester_f94ec2b")
SNAP2 = os.path.join(REPO, "tests", "oracle", "jax2_3bc5775")
SNAP3 = os.path.join(REPO, "tests", "oracle", "jax2_6884fc8")
SNAP_TESTER = os.path.join(REPO, "tests", "oracle", "tester_f94ec2b")
FROZEN = ("src/sraging/compat/v033", "src/sraging/compat/jax2", "tests/oracle/v033", "tests/oracle/jax2_3bc5775",
          "tests/oracle/fp_piecewise_cb191f5",
          # spec 006 (R1, R3): W is a class outside compat/, so 003b's port and both its snapshots stay frozen too
          "src/sraging/compat/jax2_101", "tests/oracle/jax2_6884fc8", "tests/oracle/tester_f94ec2b")


def decl(port):
    return tomllib.load(open(os.path.join(COMPAT, port, "VERBATIM.toml"), "rb"))["module"]


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def snapshot_md5s():
    """{repo-relative source path: md5} over both frozen snapshots' manifests."""
    out = {f"tests/oracle/v033/{f['repo']}/{f['path']}": f["md5"]
           for f in json.load(open(os.path.join(REPO, "tests", "oracle", "v033", "MANIFEST.json")))["files"]}
    for snap in (SNAP2, SNAP3):
        out.update({f"tests/oracle/{os.path.basename(snap)}/{f['path']}": f["md5"]
                    for f in json.load(open(os.path.join(snap, "MANIFEST.json")))["files"]})
    return out


def snapshot_sha256s():
    """{repo-relative source path: sha256} over the tester snapshot's manifest (spec 006 R3)."""
    return {f"tests/oracle/tester_f94ec2b/{f['path']}": f["sha256"]
            for f in json.load(open(os.path.join(SNAP_TESTER, "MANIFEST.json")))["files"]}


def _top(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out[n.targets[0].id] = n
    return out


def _edit_statements(node, fn):
    """Apply fn(stmt) -> list[stmt] to every statement list under `node`, recursively."""
    for field, value in ast.iter_fields(node):
        if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
            new = []
            for st in value:
                _edit_statements(st, fn)
                new.extend(fn(st))
            setattr(node, field, new)
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, ast.AST):
                    _edit_statements(v, fn)
        elif isinstance(value, ast.AST):
            _edit_statements(value, fn)


def _apply(node, edits):
    for e in edits:
        kind = e["kind"]
        if kind == "delete_statements":
            drop = set(e["match"])
            _edit_statements(node, lambda st: [] if ast.unparse(st) in drop else [st])
        elif kind in ("import_rewrite", "replace_statement"):
            old, new = e["old"], ast.parse(e["new"]).body
            _edit_statements(node, lambda st: list(new) if ast.unparse(st) == old else [st])
        elif kind == "note":
            pass
        else:
            raise AssertionError(f"undeclared edit kind {kind!r}")
    return node


def compare(entry, source_text, port_text):
    """Return a list of mismatch descriptions (empty = verbatim)."""
    src, port = ast.parse(source_text), ast.parse(port_text)
    mode, bad = entry["mode"], []
    if mode == "subset":
        s_top, p_top = _top(src), _top(port)
        for name in entry["names"]:
            if name not in p_top:
                bad.append(f"{name}: missing from port")
                continue
            s_node = _apply(s_top[name], [e for e in entry.get("edit", []) if e.get("name") in (None, name)])
            if ast.dump(s_node) != ast.dump(p_top[name]):
                bad.append(f"{name}: differs from the snapshot after the declared edits")
        extra = set(p_top) - set(entry["names"])
        if extra:
            bad.append(f"port defines undeclared names: {sorted(extra)}")
    elif mode == "whole":
        if ast.dump(_apply(src, entry.get("edit", []))) != ast.dump(port):
            bad.append("module differs from the snapshot after the declared edits")
    elif mode == "logic_verbatim":
        s_top, p_top = _top(src), _top(port)
        for name in entry["names"]:
            if ast.dump(s_top[name]) != ast.dump(p_top.get(name, ast.Pass())):
                bad.append(f"{name}: differs")
        cls_name, meth = entry["source_function"].split(".")
        meth_node = next(n for n in s_top[cls_name].body if isinstance(n, ast.FunctionDef) and n.name == meth)
        a, b = entry["source_lines"]
        s_stmts = [st for st in meth_node.body if st.lineno >= a and st.end_lineno <= b]
        p_body = p_top[entry["function"]].body
        p_stmts = [st for st in p_body if not (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant))][:-1]
        if [ast.dump(s) for s in s_stmts] != [ast.dump(p) for p in p_stmts]:
            bad.append(f"{entry['function']}: body differs from {entry['source_function']} lines {a}-{b}")
    elif mode == "class_constants":                 # spec 003: the jax2 classes' constant assignments only
        s_top, p_top = _top(src), _top(port)
        consts = set(entry["constants"])
        def kept(cls):
            return [ast.dump(st) for st in cls.body if isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name) and st.targets[0].id in consts]
        for name in entry["names"]:
            if name not in p_top:
                bad.append(f"{name}: missing from port")
                continue
            sc, pc = s_top[name], p_top[name]
            if [ast.dump(b) for b in sc.bases] != [ast.dump(b) for b in pc.bases]:
                bad.append(f"{name}: bases differ")
            if kept(sc) != kept(pc) or len(kept(pc)) != len(pc.body):
                bad.append(f"{name}: constants differ from the snapshot (or the port class holds anything else)")
        extra = set(p_top) - set(entry["names"])
        if extra:
            bad.append(f"port defines undeclared names: {sorted(extra)}")
    else:
        bad.append(f"unknown mode {mode!r}")
    return bad


ENTRIES = [(port, e) for port in PORTS for e in decl(port)]


def test_sources_are_the_frozen_snapshots():
    manifest, sha = snapshot_md5s(), snapshot_sha256s()
    for port, entry in ENTRIES:
        path = os.path.join(REPO, entry["source"])
        if "source_sha256" in entry:                    # the tester snapshot pins sha256s (spec 006 R3)
            assert entry["source"] in sha, (port, entry["source"])
            assert _sha256(path) == sha[entry["source"]] == entry["source_sha256"], (port, entry["source"])
            continue
        assert entry["source"] in manifest, (port, entry["source"])
        assert _md5(path) == manifest[entry["source"]] == entry["source_md5"], (port, entry["source"])


@pytest.mark.parametrize("snap,version", [(SNAP2, "1.0.0"), (SNAP3, "1.0.1")])
def test_the_jax2_snapshot_is_the_code_its_version_lock_records(snap, version):
    """Spec 003 R1: the runtime lock guard is a declared deletion, so this test carries its check."""
    pkg = os.path.join(snap, "mle_profile_jax2")
    lock = json.load(open(os.path.join(pkg, "VERSION_LOCK.json")))
    for name, ent in lock["names"].items():
        assert ent["version"] == version, name
        assert {f: _sha256(os.path.join(pkg, f)) for f in ent["sha256"]} == ent["sha256"], name


@pytest.mark.parametrize("port,entry", ENTRIES, ids=[f"{p}:{e['port']}" for p, e in ENTRIES])
def test_port_is_verbatim(port, entry):
    source_text = open(os.path.join(REPO, entry["source"])).read()
    port_text = open(os.path.join(COMPAT, port, entry["port"])).read()
    assert compare(entry, source_text, port_text) == []


@pytest.mark.parametrize("port,extra", [("v033", {"runtime.py"}), ("jax2", set()), ("jax2_101", set()),
                                        ("tester_f94ec2b", set())])
def test_every_port_module_is_declared(port, extra):
    declared = {e["port"] for e in decl(port)} | {"__init__.py"} | extra
    present = {f for f in os.listdir(os.path.join(COMPAT, port)) if f.endswith(".py")}
    assert present == declared


@pytest.mark.parametrize("port,module,old,new", [
    ("v033", "fp_likelihood.py", "tm / float(target_steps)", "tm / float(target_steps + 1)"),   # arithmetic
    ("v033", "fp_likelihood.py", "1.15 * float(np.max(t))", "1.16 * float(np.max(t))"),         # a constant
    ("jax2", "claims.py", "SC_EDGE_DEX = 0.01", "SC_EDGE_DEX = 0.02"),                          # a constant
    ("jax2", "claims.py", "kw['lower'] = float(max(lo, BOUNDS[p][0]))", "kw['lower'] = float(min(lo, BOUNDS[p][0]))"),
    ("jax2", "estimator.py", "return True   #", "return False   #"),                         # the fixed policy
    ("jax2", "settings.py", "DT_RULE = False", "DT_RULE = True"),                              # a class constant
    ("jax2_101", "claims.py", "SC_EDGE_DEX = 0.01", "SC_EDGE_DEX = 0.02"),                      # spec 003b
    ("jax2_101", "claims.py", "ESTIMATE_SCHEMA = 1", "ESTIMATE_SCHEMA = 2"),                    # estimate_block's
    ("jax2_101", "estimator.py", "return True   #", "return False   #"),
    ("jax2_101", "settings.py", "BOX_FLAG = False", "BOX_FLAG = True"),                         # a 1.0.1 constant
    ("tester_f94ec2b", "observation.py", "past = ages > hi", "past = ages >= hi"),             # spec 006: the edge
    ("tester_f94ec2b", "observation.py", "keep = ages >= lo", "keep = ages > lo"),
])
def test_check_detects_a_one_token_change(port, module, old, new):
    """FR-020 (spec 001) and T005 (spec 003): the verbatim check must FAIL on a planted perturbation."""
    entry = next(e for e in decl(port) if e["port"] == module)
    source_text = open(os.path.join(REPO, entry["source"])).read()
    port_text = open(os.path.join(COMPAT, port, module)).read()
    if old not in port_text:                      # the claim builders are written with double quotes
        old, new = old.replace("'", '"'), new.replace("'", '"')
    assert port_text.count(old) == 1, (module, old)
    assert compare(entry, source_text, port_text.replace(old, new)) != []


def test_the_three_unchanged_sources_port_to_the_same_code():
    """Spec 003b R1: fp_jax.py and observation_model.py are byte-identical between 1.0.0 and 1.0.1 (handoff section 3),
    and fp_curvefit_h comes from the same SRK file, so the two packages hold the same code (banners aside)."""
    def code(path):
        tree = ast.parse(open(path).read())
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            tree.body = tree.body[1:]                     # the generated module docstring names its own source
        return ast.dump(tree)
    for f in ("fp_jax.py", "observation_model.py", "fp_curvefit_h.py"):
        assert code(os.path.join(COMPAT, "jax2", f)) == code(os.path.join(COMPAT, "jax2_101", f)), f


def frozen_changed(repo, base, paths):
    """The FROZEN paths that differ between `base` and the working tree (tracked or not); [] = unchanged."""
    r = subprocess.run(["git", "-C", repo, "diff", "--name-only", base, "--", *paths], capture_output=True, text=True)
    u = subprocess.run(["git", "-C", repo, "ls-files", "--others", "--exclude-standard", "--", *paths],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return sorted(set(r.stdout.split()) | set(u.stdout.split()))


def test_the_frozen_ports_and_oracles_never_change():
    """Constitution VI: spec 003b adds a port and snapshots; it never edits compat/jax2, compat/v033 or an existing
    oracle. Compared with `main`, the branch every spec starts from."""
    assert frozen_changed(REPO, "main", FROZEN) == []


def test_the_frozen_check_can_fail(tmp_path):
    repo = str(tmp_path / "r")
    os.makedirs(os.path.join(repo, "frozen"))
    open(os.path.join(repo, "frozen", "a.py"), "w").write("x = 1\n")
    for a in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "a"]):
        subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True)
    assert frozen_changed(repo, "HEAD", ["frozen"]) == []                       # the twin
    open(os.path.join(repo, "frozen", "a.py"), "a").write("y = 2\n")           # planted
    assert frozen_changed(repo, "HEAD", ["frozen"]) == ["frozen/a.py"]

