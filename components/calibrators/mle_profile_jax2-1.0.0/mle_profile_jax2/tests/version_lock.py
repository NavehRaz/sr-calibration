"""VERSION_LOCK.json: registered name -> {version, sha256 of every runtime .py file of the package}.

WHY. `version` is part of the result-store key, so a behaviour change under an unchanged version makes the
store serve the old result SILENTLY -- the one quiet failure of the method contract. The lock turns it loud.

THE RULE. If a name's version equals its locked version, every runtime file's sha256 must equal the locked
one; otherwise the code changed but the version did not, and the check FAILS. The two names share every
runtime file, so ANY code change needs BOTH versions bumped (conservative on purpose: proving that a change
leaves one name's behaviour bit-identical is the identity harness's job, and is not encoded here).
"Runtime files" = every *.py directly in the package directory; tests/ and fixtures are not runtime.

usage (Python only, stdlib only; no import of the package, which would register it):
    python version_lock.py            check; exit 1 and print every violation
    python version_lock.py --write    rewrite the lock after a version bump. REFUSES if any name's code
                                      changed while its version did not.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import sys

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = os.path.join(PKG, "VERSION_LOCK.json")
RULE = ("If a name's version equals its locked version, every runtime file's sha256 must equal the locked "
        "one. The names share every runtime file, so any code change needs every version bumped. "
        "Regenerate with: python tests/version_lock.py --write (refuses on a code change under an "
        "unchanged version).")


def runtime_files(pkg=PKG):
    return sorted(f for f in os.listdir(pkg) if f.endswith(".py") and os.path.isfile(os.path.join(pkg, f)))


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def versions(pkg=PKG):
    """The literal `VERSIONS = {...}` dict of the package's __init__.py, read without importing it."""
    tree = ast.parse(open(os.path.join(pkg, "__init__.py")).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VERSIONS"
                                                for t in node.targets):
            return dict(ast.literal_eval(node.value))
    raise LookupError("no literal VERSIONS = {...} at the top level of __init__.py")


def current(pkg=PKG):
    files = {f: sha256(os.path.join(pkg, f)) for f in runtime_files(pkg)}
    return dict(package=os.path.basename(pkg), files=sorted(files), rule=RULE,
                names={n: dict(version=v, sha256=dict(files)) for n, v in sorted(versions(pkg).items())})


def problems(lock=None, cur=None):
    """Every violation, as text ([] = the lock is consistent with the code and the versions)."""
    cur = current() if cur is None else cur
    if lock is None:
        if not os.path.exists(LOCK):
            return [f"no lock file at {LOCK}; run: python tests/version_lock.py --write"]
        lock = json.load(open(LOCK))
    out = []
    ln, cn = set(lock.get("names", {})), set(cur["names"])
    if ln != cn:
        out.append(f"registered names {sorted(cn)} != locked names {sorted(ln)}: regenerate the lock")
    for name in sorted(cn & ln):
        ent, now = lock["names"][name], cur["names"][name]
        if ent["version"] == now["version"]:
            diff = sorted(set(ent["sha256"]) ^ set(now["sha256"]) |
                          {f for f in set(ent["sha256"]) & set(now["sha256"]) if ent["sha256"][f] != now["sha256"][f]})
            if diff:
                out.append(f"CODE CHANGED BUT VERSION DID NOT: {name} is still {now['version']} but "
                           f"{diff} changed since the lock. Bump the version (all names share the code).")
        else:
            out.append(f"lock is stale: {name} {ent['version']} -> {now['version']}; run "
                       "python tests/version_lock.py --write")
    return out


def write():
    cur = current()
    if os.path.exists(LOCK):
        bad = [p for p in problems(cur=cur) if p.startswith("CODE CHANGED")]
        if bad:
            raise SystemExit("REFUSING to rewrite the lock:\n  " + "\n  ".join(bad))
    tmp = LOCK + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cur, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, LOCK)
    return cur


if __name__ == "__main__":
    if "--write" in sys.argv[1:]:
        c = write()
        print(f"wrote {LOCK}: " + ", ".join(f"{n} {e['version']}" for n, e in c["names"].items()))
    else:
        p = problems()
        print("\n".join(p) if p else "VERSION_LOCK ok")
        sys.exit(1 if p else 0)
