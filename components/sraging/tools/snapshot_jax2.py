"""Build the frozen jax2 1.0.0 oracle snapshot (spec 003, tasks T001-T002; research R2;
contracts/oracle-snapshot.md). Permitted by Naveh on 2026-10-05.

Run ONCE:  .venv/bin/python tools/snapshot_jax2.py

Writes `tests/oracle/jax2_3bc5775/`: byte copies of the four method files and VERSION_LOCK.json at the frozen
commit, MANIFEST.json and README.md. The SRK and srbench files the fit loads are hash-equal to
`tests/oracle/v033` and are CITED (with sha256), not copied: the SRK list is the tester's own record of the
files a 1.0.0 fit loaded (`diagnostics.srk_sha` of a stored identity fit), and every srbench file of the v033
snapshot is compared with J2's copy. Once built the snapshot is frozen: never edit it. The builder refuses to
overwrite one, and reads J2 only.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J2 = os.environ.get("SRAGING_J2_ROOT", os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset",
                                                  "sr_calibration_tester_jax2"))   # release 0.1.1
SNAP = os.path.join(REPO, "tests", "oracle", "jax2_3bc5775")
V033 = os.path.join(REPO, "tests", "oracle", "v033")
FROZEN = "3bc57751a6617e5bbdc76205e58d158c9d434990"
METHOD = "methods/mle_profile_jax2"
PY = ("__init__.py", "estimator.py", "fp_jax.py", "observation_model.py")
LOCK = "VERSION_LOCK.json"
NAMES = ("mle_profile_jax2", "mle_profile_jax2_h")
IDENTITY = "reports/rc2/C/freeze_1.0.0/identity/pure-k0.01-p02-c1-s1__jax2__v100on.json"


class Refused(RuntimeError):
    """The snapshot cannot be built faithfully; the message says why."""


def sha256(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _git(repo, *args, check=True):
    r = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and r.returncode:
        raise Refused(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
    return r


def _find(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find(v, key)
            if r is not None:
                return r
    return None


def build(*, j2=J2, snap=SNAP, v033=V033, frozen=FROZEN) -> dict:
    """Build the snapshot and return its manifest; raise `Refused` on any doubt (nothing is written then)."""
    if os.path.exists(snap):
        raise Refused(f"{snap} exists (the snapshot is frozen; never rebuild or edit it)")
    meth = os.path.join(j2, METHOD)

    # 1. J2's method directory is exactly the frozen commit's, and clean
    _git(j2, "cat-file", "-e", f"{frozen}^{{commit}}")
    if _git(j2, "merge-base", "--is-ancestor", frozen, "HEAD", check=False).returncode:
        raise Refused(f"J2's HEAD does not contain the frozen commit {frozen[:7]}")
    if _git(j2, "status", "--porcelain", "--", METHOD).stdout.strip():
        raise Refused(f"J2's {METHOD} is dirty")
    if _git(j2, "diff", "--quiet", frozen, "HEAD", "--", METHOD, check=False).returncode:
        raise Refused(f"J2's {METHOD} at HEAD differs from the frozen commit {frozen[:7]}")
    head = _git(j2, "rev-parse", "HEAD").stdout.strip()

    # 2. the files match VERSION_LOCK.json for both names
    lock = json.load(open(os.path.join(meth, LOCK)))
    now = {f: sha256(os.path.join(meth, f)) for f in PY}
    for name in NAMES:
        locked = lock["names"][name]["sha256"]
        bad = sorted(f for f in PY if locked.get(f) != now[f])
        if bad or set(locked) != set(PY):
            raise Refused(f"{name}: {bad or sorted(set(locked) ^ set(PY))} do not match VERSION_LOCK.json")

    # 3. the cited files are hash-equal to tests/oracle/v033
    cited = []
    srk = _find(json.load(open(os.path.join(j2, IDENTITY))), "srk_sha")["files"]
    for rel, h in sorted(srk.items()):
        p = os.path.join(v033, "SRtools_kappa", "SRK", rel)
        if not os.path.exists(p) or sha256(p) != h:
            raise Refused(f"SRK/{rel} loaded by the 1.0.0 fit differs from tests/oracle/v033's copy")
        cited.append({"repo": "SRtools_kappa", "path": f"SRK/{rel}", "sha256": h,
                      "v033_path": f"SRtools_kappa/SRK/{rel}", "source": "srk_sha of " + IDENTITY})
    v033_manifest = json.load(open(os.path.join(v033, "MANIFEST.json")))
    for f in v033_manifest["files"]:
        if f["repo"] != "sr_calibration_tester" or not f["path"].startswith("srbench/"):
            continue
        a, b = os.path.join(v033, "sr_calibration_tester", f["path"]), os.path.join(j2, f["path"])
        if not os.path.exists(b) or sha256(a) != sha256(b):
            raise Refused(f"{f['path']} differs between J2 and tests/oracle/v033")
        cited.append({"repo": "sr_calibration_tester", "path": f["path"], "sha256": sha256(a),
                      "v033_path": f"sr_calibration_tester/{f['path']}", "source": "J2 working tree"})

    # 4. copy byte for byte and verify
    dest = os.path.join(snap, "mle_profile_jax2")
    os.makedirs(dest)
    files = []
    for f in PY + (LOCK,):
        shutil.copyfile(os.path.join(meth, f), os.path.join(dest, f))
        if md5(os.path.join(dest, f)) != md5(os.path.join(meth, f)):
            raise Refused(f"copy of {f} does not match its source")
        files.append({"path": f"mle_profile_jax2/{f}", "md5": md5(os.path.join(dest, f)),
                      "sha256": sha256(os.path.join(dest, f))})
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "purpose": ("Byte-level oracle for spec 003's port of mle_profile_jax2 / mle_profile_jax2_h 1.0.0 (the tester's "
                    "successor candidate). NEVER EDIT. Compare side by side on the same host only."),
        "source": {"repo": j2, "method_dir": METHOD, "frozen_commit": frozen, "head": head, "diff_empty": True},
        "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()},
        "python": sys.version,
        "files": files,
        "cited": cited,
        "environment_pins": "tests/oracle/v033/pip_freeze.txt (the same pinned environment)",
        "version_lock_check": now,
    }
    json.dump(manifest, open(os.path.join(snap, "MANIFEST.json"), "w"), indent=1)
    open(os.path.join(snap, "README.md"), "w").write(README.format(frozen=frozen, n_srk=len(srk),
                                                                     n_srb=len(cited) - len(srk)))
    return manifest


def _blob(repo, commit, path) -> bytes:
    """The bytes of `path` at `commit`, from git's object store (never a working tree). Raises Refused if absent."""
    r = subprocess.run(["git", "--no-optional-locks", "-C", repo, "show", f"{commit}:{path}"], capture_output=True)
    if r.returncode:
        raise Refused(f"{path} is missing at {commit[:7]}")
    return r.stdout


def _gitq(repo, *args):
    r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, text=True)
    if r.returncode:
        raise Refused(f"git {' '.join(args)} failed in {repo}: {r.stderr.strip()}")
    return r.stdout.strip()


def build_git(*, repo, commit, tag, snap, expect, v033=V033, srk_record, srbench_snapshot, names=None) -> dict:
    """Spec 003b: build a method snapshot from git OBJECTS at `commit` (the tag's), checking every byte before
    writing anything. `expect`: {"runtime": {file: sha256}, "version_lock": sha256, "listing": the sha256 of the
    method subtree's `sha256  path` lines (git's order, each newline-terminated)}. SRK files are cited from
    `srk_record`'s srk_sha (hash-equal to tests/oracle/v033); the srbench tree is cited from `srbench_snapshot`'s
    MANIFEST. Raises Refused on any doubt; nothing is written then."""
    if os.path.exists(snap):
        raise Refused(f"{snap} exists (the snapshot is frozen; never rebuild or edit it)")
    tagged = _gitq(repo, "rev-parse", f"{tag}^{{commit}}")
    if tagged != commit:
        raise Refused(f"the tag {tag} points to {tagged[:7]}, not to {commit[:7]}")
    files = list(expect["runtime"])
    data = {f: _blob(repo, commit, f"{METHOD}/{f}") for f in files}
    for f, b in data.items():
        if hashlib.sha256(b).hexdigest() != expect["runtime"][f]:
            raise Refused(f"{f} at {commit[:7]} differs from the expected sha256")
    lock_b = _blob(repo, commit, f"{METHOD}/{LOCK}")
    if hashlib.sha256(lock_b).hexdigest() != expect["version_lock"]:
        raise Refused(f"{LOCK} at {commit[:7]} differs from the expected sha256")
    lock = json.loads(lock_b)
    for name, ent in lock["names"].items():
        if names is not None and name not in names:
            continue
        bad = sorted(f for f in files if ent["sha256"].get(f) != hashlib.sha256(data[f]).hexdigest())
        if bad or set(ent["sha256"]) != set(files):
            raise Refused(f"VERSION_LOCK: {name} does not pin the files ({bad or sorted(set(ent['sha256']) ^ set(files))})")
    rows = []
    for line in _gitq(repo, "ls-tree", "-r", commit, METHOD).splitlines():
        meta, path = line.split("\t")
        rows.append(f"{hashlib.sha256(_blob(repo, commit, path)).hexdigest()}  {path}")
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()
    if digest != expect["listing"]:
        raise Refused(f"the listing digest of {METHOD} at {commit[:7]} is {digest[:12]}, not {expect['listing'][:12]}")
    cited = []
    srk = _find(json.load(open(srk_record)), "srk_sha")["files"]
    for rel, h in sorted(srk.items()):
        p = os.path.join(v033, "SRtools_kappa", "SRK", rel)
        if not os.path.exists(p) or sha256(p) != h:
            raise Refused(f"SRK/{rel} loaded by the fit differs from tests/oracle/v033's copy")
        cited.append({"repo": "SRtools_kappa", "path": f"SRK/{rel}", "sha256": h,
                      "v033_path": f"SRtools_kappa/SRK/{rel}", "source": "srk_sha of " + os.path.basename(srk_record)})
    srb_man = os.path.join(srbench_snapshot, "MANIFEST.json")
    if not os.path.exists(srb_man):
        raise Refused(f"the srbench snapshot {srbench_snapshot} has no MANIFEST.json (build it first)")
    srb = json.load(open(srb_man))
    dest = os.path.join(snap, "mle_profile_jax2")
    os.makedirs(dest)
    out = []
    for f, b in list(data.items()) + [(LOCK, lock_b)]:
        open(os.path.join(dest, f), "wb").write(b)
        out.append({"path": f"mle_profile_jax2/{f}", "md5": md5(os.path.join(dest, f)),
                    "sha256": sha256(os.path.join(dest, f))})
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "purpose": "Byte-level oracle for spec 003b's port of jax2 (see README). NEVER EDIT. Compare side by side on "
                   "the same host only.",
        "source": {"repo": repo, "method_dir": METHOD, "commit": commit, "tag": tag,
                   "tag_object": _gitq(repo, "rev-parse", tag), "read_from": "git objects"},
        "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()},
        "python": sys.version,
        "files": out,
        "version_lock_sha256": expect["version_lock"],
        "listing_digest": digest,
        "listing": rows,
        "cited": cited,
        "srk_record": {"file": os.path.basename(srk_record), "sha256": sha256(srk_record)},
        "cited_srbench": {"snapshot": os.path.basename(srbench_snapshot), "tree": srb.get("tree"),
                          "listing_digest": srb.get("listing_digest")},
        "environment_pins": "tests/oracle/v033/pip_freeze.txt; the handoff's section 5 environment",
    }
    json.dump(manifest, open(os.path.join(snap, "MANIFEST.json"), "w"), indent=1)
    open(os.path.join(snap, "README.md"), "w").write(README_GIT.format(
        name=os.path.basename(snap), tag=tag, commit=commit, n_srk=len(cited), srb=os.path.basename(srbench_snapshot)))
    return manifest


README_GIT = """# Oracle snapshot `{name}/` — NEVER EDIT

`mle_profile_jax2` and its sibling names at tag `{tag}`, commit `{commit}`, read from git objects (spec 003b; the
snapshot permitted by Naveh, 2026-10-19).

- `mle_profile_jax2/` holds the lock and the runtime files, byte for byte as the commit holds them.
- `MANIFEST.json` gives each file's md5 and sha256, the method subtree's listing (sha256 of every file, tests and
  fixtures included) and its digest, the {n_srk} SRK files the fit loads (hash-equal to `tests/oracle/v033`, used from
  there), and the srbench tree, used from `tests/oracle/{srb}`.

Load it with `tools/oracle_jax2_101.py` and compare the port side by side, in the same process, on the same host, in
the pinned environment. It may only be replaced by a new versioned directory, never edited.
"""


README = """# Oracle snapshot `jax2_3bc5775/` — NEVER EDIT

The tester's successor candidate `mle_profile_jax2` and `mle_profile_jax2_h` 1.0.0, frozen at
`sr_calibration_tester_jax2` commit `{frozen}` (spec 003; permitted by Naveh, 2026-10-05).

- `mle_profile_jax2/` holds byte copies of the four method files and `VERSION_LOCK.json`.
- `MANIFEST.json` gives each file's md5 and sha256, the source commit and HEAD, and the host. It also cites the
  {n_srk} SRK and {n_srb} srbench files the fit loads, which are hash-equal to `tests/oracle/v033` and are used
  from there, not copied.

Load it with `tools/oracle_jax2.py` (v033's srbench and SRK on `sys.path`, this package by path under
`srbench_methods.mle_profile_jax2`) and compare the port side by side in the same process, on the same
host, in the pinned environment. It may only be replaced by a new versioned directory, never edited.
"""


def main(argv=None):
    """No arguments: spec 003's 1.0.0 build, unchanged. `--git`: spec 003b's build from git objects, the expected
    sha256s read from a TOML table (tests/acceptance/panel_003b.toml [pins])."""
    import argparse
    import tomllib
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--git", action="store_true")
    for a in ("--repo", "--commit", "--tag", "--dest", "--pins", "--srk-record", "--srbench-snapshot"):
        ap.add_argument(a)
    a = ap.parse_args(argv)
    try:
        if not a.git:
            m = build()
            dest = SNAP
        else:
            pins = tomllib.load(open(a.pins, "rb"))["pins"]
            expect = {"runtime": dict(pins["runtime"]), "version_lock": pins["version_lock_sha256"],
                      "listing": pins["listing_digest"]}
            m = build_git(repo=a.repo, commit=a.commit, tag=a.tag, snap=a.dest, expect=expect,
                          srk_record=a.srk_record, srbench_snapshot=a.srbench_snapshot)
            dest = a.dest
    except Refused as ex:
        sys.exit(f"refusing: {ex}")
    print(f"wrote {dest}: {len(m['files'])} files, {len(m['cited'])} cited")


if __name__ == "__main__":
    main()
