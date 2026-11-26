"""Build a frozen srbench snapshot from git objects (spec 003b, T004; contracts/oracle-snapshots.md; permitted by Naveh,
2026-10-19): every file of `<commit>:srbench`, checked against a `sha256  path` sums file before anything is written.

    .venv/bin/python tools/snapshot_srbench.py --repo <jax2 repo> --commit 6884fc8 \\
        --sums <freeze>/reports/freeze101/snapshot/SRBENCH_FILES.sha256 --dest tests/oracle/tester_f94ec2b

Reads the repository's object store only (git rev-parse / ls-tree / show); never its working tree; writes nothing
there. Once built the snapshot is frozen: never edit it.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys


class Refused(RuntimeError):
    """The snapshot cannot be built faithfully; the message says why."""


def _git(repo, *args, binary=False):
    r = subprocess.run(["git", "--no-optional-locks", "-C", repo, *args], capture_output=True, text=not binary)
    if r.returncode:
        raise Refused(f"git {' '.join(args)} failed in {repo}: {(r.stderr if not binary else r.stderr.decode()).strip()}")
    return r.stdout if binary else r.stdout.strip()


def build(*, repo, commit, sums, snap, expect_tree, expect_sums_sha256, expect_listing) -> dict:
    if os.path.exists(snap):
        raise Refused(f"{snap} exists (the snapshot is frozen; never rebuild or edit it)")
    raw = open(sums, "rb").read()
    if hashlib.sha256(raw).hexdigest() != expect_sums_sha256:
        raise Refused(f"SRBENCH_FILES sums file {sums} differs from the expected sha256")
    rows = [ln for ln in raw.decode().splitlines() if ln and not ln.startswith("#")]
    digest = hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest()
    if digest != expect_listing:
        raise Refused(f"the sums file's listing digest is {digest[:12]}, not {expect_listing[:12]}")
    want = {}
    for ln in rows:
        h, path = ln.split("  ", 1)
        want[path] = h
    tree = _git(repo, "rev-parse", f"{commit}:srbench")
    if tree != expect_tree:
        raise Refused(f"the srbench tree at {commit[:7]} is {tree[:12]}, not {expect_tree[:12]}")
    have = [line.split("\t")[1] for line in _git(repo, "ls-tree", "-r", commit, "srbench").splitlines()]
    extra, missing = sorted(set(have) - set(want)), sorted(set(want) - set(have))
    if extra or missing:
        raise Refused(f"the tree and the sums file disagree: extra {extra}, missing {missing}")
    data = {}
    for path in have:
        b = _git(repo, "show", f"{commit}:{path}", binary=True)
        if hashlib.sha256(b).hexdigest() != want[path]:
            raise Refused(f"{path} at {commit[:7]} differs from its line in the sums file")
        data[path] = b
    files = []
    for path, b in data.items():
        out = os.path.join(snap, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(b)
        files.append({"path": path, "sha256": hashlib.sha256(b).hexdigest()})
    manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                "purpose": "The srbench tree jax2 1.0.1 runs against (spec 003b). NEVER EDIT.",
                "repo": repo, "commit": commit, "tree": tree, "read_from": "git objects",
                "sums_file": os.path.basename(sums), "sums_sha256": expect_sums_sha256, "listing_digest": digest,
                "files": files, "host": {"node": platform.node(), "machine": platform.machine()}}
    json.dump(manifest, open(os.path.join(snap, "MANIFEST.json"), "w"), indent=1)
    open(os.path.join(snap, "README.md"), "w").write(
        f"# Oracle snapshot `{os.path.basename(snap)}/` — NEVER EDIT\n\nThe srbench tree `{tree}` at commit `{commit}`, "
        f"read from git objects and checked against `{os.path.basename(sums)}` (sha256 {expect_sums_sha256}; listing "
        f"digest {digest}): {len(files)} files (spec 003b; permitted by Naveh, 2026-10-19).\n\nIt is the srbench "
        "`tools/oracle_jax2_101.py` puts on `sys.path` for the side-by-side 1.0.1 oracle. It may only be replaced by a "
        "new versioned directory, never edited.\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for a in ("--repo", "--commit", "--sums", "--dest", "--tree", "--sums-sha256", "--listing"):
        ap.add_argument(a, required=True)
    a = ap.parse_args(argv)
    try:
        m = build(repo=a.repo, commit=a.commit, sums=a.sums, snap=a.dest, expect_tree=a.tree,
                  expect_sums_sha256=a.sums_sha256, expect_listing=a.listing)
    except Refused as ex:
        sys.exit(f"refusing: {ex}")
    print(f"wrote {a.dest}: {len(m['files'])} files, tree {m['tree'][:12]}")


if __name__ == "__main__":
    main()
