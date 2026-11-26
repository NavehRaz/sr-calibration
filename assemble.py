"""Assemble the release from VERSIONS.toml. Read-only on every source repository.

  python assemble.py plan                 # list components and data with their resolved commits and sizes; writes nothing
  python assemble.py code                 # extract the included code components into components/ (tracked here)
  python assemble.py data --out DIR       # stage the data deposit into DIR (outside this repository) + DATA_MANIFEST.json
  python assemble.py pack --out DIR       # zip the staged deposit into DIR/upload/: one zip per data entry + manifest,
                                          #   licence and README (Zenodo takes at most 100 files per record)
  python assemble.py lock                 # write MANIFEST.lock.json: resolved commit, archive sha256 and file count per component

Sources are read with `git archive` (committed files at a pinned ref) or, for files a repository does not track, copied
from its working tree. Nothing is ever written to a source repository. Run with Python >= 3.11 (tomllib).
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_LICENCE = """The data in this deposit ({name} {version}) are licensed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/
Attribution: {authors}, {name} {version} (cite the Zenodo version DOI of this deposit and the methods preprint).
The code that produced and reads them is licensed separately under the BSD 3-Clause licence (software record).
"""
# the authors' aging folder (`aging:` in VERSIONS.toml): this repository is <aging>/code_5/sr_methods_release
AGING = os.environ.get("SR_AGING", os.path.dirname(os.path.dirname(HERE))).rstrip("/") + "/"


def resolve(path: str) -> str:
    if path.startswith("aging:"):
        return AGING + path[len("aging:"):]
    return os.path.expanduser(path)


def git(repo: str, *args: str, binary: bool = False):
    out = subprocess.run(["git", "-C", resolve(repo), *args], capture_output=True, check=True)
    return out.stdout if binary else out.stdout.decode().strip()


def archive(repo: str, ref: str, paths: list[str]) -> bytes:
    return git(repo, "archive", "--format=tar", ref, "--", *paths, binary=True)


def excluded(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def extract(tar_bytes: bytes, dest: str, strip: str = "", exclude: list[str] | None = None) -> int:
    n = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if exclude and excluded(m.name, exclude):
                continue
            if strip and m.name.startswith(strip):
                m.name = m.name[len(strip):]
            if not m.name:
                continue
            tf.extract(m, dest, filter="data")
            n += m.isfile()
    return n


def sync_tree(src: str, dest: str) -> int:
    """Make `dest` equal to `src`, touching only files whose bytes differ and removing files `src` lacks."""
    changed = 0
    want = set()
    for d, _, fs in os.walk(src):
        for f in fs:
            rel = os.path.relpath(os.path.join(d, f), src)
            want.add(rel)
            s, t = os.path.join(src, rel), os.path.join(dest, rel)
            if os.path.isfile(t) and open(s, "rb").read() == open(t, "rb").read():
                continue
            os.makedirs(os.path.dirname(t), exist_ok=True)
            shutil.copy2(s, t)
            changed += 1
    for d, _, fs in os.walk(dest, topdown=False):
        for f in fs:
            if os.path.relpath(os.path.join(d, f), dest) not in want:
                os.remove(os.path.join(d, f))
                changed += 1
        if d != dest and not os.listdir(d):
            os.rmdir(d)
    return changed


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load() -> dict:
    with open(os.path.join(HERE, "VERSIONS.toml"), "rb") as f:
        return tomllib.load(f)


def store_files(entry: dict) -> list[tuple[str, str]]:
    """(source path, relative destination) for every record of the listed tester runs."""
    root = resolve(entry["repo"])
    out = []
    for run in entry["runs"]:
        keys = json.load(open(os.path.join(root, "runs", run + ".json")))["keys"]   # cohort id -> record hash
        for cohort, k in keys.items():
            src = os.path.join(root, "data", "store", k[:2], k[2:4], k + ".json")
            out.append((src, os.path.join(run, cohort + ".json")))
    return out


def working_tree_files(entry: dict) -> list[tuple[str, str]]:
    root = resolve(entry["repo"])
    strip = entry.get("strip", "")
    out = []
    for p in entry["paths"]:
        full = os.path.join(root, p)
        walk = [(full, None)] if os.path.isfile(full) else [(os.path.join(d, f), None) for d, _, fs in os.walk(full) for f in fs]
        for f, _ in walk:
            rel = os.path.relpath(f, root)
            out.append((f, rel[len(strip):] if strip and rel.startswith(strip) else rel))
    return out


def cfg_component(cfg: dict, cid: str) -> dict:
    return next(c for c in cfg["component"] if c["id"] == cid)


def paper_inputs(cfg: dict) -> dict:
    """The list of the paper's inputs that go into the data deposit (release_inputs.json at the paper's ref)."""
    p = cfg_component(cfg, "paper")
    return json.loads(git(p["repo"], "show", f"{p['ref']}:release_inputs.json"))


def cmd_pack(cfg: dict, out: str) -> None:
    """Zip the staged deposit (from `data --out DIR`): one zip per data entry, so the record stays under Zenodo's
    100-file limit. Unzipping every zip into one folder gives back data/ exactly as staged."""
    out = os.path.abspath(out)
    man = json.load(open(os.path.join(out, "DATA_MANIFEST.json")))
    up = os.path.join(out, "upload")
    os.makedirs(up, exist_ok=True)
    by_set: dict[str, list[str]] = {}
    for f in man["files"]:
        by_set.setdefault(f["dataset"], []).append(f["path"])
    zips = []
    for ds, paths in by_set.items():
        z = os.path.join(up, f"{ds}.zip")
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for p in sorted(paths):
                zf.write(os.path.join(out, p), os.path.relpath(p, "data"))
        zips.append({"zip": f"{ds}.zip", "files": len(paths), "bytes": os.path.getsize(z),
                     "sha256": sha256(open(z, "rb").read())})
        print(f"pack  {ds}.zip: {len(paths)} files, {os.path.getsize(z) / 1e6:.1f} MB")
    man["zips"] = zips
    with open(os.path.join(up, "DATA_MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    shutil.copy2(os.path.join(out, "LICENSE-DATA.txt"), up)
    with open(os.path.join(up, "README-DATA.md"), "w") as f:
        f.write(DATA_README.format(name=cfg["release"]["name"], version=cfg["release"]["version"],
                                   repo=cfg["release"].get("repository", ""),
                                   table="\n".join(f"| `{z['zip']}` | {z['files']} | {next(d['note'] for d in cfg['data'] if d['id'] == z['zip'][:-4])} |"
                                                    for z in zips)))
    n = len(os.listdir(up))
    print(f"upload/: {n} files ({sum(os.path.getsize(os.path.join(up, x)) for x in os.listdir(up)) / 1e6:.1f} MB)"
          + ("" if n <= 100 else "  TOO MANY FILES FOR ZENODO (max 100)"))


DATA_README = """# {name} {version}: data deposit

The stored results behind the methods preprint "Calibrating the saturating-removal model of aging from survival data:
exact likelihood, identifiability and intervention effects" (Raz, Mayo and Alon). The code is the linked software
record ({repo}).

Unzip every zip into one folder named `data/` at the top of the code repository (or anywhere, and set the
environment variable `SR_DATA` to that folder). The scripts of the preprint (`components/paper/`) and its number
check (`verify_claims.py`) then read their inputs from there. `DATA_MANIFEST.json` lists every file with its sha256,
the data entry it belongs to and the commit or folder it came from; `zips` lists each zip's checksum.

| Zip | Files | What |
|---|---|---|
{table}

Licence: CC BY 4.0 (`LICENSE-DATA.txt`).
"""


def cmd_plan(cfg: dict) -> None:
    print(f"{cfg['release']['name']} {cfg['release']['version']}")
    for c in cfg["component"]:
        try:
            commit = git(c["repo"], "rev-parse", c["ref"] + "^{commit}")[:10]
            size = f"{len(archive(c['repo'], c['ref'], c['paths'])) / 1e6:7.1f} MB"
        except subprocess.CalledProcessError:
            commit, size = "UNRESOLVED", "      -   "
        print(f"  code {'+' if c['include'] else '-'} {c['id']:<28} {commit:<10}  {size}  -> {c['dest']}")
    for d in cfg["data"]:
        if d["source"] == "git":
            size, what = len(archive(d["repo"], d["ref"], d["paths"])), git(d["repo"], "rev-parse", d["ref"] + "^{commit}")[:10]
        elif d["source"] == "paper-inputs":
            lst = paper_inputs(cfg)
            size = sum(os.path.getsize(resolve(x["from"])) for x in lst["copy"])
            what = f"{len(lst['copy'])}+{len(lst['git'])} files"
        else:
            files = store_files(d) if d["source"] == "store" else working_tree_files(d)
            missing = [s for s, _ in files if not os.path.exists(s)]
            size = sum(os.path.getsize(s) for s, _ in files if os.path.exists(s))
            what = f"{len(files)} files" + (f", {len(missing)} MISSING" if missing else "")
        print(f"  data   {d['id']:<28} {what:<12} {size / 1e6:7.1f} MB  -> data/{d['dest']}")


def cmd_code(cfg: dict) -> None:
    for c in cfg["component"]:
        dest = os.path.join(HERE, c["dest"])
        if not c["include"]:
            print(f"skip  {c['id']} (include = false)")
            continue
        # Extract outside this folder, then write only what differs: deleting and rewriting the whole component in place
        # made Dropbox restore the old files and set the new ones aside as "conflicted copies" (2026-11-25).
        with tempfile.TemporaryDirectory() as tmp:
            n = extract(archive(c["repo"], c["ref"], c["paths"]), tmp, c.get("strip", ""), c.get("exclude"))
            changed = sync_tree(tmp, dest)
        print(f"code  {c['id']}: {n} files -> {c['dest']}, {changed} written or removed"
              + (f" ({len(c['exclude'])} patterns left out)" if c.get("exclude") else ""))


def cmd_data(cfg: dict, out: str) -> None:
    out = os.path.abspath(out)
    if out.startswith(HERE):
        sys.exit("the data deposit goes outside this repository (--out)")
    os.makedirs(out, exist_ok=True)
    manifest = {"release": cfg["release"]["version"], "files": []}
    for d in cfg["data"]:
        base = os.path.join(out, "data", d["dest"])
        if d["source"] == "git":
            extract(archive(d["repo"], d["ref"], d["paths"]), base, d.get("strip", ""))
            origin = {"repo": d["repo"], "commit": git(d["repo"], "rev-parse", d["ref"] + "^{commit}")}
        elif d["source"] == "paper-inputs":
            lst = paper_inputs(cfg)
            for x in lst["copy"]:
                dst = os.path.join(out, "data", x["to"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(resolve(x["from"]), dst)
            for x in lst["git"]:
                dst = os.path.join(out, "data", x["to"])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "wb") as f:
                    f.write(git(cfg_component(cfg, "sraging")["repo"], "show", f"{x['rev']}:{x['path']}", binary=True))
            origin = {"repo": d["repo"], "list": "components/paper/release_inputs.json"}
        else:
            files = store_files(d) if d["source"] == "store" else working_tree_files(d)
            for src, rel in files:
                os.makedirs(os.path.dirname(os.path.join(base, rel)), exist_ok=True)
                shutil.copy2(src, os.path.join(base, rel))
            origin = {"repo": d["repo"], "working_tree": True}
        for dp, _, fs in os.walk(base):
            for f in sorted(fs):
                p = os.path.join(dp, f)
                manifest["files"].append({"path": os.path.relpath(p, out), "bytes": os.path.getsize(p),
                                          "sha256": sha256(open(p, "rb").read()), "dataset": d["id"], **origin})
        print(f"data  {d['id']} -> data/{d['dest']}")
    with open(os.path.join(out, "LICENSE-DATA.txt"), "w") as f:
        f.write(DATA_LICENCE.format(name=cfg["release"]["name"], version=cfg["release"]["version"],
                                    authors=", ".join(cfg["release"]["authors"])))
    with open(os.path.join(out, "DATA_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(x["bytes"] for x in manifest["files"])
    print(f"DATA_MANIFEST.json: {len(manifest['files'])} files, {total / 1e6:.1f} MB")


def cmd_lock(cfg: dict) -> None:
    lock = {"release": cfg["release"]["version"], "components": []}
    for c in cfg["component"]:
        if not c["include"]:
            lock["components"].append({"id": c["id"], "include": False, "repo": c["repo"], "ref": c["ref"]})
            continue
        tar = archive(c["repo"], c["ref"], c["paths"])
        ex = c.get("exclude", [])
        with tarfile.open(fileobj=io.BytesIO(tar)) as tf:
            nfiles = sum(m.isfile() and not excluded(m.name, ex) for m in tf.getmembers())
        lock["components"].append({"id": c["id"], "include": c["include"], "repo": c["repo"], "ref": c["ref"],
                                   "commit": git(c["repo"], "rev-parse", c["ref"] + "^{commit}"),
                                   "paths": c["paths"], "exclude": ex, "files": nfiles,
                                   "archive_sha256": sha256(tar)})
    with open(os.path.join(HERE, "MANIFEST.lock.json"), "w") as f:
        json.dump(lock, f, indent=1)
    print(f"MANIFEST.lock.json: {len(lock['components'])} components")


if __name__ == "__main__":
    cfg = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        cmd_plan(cfg)
    elif cmd == "code":
        cmd_code(cfg)
    elif cmd == "lock":
        cmd_lock(cfg)
    elif cmd == "data" and "--out" in sys.argv:
        cmd_data(cfg, sys.argv[sys.argv.index("--out") + 1])
    elif cmd == "pack" and "--out" in sys.argv:
        cmd_pack(cfg, sys.argv[sys.argv.index("--out") + 1])
    else:
        sys.exit(__doc__)
