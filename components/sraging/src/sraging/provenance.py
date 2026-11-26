"""Provenance recorded on every FitResult (FR-008; research R14)."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import os
import platform
import subprocess
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _git() -> dict:
    """SHA and dirty flag of the sraging checkout, or an explicit 'unknown' (never another repo's SHA)."""
    def run(*args):
        return subprocess.run(["git", "-C", _PKG_DIR, *args], capture_output=True, text=True, timeout=20)
    try:
        top = run("rev-parse", "--show-toplevel")
        if top.returncode != 0:
            return {"git_sha": "unknown: not a git checkout", "dirty": None}
        root = top.stdout.strip()
        pyproject = os.path.join(root, "pyproject.toml")
        if not (os.path.exists(pyproject) and 'name = "sraging"' in open(pyproject).read()):
            return {"git_sha": "unknown: installed outside the sraging checkout", "dirty": None}
        sha = run("rev-parse", "HEAD").stdout.strip() or "unknown: no commit"
        dirty = bool(run("status", "--porcelain", "--untracked-files=no").stdout.strip())
        return {"git_sha": sha, "dirty": dirty}
    except (OSError, subprocess.SubprocessError) as ex:
        return {"git_sha": f"unknown: {type(ex).__name__}", "dirty": None}


def _version(dist: str) -> str:
    try:
        return _md.version(dist)
    except _md.PackageNotFoundError:
        return "not installed"


def collect(calibrator, grid: dict, runtime_state: dict) -> dict:
    return {
        "package": {"version": _version("sraging"), **_git()},
        "numerics": calibrator.numerics,
        "calibrator": {"name": calibrator.name, "version": calibrator.version},
        "engine": "fp",
        "grid": grid,
        "prng": "none (deterministic path)",
        "runtime": runtime_state,
        "libraries": {"python": sys.version.split()[0],
                      **{d: _version(d) for d in ("numpy", "scipy", "jax", "jaxlib")}},
        "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()},
        "created_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def code_line(prov: dict) -> str:
    """The one-line code provenance put in diagnostics['code'] (the reference's format_code_line role)."""
    p = prov["package"]
    dirty = "+dirty" if p.get("dirty") else ""
    return (f"sraging {p['version']} @ {p['git_sha'][:12]}{dirty}; numerics {prov['numerics']}; "
            f"calibrator {prov['calibrator']['name']}@{prov['calibrator']['version']}")
