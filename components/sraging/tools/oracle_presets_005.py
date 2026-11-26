"""SRK's typed getTheta from the frozen snapshot, read only (spec 005 HR-7, HR-8; tasks T039).

    rows = get_theta([{"name": "mice_F", "type": "mode", "ExtH": True, "time_unit": "weeks"}, ...])
    rows[i]["theta"]       # getTheta's array as floats (exact: sent as float.hex), or rows[i]["error"]

The snapshot `tests/oracle/fp_piecewise_cb191f5/` holds SRK's presets.py and Preset_values (v033's has none). Its SRK
is imported as top-level `SRK`, as v033's is, so the two cannot share a process: every call runs in one worker
subprocess (`python tools/oracle_presets_005.py serve`, a JSON list in on stdin, a JSON list out on stdout). The
worker redirects bytecode first, checks tools/oracle_fp_piecewise.py's preflight (manifest checksums, pins, host), and
asserts that SRK and SRK.presets were imported from the snapshot root, as oracle_fp_piecewise does. getTheta's prints
go to stderr. Nothing is written into the snapshot.

This is a new tool: it is not one of spec 003's UNTOUCHED oracle tools (tools/acceptance_003.py) and not in spec 001's
claims closure, so it needs no SPEC_ORACLE_ADDITIONS entry and triggers no claims re-run.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THIS = os.path.abspath(__file__)
SNAP_SRK_ROOT = os.path.join(REPO, "tests", "oracle", "fp_piecewise_cb191f5", "SRtools_kappa")
PYCACHE = os.path.join(REPO, ".cache", "pycache")
TYPES = ("mode_overall", "mode", "max_likelihood")
UNITS = ("hours", "days", "weeks", "years", "generations")


def _of():
    if os.path.join(REPO, "tools") not in sys.path:
        sys.path.insert(0, os.path.join(REPO, "tools"))
    import oracle_fp_piecewise as OF
    return OF


def preflight() -> list[str]:
    """[] or why the snapshot cannot be used validly here (tools/oracle_fp_piecewise.py's own preflight)."""
    return _of().preflight()


def _enc(x) -> str:
    return float(x).hex()


def get_theta(requests) -> list[dict]:
    """getTheta(name, type=, ExtH=, time_unit=) for each request, in one worker; NotApplicable (oracle_fp_piecewise's)
    when the snapshot cannot be used here."""
    OF = _of()
    problems = OF.preflight()
    if problems:
        raise OF.NotApplicable(problems)
    reqs = [{"name": str(r["name"]), "type": str(r["type"]), "ExtH": bool(r["ExtH"]),
             "time_unit": str(r["time_unit"])} for r in requests]
    p = subprocess.run([sys.executable, THIS, "serve"], input=json.dumps(reqs), capture_output=True, text=True,
                       env=OF._worker_env(), cwd=REPO)
    if p.returncode != 0:
        raise OF.OracleError(f"the presets worker exited {p.returncode}: {p.stderr[-2000:]}")
    out = []
    for r in json.loads(p.stdout):
        out.append({**r, "theta": None if r.get("theta") is None else [float.fromhex(x) for x in r["theta"]]})
    return out


def _serve():
    """Worker side: import the snapshot's SRK only, then answer every request."""
    sys.pycache_prefix = PYCACHE
    OF = _of()
    problems = OF.preflight(worker=True)
    if problems:
        sys.stderr.write("NOT APPLICABLE: " + "; ".join(problems) + "\n")
        sys.exit(3)
    sys.path.insert(0, SNAP_SRK_ROOT)
    import numpy as np
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid
    import SRK
    from SRK import presets
    root = os.path.realpath(SNAP_SRK_ROOT) + os.sep
    for m in (SRK, presets):
        assert os.path.realpath(m.__file__).startswith(root), f"{m.__name__} imported from {m.__file__}, not the " \
                                                              f"snapshot"
    out = []
    for r in json.loads(sys.stdin.read()):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                th = presets.getTheta(r["name"], type=r["type"], time_unit=r["time_unit"], ExtH=r["ExtH"])
            out.append({**r, "theta": [_enc(x) for x in np.asarray(th, float)], "error": None})
        except Exception as ex:                                 # noqa: BLE001 -- returned as this request's error
            out.append({**r, "theta": None, "error": f"{type(ex).__name__}: {ex}"})
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1:] == ["serve"]:
        _serve()
    else:
        sys.exit("usage: oracle_presets_005.py serve  (a JSON list of requests on stdin)")
