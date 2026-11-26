"""Build the frozen `fp_piecewise` oracle snapshot (spec 002, task T014; research R10).

Run ONCE:  .venv/bin/python tools/snapshot_fp_piecewise.py

Writes `tests/oracle/fp_piecewise_cb191f5/` (SRK files, MANIFEST.json, patch, pip_freeze.txt), then
verifies it (v033 md5s; SRK's own tests/test_fp_piecewise.py run against the COPY). Once built it is
frozen: never edit it. It refuses to overwrite an existing snapshot. Live SRK is only ever read, and
every process that imports SRK sets `sys.pycache_prefix` first.

`fingerprint(root)` (md5 of every file + the file list) is reused by tools/oracle_fp_piecewise.py.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")
LIVE = os.environ.get(
    "SRAGING_SRK_ROOT",
    os.path.join(os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset", "SRtools_kappa"))   # release 0.1.1
SNAP = os.path.join(REPO, "tests", "oracle", "fp_piecewise_cb191f5")
SNAP_SRK = os.path.join(SNAP, "SRtools_kappa")
V033 = os.path.join(REPO, "tests", "oracle", "v033")
HEAD = "cb191f5"
TEST_FILE = "tests/test_fp_piecewise.py"
PRESET_DIR = "SRK/Preset_values"
PRESET_MAX_BYTES = 5 * 1024 * 1024


def md5(p):
    with open(p, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def fingerprint(root) -> dict:
    """{relative path: md5} of every file under `root` (so new, missing and changed files all show)."""
    out = {}
    for d, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = md5(p)
    return out


def _git(*args):
    return subprocess.run(["git", "-C", LIVE, *args], capture_output=True, text=True, check=True).stdout


# The recorder runs in a fresh subprocess against LIVE SRK (read-only; bytecode redirected).
RECORDER = r'''
import sys, os, json, warnings
sys.pycache_prefix = sys.argv[1]
live = sys.argv[2]
sys.path.insert(0, live)
import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
warnings.simplefilter("ignore")
import SRK.fp_piecewise as fpp
from SRK import fp_likelihood as fpl, intervention_sim as isim, presets, curves
from SRK.intervention_sim import ExtHazard, Intervention
rng = np.random.default_rng(0)
n = 50
t = np.round(rng.gamma(9.0, 14.0, n)) + 1.0
e = (rng.random(n) > 0.05).astype(int)
th = presets.getTheta("mice_F", type="max_likelihood", as_theta=True)
kw = dict(N=100, dt=1e-2, tau_max=3.0)
ivs = [Intervention("beta", -0.3, (30.0, 80.0)), Intervention("A", 0.2, (20.0, 60.0)),
       Intervention("xc", 0.1, (40.0, 90.0)), ExtHazard(5.0, (50.0, 100.0))]
L, T_END = 124.0, 214.0
for sdt in (None, 1.0):
    fpp.piecewise_loglik(t, e, th, ivs, L=L, t_end=T_END, sampling_dt=sdt, **kw)
    fpp.piecewise_loglik_terms(t, e, th, ivs, L=L, t_end=T_END, sampling_dt=sdt, **kw)
fpp.piecewise_curves(th, ivs, L=L, t_end=T_END, external_hazard=6.0, **kw)
isim.sr_metrics(th, ivs, evaluator="piecewise_fp", t_end=T_END, N=100, dt=1e-2, tau_max=3.0)
fpl.fp_loglik_physical(t, e, th.eta, th.beta, th.epsilon, th.xc, th.kappa, L=L,
                       external_hazard=6.0, sampling_dt=1.0, **kw)
fpl.fp_loglik_physical(t, e, th.eta, th.beta, th.epsilon, th.xc, th.kappa, L=L,
                       time_range=(20.0, 150.0), **kw)
fpl.fp_loglik_het_physical(t, e, th.eta, th.beta, th.epsilon, th.xc, th.kappa, het_param="xc",
                           het_var=0.2, M=3, L=L, time_range=(20.0, 150.0), sampling_dt=1.0, **kw)
fpl.fp_mixture_curves_physical(th.eta, th.beta, th.epsilon, th.xc, th.kappa, L, np.linspace(0, 2, 20),
                               het_param="xc", het_var=0.2, M=3, **kw)
rb, rh, re, kd = fpl.to_dimensionless(th.eta, th.beta, th.epsilon, 1.0, L, k=th.kappa)
fpl.fpt_full(rh, rb, re, kd, return_survival=True, **kw)
fpl.fpt_full_density(rh, rb, re, kd, **kw)
root = os.path.realpath(live)
mods = sorted({os.path.relpath(os.path.realpath(m.__file__), root) for m in list(sys.modules.values())
               if getattr(m, "__file__", None) and os.path.realpath(m.__file__).startswith(root + os.sep)})
json.dump(mods, sys.stdout)
'''

# Second recording pass: SRK's own test file against LIVE SRK, to catch lazy imports (e.g. the simulator's
# intervention_SR) that the tiny calls above do not reach.
RECORDER_TESTS = r'''
import sys, os, json
sys.pycache_prefix = sys.argv[1]
live = sys.argv[2]
sys.path.insert(0, live)
os.chdir(live)
import pytest
rc = pytest.main(["-p", "no:cacheprovider", "-c", sys.argv[3], "--rootdir", live, "-q", "-x", "-m", "",
                  "tests/test_fp_piecewise.py"])
root = os.path.realpath(live)
mods = sorted({os.path.relpath(os.path.realpath(m.__file__), root) for m in list(sys.modules.values())
               if getattr(m, "__file__", None) and os.path.realpath(m.__file__).startswith(root + os.sep)})
sys.stderr.write("RC %d\n" % rc)
json.dump(mods, sys.stdout)
'''

VERIFY = r'''
import sys, os
sys.pycache_prefix = sys.argv[1]
snap = sys.argv[2]
sys.path.insert(0, snap)
os.chdir(snap)
import pytest
sys.exit(pytest.main(["-p", "no:cacheprovider", "-c", sys.argv[3], "--rootdir", snap, "-q", "-rxXs",
                      "tests/test_fp_piecewise.py"]))
'''


def pip_freeze_text():
    lines = sorted(f"{d.metadata['Name']}=={d.version}" for d in md.distributions())
    return "\n".join(lines) + "\n"


def main():
    if os.path.exists(SNAP):
        sys.exit(f"refusing: {SNAP} exists (the snapshot is frozen; never rebuild or edit it)")
    v033 = json.load(open(os.path.join(V033, "MANIFEST.json")))
    v033_md5 = {f["path"]: f["md5"] for f in v033["files"] if f["repo"] == "SRtools_kappa"}

    # 1. refuse unless the live tree is what we expect
    head = _git("rev-parse", "HEAD").strip()
    if not head.startswith(HEAD):
        sys.exit(f"refusing: live SRK HEAD is {head}, not {HEAD}")
    if md5(os.path.join(LIVE, "SRK/fp_likelihood.py")) != v033_md5["SRK/fp_likelihood.py"]:
        sys.exit("refusing: SRK/fp_likelihood.py differs from v033's copy")

    # 2. record the import closure in a fresh subprocess
    os.makedirs(sys.pycache_prefix, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", RECORDER, sys.pycache_prefix, LIVE],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit("recorder failed:\n" + r.stdout[-2000:] + r.stderr[-4000:])
    recorded = json.loads(r.stdout[r.stdout.rindex("["):])
    print(f"recorded {len(recorded)} SRK modules in {time.time() - t0:.1f}s")

    empty_ini = os.path.join(REPO, ".cache", "empty_pytest.ini")
    open(empty_ini, "w").write("[pytest]\n")
    r2 = subprocess.run([sys.executable, "-c", RECORDER_TESTS, sys.pycache_prefix, LIVE, empty_ini],
                        capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                             if not k.startswith("SRK_")})
    if "RC 0" not in r2.stderr:
        sys.exit("test-pass recorder failed:\n" + r2.stdout[-3000:] + r2.stderr[-3000:])
    extra = [m for m in json.loads(r2.stdout[r2.stdout.rindex("["):]) if m.startswith("SRK")]
    print("modules found only via the test file:", sorted(set(extra) - set(recorded)))
    recorded = sorted(set(recorded) | set(extra))
    files = set(recorded) | set(v033_md5)
    pre = sorted(os.path.relpath(os.path.join(d, f), LIVE)
                 for d, _s, fs in os.walk(os.path.join(LIVE, PRESET_DIR)) for f in fs
                 if "__pycache__" not in d and f != ".DS_Store")
    size = sum(os.path.getsize(os.path.join(LIVE, p)) for p in pre)
    if size > PRESET_MAX_BYTES:
        sys.exit(f"Preset_values is {size} bytes (> 5 MB): choose the files by hand")
    files |= set(pre)
    files.add(TEST_FILE)
    for need in ("SRK/__init__.py",):
        files.add(need)
    for conf in ("tests/conftest.py", "conftest.py"):       # the test's conftest, if it had one
        if os.path.exists(os.path.join(LIVE, conf)):
            files.add(conf)
    files = sorted(files)

    # clean-tree check: every closure file clean, except fp_likelihood (and v033-only extras, which
    # are held to v033's md5 instead)
    status = _git("status", "--short").splitlines()
    dirty = {ln[3:] for ln in status if not ln.startswith("??")}
    untracked = {ln[3:] for ln in status if ln.startswith("??")}
    must_be_clean = set(recorded) | set(pre) | {TEST_FILE}
    bad = sorted(p for p in must_be_clean
                 if (p in dirty or p in untracked) and p != "SRK/fp_likelihood.py"
                 and not (p in v033_md5 and md5(os.path.join(LIVE, p)) == v033_md5[p]))
    if bad:
        sys.exit(f"refusing: closure files not clean at HEAD: {bad}")

    # 4. copy byte for byte; every v033 file must match v033's md5
    os.makedirs(SNAP_SRK)
    entries = []
    for p in files:
        src = os.path.join(LIVE, p)
        dst = os.path.join(SNAP_SRK, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        h = md5(dst)
        if p in v033_md5 and h != v033_md5[p]:
            sys.exit(f"{p}: md5 {h} != v033 {v033_md5[p]} (snapshot left in place for inspection)")
        entries.append({"repo": "SRtools_kappa", "path": p, "md5": h,
                        "tracked": p not in untracked, "dirty_vs_head": p in dirty or p in untracked})

    # 5. provenance
    dirty_closure = [e["path"] for e in entries if e["tracked"] and e["dirty_vs_head"]]
    patch = _git("diff", "HEAD", "--", *dirty_closure) if dirty_closure else ""
    open(os.path.join(SNAP, "SRK_working_tree_vs_HEAD.patch"), "w").write(patch)
    open(os.path.join(SNAP, "pip_freeze.txt"), "w").write(pip_freeze_text())
    pins = {k: v for k, v in v033["environment_pins"].items()}
    pins = {k: md.version(k) for k in pins}
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "purpose": "Frozen SRK (HEAD cb191f5) for T1 of spec 002: fp_piecewise, fp_likelihood and their "
                   "closure. NEVER EDIT. Used only in a worker subprocess, same host, side by side.",
        "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()},
        "python": sys.version,
        "sources": {"SRtools_kappa": {"head": head, "status": status}},
        "files": entries,
        "environment_pins": pins,
    }
    man_path = os.path.join(SNAP, "MANIFEST.json")
    json.dump(manifest, open(man_path, "w"), indent=1)

    # 6. verify: SRK's own tests, run against the copy
    before = fingerprint(SNAP)
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", VERIFY, sys.pycache_prefix, SNAP_SRK, empty_ini],
                       capture_output=True, text=True, env={k: v for k, v in os.environ.items()
                                                            if not k.startswith("SRK_")})
    wall = time.time() - t0
    after = fingerprint(SNAP)
    print(r.stdout[-3000:])
    if r.returncode not in (0,):
        print(r.stderr[-3000:])
    import re
    tail = r.stdout.strip().splitlines()[-1]
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|xfailed|skipped|xpassed|error|errors)", tail)}
    manifest["verification"] = {
        "test": TEST_FILE, "returncode": r.returncode, "summary": tail,
        "passed": counts.get("passed", 0), "failed": counts.get("failed", 0),
        "xfailed": counts.get("xfailed", 0), "skipped": counts.get("skipped", 0),
        "wall_s": round(wall, 1),
        "v033_shared_md5_identical": True,
        "fingerprint_unchanged_by_verification": {k: v for k, v in after.items() if k != "MANIFEST.json"}
        == {k: v for k, v in before.items() if k != "MANIFEST.json"},
    }
    json.dump(manifest, open(man_path, "w"), indent=1)
    ok = (r.returncode == 0 and manifest["verification"]["failed"] == 0
          and manifest["verification"]["fingerprint_unchanged_by_verification"])
    print(json.dumps(manifest["verification"], indent=1))
    print(f"files: {len(entries)} ({len(set(entries_p['path'] for entries_p in entries) - set(v033_md5))} beyond v033); "
          f"Preset_values {size} bytes")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
