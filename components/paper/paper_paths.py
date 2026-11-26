"""Where the paper's inputs are, on the author's machine and in the public release.

Every script in tables/ and figures/ and verify_claims.py names its inputs the way they were produced, e.g.
"aging:code_5/sr_calibration_tester/reports/jax2-v6-full-bign/explorer_data.json" or "~/code/sraging/tests/...".

* On the author's machine (this folder is the preprint repository) the names resolve in place:
  `aging:` = the Dropbox aging folder, `~/` = the home folder.
* In the public release (this folder is components/paper/ of the sr-calibration repository, next to VERSIONS.toml)
  the same names resolve through RELEASE_MAP into the release itself (`code:`) or into its Zenodo data deposit
  (`data:`), unpacked into <release>/data or into the folder named by the environment variable SR_DATA.
  An input that is not public in this version raises NotPublic, with the reason.

Set SR_RECORD_INPUTS=<file> to append every input a script reads to <file> (one per line); the release's
data deposit is built from that list (paper_inputs.py).
"""
from __future__ import annotations

import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
_REL = os.path.dirname(os.path.dirname(HERE))
RELEASE = os.path.basename(HERE) == "paper" and os.path.isfile(os.path.join(_REL, "VERSIONS.toml"))
DATA = os.environ.get("SR_DATA", os.path.join(_REL, "data"))
# the author's aging folder: this repository is <aging>/code_5/sr_methods_preprint (not used in the release)
AGING = None if RELEASE else os.path.dirname(os.path.dirname(HERE)) + "/"
SRAGING = "~/code/sraging"
SRAGING_TAG = "v0.1.1"          # the sraging version in the release (components/sraging); src/ as at v0.1.0

# (name prefix, place in the release or None, reason). The longest matching prefix wins.
# `paper:` = this folder; `code:` = the release checkout; `data:` = the unpacked data deposit.
RELEASE_MAP = [
    ("aging:code_5/sr_methods_preprint/", "paper:", ""),
    ("aging:code_5/sr_methods_release/", "code:", ""),
    ("~/code/sraging/specs/", "data:sraging_evidence/specs/", ""),
    ("~/code/sraging/", "code:components/sraging/", ""),
    ("aging:code_5/sr_calibration_tester/data/library/", "data:benchmark/library/", ""),
    ("aging:code_5/sr_calibration_tester/runs/", "data:benchmark/runs/", ""),
    ("aging:code_5/sr_calibration_tester/__records__/", "data:benchmark/records/", ""),
    ("aging:code_5/sr_calibration_tester/reports/validity/", "data:paper_inputs/code_5/sr_calibration_tester/reports/validity/", ""),
    ("aging:code_5/sr_calibration_tester/reports/", "data:benchmark/reports/", ""),
    ("aging:code_5/sr_calibration_tester/methods/mle_profile_jax2/", "code:components/calibrators/mle_profile_jax2-1.0.1/mle_profile_jax2/", ""),
    ("aging:code_5/sr_calibration_tester_jax2/reports/rc2/", "data:paper_inputs/code_5/sr_calibration_tester_jax2/reports/rc2/", ""),
    ("aging:code_5/sr_calibration_tester_jax2_rel/reports/rel101/", "data:paper_inputs/code_5/sr_calibration_tester_jax2_rel/reports/rel101/", ""),
    ("aging:code_5/sr_calibration_tester_kfix/__records__/", "data:kfix_records/", ""),
    ("aging:code_5/sr_calibration_tester_kfix/", "data:paper_inputs/code_5/sr_calibration_tester_kfix/", ""),
    ("aging:code_5/sr_fp_fit_share/", "data:paper_inputs/code_5/sr_fp_fit_share/", ""),
    ("git:", "data:paper_inputs/git/", ""),
    ("aging:code_5/sr_calibration_tester_jax2_freeze/reports/freeze101/HANDOFF.txt", None,
     "the hand-off notes of the calibrator freeze are internal working notes"),
    ("aging:code_4/Mire_fly_interventions/", None, "a fly study's files; released with that study"),
    ("aging:code_5/itp_analysis/", None, "the ITP effect-recovery study's files; not in this version"),
    ("aging:interventions_data/", None, "the interventions project's files; not in this version"),
    # §7d's evidence (Naveh, 2026-11-25): the ITP recovery, identification and engine-check results and the
    # interventions identifiability summary are released; the rest of those two projects is not
    ("aging:code_5/itp_analysis/recal/results/2026-10-18_J100/",
     "data:paper_inputs/code_5/itp_analysis/recal/results/2026-10-18_J100/", ""),
    ("aging:code_5/itp_analysis/recal/results/gates/", "data:paper_inputs/code_5/itp_analysis/recal/results/gates/", ""),
    ("aging:interventions_data/calib/ident/ia_validation/summary.json",
     "data:paper_inputs/interventions_data/calib/ident/ia_validation/summary.json", ""),
    ("aging:code_5/alibek_mouse/", None, "the mouse study's files; released with that study"),
    ("aging:code_5/mosquitos/", None, "the mosquito study's files; released with that study"),
]


class NotPublic(Exception):
    """An input that is not part of this release."""


def norm(name: str) -> str:
    """The canonical name of an input: absolute local paths become aging:/~/ names; relative names are paper files."""
    if AGING and name.startswith(AGING):
        return "aging:" + name[len(AGING):]
    if name.startswith(HOME + "/"):
        return "~/" + name[len(HOME) + 1:]
    if not (name.startswith("aging:") or name.startswith("~/") or name.startswith("git:") or os.path.isabs(name)):
        return "aging:code_5/sr_methods_preprint/" + name
    return name


def local(name: str) -> str:
    n = norm(name)
    if n.startswith("aging:code_5/sr_methods_preprint/"):
        return os.path.join(HERE, n[len("aging:code_5/sr_methods_preprint/"):])
    if n.startswith("aging:"):
        if AGING is None:
            raise NotPublic(f"{name}: an aging: name outside the release map")
        return AGING + n[len("aging:"):]
    if n.startswith("~/"):
        return os.path.join(HOME, n[2:])
    return n


def released(name: str) -> str:
    """Where `name` is in the release; NotPublic if it is not there."""
    n = norm(name)
    best = None
    for pre, dest, why in RELEASE_MAP:
        if n.startswith(pre) and (best is None or len(pre) > len(best[0])):
            best = (pre, dest, why)
    if best is None:
        raise NotPublic(f"{name}: not part of this release")
    pre, dest, why = best
    if dest is None:
        raise NotPublic(f"{name}: {why}")
    kind, rest = dest.split(":", 1)
    root = {"paper": HERE, "code": _REL, "data": DATA}[kind]
    return os.path.join(root, rest + n[len(pre):])


def _record(entry: str) -> None:
    rec = os.environ.get("SR_RECORD_INPUTS")
    if rec:
        with open(rec, "a") as f:
            f.write(entry + "\n")


def path(name: str) -> str:
    """The file to open for input `name`."""
    _record(norm(name))
    return released(name) if RELEASE else local(name)


def records(tester: str, run: str):
    """(cohort id, record) for every cohort of a stored calibration run, in the run's order.
    `tester` names a calibration-tester checkout, e.g. "aging:code_5/sr_calibration_tester"."""
    keys = json.load(open(path(f"{tester}/runs/{run}.json")))["keys"]
    _record(f"{norm(tester)}/__records__/{run}/")
    for cid, k in keys.items():
        if RELEASE:
            p = released(f"{tester}/__records__/{run}/{cid}.json")
        else:
            p = os.path.join(local(tester), "data", "store", k[:2], k[2:4], k + ".json")
        with open(p) as f:
            yield cid, json.load(f)


def git_show(repo: str, rev: str, relpath: str) -> bytes:
    """The bytes of `relpath` at commit `rev` of `repo` (only sraging's history is used).
    In the release: the file in components/sraging if `rev` is the released tag and the file is released code,
    else the copy in the data deposit (paper_inputs/git/<rev>/<relpath>)."""
    if norm(repo) != SRAGING:
        raise NotPublic(f"git {repo}: only sraging's history is used")
    _record(f"git:{rev}/{relpath}")
    if not RELEASE:
        return subprocess.run(["git", "-C", local(repo), "show", f"{rev}:{relpath}"],
                              capture_output=True, check=True).stdout
    copy = released(f"git:{rev}/{relpath}")
    if os.path.isfile(copy):
        return open(copy, "rb").read()
    if rev == SRAGING_TAG:
        return open(released(f"{SRAGING}/{relpath}"), "rb").read()
    raise NotPublic(f"git {rev}:{relpath}: not in this release")
