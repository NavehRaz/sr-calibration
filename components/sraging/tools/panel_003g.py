"""Spec 003g's frozen gate panel (research R1; contracts/gate-and-panel.md; tasks T005). Read-only on every source.

    .venv/bin/python tools/panel_003g.py build    # writes tests/acceptance/panel_003g_ids.tsv and the toml's [strata]
    .venv/bin/python tools/panel_003g.py check    # rebuilds in memory; exit 1 on any difference from the committed

The panel is every stratum where a registered defect of the candidate (jax2 1.0.1) or its predecessor (v0.3.3) fires
(constitution V): the 53 [T-JAX-BASIN] cohorts as they map onto library v6 (E. coli has no v6 cohort), register REV 6's
lists in J's domain (no window: both calibrators refuse windows; no hazard: J is misspecified there), the `_h` strata
(reported, not gating), and three predecessor-defect strata by a data rule (spec FR-002). Each source is read only
after its sha256 equals the toml's; each v6 cohort array only after its content digest equals the manifest's.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import tomllib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DROPBOX = os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset"   # release 0.1.1: the lab's checkouts
PANEL_PATH = os.path.join(REPO, "tests", "acceptance", "panel_003g.toml")
IDS_PATH = os.path.join(REPO, "tests", "acceptance", "panel_003g_ids.tsv")
COHORTS_V6 = "sr_calibration_tester/data/library/v6/cohorts"
J = "profile_jax2"
H = "profile_jax2_h"
FIELDS = ("cohort_id", "group", "arm", "half", "window", "calibrator", "valid", "informative", "dt_in_scope",
          "gating", "strata")
#: the register's lists, as strata: (stratum, list key, calibrator, gating)
REGISTER_STRATA = (("LB-2", "LB-2", J, True), ("LB-1x", "LB-1x", J, True), ("SRK-1", "SRK-1", J, True),
                   ("SRK-1b", "SRK-1b", J, True), ("SRK-2", "SRK-2", J, True), ("HZ-1", "HZ-1", H, False),
                   ("HZ-4", "HZ-4", H, False))
TIE_DOSE = 0.005             # the censor-tie bias's threshold (README's CENSORING TIES; warnings._TIE_FRACTION)
DT_CEILING = 1e-3            # [T-DT-CEILING]: v0.3.3's solver step is capped at delta / (3 L)


class Refused(Exception):
    """A source, list or row the panel cannot be built from."""


# ------------------------------------------------------------------------------------------------ guarded reads
def read_source(sources: dict, name: str, *, root: str = DROPBOX) -> bytes:
    """The bytes of a pinned source, refused unless their sha256 is the pinned one."""
    s = sources[name]
    b = open(os.path.join(root, s["path"]), "rb").read()
    got = hashlib.sha256(b).hexdigest()
    if got != s["sha256"]:
        raise Refused(f"source {name!r} ({s['path']}): sha256 {got[:12]}..., pinned {s['sha256'][:12]}...")
    return b


def checked_ids(entry: dict) -> list:
    """A register list's ids, refused unless they hash to its own `ids_sha256` (sorted, newline-joined)."""
    ids = sorted(entry["ids"])
    got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if got != entry["ids_sha256"]:
        raise Refused(f"register list {entry.get('key')!r}: ids_sha256 {got[:12]}... != {entry['ids_sha256'][:12]}...")
    return ids


def cohort_digest(ages, events) -> str:
    """srbench.library.build.cohort_digest, copied: repr floats joined by '|', '#', the events joined by '|'."""
    h = hashlib.sha256()
    h.update(b"|".join(repr(float(a)).encode() for a in np.asarray(ages)))
    h.update(b"#")
    h.update(b"|".join(str(int(e)).encode() for e in np.asarray(events)))
    return h.hexdigest()


def j_domain(spec: dict) -> bool:
    """J's domain: no window (both calibrators refuse windows) and no hazard (J is misspecified there)."""
    return spec["arm"] not in ("window", "hazard")


def require_no_window(rows) -> None:
    bad = [r["cohort_id"] for r in rows if r["arm"] == "window"]
    if bad:
        raise Refused(f"window ids in the panel ({len(bad)}, e.g. {bad[0]}): both calibrators refuse windows")


def _ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------- build
def build(panel: dict, *, root: str = DROPBOX) -> tuple[list, dict]:
    """(rows, strata): the panel's rows (one per cohort, FIELDS) and each stratum's registration."""
    src = panel["sources"]
    reg = json.loads(read_source(src, "register", root=root))
    man = json.loads(read_source(src, "manifest_v6", root=root))
    spec = {s["cohort_id"]: s for s in man["specs"]}
    ref = set()
    for line in read_source(src, "reference_set", root=root).decode().splitlines():
        if line and not line.startswith("#"):
            ref.add(line.split("\t")[1])
    validity = {r["cohort_id"]: r for r in csv.DictReader(io.StringIO(read_source(src, "validity_v6", root=root)
                                                                      .decode()))}
    labels = list(csv.DictReader(io.StringIO(read_source(src, "basin_labels", root=root).decode())))
    read_source(src, "basin_scope", root=root)                       # pinned for the predecessor phase
    census = {}
    for line in read_source(src, "census", root=root).decode().splitlines():
        if line.strip():
            r = json.loads(line)
            census[(r["cid"], r["view"])] = r

    def invalid(c):
        return str(validity[c]["flag_invalid"]).lower() in ("true", "1")

    def informative(c):
        return float(spec[c].get("het_var") or 0.0) == 0.0

    lists = {e["key"]: e for e in reg["v6_id_lists"]}
    strata = {}

    def stratum(name, ids, *, kind, rule, defect, calibrator, gating):
        ids = sorted(ids)
        strata[name] = {"kind": kind, "rule": rule, "defect": defect, "calibrator": calibrator, "gating": gating,
                        "count": len(ids), "ids_sha256": _ids_sha(ids), "ids": ids}

    # the 53 [T-JAX-BASIN] cohorts (label_final.csv fail_mle on v1), as they map onto v6
    basin = [r for r in labels if r["fail_mle"] == "True"]
    ecoli = {r["cohort_id"] for r in basin if "coli" in (r["cohort_id"] + (r.get("real_name") or "")).lower()}
    b48 = {r["cohort_id"] for r in basin} - ecoli
    missing = sorted(c for c in b48 if c not in spec)
    if missing:
        raise Refused(f"basin ids not in library v6: {missing[:3]}")
    stratum("BASIN-J", [c for c in b48 if c in ref and j_domain(spec[c])], kind="register",
            rule="label_final fail_mle, J's domain, inside the reference set", defect="T-JAX-BASIN",
            calibrator=J, gating=True)
    stratum("BASIN-P15", [c for c in b48 if c not in ref], kind="register",
            rule="label_final fail_mle, outside the reference set (sharp p15)", defect="T-JAX-BASIN",
            calibrator=J, gating=True)
    stratum("BASIN-H", [c for c in b48 if spec[c]["arm"] == "hazard"], kind="register",
            rule="label_final fail_mle, hazard arm (Naveh D5-D6: _h, reported)", defect="T-JAX-BASIN",
            calibrator=H, gating=False)
    strata["BASIN-ECOLI"] = {"kind": "register", "rule": "label_final fail_mle, E. coli: no v5 or v6 cohort",
                             "defect": "T-JAX-BASIN", "calibrator": None, "gating": False, "count": 0,
                             "ids_sha256": _ids_sha([]), "ids": [], "deviation": sorted(ecoli)}
    for name, key, cal, gating in REGISTER_STRATA:
        ids = checked_ids(lists[key])
        keep = [c for c in ids if (j_domain(spec[c]) if cal == J else spec[c]["arm"] == "hazard")]
        stratum(name, keep, kind="register", rule=f"register REV 6 list {key} ({lists[key]['rule'][:80]}...)",
                defect=lists[key]["defect"], calibrator=cal, gating=gating)

    # the predecessor's other registered defects, by a data rule on v6 (FR-002), J's domain
    roundoff, ties, dtceil = [], [], []
    from sraging.compat.jax2_101.observation_model import censor_tie_relabel
    for c in sorted(c for c, s in spec.items() if j_domain(s)):
        z = np.load(os.path.join(root, COHORTS_V6, f"{c}.npz"))
        a, e = np.asarray(z["ages"], float), np.asarray(z["events"], int)
        if cohort_digest(a, e) != man["checksums"][c]:
            raise Refused(f"{c}: the array's content digest is not the manifest's")
        d = float(spec[c].get("delta") or 0.0)
        if d > 0 and bool(((e == 1) & (a < d)).any()):
            roundoff.append(c)
        if d > 0 and censor_tie_relabel(a, e, d, enabled=False)[2]["dose"] >= TIE_DOSE:
            ties.append(c)
        L = float(validity[c]["L_phys"])
        if d > 0 and L > 0 and d / L < DT_CEILING:
            dtceil.append(c)
    stratum("P-ROUNDOFF", roundoff, kind="data", rule="a death in the first recording bin (event == 1 and age < delta)",
            defect="v0.3.3 round-off early deaths (fixed by 1.0.0)", calibrator=J, gating=True)
    stratum("P-CENSORTIE", ties, kind="data",
            rule="jax2's censor-tie detector (compat/jax2_101 censor_tie_relabel, enabled=False) dose >= 0.005",
            defect="v0.3.3 censor-tie bias (fixed by 1.0.0)", calibrator=J, gating=True)
    stratum("P-DTCEIL", dtceil, kind="data", rule="delta / L_phys < 1e-3 (validity_v6.csv)",
            defect="T-DT-CEILING (v0.3.3's step cap delta/(3L))", calibrator=J, gating=True)

    # one row per cohort
    member = {}
    for name, s in strata.items():
        for c in s["ids"]:
            member.setdefault((c, s["calibrator"]), []).append(name)
    rows = []
    for (c, cal), names in sorted(member.items()):
        sp = spec[c]
        half = "inside" if c in ref else "outside"
        view = "J" if cal == J else "H"
        cen = census.get((c, view)) or {}
        rows.append({"cohort_id": c, "group": sp.get("group"), "arm": sp["arm"], "half": half,
                     "window": ("W2" if cal == H and half == "inside" else "none" if cal == H
                                else "W1" if half == "inside" else "W3"),
                     "calibrator": cal, "valid": not invalid(c), "informative": informative(c),
                     "dt_in_scope": bool(cen.get("in5")), "gating": any(strata[n]["gating"] for n in names),
                     "strata": sorted(names)})
    require_no_window(rows)
    return rows, strata


# ---------------------------------------------------------------------------------------------- files: tsv, toml
def tsv_bytes(rows) -> bytes:
    out = io.StringIO()
    out.write("\t".join(FIELDS) + "\n")
    for r in rows:
        out.write("\t".join(",".join(r[f]) if f == "strata" else str(r[f]) for f in FIELDS) + "\n")
    return out.getvalue().encode()


def read_ids(path=IDS_PATH) -> list:
    rows = []
    with open(path) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(head, line.rstrip("\n").split("\t")))
            for f in ("valid", "informative", "dt_in_scope", "gating"):
                r[f] = r[f] == "True"
            r["strata"] = r["strata"].split(",") if r["strata"] else []
            rows.append(r)
    return rows


def strata_toml(strata: dict) -> str:
    lines = []
    for name in sorted(strata):
        s = strata[name]
        lines.append(f'[strata."{name}"]')
        for k in ("kind", "rule", "defect", "calibrator", "gating", "count", "ids_sha256"):
            v = s[k]
            lines.append(f"{k} = " + ("true" if v is True else "false" if v is False else '""' if v is None
                                      else str(v) if isinstance(v, int) else json.dumps(v)))
        if "deviation" in s:
            lines.append("deviation = " + json.dumps(s["deviation"]))
        lines.append("")
    return "\n".join(lines)


def write(rows, strata, *, panel_path=PANEL_PATH, ids_path=IDS_PATH) -> str:
    b = tsv_bytes(rows)
    open(ids_path, "wb").write(b)
    sha = hashlib.sha256(b).hexdigest()
    text = open(panel_path).read()
    text = re.sub(r'(ids = \{ path = "tests/acceptance/panel_003g_ids.tsv", sha256 = ")[0-9a-f]*(" \})',
                  lambda m: m.group(1) + sha + m.group(2), text)
    head = re.split(r"(?m)^\[strata\]", text, maxsplit=1)[0]        # the table header line, not a comment's mention
    if "[sources]" not in head or "[rules]" not in head:
        raise Refused("panel_003g.toml has lost its [sources] or [rules] table; refusing to write it")
    text = head + "[strata]   # written by build: { kind, rule, defect, calibrator, gating, count, ids_sha256 }\n\n" + \
        strata_toml(strata)
    open(panel_path, "w").write(text)
    return sha


def check(panel: dict, *, root: str = DROPBOX) -> list:
    """The differences between a rebuild and the committed panel ([] when equal)."""
    rows, strata = build(panel, root=root)
    out = []
    b = tsv_bytes(rows)
    if b != open(IDS_PATH, "rb").read():
        out.append("panel_003g_ids.tsv differs from the rebuild")
    if hashlib.sha256(b).hexdigest() != panel["sources"]["ids"]["sha256"]:
        out.append("the toml's ids sha256 is not the rebuild's")
    for name, s in strata.items():
        got = panel.get("strata", {}).get(name)
        if not got or got.get("count") != s["count"] or got.get("ids_sha256") != s["ids_sha256"]:
            out.append(f"stratum {name}: committed {got and (got.get('count'), got.get('ids_sha256', '')[:12])}, "
                       f"rebuilt {(s['count'], s['ids_sha256'][:12])}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("build", "check"))
    args = ap.parse_args(argv)
    panel = tomllib.load(open(PANEL_PATH, "rb"))
    if args.action == "build":
        rows, strata = build(panel)
        sha = write(rows, strata)
        print(f"wrote {os.path.relpath(IDS_PATH, REPO)} ({len(rows)} rows, sha256 {sha[:12]}...)")
        for name in sorted(strata):
            print(f"  {name:12s} {strata[name]['count']:4d}  gating={strata[name]['gating']}")
        return 0
    diffs = check(panel)
    for d in diffs:
        print("DIFF", d)
    print("panel check:", "OK" if not diffs else f"{len(diffs)} difference(s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
