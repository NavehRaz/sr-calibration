"""Spec 004's heterogeneity panel (research R9; contracts/register-and-panel.md §2; data-model "Acceptance panel"; tasks
T023). Read-only on every source.

    .venv/bin/python tools/panel_004.py build    # writes tests/acceptance/panel_004_ids.tsv and its sha256 into
                                                 # panel_004.toml [sources] (which T041 writes first)
    .venv/bin/python tools/panel_004.py check    # rebuilds in memory; exit 1 on any difference from the committed

**The source**: library v6's manifest, pinned by sha256 (MANIFEST_V6; panel_004.toml [sources] must pin the same), and
each cohort array only after its content digest equals the manifest's. The candidates are v6's 332 heterogeneous
cohorts (xc, het_var 0.2): 296 in arm het, 36 realfull.

**Per cohort**: CV_c, the CV of the age at death of one homogeneous SR march at the truth (no hazard; the engine's
march on a model grid: L from the coarse probe of `sraging.engine.curves.model_L`, then N = 300, dt = 1e-3, tau_max =
8); CV_data, the CV of the cohort's recorded death ages (events == 1, before any window); r = sqrt(CV_data^2 -
CV_c^2) / CV_c at the cohorts' sd 0.2 (the NOTE's data-based r; the engine's guide uses the model's, OD-10).

**The strata** (40 cohorts): syn-broad, syn-mid, syn-sharp 4 each (the synthetic cohorts by their manifest cv class);
real-lo-r, real-hi-r 6 each (the real non-human cohorts, split at their median r); human-10k, -25k, -100k 4 each (the
human life tables, Denmark, Sweden and combined human, by n); realfull 4 (window and hazard, human).

**The selection**: within a stratum, sort by r, cut into k equal-count bins (np.array_split), and take from each bin the
cohort with the smallest sha256(cohort_id + "spec004"). **Held out**: the het-tradeoff pilot's 9 cohorts, on which the
guide was fitted (NOTE; docs/planning/het_tradeoff/evidence/cvc.json).

**The subsets** (roles; research R9, R11 and spec HR-3 fix their sizes, not their members; this rule is the build's
proposal, frozen with Naveh's OK at freeze-H, tasks T043-T044). Each takes, within a stratum, the panel cohorts in
order of sha256(cohort_id + "spec004:" + subset):
- `subset-hr3` (12 cohorts, HR-3's 24 pairs at sd 0.2 and 0.3): the first of each stratum, then the second of
  syn-sharp, real-hi-r and human-100k (the strata that need the most nodes, where the never-raise plant must fail);
- `subset-composite@<sd>` (8 cells, 2 per sd, human and sharp first; the reference's composite cross-check): per sd in
  SDS order, the next human cohort (the 12 human cohorts in hash order) and the next syn-sharp cohort;
- `subset-hr2` (6 cohorts, HR-2's sd grid): the first of syn-broad, syn-mid, syn-sharp, real-lo-r, real-hi-r and
  human-10k.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import math
import os
import re
import sys
import tomllib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.pycache_prefix = os.path.join(REPO, ".cache", "pycache")

DROPBOX = os.environ.get("SRAGING_CODE5") or "/SRAGING_CODE5-is-unset"   # release 0.1.1: the lab's checkouts
PANEL_PATH = os.path.join(REPO, "tests", "acceptance", "panel_004.toml")
IDS_PATH = os.path.join(REPO, "tests", "acceptance", "panel_004_ids.tsv")
IDS_REL = "tests/acceptance/panel_004_ids.tsv"
#: library v6's manifest, the same bytes panel_003g.toml pins
MANIFEST_V6 = {"path": "sr_calibration_tester/data/library/v6/manifest.json",
               "sha256": "399c9c555782b4271f0e790099893180a2c5101e2e56318eb87d9c25785f7d2f"}
COHORTS_V6 = "sr_calibration_tester/data/library/v6/cohorts"

STRATA = {"syn-broad": 4, "syn-mid": 4, "syn-sharp": 4, "real-lo-r": 6, "real-hi-r": 6, "human-10k": 4,
          "human-25k": 4, "human-100k": 4, "realfull": 4}
SDS = (0.1, 0.2, 0.3, 0.5)                   # each panel cohort is scored at these (160 cells; Naveh 2026-10-29)
SALT = "spec004"
HELD_OUT = ("het-real-Guiniea_pig_VC-c0-s0", "het-real-German_Shepherd_vetCompass-c0-s0",
            "het-real-Jack_Russell_vetCompass-c0-s0", "het-real-cats_vp_F-c0-s0", "het-k0.01-p03-c1-s0",
            "het-k0.01-p01-c1-s0", "het-k0.01-p00-c1-s0", "het-real-combined_human_F-c2-s0",
            "het-real-Sweden_F_1910_homo-c0-s0")
HUMAN = ("Denmark_", "Sweden_", "combined_human_")
HUMAN_N = {10_000: "human-10k", 25_000: "human-25k", 100_000: "human-100k"}
FIELDS = ("cohort_id", "stratum", "n", "cv_c", "cv_data", "r", "roles")
#: the subsets' rule (module docstring)
HR3_EXTRA = ("syn-sharp", "real-hi-r", "human-100k")
HR2_STRATA = ("syn-broad", "syn-mid", "syn-sharp", "real-lo-r", "real-hi-r", "human-10k")
FINE = dict(N=300, dt=1e-3, tau_max=8.0, Gtol=1e-11)        # CV_c's march, in units of the model's L


class Refused(Exception):
    """A source, stratum or row the panel cannot be built from."""


# ------------------------------------------------------------------------------------------------ guarded reads
def read_source(source: dict, *, root: str = DROPBOX) -> bytes:
    """The bytes of a pinned source, refused unless their sha256 is the pinned one."""
    b = open(os.path.join(root, source["path"]), "rb").read()
    got = hashlib.sha256(b).hexdigest()
    if got != source["sha256"]:
        raise Refused(f"source {source['path']}: sha256 {got[:12]}..., pinned {source['sha256'][:12]}...")
    return b


def cohort_arrays(cid: str, manifest: dict, *, root: str = DROPBOX):
    """(ages, events) of a v6 cohort, refused unless their content digest is the manifest's."""
    from panel_003g import cohort_digest
    with np.load(os.path.join(root, COHORTS_V6, f"{cid}.npz")) as z:
        a, e = np.asarray(z["ages"], float), np.asarray(z["events"], int)
    if cohort_digest(a, e) != manifest["checksums"][cid]:
        raise Refused(f"{cid}: the array's content digest is not the manifest's")
    return a, e


# --------------------------------------------------------------------------------------------- per cohort: CV, r
def candidate(spec: dict) -> str | None:
    """A heterogeneous v6 cohort's candidate stratum ('real' before its median split), or None."""
    if not (spec.get("het_param") and float(spec.get("het_var") or 0.0) > 0.0):
        return None
    if spec["arm"] == "realfull":
        return "realfull"
    if spec["arm"] != "het":
        return None
    name = spec.get("real_name")
    if not name:
        cls = next((p[3:] for p in str(spec["stratum"]).split("/") if p.startswith("cv-")), None)
        if cls not in ("broad", "mid", "sharp"):
            raise Refused(f"{spec['cohort_id']}: no cv class in its manifest stratum {spec['stratum']!r}")
        return f"syn-{cls}"
    if name.startswith(HUMAN):
        if spec["n"] not in HUMAN_N:
            raise Refused(f"{spec['cohort_id']}: a human cohort at n = {spec['n']}, not one of {sorted(HUMAN_N)}")
        return HUMAN_N[spec["n"]]
    return "real"


def physical_truth(spec: dict) -> dict:
    """The tester's single conversion site (srbench/library/spec.py physical_truth), at xc = 1 and L = L_phys."""
    L = float(spec["L_phys"])
    return {"eta": spec["rho_eta"] / (L * L), "beta": spec["rho_beta"] / L, "epsilon": spec["rho_eps"] / L,
            "kappa": float(spec["kappa"])}


@functools.lru_cache(maxsize=None)
def _fine_march():
    import jax
    from sraging.engine import stepper as ST
    nst = int(round(FINE["tau_max"] / FINE["dt"]))
    spec = ST.MarchSpec(N=FINE["N"], nsteps=nst, dt=FINE["dt"], nstart=20, Gtol=FINE["Gtol"])
    tau = np.arange(nst + 1) * FINE["dt"]
    return jax.jit(lambda th: ST.truncate(ST.make_march(spec, "value")(th), tau, FINE["dt"], FINE["Gtol"], nst)[0]), tau


def cv_c(truth: dict) -> float:
    """The CV of the age at death of the homogeneous SR model at `truth` (no hazard): one march at L = the model's
    mean lifetime (sraging.engine.curves.model_L), CV = sqrt(m2 - m1^2) / m1 with m1 = int S, m2 = int 2 tau S."""
    import jax.numpy as jnp
    from sraging import SRParams
    from sraging.engine.curves import model_L
    p = SRParams(**truth)
    L = model_L(p, scheme="bdf2")
    march, tau = _fine_march()
    N = FINE["N"]
    yf = (np.arange(N) + 0.5) / N
    theta = (L ** 2 * p.eta, L * p.epsilon, jnp.asarray(L * p.beta * yf / (yf + p.kappa)))
    S = np.asarray(march(theta), float)
    m1, m2 = float(np.trapezoid(S, tau)), float(np.trapezoid(2.0 * tau * S, tau))
    return math.sqrt(max(m2 - m1 * m1, 0.0)) / m1


def measure(spec: dict, ages, events) -> dict:
    d = np.asarray(ages, float)[np.asarray(events) == 1]
    cvd = float(np.std(d) / np.mean(d))
    cvc = cv_c(physical_truth(spec))
    return {"cv_c": cvc, "cv_data": cvd, "r": math.sqrt(max(cvd * cvd - cvc * cvc, 0.0)) / cvc}


# ------------------------------------------------------------------------------------------------- the selection
def _key(cid: str, salt: str = SALT) -> str:
    return hashlib.sha256((cid + salt).encode()).hexdigest()


def require_held_out(rows) -> None:
    bad = sorted({r["cohort_id"] for r in rows} & set(HELD_OUT))
    if bad:
        raise Refused(f"held out of the panel (the het-tradeoff pilot's cohorts, on which the guide was fitted): {bad}")


def _pick(cands, k, stratum):
    if len(cands) < k:
        raise Refused(f"stratum {stratum}: {len(cands)} candidates for {k} cohorts")
    srt = sorted(cands, key=lambda r: (r["r"], r["cohort_id"]))
    return [min((srt[i] for i in b), key=lambda r: _key(r["cohort_id"])) for b in np.array_split(np.arange(len(srt)), k)]


def _subsets(rows) -> None:
    """Adds the subset roles in place (module docstring)."""
    def order(stratum, subset):
        return sorted((r for r in rows if r["stratum"] == stratum), key=lambda r: _key(r["cohort_id"], f"{SALT}:{subset}"))
    for s in STRATA:
        order(s, "subset-hr3")[0]["roles"].append("subset-hr3")
    for s in HR3_EXTRA:
        order(s, "subset-hr3")[1]["roles"].append("subset-hr3")
    humans = sorted((r for r in rows if r["stratum"].startswith("human-")),
                    key=lambda r: _key(r["cohort_id"], f"{SALT}:subset-composite"))
    sharp = order("syn-sharp", "subset-composite")
    for i, sd in enumerate(SDS):
        for r in (humans[i], sharp[i]):
            r["roles"].append(f"subset-composite@{sd:g}")
    for s in HR2_STRATA:
        order(s, "subset-hr2")[0]["roles"].append("subset-hr2")


def select(candidates) -> list:
    """The 40 panel rows from the candidate rows ({cohort_id, candidate, n, cv_c, cv_data, r}), in STRATA order and by
    r within a stratum, each with its roles."""
    cands = [dict(r) for r in candidates if r["cohort_id"] not in HELD_OUT]
    real = [r for r in cands if r["candidate"] == "real"]
    if real:
        med = float(np.median([r["r"] for r in real]))
        for r in real:
            r["candidate"] = "real-lo-r" if r["r"] <= med else "real-hi-r"
    rows = []
    for s, k in STRATA.items():
        for r in sorted(_pick([c for c in cands if c["candidate"] == s], k, s), key=lambda r: (r["r"], r["cohort_id"])):
            rows.append({"cohort_id": r["cohort_id"], "stratum": s, "n": int(r["n"]), "cv_c": r["cv_c"],
                         "cv_data": r["cv_data"], "r": r["r"], "roles": ["panel"]})
    _subsets(rows)
    require_held_out(rows)
    return rows


def build(panel: dict | None = None, *, root: str = DROPBOX, progress=False) -> list:
    """The panel's rows, measured from the pinned manifest and arrays (about 1 CPU-s per candidate)."""
    pin = (panel or {}).get("sources", {}).get("manifest_v6", MANIFEST_V6)
    if pin["sha256"] != MANIFEST_V6["sha256"] or pin["path"] != MANIFEST_V6["path"]:
        raise Refused(f"panel_004.toml pins manifest {pin}, not this tool's {MANIFEST_V6}")
    man = json.loads(read_source(pin, root=root))
    cands = []
    for spec in man["specs"]:
        stratum = candidate(spec)
        if stratum is None or spec["cohort_id"] in HELD_OUT:
            continue
        a, e = cohort_arrays(spec["cohort_id"], man, root=root)
        cands.append({"cohort_id": spec["cohort_id"], "candidate": stratum, "n": spec["n"], **measure(spec, a, e)})
        if progress and len(cands) % 25 == 0:
            print(f"  measured {len(cands)} cohorts", flush=True)
    return select(cands)


# ---------------------------------------------------------------------------------------------- files: tsv, toml
def _num(x) -> str:
    return f"{float(x):.6g}"


def tsv_bytes(rows) -> bytes:
    out = io.StringIO()
    out.write("\t".join(FIELDS) + "\n")
    for r in rows:
        out.write("\t".join([r["cohort_id"], r["stratum"], str(int(r["n"])), _num(r["cv_c"]), _num(r["cv_data"]),
                             _num(r["r"]), ",".join(r["roles"])]) + "\n")
    return out.getvalue().encode()


def parse_ids(text: str) -> list:
    lines = text.splitlines()
    head = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        r = dict(zip(head, line.split("\t")))
        r["n"] = int(r["n"])
        for f in ("cv_c", "cv_data", "r"):
            r[f] = float(r[f])
        r["roles"] = r["roles"].split(",") if r["roles"] else []
        rows.append(r)
    return rows


def read_ids(path: str = IDS_PATH) -> list:
    return parse_ids(open(path).read())


def load_panel(path: str = PANEL_PATH) -> dict:
    return tomllib.load(open(path, "rb"))


_IDS_LINE = re.compile(r'(ids = \{ path = "tests/acceptance/panel_004_ids\.tsv", sha256 = ")[0-9a-f]*(" \})')


def write(rows, *, panel_path=PANEL_PATH, ids_path=IDS_PATH) -> str:
    """IDS, then its sha256 into the toml's [sources] ids line (the toml is written first, tasks T041)."""
    if not os.path.exists(panel_path):
        raise Refused(f"{panel_path} does not exist: write panel_004.toml's freeze-H content first (tasks T041)")
    text = open(panel_path).read()
    if "[sources]" not in text or not _IDS_LINE.search(text):
        raise Refused(f'{panel_path} has no [sources] line ids = {{ path = "{IDS_REL}", sha256 = "" }}')
    b = tsv_bytes(rows)
    sha = hashlib.sha256(b).hexdigest()
    open(ids_path, "wb").write(b)
    open(panel_path, "w").write(_IDS_LINE.sub(lambda m: m.group(1) + sha + m.group(2), text))
    return sha


def check(panel: dict, *, root: str = DROPBOX) -> list:
    """The differences between a rebuild and the committed panel ([] when equal)."""
    b = tsv_bytes(build(panel, root=root))
    out = []
    if b != open(IDS_PATH, "rb").read():
        out.append("panel_004_ids.tsv differs from the rebuild")
    if hashlib.sha256(b).hexdigest() != panel["sources"]["ids"]["sha256"]:
        out.append("the toml's ids sha256 is not the rebuild's")
    return out


def counts(rows) -> dict:
    return {s: sum(r["stratum"] == s for r in rows) for s in STRATA}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("build", "check"))
    args = ap.parse_args(argv)
    panel = load_panel()
    if args.action == "build":
        rows = build(panel, progress=True)
        sha = write(rows)
        print(f"wrote {IDS_REL} ({len(rows)} rows, sha256 {sha[:12]}...)")
        for s, k in counts(rows).items():
            print(f"  {s:11s} {k:3d}")
        return 0
    diffs = check(panel)
    for d in diffs:
        print("DIFF", d)
    print("panel check:", "OK" if not diffs else f"{len(diffs)} difference(s)")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
