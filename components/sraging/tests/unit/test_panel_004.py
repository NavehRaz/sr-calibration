"""Spec 004's heterogeneity panel (tasks T015; research R9; contracts/register-and-panel.md §2): the committed id list is
what `tools/panel_004.py` rebuilds from the pinned v6 manifest, its strata counts are the spec's, the het-tradeoff
pilot's 9 cohorts are held out, and the builder refuses a manifest whose bytes moved and a held-out id in the panel."""
import hashlib
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import panel_004 as P  # noqa: E402

HAVE_LIBRARY = os.path.exists(os.path.join(P.DROPBOX, P.MANIFEST_V6["path"]))
HAVE_IDS = os.path.exists(P.IDS_PATH) and os.path.exists(P.PANEL_PATH)

#: research R9 / tasks T015
COUNTS = {"syn-broad": 4, "syn-mid": 4, "syn-sharp": 4, "real-lo-r": 6, "real-hi-r": 6, "human-10k": 4,
          "human-25k": 4, "human-100k": 4, "realfull": 4}
#: the het-tradeoff pilot's 9 (docs/planning/het_tradeoff/evidence/cvc.json), on which the guide was fitted
PILOT_9 = {"het-real-Guiniea_pig_VC-c0-s0", "het-real-German_Shepherd_vetCompass-c0-s0",
           "het-real-Jack_Russell_vetCompass-c0-s0", "het-real-cats_vp_F-c0-s0", "het-k0.01-p03-c1-s0",
           "het-k0.01-p01-c1-s0", "het-k0.01-p00-c1-s0", "het-real-combined_human_F-c2-s0",
           "het-real-Sweden_F_1910_homo-c0-s0"}
#: the v6 heterogeneous cohorts per candidate stratum before the hold-out [M, the manifest]
V6_SIZES = {"syn-broad": 56, "syn-mid": 56, "syn-sharp": 56, "real": 62, "human-10k": 22, "human-25k": 22,
            "human-100k": 22, "realfull": 36}


def synthetic_candidates(seed=4):
    """Candidate rows shaped like v6's (P.measure's fields), with r drawn at random; the pilot's 9 among them."""
    rng = np.random.default_rng(seed)
    rows = []
    for s, k in V6_SIZES.items():
        for i in range(k):
            rows.append({"cohort_id": f"{s}-x{i}", "candidate": s, "n": 1000, "cv_c": 0.1, "cv_data": 0.2,
                         "r": float(rng.uniform(0.2, 3.0))})
    pilot = sorted(PILOT_9)
    for i, cid in enumerate(pilot):                                     # replace some rows by the pilot's ids
        rows[i * 7] = dict(rows[i * 7], cohort_id=cid)
    return rows


def test_the_constants_are_the_specs():
    assert P.STRATA == COUNTS and sum(P.STRATA.values()) == 40
    assert set(P.HELD_OUT) == PILOT_9
    assert P.SDS == (0.1, 0.2, 0.3, 0.5)
    assert P.MANIFEST_V6["sha256"] == "399c9c555782b4271f0e790099893180a2c5101e2e56318eb87d9c25785f7d2f"


def test_the_selection_gives_the_strata_counts_and_holds_the_pilot_out():
    rows = P.select(synthetic_candidates())
    got = {s: sum(r["stratum"] == s for r in rows) for s in COUNTS}
    assert got == COUNTS and len(rows) == 40
    assert not {r["cohort_id"] for r in rows} & PILOT_9
    assert all("panel" in r["roles"] for r in rows)


def test_the_selection_rule_bins_by_r_and_takes_the_smallest_salted_hash():
    """Within a stratum: sort by r, cut into k equal-count bins, take from each the smallest sha256(id + 'spec004')."""
    cands = [r for r in synthetic_candidates() if r["candidate"] == "syn-mid" and r["cohort_id"] not in PILOT_9]
    got = sorted(r["cohort_id"] for r in P.select(synthetic_candidates()) if r["stratum"] == "syn-mid")
    srt = sorted(cands, key=lambda r: (r["r"], r["cohort_id"]))
    bins = np.array_split(np.arange(len(srt)), 4)
    want = sorted(min((srt[i]["cohort_id"] for i in b), key=lambda c: hashlib.sha256((c + "spec004").encode()).hexdigest())
                  for b in bins)
    assert got == want


def test_the_real_non_human_set_splits_at_its_median_r():
    rows = P.select(synthetic_candidates())
    real = [r for r in synthetic_candidates() if r["candidate"] == "real" and r["cohort_id"] not in PILOT_9]
    med = float(np.median([r["r"] for r in real]))
    lo = [r for r in rows if r["stratum"] == "real-lo-r"]
    hi = [r for r in rows if r["stratum"] == "real-hi-r"]
    assert all(r["r"] <= med for r in lo) and all(r["r"] > med for r in hi)


def test_the_subsets_are_inside_the_panel_with_their_sizes():
    rows = P.select(synthetic_candidates())
    roles = [x for r in rows for x in r["roles"]]
    assert roles.count("subset-hr3") == 12 and roles.count("subset-hr2") == 6
    comp = [x for x in roles if x.startswith("subset-composite@")]
    assert sorted(comp) == sorted(f"subset-composite@{sd:g}" for sd in P.SDS for _ in range(2))      # 2 per sd
    by = {r["cohort_id"]: r for r in rows}
    for sd in P.SDS:                                                        # human and sharp first
        strata = sorted(by[c]["stratum"] for c in by if f"subset-composite@{sd:g}" in by[c]["roles"])
        assert strata[0].startswith("human-") and strata[1] == "syn-sharp"
    hr3 = {r["stratum"] for r in rows if "subset-hr3" in r["roles"]}
    assert {"syn-sharp", "human-10k", "human-25k", "human-100k"} <= hr3


def test_the_tsv_round_trips():
    rows = P.select(synthetic_candidates())
    b = P.tsv_bytes(rows)
    assert b.decode().splitlines()[0].split("\t") == list(P.FIELDS)
    back = P.parse_ids(b.decode())
    assert [r["cohort_id"] for r in back] == [r["cohort_id"] for r in rows]
    assert [r["roles"] for r in back] == [r["roles"] for r in rows]


# ------------------------------------------------------------------------------------------ planted refusals + twins
def test_a_manifest_whose_sha256_differs_is_refused(tmp_path):
    f = tmp_path / "manifest.json"
    f.write_bytes(b'{"specs": []}')
    good = {"path": "manifest.json", "sha256": hashlib.sha256(b'{"specs": []}').hexdigest()}
    assert P.read_source(good, root=str(tmp_path)) == b'{"specs": []}'                           # the twin
    with pytest.raises(P.Refused, match="sha256"):
        P.read_source(dict(good, sha256=hashlib.sha256(b"other").hexdigest()), root=str(tmp_path))


def test_a_held_out_id_forced_into_the_panel_is_refused():
    rows = P.select(synthetic_candidates())
    P.require_held_out(rows)                                                                     # the twin
    forced = rows[:-1] + [dict(rows[-1], cohort_id="het-real-Jack_Russell_vetCompass-c0-s0")]
    with pytest.raises(P.Refused, match="held out"):
        P.require_held_out(forced)


def test_a_stratum_too_small_for_its_count_is_refused():
    rows = synthetic_candidates()
    realfull = [r for r in rows if r["candidate"] == "realfull"]
    few = [r for r in rows if r["candidate"] != "realfull"] + realfull[:3]
    with pytest.raises(P.Refused, match="realfull"):
        P.select(few)
    assert len(P.select(few + realfull[3:4])) == 40                                              # the twin: 4 is enough


# ---------------------------------------------------------------------------------------- the committed panel (T042)
@pytest.mark.skipif(not HAVE_IDS, reason="NOT BUILT: panel_004_ids.tsv and panel_004.toml are written at freeze-H "
                                         "(tasks T041-T042)")
def test_the_committed_ids_have_the_counts_and_hold_the_pilot_out():
    rows = P.read_ids()
    assert {s: sum(r["stratum"] == s for r in rows) for s in COUNTS} == COUNTS
    assert not {r["cohort_id"] for r in rows} & PILOT_9
    b = open(P.IDS_PATH, "rb").read()
    assert hashlib.sha256(b).hexdigest() == P.load_panel()["sources"]["ids"]["sha256"]


@pytest.mark.skipif(not HAVE_IDS, reason="NOT BUILT: panel_004_ids.tsv and panel_004.toml are written at freeze-H "
                                         "(tasks T041-T042)")
@pytest.mark.skipif(not HAVE_LIBRARY, reason="NOT APPLICABLE: the v6 library is absent on this host")
def test_a_rebuild_from_the_pinned_manifest_equals_the_committed_ids():
    assert P.check(P.load_panel()) == []
