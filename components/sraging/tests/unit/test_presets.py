"""Legacy presets (spec 005 US7; research R9; tasks T005, T038): the package-data copy of SRK's Preset_values, pinned
against the frozen snapshot (constitution VI: the library never reads tests/oracle at run time)."""
import hashlib
import json
import os
from importlib import resources

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SNAP = "tests/oracle/fp_piecewise_cb191f5/SRtools_kappa/SRK/Preset_values"
NAMES = ("summery_mode_overall.csv", "summery_mode.csv", "summery_max_likelihood.csv")


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def test_package_copy_matches_snapshot():
    """T005 (R9): each packaged CSV is a byte copy of the snapshot's, and MANIFEST.json names its sha256 and its
    snapshot path; nothing else is packaged."""
    pkg = resources.files("sraging") / "data" / "presets_legacy"
    man = json.loads((pkg / "MANIFEST.json").read_text())
    assert sorted(man["files"]) == sorted(NAMES)
    for name in NAMES:
        copy = (pkg / name).read_bytes()
        snap = open(os.path.join(REPO, SNAP, name), "rb").read()
        assert _sha(copy) == _sha(snap) == man["files"][name]["sha256"], name
        assert man["files"][name]["snapshot"] == f"{SNAP}/{name}"
    assert sorted(p.name for p in pkg.iterdir() if p.is_file()) == sorted([*NAMES, "MANIFEST.json"])


# ------------------------------------------------------------------------------------- US7: get and names (T038)
import math  # noqa: E402

import pytest  # noqa: E402

import dataclasses  # noqa: E402

from sraging import units  # noqa: E402
from sraging.errors import UnsupportedOptionError  # noqa: E402
from sraging.presets import FILES, KAPPA_RULE, PresetInfo, auto_unit, get, names  # noqa: E402
from sraging.units import convert  # noqa: E402


@pytest.fixture(scope="module")
def acc():
    """tools/acceptance_005.py, whose presets_hr7 calls SRK's getTheta from the frozen snapshot through
    tools/oracle_presets_005.py (oracle group; NOT APPLICABLE without it)."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import acceptance_005 as ACC
    import oracle_presets_005 as OP
    if OP.preflight():
        pytest.skip(f"NOT APPLICABLE: {OP.preflight()}")
    return ACC


def test_presets_are_in_days_with_the_unit_stated():
    """D-502: days, the tables' unit, always, and stated; getTheta's "auto" returned mice in weeks. Rebuilt from the
    ratio rows with the legacy kappa restatement."""
    p, info = get("mice_F")
    assert p.time_unit == info.time_unit == "days" and isinstance(info, PresetInfo)
    assert info.kappa_rule == KAPPA_RULE == "0.5/xc (legacy restatement)" and info.source == "legacy"
    assert p.xc == 1.0 and p.kappa == 0.5 / 29.7 and p.h_ext == 0.0
    assert convert(p, to="weeks").eta == pytest.approx(0.0011978543153447314, rel=1e-15)    # spec 004's weeks value
    jr, ji = get("Jack Russell", ExtH=True)                  # ExtH in getTheta's auto unit (years), stored per day
    assert (ji.ext_h, ji.ext_h_unit, jr.time_unit) == (5.577, "years", "days") and jr.h_ext == math.exp(-5.577) / 365
    assert info.file_sha256 == json.loads((resources.files("sraging") / "data" / "presets_legacy" /
                                           "MANIFEST.json").read_text())["files"]["summery_mode_overall.csv"]["sha256"]


def test_every_preset_type_and_exth_loads_and_aliases_resolve():
    for t in FILES:
        assert len(names(type=t)) == 35 and not any("95% CI" in n for n in names(type=t))
        for n in names(type=t):
            for ext in (False, True):
                p, info = get(n, type=t, ExtH=ext)
                assert p.time_unit == "days" and info.type == t and info.ExtH is ext
                assert (p.h_ext == 0.0 and info.ext_h_unit is None) if not ext or info.ext_h is None else \
                    p.h_ext == math.exp(-info.ext_h) / units.UNITS[info.ext_h_unit]
    p, info = get("Mice_F")
    assert info.name == "mice_F" and info.requested == "Mice_F" and p == get("mice_F")[0]
    jr, info = get("Jack Russell", ExtH=True)
    assert info.ext_h == 5.577 and info.ext_h_unit == "years" and jr.h_ext == math.exp(-5.577) / 365


def _exth_mismatches(reading=get) -> tuple:
    """(cells, mismatches) of (i): for every preset and type with a finite ExtH, convert(params, to=ext_h_unit).h_ext
    against exp(-ExtH) within 1e-15 relative, `reading` giving (params, info) as get does."""
    cells, bad = 0, []
    for t in FILES:
        for n in names(type=t):
            p, info = reading(n, type=t, ExtH=True)
            if info.ext_h is None:
                continue
            cells += 1
            got, want = units.convert(p, to=info.ext_h_unit).h_ext, math.exp(-info.ext_h)
            if not abs(got - want) <= 1e-15 * want:
                bad.append((n, t, got, want))
    return cells, bad


def test_the_external_hazard_is_exp_minus_exth_per_auto_unit():
    """sr-naveh_005 B1 (i): the table's ExtH was fitted in getTheta's auto unit (presets.py:391-403), so in that unit
    the rate is exp(-ExtH) for every preset and type that has one (16 presets x 3 types, all in years); a preset
    without one has h_ext = 0 and no unit."""
    cells, bad = _exth_mismatches()
    assert cells == 16 * 3 and bad == [], bad[:3]
    assert {get(n, type=t, ExtH=True)[1].ext_h_unit for t in FILES for n in names(type=t)} == {"years", None}
    p, info = get("mice_F", ExtH=True)
    assert info.ext_h is None and info.ext_h_unit is None and p.h_ext == 0.0
    assert get("Jack Russell")[1].ext_h_unit is None                                       # ExtH not asked


def test_the_auto_unit_is_gettheta_rule():
    """getTheta's time_unit="auto" (presets.py:391-403) on the table names, after the alias."""
    want = {"mice_F": "weeks", "mice_M": "weeks", "ecoli": "hours", "yeast": "generations", "celegans": "days",
            "drosophila_441": "days", "Wdah_control": "days", "Guiniea_pig_VC": "years", "cats_BPH": "years",
            "combined_human_M": "years", "Jack_Russell_vetCompass": "years", "Sweden_F_1910_homo": "years"}
    assert {n: auto_unit(n) for n in want} == want


def test_jack_russell_in_years_is_spec_004s_preset():
    """B1 (ii): convert(get("Jack Russell", ExtH=True), to="years") equals tools/presets_004.py's Jack Russell (getTheta's
    values in its auto unit, h_ext = exp(-5.577) per year) in h_ext, eta, beta, epsilon and kappa, within 1e-15."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import presets_004
    want = presets_004.PRESETS["Jack_Russell"]
    assert want["time_unit"] == "years"
    got = convert(get("Jack Russell", ExtH=True)[0], to="years")
    for k in ("h_ext", "eta", "beta", "epsilon", "kappa"):
        assert getattr(got, k) == pytest.approx(want["params"][k], rel=1e-15, abs=0), k
    assert got.xc == want["params"]["xc"] == 1.0


def test_the_per_day_reading_of_exth_fails_the_auto_unit_check():
    """B1 (iii), the planted twin: exp(-ExtH) read per day (the first build) is 365 times too large in years, on every
    cell of (i)."""
    def per_day(n, **kw):
        p, info = get(n, **kw)
        return (dataclasses.replace(p, h_ext=math.exp(-info.ext_h)) if info.ext_h is not None else p), info
    cells, bad = _exth_mismatches(per_day)
    assert cells == 16 * 3 and len(bad) == cells
    assert all(g / w == pytest.approx(365, rel=1e-12) for _, _, g, w in bad)


@pytest.mark.parametrize("name,match", [("Mice", "no preset 'Mice'"), ("beta/xc", "parameter row"),
                                        ("k/xc", "parameter row"), ("mice_F 95% CI", "no preset"),
                                        ("German Shepherd", "GermanShepherd_vetCompass")])
def test_unknown_names_and_stray_alias_keys_are_refused_naming_them(name, match):
    with pytest.raises(UnsupportedOptionError, match=match) as ex:
        get(name, type="mode")
    assert ex.value.option == "name" and repr(name) in str(ex.value)


def test_any_source_but_legacy_and_an_unknown_type_are_refused():
    for src in ("j101", "organisms_J101", "auto", None):
        with pytest.raises(UnsupportedOptionError, match="source") as ex:
            get("mice_F", source=src)
        assert repr(src) in str(ex.value)
    with pytest.raises(UnsupportedOptionError, match="type"):
        get("mice_F", type="median")


def test_the_packaged_tables_parse_as_srks_pandas_read_them():
    """getTheta reads the CSVs with pandas' default parser; sraging with float(). Equal on every numeric cell of the
    three files, so the rebuilt theta starts from the same floats (oracle group: pandas)."""
    pd = pytest.importorskip("pandas")
    from sraging import presets as P
    for t, f in FILES.items():
        df = pd.read_csv(resources.files("sraging") / "data" / "presets_legacy" / f, index_col=0)
        for row, vals in P._table(t).items():
            for col, x in vals.items():
                y = float(df.loc[row, col])
                assert x == y or (math.isnan(x) and math.isnan(y)), (t, row, col)


def test_every_preset_type_and_exth_equals_srk_gettheta_in_days_and_converted_to_each_unit(acc):
    """HR-7 (T038): 35 presets x 3 types x ExtH (True, False) x 5 units against getTheta(name, type=, ExtH=,
    time_unit=u) from the frozen snapshot: days exact; converted within 1e-15 relative; xc, kappa and the raw ExtH
    exact. And the h_ext cells (B1): 35 x 3, the 48 with an ExtH equal to exp(-ExtH) of getTheta at "auto" in SRK's
    auto unit, the others h_ext = 0."""
    r = acc.presets_hr7()
    assert r["cells"] == 35 * 3 * 2 * 5 and r["mismatches"] == [], r["mismatches"][:5]
    assert 0 < r["max_rel"] <= 1e-15
    assert (r["h_ext_cells"], r["h_ext_finite"]) == (35 * 3, 16 * 3) and 0 < r["h_ext_max_rel"] <= 1e-15


@pytest.mark.parametrize("defect", ["eta_s", "ratio"])
def test_a_wrong_unit_factor_or_ratio_rebuild_fails_hr7(acc, defect):
    r = acc.presets_hr7(types=("mode_overall",), ext=(False,), units=("days", "weeks"), planted=defect)
    assert r["mismatches"] and all(m["what"] == "eta" for m in r["mismatches"])


def test_exth_read_per_day_fails_hr7s_h_ext_cells(acc):
    """B1's planted defect: exp(-ExtH) per day is 365 times getTheta's rate in years on all 16 presets with an ExtH;
    its twin, the shipped reading, passes the same cells."""
    kw = {"types": ("mode_overall",), "ext": (True,), "units": ("days",)}
    r = acc.presets_hr7(planted="h_ext_per_day", **kw)
    assert len(r["mismatches"]) == 16 and all(m["what"] == "h_ext" for m in r["mismatches"])
    assert all(m["rel"] == pytest.approx(364, rel=1e-12) for m in r["mismatches"])
    assert acc.presets_hr7(**kw)["mismatches"] == []
