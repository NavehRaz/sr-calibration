"""`sraging fit-set` and `sraging clean` (spec 005 US6; FR-021 to FR-023; contracts/cli.md; tasks T034), and
fit-cohort unchanged but for its leverage line naming spec 006. Synthetic files; fit-set with the seconds-fast stand-in
calibrator (tests/unit/fitset_fast_calibrator.py), registered for the test only."""
import csv
import json
import os

import numpy as np
import pytest
from scipy.stats import norm

import fitset_fast_calibrator as FAST
from sraging import cli, registry
from test_cli import csv_path, fake_fit, quiet  # noqa: F401 -- fixtures


def _ages(center, n=40):
    return np.round(center * (1 + 0.2 * norm.ppf(np.linspace(0.02, 0.98, n))))


@pytest.fixture
def cage_csv(tmp_path):
    """A long CSV: cohorts x (12 cages, so the floors pool) and y (5 cages, one planted outlier)."""
    rows = ["cohort,age,event,cage_id"]
    rng = np.random.default_rng(0)
    for co, centers in (("x", 50 * (1 + 0.05 * rng.standard_normal(12))), ("y", [50, 51, 49, 52, 80])):
        for i, c in enumerate(centers):
            rows += [f"{co},{a:g},1,{i}" for a in _ages(c)]
    p = tmp_path / "cages.csv"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setitem(registry._REGISTRY, "fast_test", {"1": FAST.SPEC})
    monkeypatch.setitem(registry._CURRENT, "fast_test", "1")


CLEAN = ["--cohort-col", "cohort", "--cage-col", "cage_id", "--delta", "1", "--time-unit", "days", "--k", "2.5",
         "--seed", "3"]


def test_fit_set_and_clean_parse():
    p = cli.build_parser()
    a = p.parse_args(["fit-set", "--csv", "f.csv", "--delta", "1", "--time-unit", "days", "--workers", "2",
                      "--out", "o"])
    assert (a.command, a.workers, a.time_unit, a.cohort_col) == ("fit-set", 2, "days", None)
    b = p.parse_args(["clean", "--csv", "f.csv", *CLEAN, "--out", "o", "--axes", "median", "--floors", "0.1,0.2"])
    assert (b.command, b.k, b.seed, b.axes, b.floors, b.min_cages, b.drop) == ("clean", 2.5, 3, "median", "0.1,0.2",
                                                                               10, False)
    assert b.spread_by is None and p.parse_args(["clean", "--csv", "f.csv", *CLEAN, "--out", "o", "--spread-by",
                                                 "line"]).spread_by == "line"


@pytest.mark.parametrize("missing", ["--k", "--seed"])
def test_clean_without_k_or_seed_exits_2_naming_it(cage_csv, tmp_path, capsys, missing):
    args = list(CLEAN)
    i = args.index(missing)
    del args[i:i + 2]
    with pytest.raises(SystemExit) as ex:
        cli.main(["clean", "--csv", cage_csv, *args, "--out", str(tmp_path / "o")])
    assert ex.value.code == 2 and missing in capsys.readouterr().err


def test_clean_writes_the_report_and_only_with_drop_the_cleaned_set(cage_csv, tmp_path, capsys):
    out = tmp_path / "o"
    assert cli.main(["clean", "--csv", cage_csv, *CLEAN, "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "cleaning.csv")))
    assert len(rows) == 17 and {r["verdict"] for r in rows} == {"kept", "flagged"}
    assert [r["cage"] for r in rows if r["verdict"] == "flagged" and r["cohort"] == "y"] == ["4"]
    rep = json.load(open(out / "cleaning.json"))
    assert {c["cohort"] for c in rep["cohorts"]} == {"x", "y"} and rep["floors_source"].startswith("pooled")
    assert [n["null"] for n in rep["cohorts"][0]["noise"]] == ["floor", "own_spread"] and rep["seed"] == 3
    assert not (out / "cleaned").exists()
    out2 = tmp_path / "o2"
    assert cli.main(["clean", "--csv", cage_csv, *CLEAN, "--out", str(out2), "--drop"]) == 0
    y = list(csv.DictReader(open(out2 / "cleaned" / "y.csv")))
    assert sorted({r["cage"] for r in y}) == ["0", "1", "2", "3"] and set(y[0]) == {"age", "event", "cage"}
    assert "sha256" in capsys.readouterr().out


def test_clean_spread_by_pools_the_own_spread_null_and_the_lines_name_the_pool(cage_csv, tmp_path, capsys):
    """sr-naveh_005 B2: --spread-by COLUMN pools the own-spread null over the cohorts sharing the column's value
    (default: every cohort); each cohort's line names its pool, and the nulls are scatter, not "noise alone"."""
    head, *rows = open(cage_csv).read().splitlines()
    lined = tmp_path / "lined.csv"
    lined.write_text("\n".join([head + ",line"] + [r + (",L1" if r.startswith("x,") else ",L2") for r in rows]) + "\n")
    assert cli.main(["clean", "--csv", str(lined), *CLEAN, "--out", str(tmp_path / "o"), "--spread-by", "line"]) == 0
    out = capsys.readouterr().out
    assert "  x: " in out and "expected from scatter like the floor: " in out and "noise alone" not in out
    assert "like the group's own spread (line = L1: 1 cohorts, 12 cages): " in out
    assert "like the group's own spread (line = L2: 1 cohorts, 5 cages): " in out
    rep = json.load(open(tmp_path / "o" / "cleaning.json"))
    assert rep["spread_by"] == "line"
    assert {c["cohort"]: (c["noise"][1]["spread_by"], c["noise"][1]["pool"]) for c in rep["cohorts"]} == {
        "x": ("line", "L1"), "y": ("line", "L2")}
    assert cli.main(["clean", "--csv", str(lined), *CLEAN, "--out", str(tmp_path / "o2")]) == 0
    assert out.count("\n  ") == 2 and "(all cohorts: 2 cohorts, 17 cages): " in capsys.readouterr().out
    for bad, why in (("arm", "is not a column"), ("cage", "varies within cohort")):
        assert cli.main(["clean", "--csv", str(lined), *CLEAN, "--out", str(tmp_path / "o3"), "--spread-by",
                         bad]) == 2
        err = capsys.readouterr().err
        assert repr(bad) in err and why in err


def test_clean_csv_needs_a_cohort_column_and_refused_rows_exit_3(cage_csv, tmp_path, capsys):
    args = [a for a in CLEAN if a not in ("--cohort-col", "cohort")]
    with pytest.raises(SystemExit) as ex:
        cli.main(["clean", "--csv", cage_csv, *args, "--out", str(tmp_path / "o")])
    assert ex.value.code == 2 and "--cohort-col" in capsys.readouterr().err
    bad = tmp_path / "bad.csv"
    bad.write_text(open(cage_csv).read() + "x,abc,1,0\n")
    assert cli.main(["clean", "--csv", str(bad), *CLEAN, "--out", str(tmp_path / "o")]) == 3
    assert "non-numeric" in capsys.readouterr().err


@pytest.mark.parametrize("cmd,extra", [("fit-set", []), ("clean", CLEAN[:2] + CLEAN[2:4] + CLEAN[8:])])
def test_an_unknown_option_exits_2(cmd, extra, capsys):
    with pytest.raises(SystemExit) as ex:
        cli.main([cmd, "--csv", "f.csv", "--delta", "1", "--time-unit", "days", "--out", "o", *extra, "--bogus",
                  "1"])
    assert ex.value.code == 2 and "--bogus" in capsys.readouterr().err


def test_fit_set_fits_each_cohort_and_a_failure_exits_4(tmp_path, fast, capsys):
    d = tmp_path / "in"
    d.mkdir()
    q = (np.arange(60) + 0.5) / 60
    for name, scale in (("a", 50.0), ("b", 70.0)):
        ages = np.round(scale * (-np.log1p(-q)) ** 0.25) + 1.0
        (d / f"{name}.csv").write_text("age,event\n" + "".join(f"{x:g},1\n" for x in ages))
    out = tmp_path / "out"
    assert cli.main(["fit-set", "--dir", str(d), "--delta", "1", "--time-unit", "weeks", "--calibrator",
                     "fast_test@1", "--workers", "2", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "summary.csv")))
    assert [r["cohort"] for r in rows] == ["a", "b"] and {r["status"] for r in rows} == {"ok"}
    assert json.load(open(out / "a.json"))["theta_hat"]["time_unit"] == "weeks"            # FR-009
    (d / "bad.csv").write_text("age,event\n3,1\n4,1\n5,1\n")
    assert cli.main(["fit-set", "--dir", str(d), "--delta", "1", "--time-unit", "weeks", "--calibrator",
                     "fast_test", "--out", str(tmp_path / "out2")]) == 4
    assert "bad" in capsys.readouterr().out


def test_fit_cohort_is_unchanged_and_its_leverage_line_names_spec_006(csv_path, fake_fit, capsys):  # noqa: F811
    fake_fit(quiet())
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"])
    out = capsys.readouterr().out
    assert "leverage: NOT RUN (a profile_v033 result written before spec 006)" in out           # spec 006 T040
    assert "RESULT ok — no warnings fired" in out
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_v033", "--combinations",
              "all"])
    assert "--combinations all ignored: profile_v033 gives no combination claims" in capsys.readouterr().err   # C13


def test_importing_the_plots_extra_succeeds():
    pytest.importorskip("matplotlib")
    import sraging.plots as PL
    PL._require()
    import matplotlib
    assert matplotlib.__version__ == "3.10.8"
