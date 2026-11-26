"""`--window LO HI` in `sraging fit-cohort` and `fit-set` (spec 006 FR-012; contracts/cli.md; research R10; tasks
T018). Canned FitResults keep these fast; the last test fits for real (slow): the CLI's result is the Python call's."""
import csv
import dataclasses
import json
import math
import os

import numpy as np
import pytest

from sraging import Cohort, cli, fit_cohort, fitset
from sraging.cohorts import read_cohorts
from test_cli import csv_path, fake_fit, quiet  # noqa: F401 -- fixtures

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMALL = os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv")
WIN = {"window_given": [4.0, "inf"], "hi_mapped_from": "inf", "hi_used": 9.0, "window_eff": [4.0, 9.0],
       "n_dropped_below_lo": 1, "n_censored_above_hi": 0, "n_deaths_censored": 1, "n_censored_at_hi_eff": 1}


def windowed():
    r = quiet()
    return dataclasses.replace(r, diagnostics={**r.diagnostics, "window": WIN})


@pytest.fixture
def long_csv(tmp_path):
    p = tmp_path / "all.csv"
    p.write_text("cohort,age,event\n" + "".join(f"{c},{a},{e}\n" for c in ("a", "b")
                                                for a, e in ((3, 1), (5, 1), (6, 0), (8, 1), (9, 1), (12, 1))))
    return str(p)


@pytest.mark.parametrize("cmd", [["fit-cohort", "--csv", "x.csv", "--delta", "1"],
                                 ["fit-set", "--csv", "x.csv", "--delta", "1", "--time-unit", "days", "--out", "o"]])
def test_window_parses_two_floats_inf_included(cmd):
    assert cli.build_parser().parse_args(cmd + ["--window", "60", "inf"]).window == [60.0, math.inf]
    assert cli.build_parser().parse_args(cmd + ["--window", "60", "120"]).window == [60.0, 120.0]
    assert cli.build_parser().parse_args(cmd).window is None
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(cmd + ["--window", "60"])


@pytest.mark.parametrize("cal", ["profile_jax2", "profile_v033", "profile_jax2@1.0.0"])
def test_fit_cohort_refuses_a_window_before_any_fit_naming_w_and_h(cal, csv_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_fit", lambda *a, **k: pytest.fail("no fit may start"))
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--window", "4", "inf", "--calibrator",
                     cal]) == 4
    err = capsys.readouterr().err
    assert "profile_jax2_w" in err and "profile_jax2_h" in err and "window" in err


def test_fit_set_refuses_a_window_before_any_fit_naming_w_and_h(long_csv, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fitset, "fit_set", lambda *a, **k: pytest.fail("no fit may start"))
    out = tmp_path / "fits"
    assert cli.main(["fit-set", "--csv", long_csv, "--cohort-col", "cohort", "--delta", "1", "--time-unit", "days",
                     "--window", "4", "10", "--calibrator", "profile_jax2", "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "profile_jax2_w" in err and "profile_jax2_h" in err and not out.exists()


def test_the_report_prints_the_window_line(csv_path, fake_fit, capsys):
    fake_fit(windowed())
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--window", "4", "inf", "--calibrator",
                     "profile_jax2_w"]) == 0
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("window:"))
    assert "[4, 9]" in line and "hi = inf mapped to 9" in line and "1 row(s) dropped below lo" in line
    assert "1 death(s) censored at the window's edge" in line
    assert cli.window_line(quiet()) is None                     # the twin: no window, no line


def test_the_window_reaches_each_cohort_read(long_csv):
    s, refused = read_cohorts(long_csv, delta=1.0, time_unit="days", cohort_col="cohort", window=(4.0, 10.0))
    assert not refused and {s[k].window for k in s} == {(4.0, 10.0)}
    s2, _ = read_cohorts(long_csv, delta=1.0, time_unit="days", cohort_col="cohort")
    assert {s2[k].window for k in s2} == {None}


def canned_windowed_fit(cohort):
    return windowed()


def test_the_summary_gains_the_window_columns(tmp_path):
    assert {"window_lo_eff", "window_hi_eff", "n_edge_censored"} <= set(fitset.SUMMARY_FIELDS)
    c = Cohort.from_arrays([3.0, 5.0, 8.0], [1, 0, 1], delta=1.0, name="a", time_unit="days")
    row = fitset._fit_one("a", fitset._payload(c), canned_windowed_fit, str(tmp_path / "a.json"))
    assert (row["window_lo_eff"], row["window_hi_eff"], row["n_edge_censored"]) == (4.0, 9.0, 1)
    row0 = fitset._fit_one("a", fitset._payload(c), lambda co: quiet(), str(tmp_path / "b.json"))   # the twin
    assert row0.get("window_lo_eff") is None and row0.get("n_edge_censored") is None


@pytest.mark.slow
def test_the_cli_fit_is_the_python_fit(tmp_path, capsys):
    out = str(tmp_path / "w.json")
    assert cli.main(["fit-cohort", "--csv", SMALL, "--delta", "1", "--time-unit", "days", "--window", "40", "70",
                     "--calibrator", "profile_jax2_w", "--no-combinations", "--json", out]) == 0
    from sraging.result import FitResult
    via_cli = FitResult.from_json(out)
    py = fit_cohort(Cohort.from_csv(SMALL, delta=1.0, window=(40.0, 70.0), time_unit="days"),
                    calibrator="profile_jax2_w", combinations=False)
    assert via_cli.identity_view() == py.identity_view()
    assert via_cli.diagnostics["window"] == py.diagnostics["window"]
