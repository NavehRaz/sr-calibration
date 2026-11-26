"""`sraging fit-cohort`: options, exit codes, the summary-line rule (T024; contracts/cli.md).

The fit is monkeypatched to return canned FitResults so these stay fast; real fits are in
test_fit_small.py (slow).
"""
import json

import pytest

from sraging import cli
from sraging.result import CheckRecord, FitResult
from test_result_json import make

CSV = "death times,events\n3,1\n5,0\n8,1\n"
CODES = ("BASIN_GAIN", "BASIN_OUTSIDE", "CENSOR_TIE", "DELTA_GRID", "BOX_EDGE", "FLOORED_DEATHS", "AGE0_DEATHS")


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text(CSV)
    return str(p)


@pytest.fixture
def fake_fit(monkeypatch):
    box = {}

    def install(result):
        box["result"] = result
        monkeypatch.setattr(cli, "_fit", lambda cohort, calibrator, version, combinations=None: result)
    return install


def quiet():
    return make(warnings=(), checks=tuple(CheckRecord(c, True, False) for c in CODES))


def test_ok_clean(csv_path, fake_fit, capsys):
    fake_fit(quiet())
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"]) == 0
    out = capsys.readouterr().out
    assert "RESULT ok — no warnings fired" in out
    assert "checks:" in out and all(c in out for c in CODES)


def test_warning_blocks_clean_summary(csv_path, fake_fit, capsys):
    fake_fit(make())                                  # BASIN_GAIN fired
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"]) == 0
    out = capsys.readouterr().out
    assert "no warnings fired" not in out
    assert "1 WARNING(S): BASIN_GAIN" in out and "WARNING BASIN_GAIN" in out


def test_checks_not_run_blocks_clean_summary(csv_path, fake_fit, capsys):
    fake_fit(make(warnings=(), checks=tuple(CheckRecord(c, False, False) for c in CODES)))
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"]) == 0
    out = capsys.readouterr().out
    assert "no warnings fired" not in out and "NOT RUN" in out


def test_json_written(csv_path, fake_fit, tmp_path):
    fake_fit(quiet())
    out = tmp_path / "out.json"
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["schema"] == "sraging.fitresult/4"


@pytest.mark.parametrize("args", [[], ["--delta", "1", "--continuous"]])
def test_delta_rule_exit_2(csv_path, args, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["fit-cohort", "--csv", csv_path] + args)
    assert ei.value.code == 2
    assert "delta" in capsys.readouterr().err


@pytest.mark.parametrize("opt", ["--window", "--xc", "--engine", "--h-ext", "--heterogeneity"])
def test_unknown_options_exit_2_and_are_named(csv_path, opt, capsys):
    with pytest.raises(SystemExit) as ei:
        cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", opt, "1"])
    assert ei.value.code == 2
    assert opt in capsys.readouterr().err


def test_refused_csv_exit_3(tmp_path, capsys):
    p = tmp_path / "bad.csv"
    p.write_text("death times,events\n3,1\n0,0\n4,2\n")
    assert cli.main(["fit-cohort", "--csv", str(p), "--delta", "1"]) == 3
    err = capsys.readouterr().err
    assert "row 3:" in err and "row 4:" in err


def test_unknown_calibrator_exit_4(csv_path, capsys):
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "nope"]) == 4
    assert "profile_v033" in capsys.readouterr().err


def test_failed_fit_exit_5(csv_path, fake_fit):
    fake_fit(make(status="error:FloatingPointError"))
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"]) == 5


def test_all_starts_failed_is_reported_as_failed(csv_path, fake_fit, capsys):
    """D-015: v0.3.3 returns status ok with four none claims when every MLE start fails; the summary line
    must not say ok (T071; FR-018), and the exit code is the failed-fit code."""
    r = make(warnings=(), checks=tuple(CheckRecord(c, True, False) for c in CODES),
             diagnostics={"error": "all starts failed"})
    fake_fit(r)
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"]) == 5
    out = capsys.readouterr().out
    assert "RESULT FAILED — every optimiser start failed" in out and "RESULT ok" not in out


def test_a_fired_floored_deaths_check_prints_ages_categories_and_advice(csv_path, fake_fit, capsys):
    """Spec 002 FR-026: the CLI shows the sixth check's ages, categories and what to do."""
    from sraging import floors
    from sraging.result import Warning
    rep = {"deaths": [{"row": 1, "age": 0.0, "term_v033": -30.5, "term_v1": -36.8, "term_refined": -114.0,
                       "mass_refined": 3e-50, "converged": True, "category": "impossible"}],
           "counts": {"horizon": 0, "impossible": 1, "grid": 0, "cancellation": 0}}
    fake_fit(make(warnings=(Warning("FLOORED_DEATHS", floors.message(rep), rep),),
                  checks=tuple(CheckRecord(c, True, c == "FLOORED_DEATHS") for c in CODES)))
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"]) == 0
    out = capsys.readouterr().out
    assert "WARNING FLOORED_DEATHS 1 death(s)" in out
    assert "1 impossible (age 0: -30.5 in this fit, -36.8 corrected, -114.0 refined)" in out
    assert "refit without the deaths before a cut-off" in out and "1 WARNING(S): FLOORED_DEATHS" in out


# ---- spec 003 (T014): jax2 calibrators -------------------------------------------------------------------

def jax2_result(**over):
    from dataclasses import replace
    r = quiet()
    prov = {**r.provenance, "calibrator": {"name": "profile_jax2", "version": "1.0.0"}, "numerics": "jax2"}
    kw = dict(provenance=prov, diagnostics={**r.diagnostics, "assumptions": "profile_jax2's claims assume no "
                                            "external mortality: ..."},
              mechanisms={"basin_refit": {"fired": True, "rounds": 1}, "relabelled_rows": 6, "N": 1800,
                          "dt_floor": {"binds": False, "dt": None}, "n_fallback": 3, "widened": ["beta"]},
              gap_nats=0.3, archive_points=4000,
              combinations={"on": False, "cost_s": 0.0, "order": [], "order_source": "x"})
    kw.update(over)
    return replace(r, **kw)


def test_a_jax2_summary_prints_its_own_cost_assumption_mechanisms_and_gap(csv_path, fake_fit, capsys):
    fake_fit(jax2_result())
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2"]) == 0
    out = capsys.readouterr().out
    assert "calibrator profile_jax2@1.0.0" in out and "fit cost" in out and "reference cost" not in out
    assert "no external mortality" in out
    assert "basin re-fit fired (1 rounds)" in out and "relabel 6 rows" in out and "N=1800" in out
    assert "0.3 nats (below the 1.92-nat threshold)" in out


def test_the_hazard_twin_prints_h_as_not_scored(csv_path, fake_fit, capsys):
    fake_fit(jax2_result(nuisance={"h_ext": 1e-4, "hL": 0.09, "ExtH": 9.2, "scored": False}))
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2_h"])
    assert "external hazard (nuisance, not scored): h=0.0001" in capsys.readouterr().out


def test_a_v033_summary_has_no_jax2_lines(csv_path, fake_fit, capsys):
    fake_fit(quiet())
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"])
    out = capsys.readouterr().out
    assert "reference cost median 590" in out and "mechanisms:" not in out and "fit cost" not in out


def test_no_combinations_is_ignored_with_a_note_for_v033(csv_path, fake_fit, capsys):
    fake_fit(quiet())
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_v033",
                     "--no-combinations"]) == 0
    assert "ignored: profile_v033 gives no combination claims" in capsys.readouterr().err


def test_an_unknown_calibrator_exits_4_and_lists_the_registered_ones(csv_path, capsys):
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "nope"]) == 4
    err = capsys.readouterr().err
    assert "profile_jax2@1.0.0" in err and "profile_v033@0.3.3" in err


def test_a_reraised_fit_error_exits_5_naming_the_cohort(csv_path, monkeypatch, capsys):
    def boom(*a, **k):
        raise ValueError("planted")
    monkeypatch.setattr(cli, "_fit", boom)
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2"]) == 5
    assert "the fit failed on cohort" in capsys.readouterr().err


# ---- spec 003: which combinations (Naveh, 2026-10-06), their cost line and FR-018's note -------------------------

@pytest.fixture
def seen_fit(monkeypatch):
    seen = {}

    def install(result):
        def fit(cohort, calibrator, version, combinations=None):
            seen["combinations"] = combinations
            return result
        monkeypatch.setattr(cli, "_fit", fit)
        return seen
    return install


@pytest.mark.parametrize("flags,want", [([], None), (["--combinations", "all"], "all"),
                                        (["--combinations", "lead"], "lead"), (["--no-combinations"], False)])
def test_the_combination_flags_reach_the_fit(csv_path, seen_fit, flags, want):
    seen = seen_fit(jax2_result())
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2", *flags]) == 0
    assert seen["combinations"] == want


def test_combinations_and_no_combinations_together_are_refused(csv_path, seen_fit):
    seen_fit(jax2_result())
    with pytest.raises(SystemExit) as ex:
        cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2",
                  "--combinations", "all", "--no-combinations"])
    assert ex.value.code == 2


def test_a_jax2_summary_prints_the_combination_cost_and_which_set(csv_path, fake_fit, capsys):
    on = {"on": True, "mode": "lead", "lead_group": ["eta_over_eps", "rbar_over_eta"], "cost_s": 120.0,
          "fit_cost_s": 100.0, "ratio_to_fit": 1.2, "order": ["eta_over_eps", "rbar_over_eta"],
          "order_source": "measured (order_003.json)", "combination_claims_contradicted": []}
    fake_fit(jax2_result(combinations=on))
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2"])
    out = capsys.readouterr().out
    assert "combination claims (the lead group: eta_over_eps, rbar_over_eta; --combinations all adds the other " \
           "4): 120 CPU-s (1.2x the fit); order measured (order_003.json)" in out
    fake_fit(jax2_result(combinations={**on, "mode": "all", "order": ["eta_over_eps", "rbar_over_eta", "rbar"],
                                       "combination_claims_contradicted": [
                                           {"parameter": "eta_over_eps", "side": "lower", "claim_end": 0.3,
                                            "all_walks_end": 0.2, "dex": 0.18}]}))
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2"])
    out = capsys.readouterr().out
    assert "combination claims (all 3): 120 CPU-s" in out
    assert "the later walks evaluated points beyond 1 lead claim end(s) within the threshold (eta_over_eps lower)" in out


def test_a_v033_summary_says_combinations_are_given_only_where_truth_scored(csv_path, fake_fit, capsys):
    fake_fit(quiet())
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"])
    out = capsys.readouterr().out
    # spec 003g FR-013: the note names the version that was truth-scored (1.0.0), not 1.0.1
    assert "combination claims are given only by the jax2 calibrators, whose combination claims were truth-scored at " \
           "1.0.0" in out


def test_the_api_ignores_combinations_for_v033_with_a_note(monkeypatch):
    import sraging
    from sraging import registry
    r = quiet()
    monkeypatch.setattr(registry, "get", lambda name, version=None: type("S", (), {"name": "profile_v033",
                                                                                     "fit": staticmethod(lambda c: r)}))
    out = sraging.fit_cohort(sraging.Cohort.from_arrays([3.0, 5.0], [1, 0], delta=1.0), combinations="all")
    assert any("ignored" in n and "truth-scored" in n for n in out.diagnostics["notes"])


# ---- spec 003b (T020): the _h 1.0.1 note, the v033 leverage line, the kappa-fixed refusal -------------------------

def _h101(**over):
    from sraging.warnings import HAZARD_SHARP_LIMIT
    r = jax2_result(nuisance={"h_ext": 1e-4, "hL": 0.09, "ExtH": 9.2, "scored": False})
    d = {k: v for k, v in r.diagnostics.items() if k != "assumptions"}
    d.update(notes=[dict(HAZARD_SHARP_LIMIT)], leverage={"label": "no_flag", "reason": "nothing to flag",
                                                         "cost": {"cpu_s": 3.0, "share_of_fit": 0.01}})
    prov = {**r.provenance, "calibrator": {"name": "profile_jax2_h", "version": "1.0.1"}}
    from dataclasses import replace
    return replace(r, diagnostics=d, provenance=prov, **over)


def test_an_h_1_0_1_summary_prints_the_note_and_stays_clean(csv_path, fake_fit, capsys):
    fake_fit(_h101())
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2_h"]) == 0
    out = capsys.readouterr().out
    line = [ln for ln in out.splitlines() if ln.startswith("note: profile_jax2_h 1.0.1")]
    assert len(line) == 1 and "[T-H-HAZARD-SHARP]" in line[0] and "13.6 nats" in line[0]
    assert "69 cohorts the register INFERRED" in line[0] and "36 are valid on v6" in line[0]
    assert "none more than 1.92" in line[0]
    assert "RESULT ok — no warnings fired" in out and "WARNING" not in out


def test_an_h_1_0_0_summary_has_no_note(csv_path, fake_fit, capsys):
    from dataclasses import replace
    r = _h101()
    fake_fit(replace(r, diagnostics={k: v for k, v in r.diagnostics.items() if k != "notes"},
                     provenance={**r.provenance, "calibrator": {"name": "profile_jax2_h", "version": "1.0.0"}}))
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_jax2_h@1.0.0"])
    assert "T-H-HAZARD-SHARP" not in capsys.readouterr().out


def test_a_v033_summary_says_the_leverage_check_was_not_run_and_its_result_is_unchanged(csv_path, fake_fit, capsys):
    """Spec 006: profile_v033 has its own check; a result without its block was written before spec 006."""
    r = quiet()
    before = r.to_json()
    fake_fit(r)
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"])
    out = capsys.readouterr().out
    assert "leverage: NOT RUN (a profile_v033 result written before spec 006)" in out
    assert "RESULT ok — no warnings fired" in out and r.to_json() == before and "notes" not in r.diagnostics


def test_a_kappa_fixed_name_is_refused_with_its_reason(csv_path, capsys):
    assert cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator",
                     "mle_profile_jax2_kfix_k1"]) == 4
    err = capsys.readouterr().err
    assert "mle_profile_jax2_kfix_k1" in err and "registered in a later spec" in err


# ---- spec 003g (T051-T052): the default switch -------------------------------------------------------------------
def test_the_cli_default_is_the_registry_default_and_a_name_still_selects_v033(csv_path, monkeypatch, capsys):
    from sraging import registry
    seen = {}

    def fit(cohort, calibrator, version, combinations=None):
        seen.setdefault("names", []).append(calibrator)
        raise RuntimeError("stop after the name")
    monkeypatch.setattr(cli, "_fit", fit)
    assert cli.build_parser().parse_args(["fit-cohort", "--csv", "x", "--delta", "1"]).calibrator == registry.DEFAULT
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1"])
    cli.main(["fit-cohort", "--csv", csv_path, "--delta", "1", "--calibrator", "profile_v033"])     # the twin
    assert seen["names"] == ["profile_jax2", "profile_v033"]
    assert "profile_v033" in cli.build_parser()._subparsers._group_actions[0].choices["fit-cohort"].format_help()


def test_a_stored_v033_result_reads_back_as_v033():
    """T051: a result stored before the switch keeps its calibrator and version when read back."""
    import os
    from sraging.result import FitResult
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "examples",
                     "results", "itp_c2016_het3_control_female.json")
    cal = FitResult.from_json(open(p).read()).provenance["calibrator"]
    assert (cal["name"], cal["version"]) == ("profile_v033", "0.3.3")
