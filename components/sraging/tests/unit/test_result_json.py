"""FitResult JSON: exact round trip, schema validity, unknown schema refused (T022; FR-007, R11), and schema /4
keyed by quantity (spec 003 T007; FR-026, data-model.md)."""
import json
import math
import os

import jsonschema
import pytest

from sraging.params import SRParams
from sraging.result import CHECK_CODES, CheckRecord, Claim, Claims, FitResult, Warning, SCHEMA

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA_FILE = os.path.join(REPO, "specs", "003-calibrator", "contracts", "fitresult.schema.json")   # current (/4)


def make(**over):
    claims = (
        Claim("eta", "closed", 1.25e-4, 3.5e-3, 6.1e-4, 0.95, 1, "two-sided"),
        Claim("beta", "upper_bound", None, 192.64108103949195, 0.18386864471573383, 0.95, 1, "upper"),
        Claim("epsilon", "closed", 3.848730503032917e-05, 0.01621483900570107, 0.0017, 0.95, 1, "x"),
        Claim("kappa", "none", None, None, None, 0.95, 1, "not determined"),
    )
    kw = dict(
        status="ok",
        cohort={"name": "c.csv", "n": 300, "n_deaths": 288, "n_censored": 12, "delta": 1.0,
                "continuous": False},
        theta_hat=SRParams(eta=6.1e-4, beta=0.18386864471573383, epsilon=0.0017, kappa=12.5),
        theta_labels={"eta": "a", "beta": "b", "epsilon": "c", "kappa": "d"},
        claims=claims,
        loglik=-1234.5678901234567,
        warnings=(Warning("BASIN_GAIN", "msg", {"gain_nats": 2.5, "parameters": ["eta"]}),),
        checks=tuple(CheckRecord(c, True, c == "BASIN_GAIN") for c in CHECK_CODES),
        diagnostics={"loglik": -1234.5678901234567, "dt_ceiling": None, "neg_zero": -0.0,
                     "nan": math.nan, "pinf": math.inf, "ninf": -math.inf, "grid": {"eta": {"x": 1}},
                     "compile_s": 1.0, "code": "sraging 0.1.0"},
        timing={"wall_s": 1.0, "cpu_s": 1.0},
        provenance={"package": {"version": "0.1.0", "git_sha": "abc", "dirty": False}, "numerics": "v033",
                    "calibrator": {"name": "profile_v033", "version": "0.3.3"}, "engine": "fp",
                    "grid": {"N": 900, "dt": 0.002, "tau_max": 2.0, "Gtol": 1e-11, "scheme": "bdf2",
                             "sampling_dt": 1.0, "L_km": 48.7, "dt_ceiling": None},
                    "prng": "none (deterministic path)", "runtime": {"x64": True, "pjrt_nproc": "1"},
                    "libraries": {}, "host": {}, "created_at": "2026-10-03T00:00:00Z"},
    )
    kw.update(over)
    return FitResult(**kw)


def hexes(x):
    """Every number in a JSON-like tree, as (path, float.hex) -- distinguishes -0.0 and NaN."""
    out = []
    def walk(v, path):
        if isinstance(v, bool) or v is None or isinstance(v, str):
            out.append((path, repr(v)))
        elif isinstance(v, (int, float)):
            out.append((path, float(v).hex() if isinstance(v, float) else f"int:{v}"))
        elif isinstance(v, dict):
            for k in sorted(v):
                walk(v[k], f"{path}/{k}")
        elif isinstance(v, (list, tuple)):
            for i, w in enumerate(v):
                walk(w, f"{path}[{i}]")
        else:
            out.append((path, repr(v)))
    walk(x, "")
    return out


def test_round_trip_is_exact():
    r = make()
    back = FitResult.from_json(r.to_json())
    assert hexes(back.to_dict()) == hexes(r.to_dict())
    assert back.theta_hat == r.theta_hat
    assert isinstance(back.claims[0], Claim) and isinstance(back.checks[0], CheckRecord)


def test_round_trip_through_a_file(tmp_path):
    r = make()
    p = tmp_path / "out.json"
    r.to_json(p)
    assert hexes(FitResult.from_json(p).to_dict()) == hexes(r.to_dict())


def test_schema_valid():
    schema = json.load(open(SCHEMA_FILE))
    jsonschema.validate(json.loads(make().to_json()), schema)


def test_unknown_schema_refused():
    d = json.loads(make().to_json())
    d["schema"] = "something/9"
    with pytest.raises(ValueError, match="schema"):
        FitResult.from_json(json.dumps(d))


def test_schema_constant():
    assert json.loads(make().to_json())["schema"] == SCHEMA == "sraging.fitresult/4"


def test_identity_view_drops_timings_code_and_labels():
    v = make().identity_view()
    assert set(v["diagnostics"]) == {"loglik", "dt_ceiling", "neg_zero", "nan", "pinf", "ninf", "grid"}
    assert all("label" not in c for c in v["claims"])
    assert v["claims"][3] == {"parameter": "kappa", "shape": "none", "lower": None, "upper": None,
                              "point": None, "level": 0.95, "n_components": 1}


def test_clean_only_when_all_checks_ran_and_none_fired():
    assert not make().clean
    quiet = tuple(CheckRecord(c.code, True, False) for c in make().checks)
    assert make(warnings=(), checks=quiet).clean
    notrun = tuple(CheckRecord(c.code, False, False) for c in make().checks)
    assert not make(warnings=(), checks=notrun).clean


def test_all_starts_failed_is_never_clean():
    quiet = tuple(CheckRecord(c, True, False) for c in CHECK_CODES)
    assert make(warnings=(), checks=quiet).clean
    assert not make(warnings=(), checks=quiet, diagnostics={"error": "all starts failed"}).clean


def test_a_schema_1_file_still_loads_with_the_sixth_check_not_run():
    """Spec 002: /1 files (spec 001's results) are read; FLOORED_DEATHS and AGE0_DEATHS are NOT RUN, never quiet."""
    d = json.loads(make().to_json())
    d["schema"] = "sraging.fitresult/1"
    d["checks"] = [c for c in d["checks"] if c["code"] not in ("FLOORED_DEATHS", "AGE0_DEATHS")]
    r = FitResult.from_dict(d)
    assert [(c.code, c.ran, c.fired) for c in r.checks[-2:]] == [("FLOORED_DEATHS", False, False),
                                                                ("AGE0_DEATHS", False, False)]
    assert not r.clean


def test_a_schema_2_file_still_loads_with_the_seventh_check_not_run():
    d = json.loads(make().to_json())
    d["schema"] = "sraging.fitresult/2"
    d["checks"] = [c for c in d["checks"] if c["code"] != "AGE0_DEATHS"]
    r = FitResult.from_dict(d)
    assert (r.checks[-1].code, r.checks[-1].ran) == ("AGE0_DEATHS", False) and not r.clean


def test_the_stored_examples_follow_the_current_schema():
    """A stored example is read, and what it upgrades to is valid under the current schema (spec 003: /3 files are
    upgraded on read; a stored example that no longer reads, or upgrades to an invalid /4, is stale)."""
    import glob
    schema = json.load(open(SCHEMA_FILE))
    paths = sorted(glob.glob(os.path.join(REPO, "examples", "results", "*.json")))
    fits = [d for d in (json.load(open(p)) for p in paths) if str(d.get("schema", "")).startswith("sraging.fitresult")]
    assert fits
    for d in fits:
        jsonschema.validate(json.loads(FitResult.from_dict(d).to_json()), schema)


def test_the_schema_asks_each_numerics_for_its_own_grid_and_floor_fields():
    """v033 records L_km and the v1 gradient; jax2 records its early grid, exact_from_F and its own floor fields
    (floors.report_jax2). Each is required for its numerics only, and a numerics outside the data model is refused."""
    schema = json.load(open(SCHEMA_FILE))
    ex = {n: json.loads(FitResult.from_dict(json.load(open(os.path.join(REPO, "examples", "results", f)))).to_json())
          for n, f in (("v033", "itp_c2016_het3_control_female.json"),
                       ("jax2", "itp_c2016_het3_control_female_jax2.json"))}
    assert {n: d["provenance"]["numerics"] for n, d in ex.items()} == {"v033": "v033", "jax2": "jax2"}
    drops = {"v033": [("provenance", "grid", "L_km"), ("diagnostics", "floor_report", "grad_v1_at_theta_hat")],
             "jax2": [("provenance", "grid", "early"), ("provenance", "grid", "exact_from_F"),
                      ("diagnostics", "floor_report", "n_screened")]}
    for n, paths in drops.items():
        jsonschema.validate(ex[n], schema)                  # whole, each is valid: a drop below is what refuses it
        for a, b, key in paths:
            d = json.loads(json.dumps(ex[n]))
            del d[a][b][key]
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate(d, schema)
    d = json.loads(json.dumps(ex["jax2"]))
    d["provenance"]["numerics"] = "v1"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(d, schema)


# ---- schema /4 (spec 003 T007) ------------------------------------------------------------------------------

def combo(key, **kw):
    base = dict(parameter=key, shape="closed", lower=0.5, upper=2.0, point=1.0, level=0.95, n_components=1,
                label="", scored=True)
    base.update(kw)
    return Claim(**base)


def make4(**over):
    r = make()
    claims = tuple(r.claims) + (
        combo("eta_over_eps", label_info={"kind": "ridge", "edge": None, "log10_cap_over_floor": None, "regime": None}),
        combo("rbar_over_eta", shape="upper_bound", lower=None, upper=0.25,
              label="negligible: cap 0.25 is 10^-0.5 below the floor (sharp regime)",
              label_info={"kind": "negligible", "edge": None, "log10_cap_over_floor": -0.5, "regime": "sharp"},
              widening={"claims_profile": {"shape": "upper_bound", "lower": None, "upper": 0.2},
                        "lower_moved_dex": 0.0, "upper_moved_dex": 0.0969}),
        combo("sqrt_eta_over_eps", shape="none", lower=None, upper=None, point=None,
              decline_reason="combination walk did not converge in 24 nodes"),
    )
    kw = dict(claims=claims, nuisance={"h_ext": 1e-4, "hL": 0.087, "ExtH": 9.21, "scored": False},
              mechanisms={"basin_refit": {"fired": False, "rounds": 0}, "relabelled_rows": 0, "N": 1800,
                          "dt_floor": {"binds": False, "dt": None}, "n_fallback": 2, "widened": ["rbar_over_eta"]},
              gap_nats=0.31, archive_points=2902,
              combinations={"on": True, "cost_s": 51.0, "order": ["eta_over_eps", "rbar_over_eta"],
                            "order_source": "hypothesis"})
    kw.update(over)
    from dataclasses import replace
    return replace(r, **kw)


def test_claims_are_keyed_by_quantity_in_json_and_indexable_both_ways():
    r = make4()
    d = json.loads(r.to_json())
    assert list(d["claims"]) == ["eta", "beta", "epsilon", "kappa", "eta_over_eps", "rbar_over_eta",
                                 "sqrt_eta_over_eps"]
    assert r.claims["rbar_over_eta"] is r.claims[5] and [c.parameter for c in r.claims][:2] == ["eta", "beta"]
    with pytest.raises(KeyError):
        r.claims["beta_over_kp1"]


def test_schema_4_round_trip_is_exact_and_valid():
    r = make4()
    back = FitResult.from_json(r.to_json())
    assert hexes(back.to_dict()) == hexes(r.to_dict())
    assert back.claims["rbar_over_eta"].label_info["kind"] == "negligible"
    assert back.claims["sqrt_eta_over_eps"].decline_reason.startswith("combination walk")
    assert back.nuisance["scored"] is False and back.archive_points == 2902
    jsonschema.validate(json.loads(r.to_json()), json.load(open(SCHEMA_FILE)))


def test_a_duplicate_quantity_is_refused():
    with pytest.raises(ValueError, match="duplicate"):
        Claims((combo("eta"), combo("eta")))


def test_the_calibrator_is_read_from_provenance():
    assert make().calibrator == {"name": "profile_v033", "version": "0.3.3"}


def test_a_schema_3_file_still_loads_by_keying_its_claim_list():
    d = make().to_dict()
    d["schema"] = "sraging.fitresult/3"
    d["claims"] = list(d["claims"].values())
    for k in ("nuisance", "mechanisms", "gap_nats", "archive_points", "combinations"):
        d.pop(k)
    for c in d["claims"]:
        for k in ("scored", "label_info", "widening", "decline_reason"):
            c.pop(k)
    r = FitResult.from_dict(d)
    assert [c.parameter for c in r.claims] == ["eta", "beta", "epsilon", "kappa"] and r.claims["kappa"].shape == "none"
    assert r.schema == SCHEMA and r.nuisance is None and r.combinations is None
    assert r.provenance.get("upgraded_from") == "sraging.fitresult/3"


def test_the_identity_view_is_unchanged_by_the_new_fields():
    """Spec 001's comparison projection: the reference fields only, in order."""
    assert make4().identity_view()["claims"][:4] == make().identity_view()["claims"]
