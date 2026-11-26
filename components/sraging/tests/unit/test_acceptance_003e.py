"""Spec 003e (T007): the SC-001 machinery -- the score file cut into quantity blocks, each block re-rendered from
`build_comparison`'s data by the tester's own `AxisResult.format`, and a line-by-line comparison. Fast; the one test
that imports the tester's class runs in a subprocess (it must not meet another srbench in this process)."""
import os
import subprocess
import sys

import acceptance_003e as ACC

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXCERPT = """method     mle_profile_jax v0.3.3
reference: mle_profile_jax2 v1.0.0 (run jax2-v5-full-bign, library v5)
library    v5   paired cohorts 1776
0 calibrations performed -- scoring reads the store.

=== kappa ===  NO DIFFERENCE DETECTED
   calibration    no difference  +0.0000 +/-0.0000   n=8  paired=True  [cell average, cluster-robust on the parameter point]
   cost           cheaper        +0.6936   n=1776  paired=False  [median wall-clock seconds per calibration]
   - calibration: no difference (+0.0000)

=== external_hazard ===  NOT MEASURED
   calibration    not measured   n/a   n=0  paired=True  [cell average, cluster-robust on the parameter point]
   - calibration could not be measured on this cohort set
"""


def test_the_score_file_is_cut_into_quantity_blocks():
    header, blocks = ACC.split_score_file(EXCERPT)
    assert header[2] == "library    v5   paired cohorts 1776"
    assert list(blocks) == ["kappa", "external_hazard"]
    assert blocks["kappa"][0] == "=== kappa ===  NO DIFFERENCE DETECTED" and len(blocks["kappa"]) == 4
    assert blocks["external_hazard"][-1] == "   - calibration could not be measured on this cohort set"


class _Axis:
    """A stand-in with the tester's format(), for the block assembly (the real class is tested below)."""
    def __init__(self, **d):
        self.d = d

    def format(self):
        return f"{self.d['axis']} {self.d['outcome']}"


def test_a_block_is_assembled_from_the_comparison_data():
    entry = {"overall": "no difference detected", "reasons": ["r1"],
             "axes": {"calibration": {"axis": "calibration", "outcome": "no difference"},
                      "cost": {"axis": "cost", "outcome": "cheaper"}}}
    assert ACC.render_block("kappa", entry, _Axis) == ["=== kappa ===  NO DIFFERENCE DETECTED",
                                                       "   calibration no difference", "   cost cheaper", "   - r1"]


def test_a_one_character_difference_is_a_mismatch():
    want = {"kappa": ["=== kappa ===  WINS", "   x"]}
    assert ACC.compare_blocks(want, {"kappa": ["=== kappa ===  WINS", "   x"]}) == []
    bad = ACC.compare_blocks(want, {"kappa": ["=== kappa ===  WINS", "   y"]})
    assert bad and bad[0]["quantity"] == "kappa" and bad[0]["line"] == 1
    assert ACC.compare_blocks(want, {}) and ACC.compare_blocks({}, {"kappa": []})     # a block missing either way


def _group(skipped=(), failures=0, returncode=0):
    return {"skipped_list": list(skipped), "failures": failures, "errors": 0, "returncode": returncode}


def test_unchanged_passes_only_superseded_not_applicable_skips():
    sup = ("tests.acceptance.test_results_002::test_claims_001_rerun_is_identical_and_valid_for_this_branch",
           "NOT APPLICABLE: the fit path changed")
    other = ("tests.acceptance.test_results_003::test_claims_001_rerun_is_identical_for_this_branch",
             "NOT APPLICABLE: the claims' import closure changed")
    assert ACC.unchanged_verdict({"u": _group()}) == (True, [])                                  # the twins
    assert ACC.unchanged_verdict({"r": _group([sup], returncode=1)}) == (True, [])
    xf = ("tests.acceptance.test_results_003::test_coverage_no_combination_loses_on_a_gated_arm", "xfail: recorded")
    assert ACC.unchanged_verdict({"r": _group([sup, xf], returncode=1)}) == (True, [])        # a recorded xfail
    assert not ACC.unchanged_verdict({"r": _group([sup, ("t::x", "plain skip")], returncode=1)})[0]   # unnamed
    ok, unlisted = ACC.unchanged_verdict({"r": _group([sup, other], returncode=1)})               # planted
    assert not ok and unlisted == [f"r: {other[0]}"]
    assert not ACC.unchanged_verdict({"u": _group(failures=1, returncode=1)})[0]
    assert not ACC.unchanged_verdict({"u": _group(returncode=1)})[0]                    # exit 1 with nothing to excuse


def test_a_re_pin_refuses_drifted_tester_data():
    import copy
    import json
    pins = json.load(open(ACC.PINS_PATH))
    assert ACC.pins_drift(pins, copy.deepcopy(pins)) == []                                     # the twin
    bad = copy.deepcopy(pins)
    run = next(iter(bad["records"]))
    c = next(iter(bad["records"][run]))
    bad["records"][run][c]["sha256"] = "0" * 64
    bad["made_at"] = {"elsewhere": True}                                                        # made_at is ignored
    assert ACC.pins_drift(pins, bad) == ["records"]


def test_a_planted_provenance_omission_is_caught():
    import copy
    import json
    d = json.load(open(os.path.join(ACC.RESULTS_DIR, "known_answer_003e.json")))
    assert ACC.provenance_problems("known_answer", d) == []                                     # the twin
    for path in (("provenance", "library_manifest_sha256"), ("provenance", "tester_clone_head"),
                 ("provenance", "pins_sha256"), ("provenance", "host"), ("provenance", "runs_sha256"),
                 ("status_for_acceptance",)):
        bad = copy.deepcopy(d)
        tgt = bad
        for k in path[:-1]:
            tgt = tgt[k]
        del tgt[path[-1]]
        assert ACC.provenance_problems("known_answer", bad), path
    bad = copy.deepcopy(d)
    bad["provenance"]["runs_sha256"]["jax2-v5-full-bign"] = "0" * 64
    assert ACC.provenance_problems("known_answer", bad)
    bad = copy.deepcopy(d)
    bad["provenance"]["tester_clone_dirty"] = True
    assert ACC.provenance_problems("known_answer", bad)


def test_the_testers_own_axis_class_reproduces_a_score_file_line():
    import acceptance_003e  # noqa: F401
    from sraging.adapters import srbench as ADP
    if not os.path.isdir(ADP.TESTER_PINNED):
        import pytest
        pytest.skip("no pinned clone here (spec 003e T001)")
    code = (
        "import acceptance_003e as ACC; m = ACC._mods(); V = m['score.verdict']; "
        "d = dict(axis='calibration', outcome='no difference', statistic=0.0, uncertainty=0.0, denominator=8, "
        "cohort_set='1776 paired', paired=True, aggregation='cell average, cluster-robust on the parameter point', "
        "detail={}); print(V.AxisResult(**d).format())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO,
                         env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                              "PYTHONPATH": os.pathsep.join([os.path.join(REPO, "tools"), os.path.join(REPO, "src")])})
    assert out.returncode == 0, out.stderr[-2000:]
    line = EXCERPT.split("\n")[6]
    assert "   " + out.stdout.strip("\n") == line


def _axis(outcome, stat, n, cs="4 paired"):
    return {"outcome": outcome, "statistic": stat, "denominator": n, "cohort_set": cs}


def test_the_self_pair_gate_is_strict():
    exp = {"eta": {"calibration": 3, "cost": 4}, "external_hazard": {"calibration": 0, "cost": 4}}
    good = {"by_parameter": {"eta": {"axes": {"calibration": _axis("no difference", 0.0, 3),
                                              "cost": _axis("no difference", 1.0, 4)}},
                             "external_hazard": {"axes": {"calibration": _axis("not measured", None, 0),
                                                          "cost": _axis("no difference", 1.0, 4)}}}}
    assert ACC.self_pair_problems(good, exp, "4 paired") == []                                  # the twin
    import copy
    for q, k, d in (("eta", "calibration", _axis("not measured", None, 0)),     # dropped claims would read so
                    ("eta", "calibration", _axis("no difference", 0.0, 2)),     # n other than the count
                    ("eta", "cost", _axis("no difference", 0.0, 4)),            # cost's identity is 1
                    ("eta", "calibration", _axis("no difference", 0.0, 3, "3 paired"))):
        bad = copy.deepcopy(good)
        bad["by_parameter"][q]["axes"][k] = d
        assert ACC.self_pair_problems(bad, exp, "4 paired") == [f"{q}.{k}"], d


class _C:
    def __init__(self, **d):
        self.d = d
        self.parameter = d["parameter"]

    def to_dict(self):
        return dict(self.d)


def _cs(cid, *claims):
    import types
    return types.SimpleNamespace(cohort_id=cid, status="ok", claims=[_C(**c) for c in claims])


def test_jcross_clamps_at_the_floor_and_counts_hit_flips():
    import math

    def log_error(a, b, fl):                     # the tester's rule: both sides clamped to the floor
        a, b = (max(a, fl), max(b, fl)) if fl else (a, b)
        return abs(math.log10(a / b))
    port = [_cs("x", dict(parameter="beta", shape="closed", lower=1e-12, upper=2.0, point=3e-12),
                dict(parameter="eta", shape="closed", lower=1.0, upper=2.0, point=1.5))]
    ref = [_cs("x", dict(parameter="beta", shape="closed", lower=1e-12, upper=2.0, point=1e-11),
               dict(parameter="eta", shape="closed", lower=1.6, upper=2.0, point=1.8))]
    out = ACC._cross(port, ref, {"x": "pure"}, {"x": {"beta": 1.0, "eta": 1.2}},
                     {"beta": {"x": 1e-6}, "eta": {"x": None}}, log_error)
    assert out["per_cohort"]["x"]["params"]["beta"]["point"] == 0.0     # both below the floor: no move
    assert out["coverage_hits"] == {"claims": 2, "flips": 1, "params": ["eta", "beta", "epsilon"], "flipped": ["x:eta"]}
    assert out["n_shape_flips"] == 0
