"""Spec 003g's comparisons with jax2's records (tasks T036): the SRK files a record loaded against sraging's oracle
(FR-010), and fit 1's claims against its W3 record (SC-007). Each planted difference beside its twin."""
import hashlib
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import window_003g as WIN  # noqa: E402

ORACLE = os.path.join(REPO, "tests", "oracle", "v033", "SRtools_kappa", "SRK")


def test_srk_files_equal_to_the_oracle_pass_and_a_changed_one_is_named():
    f = "fp_likelihood.py"
    sha = hashlib.sha256(open(os.path.join(ORACLE, f), "rb").read()).hexdigest()
    same = WIN.srk_diff({f"/home/x/SRK/{f}": sha}, ORACLE)
    assert same == {"n_files": 1, "different": [], "missing_in_oracle": []}                     # twin
    bad = WIN.srk_diff({f"/home/x/SRK/{f}": "0" * 64, "/home/x/SRK/nope.py": sha}, ORACLE)
    assert bad["different"] == [f"/home/x/SRK/{f}"] and bad["missing_in_oracle"] == ["/home/x/SRK/nope.py"]


def claim(p, shape, lo=None, hi=None):
    return {"parameter": p, "shape": shape, "lower": lo, "upper": hi}


def test_a_shape_flip_fails_an_end_beyond_003es_spread_is_reported_and_the_twin_passes():
    truth = {"eta": 1.0, "epsilon": 2.0}
    rec = [claim("eta", "closed", 0.5, 2.0), claim("epsilon", "closed", 1.0, 3.0)]
    twin = WIN.claim_compare([claim("eta", "closed", 0.5, 2.0), claim("epsilon", "closed", 1.0, 3.01)], rec, truth)
    assert twin["pass"] and twin["n_ends_beyond"] == 0
    flip = WIN.claim_compare([claim("eta", "upper_bound", None, 2.0), claim("epsilon", "closed", 1.0, 3.0)], rec, truth)
    assert not flip["pass"] and not flip["shapes_equal"]
    far = WIN.claim_compare([claim("eta", "closed", 0.5, 2.5), claim("epsilon", "closed", 1.0, 3.0)], rec, truth)
    assert far["pass"] and far["n_ends_beyond"] == 1                       # reported, not failed
    miss = WIN.claim_compare([claim("eta", "closed", 1.5, 2.0), claim("epsilon", "closed", 1.0, 3.0)], rec, truth)
    assert not miss["pass"] and not miss["hits_equal"]                     # a coverage-hit flip fails


def test_claims_only_one_side_gives_are_listed_not_failed():
    """A default fit's combination claims (eta_over_eps, ...) are not in a tester record: compared parameters are the
    ones both sides claim; the rest are listed (fit 2, 2026-10-27)."""
    truth = {"eta": 1.0}
    rec = [claim("eta", "closed", 0.5, 2.0)]
    out = WIN.claim_compare([claim("eta", "closed", 0.5, 2.0), claim("eta_over_eps", "closed", 0.1, 0.2)], rec, truth)
    assert out["pass"] and out["only_mine"] == ["eta_over_eps"] and [r["parameter"] for r in out["rows"]] == ["eta"]


# ------------------------------------------------------------------------- T067: score-w3 can run (interim review)
RESULTS = os.path.join(REPO, "tests", "acceptance", "results")


def test_fit1s_stored_claims_a_dict_are_compared_as_its_records_list():
    """B1: fit1_003g.json keeps its claims as {parameter: claim}; the record keeps a list. Both shapes compare."""
    import json
    f1 = json.load(open(os.path.join(RESULTS, "fit1_003g.json")))
    assert isinstance(f1["fit"]["claims"], dict)
    rec = [dict(c) for c in f1["fit"]["claims"].values() if c["parameter"] in WIN.RAW_PARAMETERS]
    out = WIN.claim_compare(f1["fit"]["claims"], rec, {"eta": 1.0})
    assert out["pass"] and {r["parameter"] for r in out["rows"]} == set(WIN.RAW_PARAMETERS)


def test_a_raw_parameter_claimed_on_one_side_only_fails_sc007_and_a_combination_does_not():
    truth = {"eta": 1.0}
    rec = [claim("eta", "closed", 0.5, 2.0), claim("kappa", "none")]
    miss = WIN.claim_compare([claim("eta", "closed", 0.5, 2.0)], rec, truth)
    assert not miss["pass"] and miss["raw_one_side"] == ["kappa"] and miss["only_record"] == ["kappa"]
    twin = WIN.claim_compare([claim("eta", "closed", 0.5, 2.0), claim("kappa", "none")], rec, truth)
    assert twin["pass"] and twin["raw_one_side"] == []


def test_basin_p15s_pr5_rows_come_from_the_p15_result_valid_and_informative_only():
    """B2: the gate requires (BASIN-P15, PR-5); its rows are p15_v033's (sraging's v0.3.3 port, theta-only)."""
    import json
    from sraging import default_gate as G
    p15 = json.load(open(os.path.join(RESULTS, "p15_v033_003g.json")))
    meta = {r["cohort_id"]: {"valid": True, "informative": True} for r in p15["rows"]}
    rows = WIN.p15_rows(p15, meta)
    v = G.predecessor_verdict("BASIN-P15", rows)
    assert v.verdict == "shown" and v.statistic["k_shown"] == 3 and v.statistic["n"] == 3
    assert sorted(round(r["W33"], 1) for r in rows) == [11.9, 103.3, 111.4]
    one_invalid = dict(meta, **{p15["rows"][0]["cohort_id"]: {"valid": False, "informative": True}})
    assert len(WIN.p15_rows(p15, one_invalid)) == 2                         # twin: an INVALID cohort is not scored
    none = {"rows": [dict(p15["rows"][0], shown=False)]}
    assert G.predecessor_verdict("BASIN-P15", WIN.p15_rows(none, meta)).verdict == "not shown"


def test_the_scorer_files_are_compared_between_two_tester_heads(tmp_path):
    """A scorer change between W1's and W3's tester heads is named (inside and outside verdicts must come from one
    scorer); same head, nothing differs. Exercised on this repository's own history."""
    import subprocess
    sha = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%H", "--", "src/sraging/default_gate.py"],
                         capture_output=True, text=True).stdout.strip()
    files = ("src/sraging/default_gate.py", "src/sraging/leverage_rule.py")
    same = WIN.scorer_diff(REPO, sha, sha, files)
    assert same["differs"] == [] and set(same["files"]) == set(files)
    diff = WIN.scorer_diff(REPO, f"{sha}^", sha, files)
    assert "src/sraging/default_gate.py" in diff["differs"]
    assert WIN.scorer_blocking(same) == []
    (b,) = WIN.scorer_blocking(diff)
    assert b["rule"] == "scorer" and b["verdict"] == "differs" and "src/sraging/default_gate.py" in b["reasons"][0]
    assert WIN.scorer_blocking(None)[0]["verdict"] == "missing"             # score-w3 must record it


def test_scoring_an_unpinned_window_is_refused_not_read_from_w1(tmp_path):
    import json
    import pytest
    import acceptance_003g as A
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"tester": {"W1": {"head": "0" * 40, "clone": "x"}}, "runs": {}, "records": {},
                             "witness": {}}))

    class Args:
        pins = str(p)
        library = "v6"
    with pytest.raises(A.Refused, match="W3 is not pinned"):
        A._context(3, Args())


# ----------------------------------------------- T068-T071: INVALID included, the power table, declines, deviations
class _C:
    def __init__(self, shape, lower=None, upper=None):
        self.shape, self.is_claim, self.lower, self.upper = shape, shape != "none", lower, upper


class _CS:
    def __init__(self, cid, ok=True, **shapes):
        self.cohort_id, self.ok, self._s = cid, ok, shapes

    def claim_for(self, p):
        v = self._s.get(p)
        return None if v is None else _C(*v) if isinstance(v, tuple) else _C(v)


def test_with_invalid_rows_count_the_invalid_cohorts_and_the_gating_rows_do_not():
    from sraging import default_gate as G
    rows = [{"cohort_id": "a", "valid": True, "informative": True, "witness": {"W": 0.1, "fresh": True}},
            {"cohort_id": "b", "valid": False, "informative": True, "witness": {"W": 5.0, "fresh": True}}]
    assert G.witness_verdict("S", rows).verdict == "PASS"                                     # gating: valid only
    v = G.witness_verdict("S", WIN.with_invalid(rows))
    assert v.verdict == "FAIL" and v.statistic["n"] == 2                                     # FR-005: reported
    assert rows[1]["valid"] is False                                                          # not mutated


def test_shape_shares_count_what_cannot_be_capped_from_above_and_betas_cap():
    sets = [_CS("a", eta="closed", epsilon="lower_bound", beta="upper_bound"),
            _CS("b", eta="lower_bound", epsilon="closed", beta="none"),
            _CS("c", ok=False), _CS("d", eta="closed", epsilon="closed", beta="upper_bound")]
    s = WIN.shape_shares(sets)
    assert {k: s["eta"][k] for k in ("n", "failed", "closed", "lower_bound", "upper_bound", "none", "open_above")} == \
        {"n": 3, "failed": 1, "closed": 2, "lower_bound": 1, "upper_bound": 0, "none": 0, "open_above": 1}
    assert s["beta"]["upper_bound"] == 2 and s["beta"]["upper_end_at_kappa_box"] == 2 and s["beta"]["n"] == 3
    assert "capped" not in s["beta"] and "upper_end_at_kappa_box" not in s["eta"]      # N1: a box end is not a cap


def test_the_pr3_power_row_says_which_n_can_pass_and_whether_equal_coverage_does():
    """B3 at the real sizes: equal coverage 0.9 fails at 40 vs 688 closed claims and N <= 3 cannot pass at all."""
    row = WIN.pr3_power_row(k_in=619, n_in=688, n_valid_out=56, n_expected=40)
    assert row["min_k_out"][1] is None and row["min_k_out"][3] is None                       # 1 of 1, 3 of 3 fail
    assert row["equal_coverage_passes_at_expected"] is False
    big = WIN.pr3_power_row(k_in=619, n_in=688, n_valid_out=400, n_expected=300)
    assert big["equal_coverage_passes_at_expected"] is True and big["min_k_out"][300] <= 270
    assert WIN.pr3_note(row, n_out=3)["reading"] == "cannot pass at this N"
    assert WIN.pr3_note(big, n_out=300, k_out=200)["reading"].startswith("worse (p = 0.00")
    assert WIN.pr3_note(row, n_out=40, k_out=36)["reading"] == "underpowered: equal coverage fails at this N"
    none_in = WIN.pr3_power_row(k_in=0, n_in=0, n_valid_out=5, n_expected=0)        # no closed claim inside (beta)
    assert none_in["min_k_out"] == [None] * 6 and none_in["p_in"] is None
    assert WIN.pr3_note(none_in, n_out=3)["reading"] == "no comparator inside the reference set"


def test_a_verdict_deviation_is_applied_with_its_reason_and_an_unknown_stratum_is_refused():
    import pytest
    dev = [{"id": "DV-1", "kind": "verdict", "stratum": "P-DTCEIL", "rule": "PR-5", "as": "reported",
            "reason": "a library gap", "decided": "Naveh 2026-10-28"},
           {"id": "DV-2", "kind": "rule-code", "rule": "PR-2", "code": "x", "change": "y", "decided": "z"}]
    vs = [{"stratum": "P-DTCEIL", "rule": "PR-5", "verdict": "not shown", "reasons": ["r"]},
          {"stratum": "BASIN-J", "rule": "PR-5", "verdict": "shown", "reasons": []}]
    out, applied = WIN.apply_deviations(vs, dev, strata={"P-DTCEIL", "BASIN-J"})
    assert out[0]["verdict"] == "reported" and out[0]["deviation"] == "DV-1" and "a library gap" in out[0]["reasons"][-1]
    assert out[1] == vs[1] and vs[0]["verdict"] == "not shown"                               # twin; input not mutated
    assert [a["id"] for a in applied] == ["DV-1", "DV-2"] and applied[0]["n_applied"] == 1
    with pytest.raises(ValueError, match="unknown stratum"):
        WIN.apply_deviations(vs, [dict(dev[0], stratum="P-NOPE")], strata={"P-DTCEIL", "BASIN-J"})
    with pytest.raises(ValueError, match="matches no verdict"):
        WIN.apply_deviations(vs[1:], dev[:1], strata={"P-DTCEIL", "BASIN-J"})



# --------------------------------------------------------------------- the close's fixes (Naveh 2026-10-29)
def test_pr3_note_reports_the_chance_of_the_count_under_equal_coverage():
    """N3: SRK-1 censor eta, 43 of 45 against 194 of 199 inside, is a 0.31 event under exactly equal coverage."""
    assert abs(WIN.binom_cdf(43, 45, 194 / 199) - 0.312771) < 1e-6                       # scipy binom.cdf
    assert WIN.binom_cdf(5, 5, 0.3) == 1.0 and abs(WIN.binom_cdf(0, 2, 0.5) - 0.25) < 1e-12
    row = WIN.pr3_power_row(k_in=194, n_in=199, n_valid_out=56, n_expected=46)
    note = WIN.pr3_note(row, n_out=45, k_out=43)
    assert note["reading"] == "consistent with equal coverage (p = 0.31)"
    assert WIN.pr3_note(row, n_out=45, k_out=38)["reading"].startswith("worse (p = ")        # planted: a real drop


def test_decisions_attach_to_their_blocking_entries_and_an_undecided_one_shows():
    """N9, planted: every entry decided; one left undecided; a decision that matches nothing is refused."""
    blocking = [{"stratum": "A/outside", "rule": "PR-3", "arm": "pure", "quantity": "eta"},
                {"stratum": "B/outside", "rule": "PR-1", "arm": "pure", "quantity": None}]
    dec = [{"stratum": "A/outside", "rule": "PR-3", "arm": "pure", "quantity": "eta", "decision": "stated limit",
            "decided": "N", "reason": "r"},
           {"stratum": "B/outside", "rule": "PR-1", "arm": "pure", "decision": "stated limit", "decided": "N",
            "reason": "r"}]
    out, n = WIN.apply_decisions(blocking, dec)
    assert n == {"decided": 2, "of": 2} and out[1]["decision"]["decision"] == "stated limit"
    out, n = WIN.apply_decisions(blocking, dec[:1])
    assert n == {"decided": 1, "of": 2} and out[1]["decision"] is None
    with pytest.raises(ValueError, match="matching no blocking entry"):
        WIN.apply_decisions(blocking[:1], dec)


def test_the_limits_widths_crosstab_and_one_sided_coverage():
    """N4 and N8 on planted claim sets."""
    sets = [_CS("a", eta=("closed", 1.0, 10.0), beta="upper_bound"), _CS("b", eta=("closed", 1.0, 1000.0), beta="none"),
            _CS("c", eta=("lower_bound", 1.0, None), beta="none"), _CS("d", ok=False)]
    w = WIN.closed_widths(sets, "eta")
    assert w["n_closed"] == 2 and w["median_dex"] == 2.0 and w["q25_dex"] <= 1.0 and w["q75_dex"] >= 3.0
    assert WIN.closed_widths(sets, "epsilon") == {"n_closed": 0, "median_dex": None, "q25_dex": None, "q75_dex": None}
    assert WIN.eta_beta_crosstab(sets) == {"eta closed / beta none": 1, "eta closed / beta upper_bound": 1,
                                           "eta lower_bound / beta none": 1}

    class _Cov:
        def __init__(self, d):
            self.d = d

        def to_dict(self):
            return self.d

    class _CL:
        @staticmethod
        def coverage(sets_, truths, q, shape_class):
            assert shape_class == "one_sided"
            return _Cov({"hits": 3, "claimed": 4, "total": 5, "coverage": 0.75})
    assert WIN.one_sided_coverage(_CL, sets, {}, quantities=("eta",)) == \
        {"eta": {"hits": 3, "claimed": 4, "total": 5, "coverage": 0.75}}


# ------------------------------------------------------------------------- T087 (N13): |g|/n on AC-11's small-n null
def _small_n_row(cid, gpi, *, scipy_status=2, fit="ok", label="no_flag", now=None):
    r = {"cohort_id": cid, "n": 150, "label_003b": label, "fit_status_003b": fit, "measured": gpi is not None,
         "not_measured_reason": None if gpi is not None else "moved"}
    if gpi is not None:
        r.update(label=now or label, free_gradient_norm=gpi * 150, free_gradient_per_individual=gpi, above=gpi > 5e-5,
                 converged=scipy_status == 0, warm_termination={"status": "ok", "scipy_status": scipy_status})
    return r


def test_the_small_n_summary_counts_above_and_converged_yet_above_and_leaves_out_the_unmeasured():
    import pytest
    import lev_small_n_003g as LS
    rows = [_small_n_row("a", 1e-8), _small_n_row("b", 2e-4, scipy_status=0, now="not_assessed"),
            _small_n_row("c", 6e-5, now="not_assessed"), _small_n_row("d", 3e-6, scipy_status=0),
            _small_n_row("e", None), _small_n_row("f", 9e-5, fit="failed", now="not_assessed")]
    refs = [{"source": "x", "cohort_id": "ref", "n": 1500, "label": "flag", "free_gradient_norm": 0.09,
             "free_gradient_per_individual": 6e-5, "scipy_status": 2},
            {"source": "x", "cohort_id": "stuck", "n": 25000, "label": "not_assessed", "free_gradient_norm": 17.0,
             "free_gradient_per_individual": 6.8e-4, "scipy_status": 2}]
    s = LS.summarise(rows, 5e-5, refs)
    assert s["n_rows"] == 6 and s["n_measured"] == 5 and s["not_measured"] == {"e": "moved"}
    assert s["n_above"] == 3 and s["above"] == ["b", "c", "f"]
    assert s["n_converged"] == 2 and s["n_converged_above"] == 1                   # b: scipy CONVERGENCE, yet above
    assert s["n_fit_ok_above"] == 2                                                # f's fit failed: not counted
    d = s["free_gradient_per_individual"]
    assert (d["min"], d["median"], d["max"], d["n"]) == (1e-8, 6e-5, 2e-4, 5)
    assert s["label_changes"] == {"no_flag -> not_assessed": 3} and s["scipy_status_counts"] == {"2": 3, "0": 2}
    sc = s["n_scaling"]                                         # median |g| 0.009 at n = 150; the ref 10x at 10x n
    assert sc["slope_to_assessed_reference"] == {"ref": pytest.approx(1.0)}          # a not_assessed ref is left out
    assert sc["n_cross_if_constant"]["max"] == pytest.approx(2e-4 * 150 / 5e-5)
    assert [b["n"] for b in s["beside"]] == [150] * 5 + [1500, 25000]
    assert LS.summarise([_small_n_row("e", None)], 5e-5)["free_gradient_per_individual"] is None


def test_theta_hat_is_recovered_only_where_the_warm_start_did_not_move_and_a_rerun_must_reproduce_ac11():
    import lev_small_n_003g as LS
    th = [0.0001886225855500526, 1.000000000000001e-12, 0.00012485919225836963, 0.0014400412675729395]  # AC-11's
    ok = LS.recover({"status": "ok", "g0": {"status": "ok", "g0": 0.0, "theta": th}})        # nlad-k0.1-p01-n150-s0
    assert ok["ok"] and list(ok["theta"].values()) == th
    moved = LS.recover({"status": "ok", "g0": {"status": "ok", "g0": 8.3e-12, "theta": th}})
    assert not moved["ok"] and "moved" in moved["reason"]
    off = LS.recover({"status": "ok", "g0": {"status": "ok", "g0": 0.0, "theta": [0.1] + th[1:]}})  # exp(log .1) != .1
    assert not off["ok"] and "eta" in off["reason"]
    stored = {"g0": {"g0": 0.0, "theta": th, "ll_ref": -521.3, "ll_warm": -521.3}, "labels": {"all4": "no_flag"},
              "R": {"all4": 1.2}, "delta_total": 0.1, "dropped": {"row": 3, "age": 1.0, "event": 1}}
    assert LS.reproduced(stored, dict(stored)) == {"equal": True, "differs": []}
    assert LS.reproduced(stored, dict(stored, R={"all4": 1.2000000000000002}))["differs"] == ["R"]
