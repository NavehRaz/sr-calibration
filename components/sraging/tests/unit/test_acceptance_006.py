"""Spec 006's acceptance frame (tools/acceptance_006.py; tasks T002, T003; contracts/acceptance.md): the phase registry
and its caps, the rules file that repeats them, the dirty-tree refusal, the freeze (with the two phases that read no
rule), the missing caps of a1 and a2, and `capped`. Each planted case beside its twin."""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "tools"))

import acceptance_003 as A3  # noqa: E402
import acceptance_006 as ACC  # noqa: E402

H = 3600.0


class Clock:
    """A fake process clock: CPU seconds, advanced by hand."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _rules(frozen, **caps):
    return {"frozen": frozen, "caps": {**{k: v for k, v in ACC.CAPS.items() if v is not None}, **caps}, "rules": {}}


# --------------------------------------------------------------------------------------------- the phase registry
def test_the_phases_and_their_caps_are_the_contracts():
    assert ACC.PHASES == ("parity", "identity", "stored", "pilot", "a1", "a2", "realfull", "v033-leverage", "unchanged")
    assert ACC.CAPS == {"parity": 0.5, "identity": 5.0, "stored": 0.1, "pilot": 12.0, "a1": None, "a2": None,
                        "realfull": 24.0, "v033-leverage": 8.0, "unchanged": 1.0}
    assert ACC.RULE_FREE == ("parity", "identity")
    assert set(ACC.WHERE) == set(ACC.PRICE) == set(ACC.PHASES) == set(ACC.PHASE_FUNCS)
    mac = sum(v for k, v in ACC.CAPS.items() if v is not None and ACC.WHERE[k] == "mac")
    wexac = sum(v for k, v in ACC.CAPS.items() if v is not None and ACC.WHERE[k] == "wexac")
    assert mac <= 15.0 and wexac <= 230.0                 # FR-020's spec caps until the freeze (a1, a2 not yet set)


def test_the_rules_file_repeats_the_caps():
    """T003: every cap the tool has is the rules file's, and the rules file sets no other cap except a1 and a2 (the
    freeze's). A drift between the tool and the rules fails here."""
    caps = ACC.read_rules()["caps"]
    for k, v in ACC.CAPS.items():
        assert caps.get(k) == v or (v is None and k in ("a1", "a2")), k
    assert set(caps) <= set(ACC.PHASES)


def test_the_draft_rules_hold_every_rule_and_the_panel():
    r = ACC.read_rules()
    assert {f"R-{i}" for i in range(1, 10)} <= set(r["rules"])
    assert r["rules"]["R-1"]["gated"] == ["eta", "epsilon"] and r["rules"]["R-1"]["D"] == 1.05
    assert len(r["panel"]["pilot"]) == 6 and r["panel"]["a1"]["expect"] + r["panel"]["a2"]["expect"] == 240
    assert r["pins"]["file"] == "tests/acceptance/pins_003g.json"


# ---------------------------------------------------------------------------------------- the envelope (Run, XI)
def test_a_dirty_tree_is_refused_and_a_dev_run_goes_on_labelled_development(monkeypatch, tmp_path):
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(True))                       # only the dirt
    monkeypatch.setattr(A3, "tree_dirty", lambda: True)
    with pytest.raises(SystemExit) as ex:
        ACC.start("stored", dev=False)
    assert "REFUSED" in str(ex.value) and "dirty" in str(ex.value)
    monkeypatch.setattr(A3, "RESULTS", str(tmp_path))
    run = ACC.start("stored", dev=True)                                                  # the twin
    out = run.write({"status": "pass"})
    assert out["status_for_acceptance"] == "development" and out["phase"] == "stored"


def test_a_rule_reading_phase_is_refused_non_dev_before_the_freeze(monkeypatch):
    monkeypatch.setattr(A3, "tree_dirty", lambda: False)
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(False))
    for phase in ("stored", "pilot", "realfull", "v033-leverage", "unchanged"):
        with pytest.raises(SystemExit) as ex:
            ACC.start(phase, dev=False)
        assert "REFUSED" in str(ex.value) and "frozen" in str(ex.value) and "T058" in str(ex.value), phase


def test_parity_and_identity_read_no_rule_and_run_non_dev_before_the_freeze(monkeypatch, tmp_path):
    """The twin: the two rule-free phases pass the freeze check (a clean tree still required, by Run)."""
    monkeypatch.setattr(A3, "tree_dirty", lambda: False)
    monkeypatch.setattr(A3, "committed", lambda m: True)
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(False))
    monkeypatch.setattr(A3, "RESULTS", str(tmp_path))
    for phase in ACC.RULE_FREE:
        assert ACC.start(phase, dev=False).phase == phase


def test_a1_and_a2_refuse_until_the_freeze_sets_their_cap(monkeypatch):
    monkeypatch.setattr(A3, "tree_dirty", lambda: False)
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(True))
    for phase in ("a1", "a2"):
        for dev in (True, False):
            with pytest.raises(SystemExit) as ex:
                ACC.start(phase, dev=dev)
            assert "no cap yet" in str(ex.value) and "T058" in str(ex.value)
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(True, a1=100.0))                # the twin
    assert ACC.cap_for("a1") == 100.0 and ACC.cap_for("pilot") == 12.0


# ------------------------------------------------------------------------------------------------ caps (FR-020)
def test_a_phase_past_its_cap_reads_capped_keeping_the_finished_cells():
    clock = Clock()
    b = ACC.Budget("stored", ACC.CAPS["stored"], clock=clock, children=False)
    ran = []

    def work(cell):
        clock.t += 0.04 * H
        ran.append(cell)
        return {"cell": cell}
    rows = ACC.run_capped(b, range(5), work)
    assert ran == [0, 1, 2] and len(rows) == 3                    # 0.12 CPU-h after the third: past 0.1
    assert b.capped and ACC.phase_status("pass", b) == "capped" and not ACC.counts_as_pass("capped")


def test_the_cap_counts_a_remote_jobs_cpu():
    clock = Clock()
    b = ACC.Budget("pilot", ACC.CAPS["pilot"], clock=clock, children=False)
    b.remote_cpu_s = 11.9 * H
    assert not b.reached()
    b.remote_cpu_s = 12.0 * H
    assert b.reached()


def test_every_phase_is_built():
    assert all(f.__name__ != "phase" for f in ACC.PHASE_FUNCS.values())


def test_the_claims_reading_needs_40_of_40_and_no_closure_change_since(tmp_path, monkeypatch):
    """FR-030: a 40/40 claims re-run at 004's head (b92c48d) predates 006's closure edits, so it does not stand for
    006; the same result at HEAD does (the twin), and 39/40 does not. Planted from the committed result, so the test
    reads the same before and after 006's own re-run (T076)."""
    import json
    d = json.load(open(os.path.join(REPO, ACC.CLAIMS_RESULT)))
    p = tmp_path / "claims.json"
    monkeypatch.setattr(A3, "untouched_problems", lambda *a, **k: [])
    d["repo_head"] = A3.git("rev-parse", "b92c48d")
    p.write_text(json.dumps(d))
    r = ACC.claims_reading(str(p))
    assert r["checks_40_of_40"] and r["closure_changed_since"] and not r["ok"]
    d["repo_head"] = A3.git("rev-parse", "HEAD")
    p.write_text(json.dumps(d))
    assert ACC.claims_reading(str(p))["ok"]
    d["checks"]["theta_hat"]["passed"] = 39                                           # planted: one cohort off
    p.write_text(json.dumps(d))
    assert not ACC.claims_reading(str(p))["ok"]


def test_dry_run_and_resume_are_the_wexac_phases():
    with pytest.raises(SystemExit):
        ACC.main(["parity", "--dry-run"])
    with pytest.raises(SystemExit):
        ACC.main(["pilot", "--resume", "1"])                   # --resume without --job-head


# --------------------------------------------------------------------------------------- identity (R-5, R-6; T020)
EXAMPLE = os.path.join(REPO, "examples", "results", "itp_c2016_het3_control_female_jax2.json")


def _example():
    from sraging.result import FitResult
    return FitResult.from_json(EXAMPLE)


def _with_diag(r, path, value):
    import copy
    import dataclasses
    d = copy.deepcopy(dict(r.diagnostics))
    node = d
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return dataclasses.replace(r, diagnostics=d)


def test_the_identity_panel_is_the_rules_and_has_seven_pairs():
    ident = ACC.read_rules()["panel"]["identity"]
    assert list(ACC.IDENTITY_UNWINDOWED) == ident["unwindowed"] and list(ACC.IDENTITY_WINDOWED) == ident["windowed"]
    pairs = ACC.identity_pairs()
    assert len(pairs) == 7 and sum(p["mode"] == "r5" for p in pairs) == 3
    assert {p["name"] for p in pairs if p["mode"] == "r6"} == {"profile_jax2_w", "profile_jax2_h"}


def test_r5_sees_a_one_ulp_change_of_theta_hat_and_not_the_excluded_paths():
    import dataclasses
    import numpy as np
    r = _example()
    assert ACC.r5_diff(r, r) == []
    th = dataclasses.replace(r.theta_hat, eta=float(np.nextafter(r.theta_hat.eta, 1.0)))
    d = ACC.r5_diff(dataclasses.replace(r, theta_hat=th), r)                                     # planted
    assert len(d) == 1 and d[0].startswith("/theta_hat/eta: ")
    w = _with_diag(_with_diag(r, ("switches", "window"), True), ("observation_model", "window_on"), True)
    assert ACC.r5_diff(w, r) == []                                                                  # the twin
    d = ACC.r5_diff(_with_diag(r, ("observation_model", "window"), [1.0, 2.0]), r)
    assert len(d) == 1 and d[0].startswith("/diagnostics/observation_model/window")


def test_any_unequal_or_crashed_pair_fails_the_phase():
    ok = {"cohort_id": "a", "rows": [{"rule": "R-6", "equal": True}]}
    assert ACC.identity_verdict([ok], 1, False)["status"] == "pass"
    bad = {"cohort_id": "b", "rows": [{"rule": "R-6", "equal": False, "diff_paths": ["/claims/0/upper"]}]}
    assert ACC.identity_verdict([ok, bad], 2, False)["status"] == "fail"                        # a changed claim byte
    assert ACC.identity_verdict([ok, {"cohort_id": "c", "error": "x"}], 2, False)["status"] == "fail"
    assert ACC.identity_verdict([ok], 2, False)["status"] == "fail"                            # a pair missing
    assert ACC.identity_verdict([ok], 1, True)["status"] == "fail"                             # stopped


# ------------------------------------------------------------------------------- the WEXAC cells (T023, T027, T028)
def test_waves_are_packed_cheapest_first_within_a_quarter_of_the_cell():
    prices = {"a": 100.0, "b": 200.0, "c": 300.0, "d": 400.0, "e": 1000.0}
    waves = ACC.pack_waves(prices, 1.0, 2000.0 / 4)
    assert waves == [["a", "b"], ["c"], ["d"], ["e"]]                   # e alone exceeds the quarter: its own wave
    assert all(sum(prices[c] for c in w) <= 500.0 for w in waves if len(w) > 1)
    assert sorted(c for w in waves for c in w) == sorted(prices)
    assert ACC.pack_waves(prices, 2.0, 1e9) == [["a", "b", "c", "d", "e"]]


def test_a_wave_goes_only_within_the_cap_and_the_cell_stops_at_it():
    assert ACC.wave_allowed(8.0, 3.9, 12.0) and ACC.wave_allowed(8.0, 4.0, 12.0)
    assert not ACC.wave_allowed(8.0, 4.01, 12.0)                         # planted: one wave past the cap


def test_the_ledger_counts_every_wave_of_every_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(ACC, "RESULTS_GATE", str(tmp_path))
    assert ACC.ledger_spent_h("pilot") == 0.0
    ACC.ledger_add("pilot", {"wave": 1, "head": "aaaaaaa", "element_cpu_s": 3600.0})
    ACC.ledger_add("pilot", {"wave": 1, "head": "bbbbbbb", "element_cpu_s": 1800.0})     # a superseded attempt
    assert ACC.ledger_spent_h("pilot") == 1.5


def test_rows_from_another_head_are_not_this_runs(tmp_path):
    import json
    (tmp_path / "a.json").write_text(json.dumps({"cohort_id": "a", "head": "h1"}))
    (tmp_path / "b.json").write_text(json.dumps({"cohort_id": "b", "head": "h0"}))
    assert set(ACC.read_rows(str(tmp_path), ["a", "b", "c"], "h1")) == {"a"}


def test_the_array_is_done_when_every_element_is_done_or_exited_and_failed_polls_are_waited_out():
    assert ACC.array_states("DONE\nDONE\nRUN\n") == {"DONE": 2, "RUN": 1}
    assert not ACC.array_done({"DONE": 2, "RUN": 1}, 3) and ACC.array_done({"DONE": 2, "EXIT": 1}, 3)
    calls = []

    def remote(cmd):
        calls.append(cmd)
        if len(calls) == 1:
            raise RuntimeError("ssh: connection reset")
        return "DONE\nEXIT\n" if len(calls) > 2 else "RUN\nPEND\n"
    assert ACC.poll_array("123", 2, remote=remote, wait_s=0) == {"DONE": 1, "EXIT": 1} and len(calls) == 3
    with pytest.raises(RuntimeError):
        ACC.poll_array("123", 2, remote=lambda c: (_ for _ in ()).throw(RuntimeError("down")), wait_s=0, retries=2)


def test_the_lsf_cpu_cross_check_reads_jobindex_and_cpu_used():
    text = "1 1234.5\n2 99.0\nbad line\n"
    assert ACC.lsf_cpu("9", remote=lambda c: text) == {1: 1234.5, 2: 99.0}


W = " " * 21
#: `bhist -l` of a 2-element array, wrapped at 80 columns as LSF prints it (element 1 from the pilot's job 614472)
BHIST = (f"Job <614472[1]>, Job Name <g006_pilot_7398c32_w5[1]>, User <navehr>, Project <d\n{W}efault>\n"
         f"Wed Nov 11 02:06:57: Done successfully. The CPU time used is 11017.3 secon\n{W}ds;\n"
         f"Wed Nov 11 02:06:57: Post job process done successfully;\n\n"
         f"Job <614472[2]>, Job Name <g006_pilot_7398c32_w5[2]>, User <navehr>\n"
         f"Wed Nov 11 01:00:00: Exited with exit code 1. The CPU time used is 12.5 seconds.\n")


def test_a_purged_array_is_read_from_lsfs_history():
    """2026-11-11: the network was down past LSF's clean period; bjobs then answers `Job <id> is not found` (exit 0)
    and a poller reading bjobs alone waited forever, its last wave's CPU uncounted."""
    assert ACC.array_states("Job <614472> is not found\n") == {}
    assert ACC.bhist_elements(BHIST) == {1: ("DONE", 11017.3), 2: ("EXIT", 12.5)}

    def remote(cmd):
        return BHIST if "bhist" in cmd[-1] else "Job <614472> is not found\n"
    assert ACC.poll_array("614472", 2, remote=remote, wait_s=0) == {"DONE": 1, "EXIT": 1}
    assert ACC.lsf_cpu("614472", remote=remote) == {1: 11017.3, 2: 12.5}
    calls = []                                                  # the twin: bjobs still has it, bhist is never read

    def live(cmd):
        calls.append(cmd[-1])
        return "RUN\nRUN\n" if len(calls) < 3 else "DONE\nDONE\n"
    assert ACC.poll_array("614472", 2, remote=live, wait_s=0) == {"DONE": 2}
    assert len(calls) == 3 and not any("bhist" in c for c in calls)


def test_a_partly_purged_array_is_read_from_lsfs_history():
    """2026-11-12: wave 4's elements ended at different times; bjobs listed the last one only (the others were past
    LSF's clean period), and a poller that read history only when bjobs listed nothing waited an hour."""
    hist = "".join(f"Job <703598[{i}]>, Job Name <g006_a1_966bbe7_w4[{i}]>\n"
                   f"Thu Nov 12 22:47:00: Done successfully. The CPU time used is 10.0 seconds.\n\n" for i in range(1, 7))
    calls = []

    def remote(cmd):
        calls.append(cmd[-1])
        return hist if "bhist" in cmd[-1] else ("RUN\n" if len(calls) == 1 else "DONE\n")
    assert ACC.poll_array("703598", 6, remote=remote, wait_s=0) == {"DONE": 6}
    assert not any("bhist" in c for c in calls[:1])                     # an element still live: history not read
    calls.clear()                                                         # the twin: history one element short
    lag = hist.rsplit("Job <", 1)[0]

    def lagging(cmd):
        calls.append(cmd[-1])
        return (lag if sum("bhist" in c for c in calls) < 3 else hist) if "bhist" in cmd[-1] else "DONE\n"
    assert ACC.poll_array("703598", 6, remote=lagging, wait_s=0) == {"DONE": 6}
    assert sum("bhist" in c for c in calls) == 3                          # 5 from history + 1 listed is not 6


def test_every_remote_call_carries_ssh_timeouts(monkeypatch):
    """2026-11-12: the VPN dropped and the poller's ssh hung on the dead connection; with keepalives it fails within a
    minute, as a counted failed poll."""
    seen = []
    monkeypatch.setattr(ACC.A5, "_remote", lambda cmd, capture=True: seen.append(cmd) or "")
    ACC._remote(["ssh", "wexac", "bjobs"])
    ACC._remote(["rsync", "-az", "wexac:out/", "local/"])
    assert seen[0][:-2] == ["ssh", *ACC.SSH_OPTS] and seen[0][-2:] == ["wexac", "bjobs"]
    assert seen[1][:3] == ["rsync", "-e", "ssh " + " ".join(ACC.SSH_OPTS)] and seen[1][3:] == ["-az", "wexac:out/",
                                                                                              "local/"]
    assert "ServerAliveInterval=15" in ACC.SSH_OPTS and "ConnectTimeout=20" in ACC.SSH_OPTS


def test_a_mac_steps_spend_counts_for_its_cell_not_the_specs_wexac_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(ACC, "RESULTS_GATE", str(tmp_path))
    ACC.ledger_add("v033-leverage", {"wave": 1, "job_id": "1", "element_cpu_s": 4.93 * H})
    ACC.ledger_add("v033-leverage", {"wave": "r9_control", "where": "mac", "job_id": None, "element_cpu_s": 3.04 * H})
    assert ACC.ledger_spent_h("v033-leverage") == pytest.approx(7.97)
    assert ACC.ledger_spent_h("v033-leverage", "wexac") == pytest.approx(4.93)
    assert ACC.ledger_spent_h("v033-leverage", "mac") == pytest.approx(3.04)
    assert ACC.spec_spent_wexac_h() == pytest.approx(4.93)


def test_r9s_control_on_the_mac_keeps_the_cells_cap(monkeypatch):
    """2026-11-11: the control ran 3.04 Mac CPU-h after the cell's 4.93 on WEXAC, with no cap check, to the 8 cap.
    Now it starts only if its price fits, and no fit starts once the cap is reached; not run, R-9 fails."""
    fits = []
    monkeypatch.setattr(ACC, "itp_control", lambda b=None: fits.append("itp") or {"label": "not_assessed",
                                                                                 "warnings": ["LEVERAGE"]})
    clock = Clock()
    budget = ACC.Budget("v033-leverage", 8.0, clock=clock, children=False)
    budget.remote_cpu_s = 7.97 * H                                        # WEXAC 4.93 + the Mac's 3.04
    out = ACC.r9_control(budget)
    assert out["not_run"].startswith("cell capped") and budget.capped and not fits
    assert ACC.score_v033(_null(1, 25), list(_null(1, 25)), {}, out, R9)["status"] == "fail"
    budget = ACC.Budget("v033-leverage", 8.0, clock=clock, children=False)  # the twin: room for the price
    budget.remote_cpu_s = 4.93 * H

    def itp(b=None):                                                      # the ITP fit spends past the cap
        clock.t += 3.1 * H
        fits.append("itp")
        return {"label": "not_assessed", "warnings": ["LEVERAGE"]}
    monkeypatch.setattr(ACC, "itp_control", itp)
    monkeypatch.setattr(ACC, "r9_candidate", lambda cid: fits.append(cid) or (cid, 9))
    out = ACC.r9_control(budget)
    assert fits == ["itp"] and budget.capped and out["candidates"] == [] and out["itp_age0"]["label"] == "not_assessed"
    assert "after 0 candidate(s)" in out["not_run"]


def test_rows_whose_job_the_ledger_never_counted_are_found(tmp_path, monkeypatch):
    monkeypatch.setattr(ACC, "RESULTS_GATE", str(tmp_path))
    ACC.ledger_add("pilot", {"wave": 1, "job_id": "606819", "element_cpu_s": 1.0})
    rows = {"a": {"lsf_jobid": "606819"}, "b": {"lsf_jobid": "614472"}, "c": {"lsf_jobid": "614472"}, "d": {}}
    assert ACC.unrecorded_jobs("pilot", rows) == {"614472": ["b", "c"]}
    rec = ACC.wave_record(5, "614472", "h", ["b", "c"], {"b": {"element_cpu_s": 10.0}}, {"DONE": 1}, {1: 12.0})
    assert rec["element_cpu_s"] == 12.0 and rec["missing"] == ["c"] and rec["lsf_cpu_s"] == 12.0


def test_resume_at_another_head_is_development_only(monkeypatch):
    heads = {"HEAD": "a" * 40, "7398c32": "b" * 40}
    monkeypatch.setattr(ACC.A3, "git", lambda *a, **k: heads[a[-1]])
    monkeypatch.setattr(ACC, "read_rules", lambda: {"frozen": True, "rules": {}})
    with pytest.raises(SystemExit, match="rows of another commit never count"):
        ACC.main(["pilot", "--resume", "614472", "--job-head", "7398c32"])


def test_the_gate_cells_refuse_before_the_freeze_dev_or_not(monkeypatch):
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(False))
    for phase in ACC.GATE_PHASES:
        for extra in ([], ["--dev"]):
            with pytest.raises(SystemExit) as ex:
                ACC.main([phase, *extra])
            assert "REFUSED" in str(ex.value) and "after the freeze" in str(ex.value), phase


def test_the_factor_is_the_pilots_until_the_freeze_sets_r():
    assert ACC.factor_for("pilot", _rules(False)) == ACC.PILOT_FACTOR
    assert ACC.factor_for("a1", {**_rules(True), "pricing": {"r": 1.2}}) == 1.2


def test_the_job_script_and_submit_line_are_single_threaded_on_one_epyc_core():
    paths = ACC.cell_paths("abcdef0", "pilot")
    js = ACC.job_script(paths)
    assert "PJRT_NPROC=1" in js and "OMP_NUM_THREADS=1" in js and "tools/gate_006.py element" in js
    sub = " ".join(ACC.submit_command(paths, "abcdef0", "pilot", 2, 3))
    assert "-n 1" in sub and "select[model==AMD_EPYC]" in sub and "span[hosts=1]" in sub and "[1-3]" in sub
    assert "wave_2.txt" in sub and "-q long" in sub


class _CS:
    def __init__(self, cid, profile, box=(9.98e-09, 0.0998)):
        self.cohort_id = cid
        self.diagnostics = {"hazard": {"profile": profile, "box": list(box)}}


def test_rs_node_is_hs_lowest_h_node_on_every_cohort(monkeypatch):
    """R-4, exact (Naveh, 2026-11-11): no box-floor filter; the node 1 % above the floor (the 1e-9 filter's 149
    "not applicable" cohorts) is read like the one at it."""
    sets = [_CS("a", [{"value": 9.98e-09, "loglik": -10.0}, {"value": 1e-4, "loglik": -9.0}]),
            _CS("b", [{"value": 1.008e-08, "loglik": -20.0}, {"value": 1e-6, "loglik": -19.5}]), _CS("c", [])]
    monkeypatch.setattr(ACC, "pinned_sets", lambda *a, **k: sets)
    nodes = ACC.h_nodes(None, None, {"pins": {"H": "jax2h-v6-full-bign"}}, ["a", "b", "c"])
    assert nodes["a"]["loglik"] == -10.0 and nodes["b"] == {"value": 1.008e-08, "loglik": -20.0,
                                                            "run": "jax2h-v6-full-bign"}
    assert "c" not in nodes                                              # no profile: R-4 fails it by name


class _Lib:
    def __init__(self, clusters):
        self._c = clusters

    def cluster_map(self, ids):
        return {c: self._c[c] for c in ids}


def test_dg_draws_one_cohort_per_cluster_by_the_frozen_seed():
    lib = _Lib({"a1": "k1", "a2": "k1", "a3": "k1", "b1": "k2", "c1": "k3", "c2": "k3"})
    rules = {"panel": {"dg": {"seed": 20261111}}}
    ids = list(lib._c)
    pick = ACC.dg_ids("a1", lib, rules, ids)
    assert len(pick) == 3 and {lib._c[c] for c in pick} == {"k1", "k2", "k3"}
    assert pick == ACC.dg_ids("a1", lib, rules, list(reversed(ids)))           # the order given does not matter
    assert ACC.dg_ids("realfull", lib, rules, ids) == [] and ACC.dg_ids("pilot", lib, rules, ids) == []


def test_a_lead_pair_cohort_is_priced_at_r_lead():
    rules = {**_rules(True), "pricing": {"r": 1.162, "r_lead": 3.627}}
    assert ACC.factor_for("a1", rules) == 1.162 and ACC.factor_for("a1", rules, lead=True) == 3.627
    waves = ACC.pack_waves({"x": 100.0, "y": 100.0, "z": 100.0}, {"x": 1.0, "y": 1.0, "z": 3.0}, 250.0)
    assert waves == [["x", "y"], ["z"]]                                         # z alone: 300 > 250


def test_an_element_writes_every_field_of_a_gate_row(tmp_path, monkeypatch):
    """T028 on a tiny fixture, the fit canned: every field of data-model.md's gate row, and a failure is a row."""
    import json
    import numpy as np
    import gate_006 as G
    import sraging
    ages, events = np.array([3.0, 5.0, 8.0, 9.0, 12.0, 15.0]), np.array([1, 1, 1, 1, 1, 0])
    (tmp_path / "cohorts").mkdir()
    np.savez(tmp_path / "cohorts" / "x.npz", ages=ages, events=events)
    entry = {"delta": 1.0, "window": [3.0, 14.0], "truth": {"eta": 1e-3, "beta": 0.1, "epsilon": 0.01, "kappa": 1.0},
             "digest": G.tester_cohort_digest()(ages, events), "n": 6, "cluster": "k1", "kappa": 1.0}
    r = _example()
    seen = {}
    monkeypatch.setattr(sraging, "fit_cohort", lambda c, **k: seen.update(k) or r)
    monkeypatch.setattr(G, "h_loglik_at", lambda c, th, h: -1.5 if h == 1e-8 else None)
    row = G.element_row("x", entry, cell="pilot", head="h", v6_dir=str(tmp_path), lead_pair=False)
    assert set(G.ROW_FIELDS) <= set(row) and row["status"] == "ok" and row["element_cpu_s"] > 0
    assert row["loglik_hat"] == r.loglik and row["theta_hat"] == r.theta_hat.as_dict()
    assert row["h_node"] is None and row["loglik_H_at_theta_w"] is None and seen["combinations"] is False
    assert {c["parameter"] for c in row["claims"]} >= {"eta", "beta", "epsilon", "kappa"}     # every claim is kept
    node = {"value": 1e-8, "loglik": -1.4}
    row = G.element_row("x", {**entry, "h_node": node, "lead_pair": True}, cell="a1", head="h", v6_dir=str(tmp_path),
                        lead_pair=False)
    assert row["h_node"] == node and row["loglik_H_at_theta_w"] == -1.5 and seen["combinations"] == "lead"
    bad = G.element_row("x", {**entry, "digest": "0" * 64}, cell="pilot", head="h", v6_dir=str(tmp_path),
                        lead_pair=False)
    assert bad["status"] == "error" and "digest" in bad["error"]                # planted: changed arrays
    G.write_atomic(str(tmp_path / "x.json"), row)
    assert json.load(open(tmp_path / "x.json"))["cohort_id"] == "x"


def test_a_rerun_collects_a_live_wave_instead_of_resubmitting_it():
    assert ACC.running_job("w", remote=lambda c: "4711 RUN\n4711 PEND\n") == "4711"
    assert ACC.running_job("w", remote=lambda c: "4711 DONE\n") is None                      # the twin
    assert ACC.running_job("w", remote=lambda c: "") is None


# ----------------------------------------------------------------------------- SPEC_ENGINE_CHANGES["006"] (T036)
def test_006s_engine_files_are_spec_exempt_and_an_unlisted_one_stays_a_violation(monkeypatch):
    """FR-017: spec 003's untouched check records the two engine files 006 edits as spec-exempt (owing the claims
    re-run's 40/40), never as violations; an engine file no spec names stays a violation."""
    assert dict(A3.SPEC_ENGINE_CHANGES["006"]) == {"src/sraging/engine/observation.py": "T034",
                                                   "src/sraging/engine/evaluator.py": "T034"}
    lines = "M\tsrc/sraging/engine/observation.py\nM\tsrc/sraging/engine/evaluator.py\nM\tsrc/sraging/engine/grid.py\n"
    monkeypatch.setattr(A3, "git", lambda *a: lines if a[:2] == ("diff", "--name-status") else "")
    monkeypatch.setattr(A3, "docstring_only", lambda path, base: False)
    exempt = []
    bad = A3.untouched_problems(base="d604a97", spec_exempt_out=exempt)
    assert {e["path"] for e in exempt} == {"src/sraging/engine/observation.py", "src/sraging/engine/evaluator.py"}
    by = {e["path"]: e["spec"] for e in exempt}          # evaluator.py is 004's too: the first spec listing it names it
    assert by["src/sraging/engine/observation.py"] == "006" and bad == ["M\tsrc/sraging/engine/grid.py"]


# --------------------------------------------------------------------------------------- parity (R-7, R-8; T037)
@pytest.fixture(scope="module")
def small_case():
    from sraging import Cohort, SRParams
    c = Cohort.from_csv(os.path.join(REPO, "tests", "unit", "data", "small_cohort.csv"), delta=1.0, window=(45.0, 70.0),
                        time_unit="days")
    return c, SRParams(eta=1.2e-3, beta=0.15, epsilon=0.01, kappa=0.5, time_unit="days")


def test_parity_passes_unplanted_and_reaches_the_underflow_point(small_case):
    r = ACC.parity_case(*small_case)
    assert r["underflow_point"] and r["FR-005"]["passes"], [x["S_lo"] for x in r["rows"]]
    assert r["R-7"]["v1"]["passes"] and r["R-7"]["v033"]["passes"], r["R-7"]
    assert r["passes"]


@pytest.mark.parametrize("plant", ACC.PLANTS)
def test_a_wrong_conditioning_term_edge_tie_or_missing_clip_fails(small_case, plant):
    r = ACC.parity_case(*small_case, plant=plant)
    assert not r["passes"], (plant, r["R-7"])
    if plant == "clip":
        assert not r["R-7"]["v1"]["passes"] and r["R-7"]["v033"]["passes"]       # only v1's rule was undone


def test_the_engine_edges_pass(small_case):
    r = ACC.parity_edges(*small_case)
    assert r["passes"], r


# --------------------------------------------------------------------------------------- v033's leverage (R-9; T041)
def test_r9s_null_panel_is_003fs_p3_until_the_freeze_names_one():
    ids = ACC.v033_ids(_rules(False))
    assert len(ids) == 26 and len(set(ids)) == 26
    assert ACC.v033_ids({**_rules(True), "panel": {"v033": {"ids": ["a", "b"]}}}) == ["a", "b"]


R9 = {"null_max_flags": 1, "null_min_assessed": 20}


def _ctl(option="C", flagged=True, refuses=True):
    return {"itp_refuses": refuses, "option": option, "control_flagged": flagged}


def _null(n_flag, n_slide, n_not=0):
    rows = {f"f{i}": {"status": "ok", "leverage_flag": "flag"} for i in range(n_flag)}
    rows.update({f"s{i}": {"status": "ok", "leverage_flag": "flat_valley_slide"} for i in range(n_slide)})
    rows.update({f"n{i}": {"status": "ok", "leverage_flag": "not_assessed"} for i in range(n_not)})
    return rows


def test_r9_under_option_c_at_p3s_26():
    """Naveh, 2026-11-11: at most 1 flag among the assessed fits, at least 20 of 26 assessed (P 0.91 at a true 2 %
    flag rate, 0.25 at 10 % [M]); the first assessed candidate flagged; ITP's age-0 control refused."""
    ids = lambda rows: list(rows)                                                     # noqa: E731
    one = _null(1, 25)
    assert ACC.score_v033(one, ids(one), {}, _ctl(), R9)["status"] == "pass"
    two = _null(2, 24)                                                                 # planted: 2 null flags
    assert ACC.score_v033(two, ids(two), {}, _ctl(), R9)["status"] == "fail"
    few = _null(0, 19, 7)                                                              # 19 assessed: inconclusive
    assert ACC.score_v033(few, ids(few), {}, _ctl(), R9)["status"] == "inconclusive"
    assert ACC.score_v033(one, ids(one), {}, _ctl(flagged=False), R9)["status"] == "fail"    # control not flagged
    out = ACC.score_v033(one, ids(one), {}, _ctl(option="B", flagged=False), R9)              # nothing assessed: B
    assert out["status"] == "fail" and out["R-9"]["option"] == "B"
    assert ACC.score_v033(one, ids(one), {}, _ctl(refuses=False), R9)["status"] == "fail"    # ITP must refuse
    assert ACC.score_v033(one, ids(one) + ["missing"], {}, _ctl(), R9)["status"] == "fail"


def test_the_first_assessed_candidate_is_the_control(monkeypatch):
    """Option C's order: a candidate J does not flag is skipped; not_assessed moves on; the first assessed stops."""
    monkeypatch.setattr(ACC, "itp_control", lambda b=None: {"label": "not_assessed", "warnings": ["LEVERAGE"]})
    monkeypatch.setattr(ACC, "r9_candidate", lambda cid: (cid, 9))
    labels = {"itp-late": "not_assessed", "planted-k0.1-p03-n300-late": "flag", "planted-k1-p03-n300-late": "flag"}
    monkeypatch.setattr(ACC, "leverage_of", lambda coh, cal, row: {
        "label": "flag" if cal == "profile_jax2" else labels[coh], "flagged_planted": True})
    monkeypatch.setattr(ACC, "j_flag_measured", lambda cid: True)
    out = ACC.r9_control()
    assert out["control"] == "planted-k0.1-p03-n300-late" and out["option"] == "C" and out["control_flagged"]
    assert [c["cid"] for c in out["candidates"]] == ["itp-late", "planted-k0.1-p03-n300-late"]
    labels.update({k: "not_assessed" for k in labels})
    out = ACC.r9_control()
    assert out["option"] == "B" and out["control"] is None and len(out["candidates"]) == 3


def test_v033s_resume_collects_the_cell_and_never_reruns_the_pilot(monkeypatch):
    """2026-11-14: `v033-leverage --dev --resume <job> --job-head <sha>` meant to score the stored P3 rows; the --dev
    branch ran the ITP pilot first (0.49 Mac CPU-h). --dev alone is still the pilot."""
    ran = []
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(True))
    monkeypatch.setattr(ACC, "phase_v033_pilot", lambda a: ran.append("pilot"))
    monkeypatch.setattr(ACC, "connect_v6", lambda: ran.append("cell") or (_ for _ in ()).throw(SystemExit("cell")))
    monkeypatch.setattr(ACC.A3, "git", lambda *a, **k: "966bbe7")
    with pytest.raises(SystemExit, match="cell"):
        ACC.main(["v033-leverage", "--dev", "--resume", "679398", "--job-head", "966bbe7"])
    assert ran == ["cell"]
    ACC.main(["v033-leverage", "--dev"])
    assert ran == ["cell", "pilot"]


def test_r9s_null_panel_refuses_before_the_freeze(monkeypatch):
    monkeypatch.setattr(ACC, "read_rules", lambda: _rules(False))
    with pytest.raises(SystemExit) as ex:
        ACC.main(["v033-leverage"])
    assert "after the freeze" in str(ex.value)


def test_a_wave_waits_for_the_specs_wexac_bound(tmp_path, monkeypatch):
    """Naveh, 2026-11-11: 300 WEXAC CPU-h for the spec; the cells' caps (1.25 x each) sum past it, so every wave also
    checks the spec's total across the cells' ledgers."""
    monkeypatch.setattr(ACC, "RESULTS_GATE", str(tmp_path))
    ACC.ledger_add("pilot", {"wave": 1, "job_id": "1", "element_cpu_s": 8.26 * 3600})
    ACC.ledger_add("a1", {"wave": 1, "job_id": "2", "element_cpu_s": 150.0 * 3600})
    rules = {"budget": {"wexac_cpu_h": 300.0}}
    assert ACC.spec_spent_wexac_h() == pytest.approx(158.26)
    assert ACC.spec_allows(141.0, rules) and not ACC.spec_allows(142.0, rules)
    assert ACC.spec_allows(1e6, {})                                              # before the freeze sets one


def test_the_proposal_holds_the_decisions_of_2026_11_11():
    import gate_006 as G
    r = ACC.read_rules()
    R1 = r["rules"]["R-1"]
    assert R1["collapse_p_bad"] == 0.70 and R1["a2_truncated_max"] == 0.10 and R1["gate_cell"] == "a1"
    assert R1["one_sided_min"] == G.ONE_SIDED_MIN and R1["truncated_dex"] == G.TRUNCATED_DEX
    assert r["rules"]["R-2"]["max_dex"] == 0.1 and r["rules"]["R-4"]["tolerance_nats"] == 0.1
    assert r["rules"]["FR-022"]["max_rate"] == 0.05
    assert (r["rules"]["R-9"]["null_max_flags"], r["rules"]["R-9"]["null_min_assessed"]) == (1, 20)
    assert len(r["rules"]["R-9"]["candidates"]) == len(ACC.R9_CANDIDATES) == 3
    assert r["budget"]["wexac_cpu_h"] == 350.0 and r["budget"]["mac_cpu_h"] == 21.0 and r["pricing"]["r"] == 1.162      # 300, then 350 for A2 (11-14) and r["panel"]["dg"]["seed"] == 20261111
    assert r["caps"]["a1"] == pytest.approx(1.25 * 153.3, abs=0.1) and r["caps"]["a2"] == pytest.approx(1.25 * 105.4,
                                                                                                         abs=0.1)
