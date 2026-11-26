"""The `fp_piecewise` oracle snapshot and its worker (spec 002, T015)."""
import numpy as np
import pytest

import oracle_fp_piecewise as OP
from conftest import not_applicable

pytestmark = pytest.mark.parity


def _tiny():
    rng = np.random.default_rng(0)
    t = np.round(rng.gamma(9.0, 14.0, 50)) + 1.0
    e = (rng.random(50) > 0.05).astype(int)
    return t, e


THETA = dict(eta=0.000943849, beta=0.106012, epsilon=0.00885909, xc=1.0, kappa_phys=0.00525768)
GRID = dict(L=124.0, N=100, dt=1e-2, tau_max=3.0)


def _preflight_or_skip():
    problems = OP.preflight()
    if problems:
        not_applicable("; ".join(problems))


def test_tampered_md5_is_not_applicable(monkeypatch):
    man = OP._read_manifest()
    man["files"][0]["md5"] = "0" * 32
    monkeypatch.setattr(OP, "_read_manifest", lambda: man)
    assert any("checksum mismatch" in p for p in OP.preflight())
    with pytest.raises(OP.NotApplicable):
        OP.call("ping")
    with pytest.raises(OP.NotApplicable):
        OP.Session()


def test_numerics_override_and_missing_pycache_are_refused(monkeypatch):
    monkeypatch.setenv("SRK_DT_CEILING", "1")
    assert any("SRK_DT_CEILING" in p for p in OP.preflight())
    monkeypatch.undo()
    monkeypatch.setattr("sys.pycache_prefix", None)
    assert any("pycache_prefix" in p for p in OP.preflight(worker=True))


def test_round_trip_matches_v033_snapshot_to_zero_ulps(oracle_v033):
    _preflight_or_skip()
    before = OP.fingerprint()
    t, e = _tiny()
    from SRK import fp_likelihood as fpl                      # the v033 snapshot's, in THIS process
    cases = [dict(), dict(external_hazard=6.0), dict(time_range=(20.0, 150.0)), dict(sampling_dt=1.0),
             dict(return_parts=True)]
    with OP.session() as s:
        assert s.ready["pycache"]
        for extra in cases:
            kw = dict(THETA, **GRID, **extra)
            want = fpl.fp_loglik_physical(t, e, **{**kw, "time_range": kw.get("time_range")}) \
                if "time_range" in kw else fpl.fp_loglik_physical(t, e, **kw)
            got = s.call("fp_loglik_physical", death_times=t, events=e, **kw)
            if isinstance(want, dict):
                for k, v in want.items():
                    assert got["loglik" if k == "loglik" else k] == v or (np.isnan(v) and np.isnan(got[k])), k
            else:
                assert got["value"] == want, extra
            assert got["cpu_s"] >= 0.0
        # one-shot mode agrees with the session
        one = OP.call("fp_loglik_physical", death_times=t, events=e, **dict(THETA, **GRID))
        assert one["value"] == fpl.fp_loglik_physical(t, e, **dict(THETA, **GRID))
        # the dimensionless solver ops
        rb, rh, re, kd = fpl.to_dimensionless(THETA["eta"], THETA["beta"], THETA["epsilon"], 1.0, 124.0,
                                              k=THETA["kappa_phys"])
        d = s.call("fpt_full_density", rho_eta=rh, rho_beta=rb, rho_eps=re, kappa=kd, N=100, dt=1e-2,
                   tau_max=3.0)
        tau, S, p, *_ = fpl.fpt_full_density(rh, rb, re, kd, N=100, dt=1e-2, tau_max=3.0)
        assert np.array_equal(d["tau"], tau) and np.array_equal(d["S"], S) and np.array_equal(d["f"], p)
        # a worker-side error comes back as OracleError and leaves the session usable
        with pytest.raises(OP.OracleError):
            s.call("no_such_op")
        assert s.call("ping")["value"] == 1
    assert OP.fingerprint() == before, "the snapshot changed (or grew a file) during the test"


def test_piecewise_ops_and_build_segments():
    _preflight_or_skip()
    before = OP.fingerprint()
    t, e = _tiny()
    th = dict(eta=0.000943849, beta=0.106012, epsilon=0.00885909, kappa=0.00525768, xc=1.0)
    ivs = [{"kind": "Intervention", "param": "beta", "effect": -0.3, "window": [30.0, 80.0]},
           {"kind": "ExtHazard", "level": 5.0, "window": [50.0, 100.0]}]
    kw = dict(theta=th, interventions=ivs, L=124.0, t_end=214.0, N=100, dt=1e-2, tau_max=3.0)
    with OP.session() as s:
        seg = s.call("build_segments", theta=th, interventions=ivs, L=124.0, t_end=214.0, return_exth=True)
        assert seg["segments"].shape[1] == 8 and seg["exth_level"].tolist() == [5.0]
        terms = s.call("piecewise_loglik_terms", death_times=t, events=e, **kw)["terms"]
        ll = s.call("piecewise_loglik", death_times=t, events=e, **kw)["value"]
        assert terms.shape == t.shape and np.isfinite(ll)
        assert ll == float(terms[e == 1].sum() + terms[e != 1].sum())
        c = s.call("piecewise_curves", **kw)
        assert c["S"][0] == 1.0 and c["n_segments"] == 3
    assert OP.fingerprint() == before
