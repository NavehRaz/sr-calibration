"""Spec 006's adapter (FR-013; research R9; register D-104 revised; tasks T021): each method row declares its
calibrator's capabilities in the tester's names, the table names 1.0.1 and W on library v6 (the 1.0.0 rows stay,
pinned to the version their description names), H 1.0.1 is not refused off v2, the observation loader passes a window
into the Cohort, and a method whose calibrator does not fit windows returns error:WindowUnsupported without fitting."""
import ast
import os
import types

import pytest

from sraging.adapters import srbench as A
from sraging.adapters.observation import cohort_from_observation
from sraging.errors import UnsupportedOptionError
from test_adapter_offline import fake_types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLONE_TYPES = os.path.join(REPO, ".cache", "tester_0805d36", "srbench", "contract", "types.py")


def _tester_capabilities():
    if not os.path.exists(CLONE_TYPES):
        pytest.skip("the pinned tester clone is not here")
    tree = ast.parse(open(CLONE_TYPES).read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CAPABILITIES")
    return ast.literal_eval(node.value)


def obs(window=None, delta=1.0):
    return types.SimpleNamespace(cohort_id="x", ages=(3.0, 5.0, 8.0, 9.0, 12.0), events=(1, 1, 1, 1, 0), delta=delta,
                                 window=window)


def test_each_row_declares_its_calibrators_capabilities_in_the_testers_names():
    caps = _tester_capabilities()
    assert caps == ("external_hazard", "fit_window", "heterogeneity", "censoring")
    want = {"sraging_profile_v033": (), "sraging_profile_jax2": (), "sraging_profile_jax2_h": (),
            "sraging_profile_jax2_101": (), "sraging_profile_jax2_h_101": ("fit_window", "external_hazard"),
            "sraging_profile_jax2_w": ("fit_window",)}
    for name, c in want.items():
        assert A.descriptor_fields(name)["capabilities"] == c and set(c) <= set(caps), name


def test_the_table_names_1_0_1_and_w_on_v6_and_pins_the_1_0_0_rows():
    m = A.methods()
    assert m["sraging_profile_jax2_w"]["calibrator"] == "profile_jax2_w@1.0.1+w1"
    assert m["sraging_profile_jax2_101"]["calibrator"] == "profile_jax2@1.0.1"
    assert m["sraging_profile_jax2_h_101"]["calibrator"] == "profile_jax2_h@1.0.1"
    assert m["sraging_profile_jax2"]["calibrator"] == "profile_jax2@1.0.0"           # what its description says
    assert m["sraging_profile_jax2_h"]["calibrator"] == "profile_jax2_h@1.0.0"
    for name in ("sraging_profile_jax2_w", "sraging_profile_jax2_101", "sraging_profile_jax2_h_101"):
        assert m[name]["library"] == "v6" and m[name]["tester_head"] == A.TESTER_V6_HEAD, name
    for name, row in m.items():
        if name != "sraging_profile_v033":
            version = row["calibrator"].split("@")[1]
            assert row["version_suffix"].endswith(version) and version in row["description"], name


def test_h_1_0_1_is_not_refused_off_v2_and_h_1_0_0_still_is():
    A._refuse_h_off_v2("sraging_profile_jax2_h_101", "v6")
    with pytest.raises(UnsupportedOptionError, match="v2"):
        A._refuse_h_off_v2(A.H_PORT, "v6")


def test_the_loader_passes_a_window_into_the_cohort():
    """D-104 revised: the window reaches the calibrator, which fits it (W, H 1.0.1) or refuses it by name."""
    assert cohort_from_observation(obs(window=(3.0, 9.0))).window == (3.0, 9.0)
    assert cohort_from_observation(obs()).window is None
    with pytest.raises(UnsupportedOptionError, match="window"):                    # off the delta grid
        cohort_from_observation(obs(window=(3.5, 9.0)))


@pytest.mark.parametrize("name", ["sraging_profile_jax2", "sraging_profile_jax2_101"])
def test_a_method_that_does_not_fit_windows_returns_window_unsupported_without_fitting(name, monkeypatch):
    import sraging
    calls = []
    real = sraging.fit_cohort
    monkeypatch.setattr(sraging, "fit_cohort", lambda c, **k: calls.append(k) or real(c, **k))
    cs = A.make_jax2_method(fake_types(), name).calibrate(obs(window=(3.0, 9.0)))
    assert cs.status == "error:WindowUnsupported" and "profile_jax2_w" in cs.diagnostics["exception"]
    assert len(calls) == 1 and calls[0]["combinations"] is False               # refused inside, before any fit


def test_v033s_method_returns_window_unsupported():
    cs = A.make_method(fake_types()).calibrate(obs(window=(3.0, 9.0)))
    assert cs.status == "error:WindowUnsupported" and "window" in cs.diagnostics["exception"]
