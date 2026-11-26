"""FR-005 labels: weakly determined values are labelled, never suppressed (T023)."""
from sraging.labels import labels
from sraging.params import SRParams
from sraging.result import Claim

THETA = SRParams(eta=1e-3, beta=0.2, epsilon=4e-3, kappa=7.0)


def claims(**shapes):
    out = []
    for p in ("eta", "beta", "epsilon", "kappa"):
        s = shapes.get(p, "closed")
        out.append(Claim(p, s, 1.0 if s in ("closed", "lower_bound") else None,
                         2.0 if s in ("closed", "upper_bound") else None,
                         None if s == "none" else 1.5, 0.95, 1, ""))
    return tuple(out)


def test_none_says_not_determined_and_representative():
    lab = labels(claims(kappa="none"), THETA)
    assert "not determined by the data inside the prior box" in lab["kappa"]
    assert "representative point" in lab["kappa"]


def test_one_sided_says_which_side():
    lab = labels(claims(epsilon="upper_bound", eta="lower_bound"), THETA)
    assert "only an upper bound" in lab["epsilon"]
    assert "only a lower bound" in lab["eta"]


def test_kappa_and_raw_beta_always_carry_the_ridge_note():
    lab = labels(claims(), THETA)
    for p in ("kappa", "beta"):
        assert "flat ridge" in lab[p] and "not a regression target" in lab[p]
    for p in ("eta", "epsilon"):
        assert "flat ridge" not in lab[p]


def test_every_raw_beta_upper_end_carries_the_measured_box_note():
    """FR-025, after spec 003's kappa_max = 3000 study (results/kmax_003.json: box-limited; the closed negative
    controls' upper ends moved with the box too) and Naveh's choice (2026-10-06): every raw-beta upper end, an upper
    bound or a closed claim's, says the kappa box sets it; no "untested" left."""
    for shape in ("upper_bound", "closed"):
        lab = labels(claims(beta=shape), THETA)["beta"]
        assert "upper end is set by the kappa box (kappa_max = 300), not by the data" in lab, shape
        assert "untested" not in lab and "may be set" not in lab, shape
        assert ("read it as a lower bound" in lab) == (shape == "closed"), shape
        assert ("on neither side" in lab) == (shape == "upper_bound"), shape
    for shape in ("lower_bound", "none"):
        assert "kappa box" not in labels(claims(beta=shape), THETA)["beta"], shape
    for p in ("eta", "epsilon", "kappa"):
        assert "kappa box" not in labels(claims(**{p: "upper_bound"}), THETA)[p]


def test_every_quantity_has_a_label():
    lab = labels(claims(eta="none", beta="none", epsilon="none", kappa="none"), THETA)
    assert set(lab) == {"eta", "beta", "epsilon", "kappa"} and all(lab.values())
