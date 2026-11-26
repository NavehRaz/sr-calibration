"""The tensor-product FP reference (tasks T104; research R28; data-model "Tensor-product reference"), internal: the
per-node program of R4 over the product grid of d <= 3 parameters, folded lexicographically over the parameter-sorted
node indices with weights prod_j w_(j, k_j); node counts per dimension by R26's check (only failing dimensions climb);
a node product above the 20,000 budget reads `none (budget)`; one dimension at sd 0 equals the single-parameter curve
bit for bit."""
import numpy as np
import pytest

from sraging import SRParams
from sraging.engine import mixture as MIX
from sraging.engine import quadrature as Q
from sraging.engine import tensor_reference as T
from sraging.engine.inputs import Heterogeneity

pytestmark = pytest.mark.engine
P = SRParams(eta=6e-4, beta=0.08, epsilon=8e-3, kappa=0.1)
XC, ETA, EPS = (Heterogeneity(parameter="xc", sd=0.2), Heterogeneity(parameter="eta", sd=0.2),
                Heterogeneity(parameter="epsilon", sd=0.1))
ETA0 = Heterogeneity.zero("eta")


@pytest.fixture(scope="module")
def single():
    """The Langevin's single-parameter reference (xc), whose evaluator the tensor shares."""
    pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")
    from sraging.langevin import sample as LS
    return LS.Reference(LS.LangevinConfig(params=P, n=1000, key=0, dt=1.0, heterogeneity=XC, error="ks"))


def test_one_dimension_at_sd_0_equals_the_single_parameter_curve_bit_for_bit(single):
    tr = T.TensorReference(P, (XC, ETA0), M={"xc": 11}, ev=single.ev)
    tau, S = single.survival_grid(11)
    assert tr.kind == "tensor" and tr.M == {"eta": 1, "xc": 11} and tr.node_count == 11
    assert tr.S.tobytes() == S.tobytes() and tr.tau.tobytes() == tau.tobytes()


def test_the_fold_is_lexicographic_over_the_parameter_sorted_node_indices(single):
    tr = T.TensorReference(P, (XC, ETA), M={"eta": 2, "xc": 3}, ev=single.ev)
    assert tr.parameters == ("eta", "xc") and tr.fold_order == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    progs, z = MIX.programs(single.ev), single.ev._z(single.p0)
    re, rx = Q.truncated_normal_rule(0.2, 2), Q.truncated_normal_rule(0.2, 3)
    S = None
    for i in range(2):
        for j in range(3):
            Sk = progs.survival(z, T.scale_vector({"eta": float(re.s[i]), "xc": float(rx.s[j])}))
            w = float(re.w[i]) * float(rx.w[j])
            S = w * Sk if S is None else S + w * Sk
    assert tr.S.tobytes() == np.asarray(S, float).tobytes()


def test_a_node_product_above_the_budget_reads_none_budget():
    tr = T.TensorReference(P, (XC, ETA, EPS), M={"xc": 128, "eta": 128, "epsilon": 5})
    assert tr.kind == "none (budget)" and tr.S is None and tr.node_count == 128 * 128 * 5 > T.BUDGET == 20_000
    assert T.TensorReference(P, (XC, ETA, EPS), M={"xc": 11, "eta": 11, "epsilon": 5}).node_count == 605   # the twin


def test_more_than_three_dimensions_is_refused():
    with pytest.raises(ValueError, match="d <= 3"):
        T.TensorReference(P, (XC, ETA, EPS, Heterogeneity(parameter="beta", sd=0.1)))


def test_the_per_dimension_check_climbs_only_the_failing_dimension(single):
    kap, xc3 = Heterogeneity(parameter="kappa", sd=0.05), Heterogeneity(parameter="xc", sd=0.3)
    tr = T.TensorReference(P, (kap, xc3), bar=1e-4, ev=single.ev)
    assert tr.kind == "tensor" and tr.check["verdict"] == "converged"
    assert tr.M["kappa"] == MIX.RUNGS[0] and tr.M["xc"] > MIX.RUNGS[0]
    assert all(r["passed"] for r in tr.check["per_dimension"]["kappa"])
    assert not tr.check["per_dimension"]["xc"][0]["passed"] and tr.check["per_dimension"]["xc"][-1]["passed"]


def test_the_single_parameter_program_passes_one_in_the_other_four_slots():
    for p in MIX.SCALED:
        s = MIX._scales(p, 0.731)
        assert s.tobytes() == T.scale_vector({p: 0.731}).tobytes()
        assert s.tobytes() == T.scale_vector({p: 0.731, **{q: 1.0 for q in MIX.SCALED if q != p}}).tobytes()
