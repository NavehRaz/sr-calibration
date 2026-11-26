"""S5's wall (tasks T084; gate C5, Naveh's three-branch rule; research R20, R30): s5_wall_branch(proj_clamp,
proj_reflect, reflect_converged, cap=86, smallest_detectable, plant_size), a pure function, one planted case per branch
beside its twin: 1, clamp if it fits under the cap; 2, else reflect if measured converged and under the cap; 3, else
clamp at the cap, asking Naveh first when the smallest detectable shift is above the plant's."""
from sraging import sampler_rules as R

def test_s5_branch_1_clamp_fits_under_the_cap():
    b = R.s5_wall_branch(80.0, 60.0, True, cap=86.0, smallest_detectable=1.0, plant_size=1.92)
    assert (b["branch"], b["wall"], b["action"]) == (1, "clamp", "run")
    assert R.s5_wall_branch(90.0, 60.0, True, smallest_detectable=1.0, plant_size=1.92)["branch"] == 2      # twin


def test_s5_branch_2_reflect_when_converged_and_under_the_cap():
    b = R.s5_wall_branch(120.0, 70.0, True, smallest_detectable=1.0, plant_size=1.92)
    assert (b["branch"], b["wall"], b["action"]) == (2, "reflect", "run")
    assert R.s5_wall_branch(120.0, 70.0, False, smallest_detectable=1.0, plant_size=1.92)["branch"] == 3     # twin


def test_s5_branch_3_clamp_at_the_cap_asks_naveh_when_the_plant_is_below_reach():
    b = R.s5_wall_branch(120.0, 70.0, False, smallest_detectable=2.5, plant_size=1.92)
    assert (b["branch"], b["wall"], b["action"]) == (3, "clamp", "ask_naveh")
    t = R.s5_wall_branch(120.0, 70.0, False, smallest_detectable=1.5, plant_size=1.92)                     # twin
    assert (t["branch"], t["wall"], t["action"]) == (3, "clamp", "run")
    assert R.s5_wall_branch(120.0, 100.0, True, smallest_detectable=1.5, plant_size=1.92)["branch"] == 3
