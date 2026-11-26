"""Spec 004's Langevin presets: the ONE place the pilots and the Langevin phases read them from (P1's "mice_F, Jack
Russell, E. coli, human"; S4's grid, P3, P4, the plants, HR-8 and HR-10 name presets from it).

**Decided** (spec.md Clarifications, Naveh 2026-10-29, "on the Langevin presets", corrected and re-decided on
2026-10-30 at 4075cd5: keep getTheta's values; at 004's close, T143 reminds Naveh to move them to values calibrated
with the new calibrator):
- **The values** are `SRK.presets.getTheta(name, ExtH=True)` at its defaults (type "mode_overall", time_unit "auto"),
  read once from the oracle snapshot (tests/oracle/fp_piecewise_cb191f5, through tools/oracle_fp_piecewise.py's
  read-only op `preset_theta`), in the unit getTheta's "auto" converts to. Hard-coded below; tests/unit/
  test_presets_004.py re-derives every value from the snapshot, exactly.
  - Measured on reading: summery_mode_overall.csv has no `kappa` row, so getTheta's `_kappa_convention` takes its
    legacy branch for all four (eta, beta, epsilon divided by xc and xc**2; kappa = 0.5 / xc_original; xc = 1). The
    kappa values below are therefore 0.5 / xc of that table (0.5 / 29.7, 0.5 / 0.443, 0.5 / 4.408, 0.5 / 6.315).
- **The human** is combined_human_M (T6a's): its theta is combined_human_M's own column (xc 6.315); only SRK's
  simulation configuration (t_end, nsteps, time_step_multiplier: P1's base step) is Denmark_M_1900_hetro's, to which
  get_config_params maps it.
- **The external hazard:** getTheta's ExtH where the table has one (Jack Russell, 5.577, in the converted unit), as
  the rate h_ext = exp(-ExtH), the v6 cases' and SRK's convention (SR_hetro.py:460-462: a finite external_hazard kills
  with probability exp(-external_hazard) dt per substep; a missing one, NaN, kills nothing). The others: h_ext = 0.
- **P1's base step** Delta t = t_end / (nsteps - 1) / time_step_multiplier from get_config_params(name) (All_config.csv
  through SRK's own reader), whose t_end is in the preset's own unit, the same unit getTheta converts to (both
  auto-detect it from the name; get_config_params converts only when given another unit).

**Pilot configurations (ROLES)** (spec.md Clarifications, Session 2026-10-30, 4075cd5: Naveh accepted the four
proposals): p3, p4, hr8 and plants, with "paths" from the documents. A value that comes from P1's result (P1's clamp
step, B1's lag-sized step, the first HR-10 age) is a named reference {"p1": [key, ...]} into
tests/acceptance/results/p1_langevin_004.json, resolved by `resolve` when P1 is written; nothing stands in for it.
The plants' table is the source freeze-L's `[plants]` is written from (tools/plants_004.py's schema; the phase reads
the frozen table). A step is a reference {"step": [cell, field]} to the frozen steps (p1-pooled's where written,
tools/acceptance_004.frozen_steps), never P1's top-level "steps". s5 (8f0f269): spec 002's score-het cohort, its
steps p1-pooled's two S5 cells.
"""
from __future__ import annotations

import math

SOURCE = ("SRK.presets.getTheta(name, ExtH=True) (mode_overall, time_unit auto) and get_config_params(name), oracle "
          "snapshot tests/oracle/fp_piecewise_cb191f5, read 2026-10-30; spec.md Clarifications (Naveh 2026-10-29)")


def _step(t_end, nsteps, tsm) -> float:
    """SRK's effective step, t_end / (nsteps - 1) / time_step_multiplier (register B2's kernel convention)."""
    return float(t_end) / (int(nsteps) - 1) / int(tsm)


def _preset(srk_name, unit, eta, beta, epsilon, xc, kappa, ext_h, t_end, nsteps, tsm):
    return {"srk_name": srk_name, "time_unit": unit,
            "params": {"eta": eta, "beta": beta, "epsilon": epsilon, "kappa": kappa, "xc": xc,
                       "h_ext": 0.0 if ext_h is None else math.exp(-ext_h)},
            "ext_h": ext_h, "srk_step": {"t_end": t_end, "nsteps": nsteps, "time_step_multiplier": tsm},
            "base_dt": _step(t_end, nsteps, tsm), "source": SOURCE}


#: name -> {"srk_name", "time_unit", "params" {eta, beta, epsilon, kappa, xc, h_ext}, "ext_h", "srk_step", "base_dt"}
PRESETS: dict = {
    "mice_F": _preset("mice_F", "weeks", 0.0011978543153447314, 0.14843519768275326, 0.011737266323603139, 1.0,
                      0.016835016835016835, None, 214, 5000, 1),
    "Jack_Russell": _preset("Jack_Russell_vetCompass", "years", 6.461176440918731, 586.7818639421007,
                            8.738538150302855, 1.0, 1.1286681715575622, 5.577, 24, 5000, 2),
    "ecoli": _preset("ecoli", "hours", 0.04000256016385049, 7.467357910906298, 0.4850601474582848, 1.0,
                     0.11343012704174228, None, 150, 5000, 1),
    "human": _preset("combined_human_M", "years", 0.23958347603888475, 27.62138466016985, 0.8978162157907424, 1.0,
                     0.0791765637371338, None, 110, 5000, 7),
}
NAMES = ("mice_F", "Jack_Russell", "ecoli", "human")       # P1's four, by role

#: the configuration a pilot or phase draws at, beyond its preset; only what the documents define
ROLES: dict = {
    # HR-10 (tasks.md T098; contracts/acceptance.md §4 `paths`; research R25 "Power at the real n"): the 4 presets,
    # homogeneous and xc sd 0.2, two ages near S = 0.75 and 0.4 on the dt grid, n = 25,000, G = 16 per wall. P1
    # measures the clamp's lump p0 and the paths' step bias at these ages (pick the ages with paths_004.pick_ages).
    "paths": {"presets": list(NAMES), "survival_targets": [0.75, 0.4], "n": 25_000, "G": 16, "het_sd": 0.2,
              "source": "tasks.md T098; contracts/acceptance.md §4 paths; research.md R25"},
    # P3 (research R19; contracts/acceptance.md §1): the human preset, recording delta 1 yr, key 3003; n in {2k, 20k,
    # 100k}; one eta (out of gen-1's table) and one xc (in it) configuration at sd 0.2 (tools/p3_cost_004.py)
    "p3": {"preset": "human", "delta": 1.0, "key": 3003, "ns": [2_000, 20_000, 100_000],
           "configs": [["out-of-table", "eta"], ["in-table", "xc"]], "sd": 0.2,
           "source": "spec.md Clarifications 2026-10-30 (4075cd5); research R19"},
    # P4 (OD-7 (a)): the human preset, n = 25,000, 20 paired repeats
    "p4": {"preset": "human", "n": 25_000, "source": "spec.md Clarifications 2026-10-30 (4075cd5); research R19"},
    # HR-8 (research R17): mice_F homogeneous at P1's clamp step, n = 25,013, W in {1, 7, 15}
    "hr8": {"preset": "mice_F", "het": "hom", "dt": {"step": ["mice_F|hom|clamp", "dt"]}, "n": 25_013,
            "W": [1, 7, 15], "source": "spec.md Clarifications 2026-10-30 (4075cd5); research R17"},
    # the plants (research R14; register D-402 to D-411; tasks T121: n = 25,000 unless R14 names another): mice_F for
    # every plant (no external hazard to confound HR-9's twin), at its own SRK configuration (PRESETS["mice_F"]
    # ["srk_step"]: t_end 214 wk, nsteps 5000); HR-9 at B1's lag-sized step from P1; recording at Delta t = 1 wk with
    # delta = 0.25 wk; D-410 at a young age; D-411 at the first HR-10 age
    "plants": {
        "preset": "mice_F", "n": 25_000,
        "source": "spec.md Clarifications 2026-10-30 (4075cd5); research R14; register D-402 to D-411",
        "hr9": {"preset": "mice_F", "dt": {"p1": ["b1", "mice_F", "hom", "dt_for_target"]}, "m": 16, "n": 25_000,
                "t_end": 214, "alpha": {"plant": 0.05, "twin": 0.025, "fix": 0.025},
                "predicted_lag": {"p1": ["b1", "mice_F", "hom", "target"]},
                # the fix arm at sraging's frozen step (Naveh 2026-11-03, sr-naveh finding 3)
                "fix_dt": {"step": ["mice_F|hom|clamp", "dt"]}},
        "b4": {"preset": "mice_F", "n": 1_003, "t_end": 214, "nsteps": 5000},
        "b11": {"preset": "mice_F", "n": 25_000, "t_end": 214, "nsteps": 5000, "sd_plant": 0.5, "sd_twin": 0.2},
        "seeding": {"preset": "mice_F", "n": 25_000, "t_end": 214, "nsteps": 5000, "W": 7},
        "hazard": {"preset": "mice_F", "n": 25_000, "nsteps": 5000, "dt": 214 / 4999, "h_delta_plant": 0.05,
                   "h_delta_twin": 1e-4},
        "recording": {"preset": "mice_F", "n": 25_000, "dt": 1.0, "delta": 0.25, "nsteps": 215},
        "d409": {"preset": "mice_F", "n": 25_000, "t_end": 214, "nsteps": 5000},
        "d411": {"preset": "mice_F", "n": 25_000, "dt": 214 / 4999,
                 "age": {"p1": ["cell_ages", "mice_F|hom|clamp", 0]}},
        # Naveh 2026-11-03 (ff794f5): B2 at mice_F's base step; D-410 at 10 base steps, at the base step
        "b2": {"preset": "mice_F", "n": 25_000, "step_size": 214 / 4999, "t_end": 214},
        "d410": {"preset": "mice_F", "n": 25_000, "age": 10 * 214 / 4999, "dt": 214 / 4999, "t_max": 214.0},
    },
}
# S5 (spec.md Clarifications 2026-11-01, 8f0f269; research R20): score_het_002's pure-k0.01-p00-c0-s0 at theta0
# (eta 2.125e-4, beta 1e-3, epsilon 3.98e-5, kappa 0.01 per day), n = 2,000, xc sd 0.2, through gen-1's transform:
# the v6 spec it names (tests/acceptance/results/score_het_002.json theta0_cohort; the v6 manifest's spec, its
# rho values at L_phys 100 days, delta 1 day, seed 100000) with n 2,000 and het xc 0.2. Its steps are p1-pooled's two
# S5 cells, read through the frozen-step accessor ({"step": [cell, field]}).
ROLES["s5"] = {
    "spec": {"name": "s5-pure-k0.01-p00-c0-s0-xc0.2", "rho_eta": 2.125427697723628, "rho_beta": 0.1,
             "rho_eps": 0.003981071705534973, "kappa": 0.01, "n": 2_000, "L_phys": 100.0, "seed": 100_000,
             "delta": 1.0, "het_param": "xc", "het_var": 0.2},
    "key": 100_000,
    "dt_clamp": {"step": ["s5|het|clamp", "dt"]},
    "dt_reflect": {"step": ["s5|het|reflect", "dt"]},
    "steps_per_individual": {"step": ["s5|het|clamp", "steps_per_individual"]},
    "source": "spec.md Clarifications 2026-11-01 (8f0f269); score_het_002.json theta0_cohort; research R20",
}
#: decided by Naveh only
UNDECIDED_ROLES = ()


class PresetsUndecided(SystemExit):
    """A preset or a configuration was read before Naveh decided it."""


def preset(name: str):
    """The preset's SRParams (in its own time unit)."""
    if name not in PRESETS:
        raise PresetsUndecided(f"preset {name!r} is not one of {sorted(PRESETS)}")
    from sraging import SRParams
    p = PRESETS[name]
    return SRParams(**p["params"], time_unit=p["time_unit"])


def base_dt(name: str) -> float:
    """P1's base step Delta t for the preset (dt = Delta t / m)."""
    if name not in PRESETS:
        raise PresetsUndecided(f"preset {name!r} is not one of {sorted(PRESETS)}")
    return PRESETS[name]["base_dt"]


def resolve(cfg, p1: dict | None = None, *, steps=None, only=None):
    """`cfg` with its references replaced, nested: {"p1": [key, ...]} by that value of P1's result, {"step": [cell,
    field]} by the frozen step's field (tools/acceptance_004.frozen_steps: p1-pooled's where written; a step that is
    not met is refused there, by name). `only`: resolve just these top-level keys (the rest stay references).
    Refused, naming the reference, when the source lacks it."""
    if only is not None:
        return {k: (resolve(v, p1, steps=steps) if k in only else v) for k, v in cfg.items()}
    if isinstance(cfg, dict):
        if set(cfg) == {"p1"}:
            v = p1
            for k in cfg["p1"]:
                try:
                    v = v[k]
                except (KeyError, IndexError, TypeError):
                    raise PresetsUndecided(f"P1's result has no {cfg['p1']!r} (needed by a pilot configuration)") \
                        from None
            return v
        if set(cfg) == {"step"}:
            cell, field = cfg["step"]
            if steps is None:
                raise PresetsUndecided(f"step {cell!r}: no frozen steps given (read them through frozen_steps)")
            return steps[cell][field]
        return {k: resolve(v, p1, steps=steps) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [resolve(v, p1, steps=steps) for v in cfg]
    return cfg


def role(name: str) -> dict:
    """A pilot's or phase's configuration (ROLES[name]), or a refusal while it is undecided."""
    if name not in ROLES:
        raise PresetsUndecided(f"{name}'s configuration: not decided (Naveh); tools/presets_004.py ROLES[{name!r}] "
                               f"is empty")
    return dict(ROLES[name])
