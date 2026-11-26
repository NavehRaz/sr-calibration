"""The leverage check's rule in force (spec 003f FR-006; the gate's C7). Decided by Naveh on 2026-10-18 after pilot P3
and revised on 2026-10-19 after the sr-naveh review and an offline re-score of P3's drops
(specs/003f-leverage-check/review/): every flag quantity (eta/eps, rbar/eta, eta and eps; the maximum over the four),
a cost in total nats (the claims' one-parameter 95 % threshold, independent of n), a refit ending below theta-hat on
its own objective labelled failed, and the warm start against the reported theta-hat in place of "MLE stage on both
sides". It must equal tests/acceptance/panel_003f.toml [rule] (tools/acceptance_003f.py refuses otherwise)."""

RULE = {
    "quantities": "all4",            # a key of sraging.leverage.QUANTITY_SETS: "all4", "lead_pair" or "raw"
    "ratio": 1.5,                    # sec13 eq. lev-ratio: R_j > ratio
    "delta_total_nats": 1.92,        # the refit's gain on the reduced objective, total nats: chi2(1, 0.95)/2
    "fail_nats": 0.1,                # a refit ending more than this below theta-hat is failed, not a measurement
    "refit": "mle_stage_and_warm_start",   # the MLE stage and L-BFGS-B from theta-hat; the better, vs theta-hat
    "provisional": False,            # Naveh, 2026-10-19: "Warm start vs reported theta-hat", "All four, 1.92 nats"
}

#: Spec 003g FR-011 (Naveh Da, 2026-10-22; per individual since DV-3, Naveh 2026-10-28): a fit whose warm start on the
#: full data did not move from theta-hat (g0 == 0) at a free-gradient norm above G_CRIT_PER_IND per individual is
#: labelled `not_assessed`: a refit without one row cannot be compared with a point the optimiser could not leave. The
#: norm is a sum over individuals, so the registered G_CRIT = 1.0 on the sum read the same pathology as no_flag at
#: n = 1,000. Measured |g|/n: assessed 1.9e-6 to 5.9e-6 (ITP+age0, fit 1), not assessed 4.0e-4 to 2.8e-2 (the 7 human
#: hazard pilot cohorts, fit 2); 5e-5 is about a decade from each. Kept apart from RULE, which must equal
#: panel_003f.toml [rule].
G_CRIT_PER_IND = 5e-5
