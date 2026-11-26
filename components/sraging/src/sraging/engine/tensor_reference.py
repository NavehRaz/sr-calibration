"""The tensor-product FP reference (spec 004 FR-024 ii; research R28; data-model "Tensor-product reference"), internal:
the FP survival of a Langevin heterogeneity tuple, for its error statement and HR-11 only (OD-11 (a): FP's public
likelihood, curves and damage density stay single-parameter).

    tr = TensorReference(params, members, *, M=None, bar=1e-4, ages=None, budget=BUDGET, ev=None)
    tr.kind, tr.M, tr.node_count, tr.check, tr.S_obs(t)

- **The node program** is R4's per-node program (`mixture.Programs.survival`), which takes the five scale factors
  (eta, beta, epsilon, kappa, xc): a node of the product grid scales each member's parameter by its own s; a member at
  sd 0 is one node, s = 1.0, w = 1.0 (exact: one dimension at sd 0 equals the single-parameter curve bit for bit).
- **The fold** is lexicographic over the parameter-sorted node indices, S = sum prod_j w_(j, k_j) S_k, accumulated in
  that order in float64 (the single-parameter fold's own pattern).
- **Node counts** (`M=None`): R26 per dimension, from the first rung of the ladder; for dimension j, S at the current
  counts against S with M_j at its next rung, sup over the check points (the march's grid where S > 1e-12, or `ages`)
  against `bar`; only the failing dimensions climb. A failing dimension at the cap's last pair reads
  `unconverged_at_cap`. A fixed `M` (a dict per parameter) is used unchecked.
- **The budget:** prod_j M_j <= 20,000 nodes; a product above it (to evaluate or to check) reads kind `none (budget)`,
  with no S.
- **d <= 3** members with a spread; more is refused (the Langevin's statement reads "no FP reference").
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from sraging.engine import mixture as MIX
from sraging.engine import quadrature as Q

BUDGET = 20_000
D_MAX = 3
S_FLOOR = 1e-12                     # the check's points: where S exceeds this on the march's grid


def scale_vector(scales: dict) -> np.ndarray:
    """The per-node program's five factors (mixture.SCALED order), 1.0 where a parameter is not scaled."""
    s = np.ones(len(MIX.SCALED))
    for p, v in scales.items():
        s[MIX.SCALED.index(p)] = v
    return s


def fold(progs, z, parameters, rules) -> tuple:
    """(S, the fold order): sum over the product of `rules` ((s, w) per parameter, in `parameters`' order) of
    prod_j w_(j, k_j) S(s), lexicographic over the node indices, accumulated in that order in float64."""
    order = list(itertools.product(*[range(np.asarray(r[0]).size) for r in rules]))
    S = None
    for idx in order:
        sc = scale_vector({p: float(rules[j][0][idx[j]]) for j, p in enumerate(parameters)})
        w = 1.0
        for j in range(len(rules)):
            w = w * float(rules[j][1][idx[j]])
        Sk = progs.survival(z, sc)
        S = w * Sk if S is None else S + w * Sk
    return np.asarray(S, float), order


def _next_rung(M):
    i = MIX.RUNGS.index(M)
    return MIX.RUNGS[i + 1] if i + 1 < len(MIX.RUNGS) else None


class TensorReference:
    def __init__(self, params, members, *, M=None, bar=1e-4, ages=None, budget=BUDGET, ev=None):
        from sraging.params import SRParams
        members = sorted(members, key=lambda h: h.parameter)
        if len({h.parameter for h in members}) != len(members):
            raise ValueError("a repeated parameter in the tensor reference")
        active = [h for h in members if h.sd > 0]
        if not active:
            raise ValueError("the tensor reference needs at least one member with a spread")
        if len(active) > D_MAX:
            raise ValueError(f"the tensor reference takes d <= {D_MAX} members with a spread, got {len(active)}")
        self.parameters = tuple(h.parameter for h in members)
        self.sd = {h.parameter: float(h.sd) for h in members}
        self.budget, self.bar, self.h = int(budget), float(bar), float(params.h_ext)
        self.p0 = SRParams(**{**params.as_dict(), "h_ext": 0.0})
        self._ev = ev
        self._ages = None if ages is None else np.asarray(ages, float)
        self.fold_order, self.S, self.tau = [], None, None
        if M is not None:
            Ms = {p: (1 if self.sd[p] == 0 else int(M[p])) for p in self.parameters}
            self.check = {"verdict": "fixed", "M": dict(Ms)}
            self._finish(Ms)
        else:
            self._climb()

    # -- the evaluator and its programs -------------------------------------------------------------------------
    def _programs(self):
        if self._ev is None:
            from sraging.cohort import Cohort
            from sraging.engine.curves import model_grid_policy
            from sraging.engine.evaluator import Evaluator
            from sraging.engine.inputs import Heterogeneity
            first = next(p for p in self.parameters if self.sd[p] > 0)
            coh = Cohort.from_arrays([1.0], [1], continuous=True, time_unit=self.p0.time_unit)
            self._ev = Evaluator(coh, grid=model_grid_policy(self.p0), fixed={"xc": self.p0.xc},
                                 heterogeneity=Heterogeneity(parameter=first, sd=self.sd[first], M=MIX.CAP,
                                                             check=False))
        ev = self._ev
        self.L = float(ev.grid.L)
        if self.tau is None:
            self.tau = np.asarray(ev._asm.tau, float)
        return MIX.programs(ev), ev._z(self.p0)

    def _rule(self, p, M):
        if self.sd[p] == 0:
            return np.ones(1), np.ones(1)
        r = Q.truncated_normal_rule(self.sd[p], M)
        return np.asarray(r.s, float), np.asarray(r.w, float)

    def _fold(self, Ms) -> tuple:
        """(S on the march grid, the fold order) at the node counts Ms."""
        progs, z = self._programs()
        return fold(progs, z, self.parameters, [self._rule(p, Ms[p]) for p in self.parameters])

    @staticmethod
    def _count(Ms) -> int:
        return int(math.prod(Ms.values()))

    def _finish(self, Ms):
        self.M, self.node_count = dict(Ms), self._count(Ms)
        if self.node_count > self.budget:
            self.kind = "none (budget)"
            return
        self.S, self.fold_order = self._fold(Ms)
        self.kind = "tensor"

    # -- R26 per dimension --------------------------------------------------------------------------------------
    def _sup(self, S0, S1) -> float:
        if self._ages is not None:
            x = self._ages / self.L
            return float(np.max(np.abs(np.interp(x, self.tau, S0) - np.interp(x, self.tau, S1)), initial=0.0))
        m = S0 > S_FLOOR
        return float(np.max(np.abs(S0[m] - S1[m]), initial=0.0))

    def _climb(self):
        Ms = {p: (1 if self.sd[p] == 0 else MIX.RUNGS[0]) for p in self.parameters}
        active = [p for p in self.parameters if self.sd[p] > 0]
        per = {p: [] for p in active}
        verdict = None
        while True:
            if self._count(Ms) > self.budget:
                verdict = "over budget"
                break
            S0, _ = self._fold(Ms)
            failing, over = [], False
            for p in active:
                nxt = _next_rung(Ms[p])
                if nxt is None:
                    continue
                trial = {**Ms, p: nxt}
                if self._count(trial) > self.budget:
                    over = True
                    break
                d = self._sup(S0, self._fold(trial)[0])
                per[p].append({"M": Ms[p], "M_next": nxt, "sup_dS": d, "passed": d <= self.bar})
                if d > self.bar:
                    failing.append(p)
            if over:
                verdict = "over budget"
                break
            if not failing:
                verdict = "converged"
                break
            if any(_next_rung(_next_rung(Ms[p])) is None for p in failing):
                verdict = "unconverged_at_cap"
                break
            for p in failing:
                Ms[p] = _next_rung(Ms[p])
        self.check = {"verdict": verdict, "bar": self.bar, "M": dict(Ms), "per_dimension": per,
                      "points": "ages" if self._ages is not None else f"the march's grid where S > {S_FLOOR:g}"}
        if verdict == "over budget":
            self.M, self.node_count, self.kind = dict(Ms), self._count(Ms), "none (budget)"
            return
        self._finish(Ms)

    # -- the reference's reads ----------------------------------------------------------------------------------
    def S_obs(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        return np.interp(t / self.L, self.tau, self.S, left=1.0, right=self.S[-1]) * np.exp(-self.h * t)

    def t_max(self, n, tail) -> tuple:
        """The age where S_obs first falls below tail / n on the grid (the grid's end if never), and S_obs there."""
        t = self.tau * self.L
        S = self.S_obs(t)
        below = np.flatnonzero(S < tail / n)
        i = int(below[0]) if below.size else t.size - 1
        return float(t[i]), float(S[i])

    def summary(self) -> dict:
        return {"kind": self.kind, "parameters": list(self.parameters), "sd": dict(self.sd), "M": dict(self.M),
                "node_count": self.node_count, "budget": self.budget, "check": self.check}
