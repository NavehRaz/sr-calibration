"""`simulate`: a seeded Langevin sample with its error statement (spec 004 FR-010 to FR-012; contracts/python-api.md
§5; data-model "LangevinConfig", "LangevinSample", "ErrorStatement"; research R13-R16, R20, R26, R30).

**The draw.** Individual i's parameters are P s_i, with s_i = 1 + sd z_i by gen-1's own inverse CDF of the truncated
normal (`generator._het_z`, so no draw is <= 0: B11) from its "het:<parameter>" stream; its external death time is
gen-1's `_hazard_times` with -ln(h_ext) from its "hazard" stream; its path runs in the numba kernel (kernels.py) to its
death or t_max. The raw times are the kernel's, after the competing hazard and before any recording.

**The observation transform** is gen-1's own functions (C8; research R20): `_censor_time` with the FP observed
survival, `_record` on delta and `_apply_time_range`, called through the module, so the test sees the same objects.

**The error statement at the user's n** (research R16; gate C1):
- **KS** of the raw times against the FP observed survival on a model grid (sraging.engine.curves' "model" policy),
  heterogeneity through the truncated normal at the node count R26's curve check chooses (sup over the sample's raw
  times of |S_M - S_next| <= crit_95 / 10), single-test crit_95 = 1.358 / sqrt(n);
- **the step bias** by the coupled half-step rung (the same streams at dt/2): D = sup|S_dt - S_dt/2| with a paired
  bootstrap SE, and the bias of the user's own sample b(dt) = D / (1 - 2^-p) with p = 1/2 for every configuration
  (the scheme's weak order at the wall; b̂ is an estimate, not a bound: T141; no test result is read: T142 C4); a b̂
  above crit_95 makes the sample not clean. `error="ks"` skips the rung: NOT MEASURED, never clean.

The default t_max is the age where the FP observed survival falls below 0.01 / n. A survivor there is censored at t_max
and counted, with LANGEVIN_SURVIVORS_AT_TMAX.

**Paths** (`record_ages`; research R24, R25): X[i, j], individual i's damage at the end of step round(a_j / dt), NaN iff
i is dead by a_j, read by the kernel with no draw (so `times` are the same with and without paths). The ages lie on the
dt grid; the (n, k) block, twice that with the coupled rung's, is capped by `max_path_bytes`. Per age the statement
tests the living's damages against `sraging.engine.density.damage_density`'s conditional CDF: under the clamp the
left-censored D_c above x_c (the FP quantile at max(0.01, 3 p0), at least 5 sqrt(2 eps_max dt)), under reflect the full
D; Bonferroni over the ages; the coupled rung's sup |F_dt - F_dt/2| on the same range beside it.
"""
from __future__ import annotations

import fractions
import importlib.util
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from sraging import generator as GEN
from sraging.engine.inputs import Heterogeneity, refuse_zero_spread
from sraging.errors import UnsupportedOptionError
from sraging.params import SRParams

KEY_MAX = 1 << 63
KERNELS = ("bridge", "euler")
WALL_NAMES = ("clamp", "reflect")
ERRORS = ("ladder", "ks")
RNG_NAME = "xoshiro256** per individual, SplitMix64-seeded"
CRIT_95 = 1.358                         # the one-sample KS critical value x sqrt(n)
PATHS_ALPHA = 0.05                      # the paths' family-wise level, Bonferroni over the ages
GRID_TOL = 1e-9                         # an age is on the dt grid iff |round(a/dt) dt - a| <= GRID_TOL max(a, 1)
QUANTILE_GRID = 20_001                  # the damage grid the FP quantile at u_c is read on
T_MAX_TAIL = 0.01                       # the default t_max: where S_obs falls below T_MAX_TAIL / n
BOOT = 100                              # the coupled rung's bootstrap resamples
P_HAT_DEFAULT = 0.5                     # the step bias's order, everywhere (FR-012; Naveh 2026-11-05, T142 C4)

#: SRK's names and options this engine does not take, each refused by name with its reason (data-model "Refused")
REFUSED = {
    "traj_points": "SRK's path recorder (register D-411); use record_ages",
    "save_trajectory": "SRK's path recorder (register D-411); use record_ages",
    "segments": "age segments and interventions are spec 007's",
    "interventions": "age segments and interventions are spec 007's (paths included)",
    "correlation": "correlated heterogeneity is not in spec 004: heterogeneous parameters are independent",
    "nsteps": "one step, dt (register B2: SRK's two step conventions)",
    "time_step_multiplier": "one step, dt (register B2: SRK's two step conventions)",
    "step_size": "one step, dt (register B2: SRK's two step conventions)",
    "parallel": "use workers (register B4: SRK's parallel path dropped n mod n_jobs)",
    "n_jobs": "use workers (register B4: SRK's parallel path dropped n mod n_jobs)",
    "seed": "the stream is set by key alone (per-individual keys, research R15)",
}


def check_key(key) -> int:
    if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or not 0 <= int(key) < KEY_MAX:
        raise UnsupportedOptionError("key", f"key is an integer in [0, 2^63), got {key!r}")
    return int(key)


def _numba_available() -> bool:
    return importlib.util.find_spec("numba") is not None


def _int(name, x, lo, hi=None):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < lo or (hi is not None and x > hi):
        raise UnsupportedOptionError(name, f"{name} is an integer >= {lo}{f' and <= {hi}' if hi else ''}, got {x!r}")
    return int(x)


def check_record_ages(ages, dt) -> tuple:
    """The ages, each on the dt grid (research R24), as a tuple; refused naming the ages and dt otherwise."""
    if isinstance(ages, (str, bytes)) or np.ndim(ages) > 1:
        raise UnsupportedOptionError("record_ages", f"record_ages is a 1-d array of ages, got {ages!r}")
    try:
        a = np.atleast_1d(np.asarray(ages, dtype=float))
    except (TypeError, ValueError):
        raise UnsupportedOptionError("record_ages", f"record_ages is a 1-d array of ages, got {ages!r}") from None
    if a.size < 1 or not np.all(np.isfinite(a)):
        raise UnsupportedOptionError("record_ages", f"record_ages is a non-empty array of finite ages, got {ages!r}")
    if np.any(a < 0):
        raise UnsupportedOptionError("record_ages", f"record ages are >= 0, got {a[a < 0].tolist()}")
    off = np.abs(np.rint(a / dt) * dt - a) > GRID_TOL * np.maximum(a, 1.0)
    if np.any(off):
        raise UnsupportedOptionError("record_ages", f"record ages {a[off].tolist()} are off the dt = {dt:g} grid "
                                     f"(|round(a/dt) dt - a| > {GRID_TOL:g} max(a, 1)); "
                                     f"sraging.langevin.grid_dt(ages, dt_max) gives a dt that divides every age")
    return tuple(float(x) for x in a)


def _pos(name, x, allow_none=False):
    if x is None and allow_none:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise UnsupportedOptionError(name, f"{name} must be a finite number > 0, got {x!r}") from None
    if not (math.isfinite(v) and v > 0):
        raise UnsupportedOptionError(name, f"{name} must be a finite number > 0, got {x!r}")
    return v


MAX_MEMBERS = 5                         # a heterogeneity tuple's members, each parameter at most once
TENSOR_D_MAX = 3                        # the tensor reference's members with a spread (research R28)


@dataclass(frozen=True)
class HetTuple:
    """data-model "Multi-parameter heterogeneity" (research R27): independent truncated normals, one per parameter,
    canonicalised by parameter name. A member at sd = 0 is s = 1.0 exactly, with no draw."""

    members: tuple

    @classmethod
    def of(cls, hets) -> "HetTuple":
        hets = tuple(hets.members if isinstance(hets, HetTuple) else hets)
        if not hets:
            raise UnsupportedOptionError("heterogeneity", "an empty heterogeneity tuple: pass None for none")
        for h in hets:
            if not isinstance(h, Heterogeneity):
                raise UnsupportedOptionError("heterogeneity", f"each member of a heterogeneity tuple is a "
                                             f"sraging.engine.Heterogeneity, got {h!r}")
            if h.definition != "truncated_normal":
                raise UnsupportedOptionError("heterogeneity", f"a normal_frac_sd member ({h.parameter}): the Langevin "
                                             f"draws the truncated normal only (register B11)")
        params = [h.parameter for h in hets]
        rep = sorted({q for q in params if params.count(q) > 1})
        if rep:
            raise UnsupportedOptionError("heterogeneity", f"a repeated parameter in the heterogeneity tuple: {rep}")
        if len(hets) > MAX_MEMBERS:
            raise UnsupportedOptionError("heterogeneity", f"at most {MAX_MEMBERS} members, got {len(hets)}")
        return cls(tuple(sorted(hets, key=lambda h: h.parameter)))

    @property
    def active(self) -> tuple:
        """The members with a spread (sd > 0)."""
        return tuple(h for h in self.members if h.sd > 0)

    def __iter__(self):
        return iter(self.members)

    def __len__(self):
        return len(self.members)


@dataclass(frozen=True)
class LangevinConfig:
    """data-model "LangevinConfig": everything a sample depends on. Refuses what the engine cannot do, by name."""

    params: SRParams
    n: int
    key: int
    dt: float
    kernel: str = "bridge"
    t_max: float | None = None
    heterogeneity: Heterogeneity | HetTuple | None = None
    record_ages: object = None
    max_path_bytes: int = 2 ** 31
    delta: float | None = None
    censor_frac: float = 0.0
    window: tuple | None = None
    error: str = "ladder"
    wall: str = "clamp"
    chunk: int = 1000
    workers: int = 1

    def __post_init__(self):
        set_ = lambda k, v: object.__setattr__(self, k, v)       # noqa: E731
        if not isinstance(self.params, SRParams):
            raise UnsupportedOptionError("params", "params is an SRParams")
        set_("n", _int("n", self.n, 1))
        set_("key", check_key(self.key))
        set_("dt", _pos("dt", self.dt))
        if self.kernel not in KERNELS:
            raise UnsupportedOptionError("kernel", f"kernel is one of {KERNELS}, got {self.kernel!r}")
        set_("t_max", _pos("t_max", self.t_max, allow_none=True))
        het = self.heterogeneity
        if isinstance(het, (tuple, list, HetTuple)):
            set_("heterogeneity", HetTuple.of(het))
        elif het is not None:
            if not isinstance(het, Heterogeneity):
                raise UnsupportedOptionError("heterogeneity", "pass a sraging.engine.Heterogeneity or a tuple of them")
            if het.definition != "truncated_normal":
                raise UnsupportedOptionError("heterogeneity", "the Langevin draws the shared definition, the truncated "
                                             "normal, only: 'normal_frac_sd' would draw non-positive parameters "
                                             "(register B11, SRK's unfiltered np.random.normal(1, var))")
            refuse_zero_spread(het, "the Langevin")
        if self.record_ages is not None:
            set_("record_ages", check_record_ages(self.record_ages, self.dt))
        set_("max_path_bytes", _int("max_path_bytes", self.max_path_bytes, 1))
        set_("delta", _pos("delta", self.delta, allow_none=True))
        cf = float(self.censor_frac)
        if not (math.isfinite(cf) and 0.0 <= cf < 1.0):
            raise UnsupportedOptionError("censor_frac", f"censor_frac is in [0, 1), got {self.censor_frac!r}")
        if cf > 0 and self.delta is None:
            raise UnsupportedOptionError("censor_frac", "censoring is placed on the recording grid: pass delta")
        set_("censor_frac", cf)
        if self.window is not None:
            lo, hi = (float(x) for x in self.window)
            if not (math.isfinite(lo) and lo >= 0 and hi > lo):
                raise UnsupportedOptionError("window", f"window is (lo, hi) with 0 <= lo < hi, got {self.window!r}")
            set_("window", (lo, hi))
        if self.error not in ERRORS:
            raise UnsupportedOptionError("error", f"error is one of {ERRORS}, got {self.error!r}")
        if self.wall not in WALL_NAMES:
            raise UnsupportedOptionError("wall", f"wall is 'clamp' (the default, SRK's max(0, .)) or 'reflect' (|.|), "
                                         f"got {self.wall!r}")
        set_("chunk", _int("chunk", self.chunk, 1))
        set_("workers", _int("workers", self.workers, 1, os.cpu_count() or 1))
        if self.record_ages is not None and self.path_bytes() > self.max_path_bytes:
            rung = ", x 2 for the coupled rung" if self.error == "ladder" else ""
            raise UnsupportedOptionError("max_path_bytes", f"the paths need {self.path_bytes():,} bytes (n "
                                         f"{self.n:,} x {len(self.record_ages)} ages x 8{rung}), above "
                                         f"max_path_bytes = {self.max_path_bytes:,}")

    def path_bytes(self) -> int:
        """n k 8 bytes, doubled while the coupled rung's paths exist (error="ladder")."""
        k = len(self.record_ages or ())
        return self.n * k * 8 * (2 if self.error == "ladder" else 1)


@dataclass(frozen=True, eq=False)
class ErrorStatement:
    fp_reference: dict
    ks_D: float
    n: int
    crit_95: float
    p_value: float
    verdict: str
    step_bias: dict
    paths: list | None = None                   # per record age (research R25), None without record_ages
    paths_reference: dict | None = None         # the damage density's grid, node count and curve check


@dataclass(frozen=True, eq=False)
class Paths:
    """X[i, j]: individual i's damage at the end of step `steps[j]` = round(ages[j] / dt), NaN iff dead by ages[j]."""
    ages: np.ndarray
    X: np.ndarray
    steps: np.ndarray


@dataclass(frozen=True, eq=False)
class LangevinSample:
    times: np.ndarray
    ages: np.ndarray
    events: np.ndarray
    n: int
    n_survived_t_max: int
    error: ErrorStatement
    cost_record: dict
    warnings: tuple
    provenance: dict
    clean: bool
    paths: Paths | None = None

    def save(self, path) -> dict:
        """An .npz (times, ages, events, record_ages, X) and a JSON sidecar (provenance, the npz's sha256, the
        warnings' codes, clean): {"npz": its path, "sidecar": its path}."""
        import hashlib
        base = os.fspath(path)
        base = base[:-4] if base.endswith(".npz") else base
        npz, side = base + ".npz", base + ".json"
        p = self.paths
        np.savez(npz, times=self.times, ages=self.ages, events=self.events,
                 record_ages=np.zeros(0) if p is None else p.ages,
                 X=np.zeros((self.n, 0)) if p is None else p.X)
        with open(npz, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        meta = {"npz": os.path.basename(npz), "npz_sha256": sha, "provenance": self.provenance,
                "warnings": [w.code for w in self.warnings], "clean": self.clean}
        with open(side, "w") as f:
            json.dump(meta, f, indent=1, default=_json_default)
        return {"npz": npz, "sidecar": side}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


# ----------------------------------------------------------------------------------------- the heterogeneity's reads
def het_members(cfg: LangevinConfig) -> tuple:
    """Every member (a single Heterogeneity is one), in canonical order."""
    het = cfg.heterogeneity
    return () if het is None else tuple(het.members) if isinstance(het, HetTuple) else (het,)


def het_active(cfg: LangevinConfig) -> tuple:
    return tuple(h for h in het_members(cfg) if h.sd > 0)


def reference_kind(cfg: LangevinConfig) -> str:
    """The FP reference a sample is judged against (tasks T108): `single` for at most one member with a spread, the
    `tensor` product for 2 to 3, `none` above (the tensor reference's budget can also turn `tensor` into `none`)."""
    d = len(het_active(cfg))
    return "single" if d <= 1 else "tensor" if d <= TENSOR_D_MAX else "none"


def single_het(cfg: LangevinConfig):
    """The one member with a spread as a Heterogeneity, or None (homogeneous): the single reference's."""
    a = het_active(cfg)
    if len(a) > 1:
        raise ValueError("more than one member with a spread: not a single reference")
    return a[0] if a else None


# ----------------------------------------------------------------------------------------------------- the draw
def draw_parameters(cfg: LangevinConfig) -> dict:
    """Per-individual arrays eta, beta, epsilon, kappa, xc and t_ext (inf without a hazard) from their own streams:
    member p's s = 1 + sd z from the stream "het:" + p (gen-1's inverse CDF); a member at sd 0 draws nothing."""
    from sraging.langevin.rng import stream_uniforms
    p, n = cfg.params, cfg.n
    idx = np.arange(n)
    out = {k: np.full(n, float(getattr(p, k))) for k in ("eta", "beta", "epsilon", "kappa", "xc")}
    for het in het_active(cfg):
        z = GEN._het_z(stream_uniforms(cfg.key, idx, "het:" + het.parameter), het.sd)
        out[het.parameter] = out[het.parameter] * (1.0 + het.sd * np.asarray(z, float))
    if p.h_ext > 0:
        out["t_ext"] = np.asarray(GEN._hazard_times(stream_uniforms(cfg.key, idx, "hazard"), -math.log(p.h_ext)), float)
    else:
        out["t_ext"] = np.full(n, math.inf)
    return out


def draw_parameters_shared_stream(cfg: LangevinConfig) -> dict:
    """HR-11's plant (research R27): one shared parameter stream, SRK's way (`SR_hetro.py:467-471` draws every
    parameter in a fixed order): member j of the canonical tuple, zero spreads included, takes the j-th draw of the
    shared sequence, so adding a zero-sd member shifts the draws of the members after it."""
    from sraging.langevin.rng import stream_uniforms
    out = draw_parameters(LangevinConfig(**{**{f: getattr(cfg, f) for f in LangevinConfig.__dataclass_fields__},
                                            "heterogeneity": None}))
    idx = np.arange(cfg.n)
    for j, het in enumerate(het_members(cfg)):
        u = stream_uniforms(cfg.key, idx, f"het#{j}")
        if het.sd > 0:
            out[het.parameter] = out[het.parameter] * (1.0 + het.sd * np.asarray(GEN._het_z(u, het.sd), float))
    return out


def record_steps(cfg: LangevinConfig) -> tuple:
    """(round(a/dt) per record age in the user's order, the order that sorts them) as int64 arrays; empty without."""
    if cfg.record_ages is None:
        e = np.zeros(0, dtype=np.int64)
        return e, e
    steps = np.rint(np.asarray(cfg.record_ages) / cfg.dt).astype(np.int64)
    return steps, np.argsort(steps, kind="stable")


def rung_job(cfg: LangevinConfig, arrays: dict, t_max: float, split: int, seeds=None) -> dict:
    """One rung's job for the pool: the streams, the draws, the step, the wall and the record steps in steps of
    h = dt / split, ascending."""
    from sraging.langevin import kernels as K
    from sraging.langevin.rng import hash64_array
    if seeds is None:
        idx = np.arange(cfg.n)
        seeds = (hash64_array(cfg.key, idx), hash64_array(cfg.key, idx, "bridge"))
    steps, order = record_steps(cfg)
    return {"seeds": seeds, "arrays": arrays, "dt": cfg.dt, "t_max": t_max, "wall": K.WALLS[cfg.wall],
            "bridge": cfg.kernel == "bridge", "split": split,
            "rec_steps": np.ascontiguousarray(steps[order] * split, dtype=np.int64)}


def run_rung(cfg: LangevinConfig, arrays: dict, t_max: float, split: int) -> tuple:
    """(times, kinds, x_end, X) of the dt rung (split 1) or the coupled dt/2 rung (split 2), over chunks, in process
    or across cfg.workers processes (sraging.langevin.pool); bit-identical either way (per-individual streams). X's
    columns are the record ages sorted by step (`unsort_paths` restores the user's order)."""
    from sraging.langevin.pool import run_chunks
    return run_chunks(rung_job(cfg, arrays, t_max, split), cfg.n, cfg.chunk, cfg.workers)


def unsort_paths(cfg: LangevinConfig, X_sorted) -> np.ndarray:
    """The kernel's columns (sorted by step) back in the user's order of record ages."""
    _, order = record_steps(cfg)
    X = np.empty_like(X_sorted)
    X[:, order] = X_sorted
    return X


# ---------------------------------------------------------------------------------------------- the FP reference
class Reference:
    """The FP observed survival S_obs(t) = S(t) e^{-h t} on a model grid; with heterogeneity, the truncated-normal
    mixture of the per-node survivals (the engine's per-node programs), its node count by R26's curve check."""

    def __init__(self, cfg: LangevinConfig):
        from sraging.cohort import Cohort
        from sraging.engine.curves import model_grid_policy
        from sraging.engine.evaluator import Evaluator
        self.cfg = cfg
        p = cfg.params
        self.h = float(p.h_ext)
        self.p0 = SRParams(**{**p.as_dict(), "h_ext": 0.0})
        policy = model_grid_policy(self.p0)
        coh = Cohort.from_arrays([1.0], [1], continuous=True, time_unit=p.time_unit)
        a = het_active(cfg)
        self.het = single_het(cfg) if len(a) <= 1 else None            # the tensor kind shares only the evaluator
        het = a[0] if a else None
        self.ev = Evaluator(coh, grid=policy, fixed={"xc": p.xc},
                            heterogeneity=None if het is None else Heterogeneity(parameter=het.parameter, sd=het.sd,
                                                                                 M=192, check=False))
        self.L = float(self.ev.grid.L)
        self._S = {}

    @classmethod
    def at(cls, base: "Reference", params: SRParams) -> "Reference":
        """The reference at other parameters on `base`'s grid and compiled evaluator (a plant's alternative, a
        bisection): only the survivals are recomputed."""
        r = cls.__new__(cls)
        r.cfg, r.ev, r.L, r._S, r.het = base.cfg, base.ev, base.L, {}, base.het
        r.h = float(params.h_ext)
        r.p0 = SRParams(**{**params.as_dict(), "h_ext": 0.0})
        return r

    def survival_grid(self, M=None) -> tuple:
        """(tau, S on the march grid) at M nodes (None: homogeneous)."""
        if M in self._S:
            return self._S[M]
        if self.het is None:
            tau, S, _, _ = self.ev.survival_on_grid(self.p0)
        else:
            from sraging.engine import mixture as MIX
            from sraging.engine import quadrature as Q
            het = self.het
            rule = Q.truncated_normal_rule(het.sd, M)
            progs = MIX.programs(self.ev)
            z = self.ev._z(self.p0)
            S = None
            for k in range(rule.M):
                Sk = progs.survival(z, MIX._scales(het.parameter, float(rule.s[k])))
                S = rule.w[k] * Sk if S is None else S + rule.w[k] * Sk
            tau = np.asarray(self.ev._asm.tau, float)
        self._S[M] = (np.asarray(tau, float), np.asarray(S, float))
        return self._S[M]

    def S_obs(self, t, M=None) -> np.ndarray:
        tau, S = self.survival_grid(M)
        t = np.asarray(t, float)
        return np.interp(t / self.L, tau, S, left=1.0, right=S[-1]) * np.exp(-self.h * t)

    def curve_check(self, ages, bar) -> dict:
        """R26: the smallest rung M with sup over `ages` of |S_M - S_next| <= bar; unconverged at the cap."""
        from sraging.engine import mixture as MIX
        if self.het is None:
            return {"M": None, "verdict": "homogeneous", "bar": bar, "sup_dS": 0.0, "ladder": []}
        ladder = []
        for j in range(len(MIX.RUNGS) - 1):
            lo, hi = MIX.RUNGS[j], MIX.RUNGS[j + 1]
            d = float(np.max(np.abs(self.S_obs(ages, lo) - self.S_obs(ages, hi)))) if len(ages) else 0.0
            ladder.append({"M": lo, "M_next": hi, "sup_dS": d})
            if d <= bar:
                return {"M": lo, "verdict": "converged", "bar": bar, "sup_dS": d, "ladder": ladder}
        return {"M": MIX.CAP, "verdict": "unconverged_at_cap", "bar": bar, "sup_dS": ladder[-1]["sup_dS"],
                "ladder": ladder}

    def t_max(self, n, M) -> tuple:
        """The age where S_obs first falls below T_MAX_TAIL / n on the grid (the grid's end if never), and S_obs there."""
        tau, _ = self.survival_grid(M)
        t = tau * self.L
        S = self.S_obs(t, M)
        below = np.flatnonzero(S < T_MAX_TAIL / n)
        i = int(below[0]) if below.size else t.size - 1
        return float(t[i]), float(S[i])


def ks_one_sample(times, kinds, S_obs) -> tuple:
    """(D, p): the raw death times against F = 1 - S_obs, survivors at t_max beyond every death."""
    from scipy.stats import kstwo
    n = times.size
    d = np.sort(times[kinds != 0])
    F = 1.0 - S_obs(d)
    i = np.arange(1, d.size + 1)
    D = float(max(np.max(i / n - F, initial=0.0), np.max(F - (i - 1) / n, initial=0.0)))
    return D, float(kstwo.sf(D, n))


def ecdf_sup(a, b) -> float:
    """sup_t |F_a(t) - F_b(t)| of two samples of equal size (the coupled rung's D)."""
    grid = np.union1d(a, b)
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def p_hat(cfg: LangevinConfig) -> tuple:
    """(p_hat, source): 1/2, the scheme's weak order at the wall, for every configuration (FR-012; Naveh 2026-11-05,
    T142 C4: the library reads no test result, so a checkout and an installed copy state the same bias). `cfg` is
    kept so the call sites stay one signature."""
    return P_HAT_DEFAULT, ("1/2, the scheme's weak order at the wall, for every configuration (FR-012); b-hat is an "
                           "estimate from one halving, not a bound")


def config_key(cfg: LangevinConfig) -> dict:
    """What identifies a P1 cell in P1's own result (pilot P1's tool records it): the parameters, the
    heterogeneity, the kernel, the wall and dt. The library does not read it (T142 C4)."""
    het = cfg.heterogeneity
    hk = (None if het is None else [{"parameter": h.parameter, "sd": h.sd} for h in het.members]
          if isinstance(het, HetTuple) else {"parameter": het.parameter, "sd": het.sd})
    return {"params": {k: v for k, v in cfg.params.as_dict().items() if k != "time_unit"},
            "heterogeneity": hk,
            "kernel": cfg.kernel, "wall": cfg.wall, "dt": cfg.dt}


def step_bias(cfg, times, times2) -> dict:
    """The coupled rung's D = sup|S_dt - S_dt/2|, its paired bootstrap SE, and b(dt) = D / (1 - 2^-p) (gate C1)."""
    D = ecdf_sup(times, times2)
    rng = np.random.default_rng(cfg.key)                          # deterministic: a function of the key alone
    boot = [ecdf_sup(times[ix], times2[ix]) for ix in (rng.integers(0, times.size, times.size) for _ in range(BOOT))]
    p, src = p_hat(cfg)
    return {"method": "coupled half-step rung", "D_dt_vs_dt2": D, "se": float(np.std(boot, ddof=1)), "p_hat": p,
            "p_hat_source": src, "bias_at_dt": D / (1.0 - 2.0 ** -p), "n": int(times.size), "bootstrap": BOOT}


# ---------------------------------------------------------------------------------------------------- simulate
def simulate(params, **kw) -> LangevinSample:
    """contracts/python-api.md §5. Every option is named; SRK's options are refused by name (REFUSED)."""
    for name in list(kw):
        if name in REFUSED:
            raise UnsupportedOptionError(name, f"{name} is refused: {REFUSED[name]}")
    known = set(LangevinConfig.__dataclass_fields__) - {"params"}
    unknown = sorted(set(kw) - known)
    if unknown:
        raise UnsupportedOptionError(unknown[0], f"unknown option {unknown[0]!r}; simulate takes {sorted(known)}")
    return run(LangevinConfig(params=params, **kw))


def require_numba():
    if not _numba_available():
        raise UnsupportedOptionError("numba", "the Langevin engine needs numba: install sraging[langevin]")


def run(cfg: LangevinConfig) -> LangevinSample:
    """One sample: prepare (the reference, t_max, the parameter draws), the dt rung and the coupled rung over cfg's
    workers, then the statement and the transform."""
    require_numba()
    st = prepare(cfg)
    c1 = time.process_time()
    rung1 = run_rung(cfg, st["arrays"], st["t_max"], 1)
    st["kernel_cpu_s"] = time.process_time() - c1
    rung2 = None
    if cfg.error == "ladder":
        c2 = time.process_time()
        rung2 = run_rung(cfg, st["arrays"], st["t_max"], 2)
        st["rung_cpu_s"] = time.process_time() - c2
    return finish(cfg, st, rung1, rung2)


def prepare(cfg: LangevinConfig) -> dict:
    """The parent's part before the kernels: the FP reference, the default t_max and the parameter draws."""
    c0, w0 = time.process_time(), time.perf_counter()
    crit = CRIT_95 / math.sqrt(cfg.n)
    kind, tensor, why = reference_kind(cfg), None, None
    ref = Reference(cfg) if kind != "none" else None
    if kind == "none":
        why = (f"{len(het_active(cfg))} heterogeneous parameters with a spread: the tensor reference takes d <= "
               f"{TENSOR_D_MAX} (research R28)")
    if kind == "tensor":
        from sraging.engine import tensor_reference as TR
        # R26 per dimension at crit_95/10, on the march's grid: the difference of two interpolants on one grid
        # peaks at a node, so this bounds the sup over any sample's raw times
        tensor = TR.TensorReference(cfg.params, het_members(cfg), bar=crit / 10, ev=ref.ev, budget=TR.BUDGET)
        if tensor.kind != "tensor":
            kind, why = "none", (f"the tensor product's node count {tensor.node_count:,} exceeds the budget "
                                 f"{tensor.budget:,} (research R28)")
    if kind == "none":
        if cfg.t_max is None:
            raise UnsupportedOptionError("t_max", f"no FP reference for this heterogeneity ({why}): the default t_max "
                                         f"reads the FP survival; pass t_max")
        if cfg.censor_frac > 0:
            raise UnsupportedOptionError("censor_frac", f"no FP reference for this heterogeneity ({why}): censoring "
                                         f"is placed with the FP survival; pass censor_frac=0")
        t_max, S_t_max = cfg.t_max, None
    elif kind == "tensor":
        t_max, S_t_max = (cfg.t_max, None) if cfg.t_max is not None else tensor.t_max(cfg.n, T_MAX_TAIL)
    else:
        # the default t_max, from the reference at the node count its check chooses on the grid's own ages
        M0 = None
        if ref.het is not None:
            tau, _ = ref.survival_grid(5)
            t_grid = tau * ref.L
            M0 = ref.curve_check(t_grid[ref.S_obs(t_grid, 5) > 1e-12], crit / 10)["M"]
        t_max, S_t_max = (cfg.t_max, None) if cfg.t_max is not None else ref.t_max(cfg.n, M0)
    if cfg.record_ages is not None:
        steps, _ = record_steps(cfg)
        past = [a for a, k in zip(cfg.record_ages, steps) if not (a < t_max and k * cfg.dt <= t_max)]
        if past:
            raise UnsupportedOptionError("record_ages", f"record ages {past} are not before t_max = {t_max:g}: the "
                                         f"paths end there (survivors are censored at t_max)")
    return {"c0": c0, "w0": w0, "ref": ref, "crit": crit, "t_max": t_max, "S_t_max": S_t_max,
            "arrays": draw_parameters(cfg), "kernel_cpu_s": 0.0, "rung_cpu_s": 0.0, "ref_kind": kind,
            "tensor": tensor, "no_reference": why}


def finish(cfg: LangevinConfig, st: dict, rung1, rung2) -> LangevinSample:
    """The statement, the transform, the warnings and the records, from the rungs' raw times."""
    from sraging.warnings import sampler_warning
    ref, crit, t_max, S_t_max, c0, w0 = st["ref"], st["crit"], st["t_max"], st["S_t_max"], st["c0"], st["w0"]
    times, kinds = rung1[0], rung1[1]
    deaths = times[kinds != 0]
    kind = st.get("ref_kind", "single")
    members = [{"parameter": h.parameter, "sd": h.sd} for h in het_members(cfg)]
    definition = "truncated_normal" if members else None
    if kind == "single":
        check = ref.curve_check(deaths, crit / 10)
        M = check["M"]
        S_obs = lambda t: ref.S_obs(t, M)                                       # noqa: E731
        fp_ref = {"kind": "single", "M": M, "curve_check": check, "node_count": M}
    elif kind == "tensor":
        tr = st["tensor"]
        check, S_obs = tr.check, tr.S_obs
        fp_ref = {"kind": "tensor", "M": dict(tr.M), "curve_check": check, "node_count": tr.node_count,
                  "budget": tr.budget, "parameters": list(tr.parameters)}
    else:
        check, S_obs = {"verdict": "none"}, None
        fp_ref = {"kind": "none", "reason": st["no_reference"], "M": None, "curve_check": None, "node_count": None}
    if ref is not None:
        fp_ref.update({"grid": ref.ev.grid.summary(), "numerics": ref.ev.numerics})
    fp_ref.update({"definition": definition, "members": members})
    D, pv = ks_one_sample(times, kinds, S_obs) if S_obs is not None else (None, None)
    if S_t_max is None and S_obs is not None:
        S_t_max = float(S_obs(np.array([t_max]))[0])
    fp_ref["S_obs_at_t_max"] = S_t_max
    if rung2 is not None:
        bias = step_bias(cfg, times, rung2[0])
        # the judged sample (gate C1): the dt/2 rung's own KS against the same reference (S4, HR-10, HR-11 read it)
        bias["ks_D_half"], bias["p_value_half"] = (ks_one_sample(rung2[0], rung2[1], S_obs) if S_obs is not None
                                                   else (None, None))
    else:
        bias = {"method": "NOT MEASURED", "D_dt_vs_dt2": None, "se": None, "p_hat": None, "p_hat_source": None,
                "bias_at_dt": None, "n": cfg.n, "ks_D_half": None, "p_value_half": None}
    verdict = "no FP reference" if D is None else "consistent at n" if D <= crit else "inconsistent at n"
    paths, rows, pref, pwarns = None, None, None, []
    if cfg.record_ages is not None:
        steps, _ = record_steps(cfg)
        X = unsort_paths(cfg, rung1[3])
        X2 = unsort_paths(cfg, rung2[3]) if rung2 is not None else None
        paths = Paths(ages=np.asarray(cfg.record_ages, float), X=X, steps=steps)
        rows, pref, pwarns = paths_statement(cfg, st["arrays"], X, X2, kind)
    error = ErrorStatement(fp_reference=fp_ref, ks_D=D, n=cfg.n, crit_95=crit, p_value=pv, verdict=verdict,
                           step_bias=bias, paths=rows, paths_reference=pref)
    ages, events = observe(cfg, times, kinds, t_max, S_obs)
    n_surv = int(np.sum(kinds == 0))
    warns = []
    if n_surv:
        warns.append(sampler_warning("LANGEVIN_SURVIVORS_AT_TMAX", n_survived=n_surv, n=cfg.n, t_max=t_max))
    if verdict == "inconsistent at n":
        warns.append(sampler_warning("LANGEVIN_INCONSISTENT_AT_N", n=cfg.n, ks_D=D, crit_95=crit, p_value=pv))
    if cfg.error == "ks":
        warns.append(sampler_warning("LANGEVIN_STEP_BIAS_NOT_MEASURED", dt=cfg.dt, n=cfg.n))
    elif bias["bias_at_dt"] is not None and bias["bias_at_dt"] > crit:
        # FR-012 (Naveh 2026-11-05, T141): the stated bias of the times, beside a KS that at small n cannot see it
        warns.append(sampler_warning("LANGEVIN_STEP_BIAS_ABOVE_CRIT", bias_at_dt=bias["bias_at_dt"], crit_95=crit,
                                     n=cfg.n, dt=cfg.dt))
    if kind == "none":
        warns.append(sampler_warning("LANGEVIN_NO_FP_REFERENCE", reason=st["no_reference"]))
    elif check["verdict"] == "unconverged_at_cap":
        what = ("sup|S_128 - S_192|" if kind == "single" else "a dimension's sup|S_M - S_next| at the cap's last pair")
        warns.append(sampler_warning("LANGEVIN_NO_FP_REFERENCE", reason=f"the reference's node count is unconverged at "
                                     f"the cap ({what} > crit_95/10 = {check['bar']:.3g})"))
    warns.extend(pwarns)
    cost = {"cpu_s": time.process_time() - c0, "wall_s": time.perf_counter() - w0, "kernel_cpu_s": st["kernel_cpu_s"],
            "rung_cpu_s": st["rung_cpu_s"], "individual_steps": int(np.sum(np.ceil(times / cfg.dt))), "workers": cfg.workers,
            "path_bytes": cfg.path_bytes(),
            "note": "the kernels' CPU in worker processes is not in cpu_s (process time of this process)"}
    return LangevinSample(times=times, ages=ages, events=events, n=cfg.n, n_survived_t_max=n_surv, error=error,
                          cost_record=cost, warnings=tuple(warns), provenance=provenance(cfg, t_max),
                          clean=not warns, paths=paths)


# ------------------------------------------------------------------------------------------ the paths statement
def fp_quantile(F, u, x_max) -> float | None:
    """The FP conditional CDF's u-quantile, read on QUANTILE_GRID points of [0, x_max]; None for u >= 1."""
    if u >= 1.0:
        return None
    xg = np.linspace(0.0, x_max, QUANTILE_GRID)
    Fg = F(xg)
    j = int(np.searchsorted(Fg, u, side="left"))
    if j == 0:
        return 0.0
    if j >= xg.size:
        return float(x_max)
    f0, f1 = Fg[j - 1], Fg[j]
    return float(xg[j - 1] + (xg[j] - xg[j - 1]) * ((u - f0) / (f1 - f0) if f1 > f0 else 1.0))


def ks_paths(x, F, x_c) -> tuple:
    """(D over all x, D_c over x >= x_c) of the living's damages x against F; x_c None reads D_c = D."""
    n = x.size
    xs = np.sort(x)
    Fx = F(xs)
    i = np.arange(1, n + 1)
    up, dn = i / n - Fx, Fx - (i - 1) / n
    D = float(max(up.max(), dn.max(), 0.0))
    if x_c is None:
        return D, D
    above = xs > x_c
    at = abs(np.searchsorted(xs, x_c, side="right") / n - float(F(np.array([x_c]))[0]))
    Dc = float(max(at, up[above].max(initial=0.0), dn[above].max(initial=0.0)))
    return D, Dc


def ecdf_sup_above(a, b, x_c) -> float:
    """sup over x >= x_c (all x for None) of |F_a(x) - F_b(x)|, two samples of any sizes."""
    if a.size == 0 or b.size == 0:
        return math.nan
    grid = np.union1d(a, b)
    if x_c is not None:
        grid = np.union1d(grid[grid >= x_c], [x_c])
    fa = np.searchsorted(np.sort(a), grid, side="right") / a.size
    fb = np.searchsorted(np.sort(b), grid, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def path_test(x, F, *, wall, step, eps_max, k, x_max) -> dict:
    """One age's test of the living's damages (one rung's): p0, the censoring point, D and D_c, the Bonferroni crit,
    the judged statistic's p-value and the verdict (clamp reads D_c, reflect D)."""
    from scipy.stats import kstwo, kstwobign
    n_alive = int(x.size)
    if n_alive == 0:
        return {"n_alive": 0, "p0": None, "u_c": None, "x_c": None, "ks_D_censored": None, "ks_D_full": None,
                "crit": None, "p_value": None, "verdict": "not assessable", "reason": "no one is alive at this age"}
    from sraging.sampler_rules import hr10_censoring
    p0 = float(np.mean(x == 0.0))
    u_c, x_c = hr10_censoring(p0, eps_max=eps_max, dt=step, quantile=lambda u: fp_quantile(F, u, x_max))
    D, Dc = ks_paths(x, F, x_c)
    crit = float(kstwobign.isf(PATHS_ALPHA / k)) / math.sqrt(n_alive)
    out = {"n_alive": n_alive, "p0": p0, "u_c": u_c, "x_c": x_c, "ks_D_censored": Dc if x_c is not None else None,
           "ks_D_full": D, "crit": crit}
    if wall == "clamp" and x_c is None:
        return {**out, "p_value": None, "verdict": "not assessable",
                "reason": f"u_c = max(0.01, 3 p0) = {u_c:.3g} >= 1: the clamp's lump leaves no law above x_c"}
    judged = Dc if wall == "clamp" else D
    return {**out, "statistic": "censored" if wall == "clamp" else "full", "p_value": float(kstwo.sf(judged, n_alive)),
            "verdict": "consistent at n" if judged <= crit else "inconsistent at n"}


def paths_statement(cfg: LangevinConfig, arrays: dict, X, X2, kind="single") -> tuple:
    """(rows per record age, the reference, warnings): each age's living damages against damage_density's
    conditional CDF among the living (research R25), with the coupled rung's step bias on the judged range and the
    dt/2 rung's own test (the sample the acceptance phases judge)."""
    from sraging.engine.density import damage_density
    from sraging.warnings import sampler_warning
    ages = np.asarray(cfg.record_ages, float)
    k = ages.size
    eps_max = float(np.max(arrays["epsilon"]))
    x_max = float(np.max(arrays["xc"]))
    want = sorted({float(a) for j, a in enumerate(ages) if a > 0 and np.isfinite(X[:, j]).any()})
    dd, pref, warns = None, {"kind": "none", "reason": "no living individual at a positive record age"}, []
    if want and kind != "single":
        pref = {"kind": "none", "reason": "the damage density of several heterogeneous parameters (R25 with R28's "
                                          "tensor product) is not built: the paths are not checked"}
        warns.append(sampler_warning("LANGEVIN_NO_FP_REFERENCE", reason=pref["reason"]))
        want = []
    het = single_het(cfg) if kind == "single" else None
    if want:
        try:
            dd = damage_density(cfg.params, want, heterogeneity=None if het is None else
                                Heterogeneity(parameter=het.parameter, sd=het.sd))
            pref = {"kind": "damage_density", "grid": dd.grid, "M": dd.M, "curve_check": dd.curve_check,
                    "grid_error_grid": dd.grid_error_grid}
            x_max = max(x_max, float(np.max(dd.nodes["xc"])))
        except UnsupportedOptionError as e:
            pref = {"kind": "none", "reason": str(e)}
            warns.append(sampler_warning("LANGEVIN_NO_FP_REFERENCE", reason=f"no damage density for the paths: {e}"))
        if dd is not None and dd.curve_check.get("verdict") == "unconverged_at_cap":
            warns.append(sampler_warning("LANGEVIN_NO_FP_REFERENCE", reason="the damage density's node count is "
                                         "unconverged at the cap"))
    p, p_src = p_hat(cfg)
    rows = []
    for j, a in enumerate(ages):
        x = X[:, j][np.isfinite(X[:, j])]
        x2 = None if X2 is None else X2[:, j][np.isfinite(X2[:, j])]
        row = {"age": float(a), "wall": cfg.wall}
        if a == 0:                                                 # the start: every path is at 0, a point mass
            D = 0.0 if np.all(x == 0.0) else 1.0
            crit = float(CRIT_95 / math.sqrt(max(x.size, 1)))
            row.update({"n_alive": int(x.size), "p0": float(np.mean(x == 0.0)) if x.size else None, "u_c": None,
                        "x_c": None, "ks_D_censored": D, "ks_D_full": D, "crit": crit, "p_value": 1.0 - D,
                        "verdict": "consistent at n" if D == 0 else "inconsistent at n",
                        "reason": "age 0: every path starts at 0 (the law is the point mass there)",
                        "step_bias": None if x2 is None else {"method": "coupled half-step rung", "D_dt_vs_dt2": 0.0},
                        "reference_grid_error": 0.0})
        elif dd is None:
            row.update({"n_alive": int(x.size), "p0": float(np.mean(x == 0.0)) if x.size else None, "u_c": None,
                        "x_c": None, "ks_D_censored": None, "ks_D_full": None, "crit": None, "p_value": None,
                        "verdict": "not assessable", "reason": pref["reason"], "step_bias": None,
                        "reference_grid_error": None})
        else:
            F = (lambda z, a=a: dd.cdf_alive(z, a))
            row.update(path_test(x, F, wall=cfg.wall, step=cfg.dt, eps_max=eps_max, k=k, x_max=x_max))
            row["reference_grid_error"] = float(dd.grid_error[want.index(float(a))])
            row["step_bias"] = None
            if x2 is not None:
                xr = None if cfg.wall == "reflect" else row["x_c"]
                d = ecdf_sup_above(x, x2, xr) if (cfg.wall == "reflect" or xr is not None) else math.nan
                row["step_bias"] = {"method": "coupled half-step rung", "D_dt_vs_dt2": d,
                                    "range": "all x" if cfg.wall == "reflect" else "x >= x_c", "p_hat": p,
                                    "p_hat_source": p_src, "bias_at_dt": d / (1.0 - 2.0 ** -p),
                                    "bias_at_dt2": d / (2.0 ** p - 1.0),
                                    "half": path_test(x2, F, wall=cfg.wall, step=cfg.dt / 2, eps_max=eps_max, k=k,
                                                      x_max=x_max)}
            if row["crit"] is not None and row["reference_grid_error"] > row["crit"] / 10:
                row["reference"] = "grid unconverged"
                warns.append(sampler_warning("LANGEVIN_NO_FP_REFERENCE", reason=f"the damage density's grid error at "
                                             f"age {a:g} ({row['reference_grid_error']:.3g}) is above crit/10 = "
                                             f"{row['crit'] / 10:.3g}: grid unconverged"))
            if cfg.wall == "clamp" and row["p0"] is not None and row["p0"] > row["crit"] / 10:
                warns.append(sampler_warning("LANGEVIN_PATHS_WALL_ATOM", p0=row["p0"], age=float(a),
                                             limit=row["crit"] / 10, x_c=row["x_c"] if row["x_c"] is not None
                                             else float("nan")))
        if row["verdict"] == "inconsistent at n":
            warns.append(sampler_warning("LANGEVIN_PATHS_INCONSISTENT_AT_N", age=float(a), n_alive=row["n_alive"],
                                         wall=cfg.wall, statistic=row.get("statistic", "point mass"),
                                         ks_D=row["ks_D_censored"] if cfg.wall == "clamp" else row["ks_D_full"],
                                         crit=row["crit"]))
        rows.append(row)
    return rows, pref, warns


def observe(cfg, times, kinds, t_max, S_obs) -> tuple:
    """gen-1's observation transform on the raw times: censoring at C (_censor_time with the FP S_obs), recording on
    delta (_record), sorted as gen-1 sorts, then the window (_apply_time_range). Without delta the ages are the raw
    times. A survivor at t_max stays censored there."""
    alive = kinds == 0
    if cfg.delta is not None:
        C = None
        if cfg.censor_frac > 0:
            C = GEN._censor_time(S_obs, cfg.delta, cfg.censor_frac, t_max)
        if C is not None and C <= t_max:                          # every survivor at t_max is past C: censored at C
            ages, events = GEN._record(times, C, cfg.delta)
        else:
            ages, events = GEN._record(times[~alive], C, cfg.delta)
            ages = np.concatenate([ages, np.full(int(alive.sum()), t_max)])
            events = np.concatenate([events, np.zeros(int(alive.sum()), int)])
    else:
        ages, events = times.copy(), np.where(alive, 0, 1).astype(int)
    order = np.lexsort((events, ages))
    ages, events = ages[order], events[order]
    if cfg.window is not None:
        ages, events, _ = GEN._apply_time_range(ages, events, cfg.window)
    return np.ascontiguousarray(ages, float), np.ascontiguousarray(events, int)


def provenance(cfg, t_max) -> dict:
    import numba
    from sraging import provenance as PV
    members = het_members(cfg)
    return {"engine": "langevin", "kernel": cfg.kernel, "wall": cfg.wall, "dt": cfg.dt, "t_max": t_max,
            "t_max_source": "given" if cfg.t_max is not None else f"S_obs < {T_MAX_TAIL:g} / n",
            "key": cfg.key, "rng": RNG_NAME, "definition": members[0].definition if members else None,
            "members": [{"parameter": h.parameter, "sd": h.sd} for h in members], "n": cfg.n,
            "record_ages": None if cfg.record_ages is None else list(cfg.record_ages),
            "chunk": cfg.chunk, "workers": cfg.workers, "error": cfg.error,
            "params": cfg.params.as_dict(), "numba": numba.__version__,
            "package": {"version": PV._version("sraging"), **PV._git()},
            "libraries": {"python": sys.version.split()[0], **{d: PV._version(d) for d in ("numpy", "scipy", "jax")}},
            "host": {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node()}}


# ------------------------------------------------------------------------------------------------------ grid_dt
def grid_dt(ages, dt_max) -> float:
    """The largest dt <= dt_max that divides every age (research R24), through their rational common divisor; refused
    when the ages have none within 1e-9."""
    dt_max = _pos("dt_max", dt_max)
    fr = []
    for a in ages:
        a = float(a)
        f = fractions.Fraction(a).limit_denominator(10 ** 9)
        if a < 0 or abs(float(f) - a) > 1e-9 * max(a, 1.0):
            raise UnsupportedOptionError("record_ages", f"age {a!r} has no rational form within 1e-9: no common dt")
        fr.append(f)
    g = fractions.Fraction(0)
    for f in fr:
        g = fractions.Fraction(math.gcd(g.numerator * f.denominator, f.numerator * g.denominator),
                               g.denominator * f.denominator)
    if g == 0:
        return dt_max
    dt = g / math.ceil(g / fractions.Fraction(dt_max))
    if float(g) < 1e-6 * max(float(max(fr)), 1.0):
        raise UnsupportedOptionError("record_ages", f"the ages {list(ages)} share no step above 1e-6 of their size "
                                     f"(common divisor {float(g):.3g}): refused")
    return float(dt)
