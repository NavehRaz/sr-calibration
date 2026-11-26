"""One switch for SRK's behavioural defaults, so a project opts into the new engine in one line.

    from SRK import config
    config.use_profile('modern')        # FP curves + automatic Langevin resolution

WHY A PROFILE RATHER THAN NEW DEFAULTS
--------------------------------------
The FP solver is exact and ~20-100x faster than the Langevin cohort, so it should be what produces
a survival/hazard curve. But flipping that default changes numbers in every existing notebook and
script, and this library's results are the published kind. A profile makes the change a single,
auditable, revertible line instead of an edit sweep:

    'legacy'  (default today) -- reproduces the current numbers exactly.
    'modern'                  -- curves from the FP solver; when the Langevin engine IS needed
                                 (trajectories, windowed interventions, damage distributions) its
                                 time step is chosen automatically instead of inherited from a
                                 seed-era config.

Every new entry point reads its defaults through :func:`resolve`, so the eventual switch of the
shipped default is a one-word change here and nothing else.

PRECEDENCE (highest first)
    1. an explicit keyword argument at the call site
    2. config.set(...) / the active profile
    3. the SRK_PROFILE environment variable   (for cluster jobs and `nbconvert` runs)
    4. the shipped default, 'legacy'

The active settings are stamped into ``CurveSet.meta['profile']`` and into the resolution report, so
no result is ever ambiguous about which defaults produced it.

Nothing here has side effects at import time beyond reading ``SRK_PROFILE``.
"""
import contextlib
import copy
import os

__all__ = ["INHERIT", "PROFILES", "use_profile", "active_profile", "get", "set", "resolve",
           "profile", "snapshot", "describe"]


class _Inherit:
    """Sentinel: 'no explicit choice at the call site -- use the configured default'.

    A plain ``None`` cannot do this job: for ``resolution``, ``None`` is itself a meaningful value
    (it means "today's behaviour: leave the step size exactly as given").
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "INHERIT"

    def __bool__(self):
        return False


INHERIT = _Inherit()


PROFILES = {
    "legacy": {
        # 'sim' = the Langevin Monte-Carlo cohort, which is what every existing call produces.
        "curve_engine": "sim",
        # None = do not touch (nsteps, time_step_multiplier); exactly today's behaviour.
        "resolution": None,
        # 'strict' = refuse to guess the meaning of an unlabelled 5-element theta (it could be
        # [eta,beta,epsilon,xc,kappa] or [eta,beta,epsilon,kappa,ExtH] -- see SRK.theta_utils).
        "theta_order": "strict",
        # Backward Euler, the scheme every fitted number in the repo was produced with.
        "fp_scheme": "be",
        # Warn when a requested time grid runs past what the FP solver actually integrated
        # (otherwise the tail is silently right-clamped and looks like real survival).
        "warn_short_horizon": True,
    },
    "modern": {
        "curve_engine": "fp",
        "resolution": "auto",
        "theta_order": "canonical",
        "fp_scheme": "be",          # deliberately NOT bdf2: still under evaluation for fitting
        "warn_short_horizon": True,
    },
}

SHIPPED_PROFILE = "legacy"

_active_profile = None
_settings = None


def _init():
    global _active_profile, _settings
    name = os.environ.get("SRK_PROFILE", SHIPPED_PROFILE).strip().lower()
    if name not in PROFILES:
        # A typo in an env var must not silently select the wrong engine for a whole cluster run.
        raise ValueError(
            f"SRK_PROFILE={name!r} is not a known profile; expected one of {sorted(PROFILES)}")
    _active_profile = name
    _settings = copy.deepcopy(PROFILES[name])


_init()


def use_profile(name):
    """Switch to a named profile, discarding any per-key overrides. Returns the new settings."""
    name = str(name).strip().lower()
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; expected one of {sorted(PROFILES)}")
    global _active_profile, _settings
    _active_profile = name
    _settings = copy.deepcopy(PROFILES[name])
    return snapshot()


def active_profile():
    """Name of the active profile, with a '+overrides' suffix if any key was changed by hand."""
    base = PROFILES[_active_profile]
    return _active_profile if _settings == base else f"{_active_profile}+overrides"


def get(key):
    try:
        return _settings[key]
    except KeyError:
        raise KeyError(f"unknown SRK config key {key!r}; known keys: {sorted(_settings)}") from None


def set(**kw):
    """Override individual keys on top of the active profile. Returns the new settings."""
    unknown = sorted(k for k in kw if k not in _settings)
    if unknown:
        raise KeyError(f"unknown SRK config key(s) {unknown}; known keys: {sorted(_settings)}")
    _settings.update(kw)
    return snapshot()


def resolve(key, value=INHERIT):
    """The value an entry point should use: an explicit argument wins, otherwise the config.

    Entry points should default their keyword to :data:`INHERIT` (not ``None``) and call this, so
    that passing ``None`` explicitly stays meaningful.
    """
    if value is INHERIT:
        return get(key)
    return value


@contextlib.contextmanager
def profile(name=None, **overrides):
    """Temporarily switch profile and/or override keys; restores everything on exit.

    Useful for A/B checks: ``with config.profile('legacy'): ...`` reproduces the old numbers inside
    a session that otherwise runs 'modern'.
    """
    global _active_profile, _settings
    saved_profile, saved_settings = _active_profile, copy.deepcopy(_settings)
    try:
        if name is not None:
            use_profile(name)
        if overrides:
            set(**overrides)
        yield snapshot()
    finally:
        _active_profile, _settings = saved_profile, saved_settings


def snapshot():
    """A copy of the active settings, safe to store alongside a result."""
    out = copy.deepcopy(_settings)
    out["profile"] = active_profile()
    return out


def describe():
    """Human-readable one-liner for logs and figure captions."""
    s = snapshot()
    return (f"SRK profile={s['profile']} curve_engine={s['curve_engine']} "
            f"resolution={s['resolution']!r} theta_order={s['theta_order']} "
            f"fp_scheme={s['fp_scheme']}")
