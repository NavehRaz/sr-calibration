"""Per-individual random streams (spec 004 research R15; Dd): a sample depends only on (key, i), so it is the same
for any worker count and chunk size, and nested in n.

- **The stream:** individual i's stream is xoshiro256**, seeded by four SplitMix64 outputs from hash64(key, i, label).
  The key is 63-bit. The label names the stream: "" (the path's normals), "bridge" (the bridge's uniforms),
  "het:<parameter>" and "hazard" (one uniform each, the parameter and the external-death draws).
- **Uniforms** are 53-bit, in [0, 1); **normals** come in Marsaglia polar pairs. A dt step uses one pair, summed
  (xi_a + xi_b) / sqrt 2; the coupled dt/2 rung uses the same pair as its two half-steps (research R16), so both rungs
  follow one Brownian path. The bridge's uniforms have their own stream, so the rungs stay coupled whatever the bridge
  draws.
- **Two implementations:** numba (`next_u64`, `uniform`, `normal_pair`, for the kernels) and a pure Python/NumPy
  reference (`RefStream`, `hash64_array`, `stream_uniforms`). tests/unit/langevin/test_rng.py asserts they agree bit
  for bit.
"""
from __future__ import annotations

import math

import numba
import numpy as np
from numba import njit

from sraging.errors import UnsupportedOptionError

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
_M1, _M2 = 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
_TWO_M53 = 1.0 / 9007199254740992.0
KEY_MAX = 1 << 63


def check_key(key) -> int:
    if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or not 0 <= int(key) < KEY_MAX:
        raise UnsupportedOptionError("key", f"key is an integer in [0, 2^63), got {key!r}")
    return int(key)


# --------------------------------------------------------------------------------------- the reference (Python)
def mix64(z: int) -> int:
    """SplitMix64's output function."""
    z = ((z ^ (z >> 30)) * _M1) & MASK
    z = ((z ^ (z >> 27)) * _M2) & MASK
    return z ^ (z >> 31)


def _label(label: str) -> int:
    """FNV-1a 64 of the label's UTF-8 bytes (0 for the path stream's empty label)."""
    if not label:
        return 0
    h = 0xCBF29CE484222325
    for b in label.encode():
        h = ((h ^ b) * 0x100000001B3) & MASK
    return h


def hash64(key: int, i: int, label: str = "") -> int:
    """The seed of individual i's stream `label` under `key`."""
    fam = mix64(((int(key) ^ _label(label)) + GOLDEN) & MASK)
    return mix64((fam + (int(i) + 1) * GOLDEN) & MASK)


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & MASK


class RefStream:
    """The reference xoshiro256** stream of (key, i, label)."""

    def __init__(self, key, i, label=""):
        x, s = hash64(key, i, label), []
        for _ in range(4):
            x = (x + GOLDEN) & MASK
            s.append(mix64(x))
        self.s = s

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (_rotl((s1 * 5) & MASK, 7) * 9) & MASK
        t = (s1 << 17) & MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 45)
        self.s = [s0, s1, s2, s3]
        return result

    def uniform(self) -> float:
        return (self.next_u64() >> 11) * _TWO_M53

    def normal_pair(self):
        while True:
            u = 2.0 * self.uniform() - 1.0
            v = 2.0 * self.uniform() - 1.0
            s = u * u + v * v
            if 0.0 < s < 1.0:
                f = math.sqrt(-2.0 * math.log(s) / s)
                return u * f, v * f


# ------------------------------------------------------------------------------ the reference, vectorized (NumPy)
def _mix64_arr(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
    return z ^ (z >> np.uint64(31))


def hash64_array(key: int, idx, label: str = "") -> np.ndarray:
    """hash64(key, i, label) for every i in idx, as uint64 (NumPy's uint64 arithmetic wraps, as the reference's)."""
    fam = np.uint64(mix64(((int(key) ^ _label(label)) + GOLDEN) & MASK))
    i = np.asarray(idx, dtype=np.uint64)
    return _mix64_arr(fam + (i + np.uint64(1)) * np.uint64(GOLDEN))


def stream_uniforms(key: int, idx, label: str) -> np.ndarray:
    """The first uniform of stream `label` for every individual in idx: one parameter or hazard draw each (R15)."""
    x = hash64_array(key, idx, label)
    s = []
    for _ in range(4):
        x = x + np.uint64(GOLDEN)
        s.append(_mix64_arr(x))
    s1 = s[1]
    r = s1 * np.uint64(5)
    r = (r << np.uint64(7)) | (r >> np.uint64(57))
    r = r * np.uint64(9)
    return (r >> np.uint64(11)).astype(np.float64) * _TWO_M53


# ------------------------------------------------------------------------------------------------ numba (kernels)
_U = numba.uint64


@njit(inline="always")
def _rotl_nb(x, k):
    return (x << _U(k)) | (x >> _U(64 - k))


@njit(inline="always")
def _mix64_nb(z):
    z = (z ^ (z >> _U(30))) * _U(_M1)
    z = (z ^ (z >> _U(27))) * _U(_M2)
    return z ^ (z >> _U(31))


@njit
def seed(state, h):
    """state[0:4] from four SplitMix64 outputs of the seed h (a hash64)."""
    x = h
    for j in range(4):
        x = x + _U(GOLDEN)
        state[j] = _mix64_nb(x)


@njit
def next_u64(state):
    s0, s1, s2, s3 = state[0], state[1], state[2], state[3]
    result = _rotl_nb(s1 * _U(5), 7) * _U(9)
    t = s1 << _U(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl_nb(s3, 45)
    state[0], state[1], state[2], state[3] = s0, s1, s2, s3
    return result


@njit
def uniform(state):
    return float(next_u64(state) >> _U(11)) * _TWO_M53


@njit
def normal_pair(state):
    while True:
        u = 2.0 * uniform(state) - 1.0
        v = 2.0 * uniform(state) - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            f = math.sqrt(-2.0 * math.log(s) / s)
            return u * f, v * f


@njit
def _draws(h, k, kind):
    state = np.zeros(4, dtype=np.uint64)
    seed(state, h)
    if kind == 0:
        out = np.zeros(k, dtype=np.uint64)
        for j in range(k):
            out[j] = next_u64(state)
        return out.view(np.float64)
    res = np.zeros(k)
    if kind == 1:
        for j in range(k):
            res[j] = uniform(state)
    else:
        for j in range(0, k, 2):
            a, b = normal_pair(state)
            res[j] = a
            if j + 1 < k:
                res[j + 1] = b
    return res


def numba_draws(key: int, i: int, k: int, kind: str = "u64", label: str = "") -> np.ndarray:
    """The first k draws ("u64", "uniform" or "normal") of the numba stream of (key, i, label): the tests' bridge to
    the reference."""
    code = {"u64": 0, "uniform": 1, "normal": 2}[kind]
    out = _draws(np.uint64(hash64(key, i, label)), int(k), code)
    return out.view(np.uint64) if kind == "u64" else out
