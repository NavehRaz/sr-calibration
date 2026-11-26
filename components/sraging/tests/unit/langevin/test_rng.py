"""The per-individual random streams (tasks T057; research R15): xoshiro256** seeded by four SplitMix64 outputs from
hash64(key, i), 53-bit uniforms and Marsaglia polar normals, in numba and in a pure NumPy/Python reference that must
agree bit for bit; the parameter and hazard streams are separate from the path stream."""
import numpy as np
import pytest

pytest.importorskip("numba", reason="NOT APPLICABLE: numba (the langevin extra) is not installed")

from sraging.langevin import rng as RNG  # noqa: E402

KEY = 20261029
N = 1_003
IDS = (0, 1, N - 1)


@pytest.mark.parametrize("i", IDS)
def test_the_numba_stream_is_the_references_on_the_first_1000_draws(i):
    ref = RNG.RefStream(KEY, i)
    want = np.array([ref.next_u64() for _ in range(1000)], dtype=np.uint64)
    assert RNG.numba_draws(KEY, i, 1000, "u64").tobytes() == want.tobytes()
    ref = RNG.RefStream(KEY, i)
    want = np.array([ref.uniform() for _ in range(1000)])
    got = RNG.numba_draws(KEY, i, 1000, "uniform")
    assert got.tobytes() == want.tobytes() and np.all((got >= 0) & (got < 1))


@pytest.mark.parametrize("i", IDS)
def test_the_polar_normals_match_too(i):
    ref = RNG.RefStream(KEY, i)
    want = np.array([x for _ in range(500) for x in ref.normal_pair()])
    got = RNG.numba_draws(KEY, i, 1000, "normal")
    assert got.tobytes() == want.tobytes()
    assert abs(got.mean()) < 0.15 and abs(got.std() - 1) < 0.1


def test_the_seed_is_four_splitmix64_outputs_of_hash64():
    h = RNG.hash64(KEY, 7)
    x, s = h, []
    for _ in range(4):
        x = (x + RNG.GOLDEN) & RNG.MASK
        s.append(RNG.mix64(x))
    assert RNG.RefStream(KEY, 7).s == s
    assert RNG.hash64(KEY, 7) != RNG.hash64(KEY, 8) != RNG.hash64(KEY + 1, 7)
    assert RNG.hash64_array(KEY, np.arange(5)).tolist() == [RNG.hash64(KEY, i) for i in range(5)]


def test_the_parameter_and_hazard_streams_are_not_the_path_stream():
    idx = np.arange(N)
    het = RNG.stream_uniforms(KEY, idx, "het:xc")
    haz = RNG.stream_uniforms(KEY, idx, "hazard")
    path = np.array([RNG.RefStream(KEY, int(i)).uniform() for i in idx])
    assert not np.array_equal(het, path) and not np.array_equal(haz, path) and not np.array_equal(het, haz)
    assert np.all((het > 0) & (het < 1)) and abs(het.mean() - 0.5) < 0.05
    for i in IDS:                                          # each stream is the reference's own first uniform
        assert het[i] == RNG.RefStream(KEY, i, "het:xc").uniform()
        assert haz[i] == RNG.RefStream(KEY, i, "hazard").uniform()
    assert RNG.stream_uniforms(KEY, idx[:50], "het:xc").tobytes() == het[:50].tobytes()        # nested in n
    assert not np.array_equal(RNG.stream_uniforms(KEY, idx, "het:eta"), het)                   # per parameter


def test_a_key_outside_63_bits_is_refused():
    from sraging.errors import UnsupportedOptionError
    for bad in (-1, 2 ** 63, 1.5, "7", True):
        with pytest.raises(UnsupportedOptionError):
            RNG.check_key(bad)
    assert RNG.check_key(2 ** 63 - 1) == 2 ** 63 - 1                                             # the twin
