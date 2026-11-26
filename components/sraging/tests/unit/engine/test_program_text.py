"""The program canonicaliser behind SC-001 (research R1): equal up to symbol names, and nothing looser."""
import jax
import jax.numpy as jnp
import numpy as np

from sraging.compat.v033 import fp_jax as V
from sraging.engine.program import canonical_text, same_program

T = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
E = np.array([1, 1, 0, 1, 1])
Z = jnp.log(jnp.array([1e-2, 1.0, 1e-2, 0.5]))


def _value_fn():
    return V.make_loglik(T, E, 4.0, N=50, dt=1e-2, tau_max=3.0, sampling_dt=1.0)[0]


def test_symbols_are_renamed_in_order_of_appearance():
    text = "module @jit_f {\n func.func public @main() { call @_where_3() }\n func.func private @_where_3() }"
    assert canonical_text(text) == "module @s0 {\n func.func public @s1() { call @s2() }\n func.func private @s2() }"


def test_the_same_operations_under_other_names_are_the_same_program():
    def one_name(z):
        return jnp.sum(jnp.where(z > 0, jnp.clip(z, 0.0, 1.0), -z))

    def another_name(z):                        # identical body; the module becomes @jit_another_name
        return jnp.sum(jnp.where(z > 0, jnp.clip(z, 0.0, 1.0), -z))

    a, b = jax.jit(one_name).lower(Z), jax.jit(another_name).lower(Z)
    assert a.as_text() != b.as_text()           # the names differ ...
    assert same_program(a, b)                   # ... the program does not


def test_a_jitted_call_inside_jit_is_a_different_program():
    """Wrapping adds a real call, so it must not compare equal (the canonicaliser renames, nothing more)."""
    f = _value_fn()
    assert not same_program(f.lower(Z), jax.jit(lambda z: f(z)).lower(Z))


def test_one_extra_operation_is_a_different_program():
    """Constitution V: the comparison must fail on the defect it exists for (an op added by a feature that
    should be off)."""
    f = _value_fn()
    planted = jax.jit(lambda z: f(z) + 0.0 * z[0])
    assert not same_program(f.lower(Z), planted.lower(Z))


def test_different_constants_are_a_different_program():
    f = _value_fn()
    g = V.make_loglik(T * 1.0000001, E, 4.0, N=50, dt=1e-2, tau_max=3.0, sampling_dt=1.0)[0]
    assert not same_program(f.lower(Z), g.lower(Z))
