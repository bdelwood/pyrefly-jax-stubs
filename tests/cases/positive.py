from typing import assert_type

import jax.numpy as jnp
from jax import Array
from shape_extensions import IntTuple, IntVar


def identity[Shape: IntTuple](x: Array[Shape]) -> Array[Shape]:
    return x


def vector_rows[N: IntVar](x: Array[[N, 3]]) -> Array[[N, 3]]:
    assert_type(x.shape, IntTuple[N, 3])
    return x


def gradual_shape_island[N: IntVar](x: Array[[N, 3]]) -> Array[[N, 3]]:
    """An unmodeled JAX operation returns bare Array and remains gradually compatible."""
    return jnp.sin(x)


def dtype_stays_gradual[N: IntVar](x: Array[[N]]) -> Array[[N]]:
    """JAX/NumPy dtype stub identities must not require downstream casts."""
    return jnp.zeros(x.shape, dtype=x.dtype)
