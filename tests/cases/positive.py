import jax.numpy as jnp
from jax import Array
from shape_extensions import SizeTuple, SymVar, assert_shape


def identity[Shape: SizeTuple](x: Array[Shape]) -> Array[Shape]:
    return x


def vector_rows[N: SymVar](x: Array[[N, 3]]) -> Array[[N, 3]]:
    assert_shape(x, (N, 3))
    return x


def gradual_shape_island[N: SymVar](x: Array[[N, 3]]) -> Array[[N, 3]]:
    """An unmodeled JAX operation returns bare Array and remains gradually compatible."""
    return jnp.sin(x)


def dtype_stays_gradual[N: SymVar](x: Array[[N]]) -> Array[[N]]:
    """JAX/NumPy dtype stub identities must not require downstream casts."""
    return jnp.zeros(x.shape, dtype=x.dtype)
