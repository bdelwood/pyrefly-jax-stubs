# pyrefly-jax-stubs

Native Pyrefly shape types for JAX.

This is a PEP 561 partial stub-only distribution. It preserves JAX's shipped typing and
specializes array-bearing APIs with Pyrefly's `SizeTuple`, `SymVar`, and shape DSL:

```python
from jax import Array
from shape_extensions import SizeTuple, SymVar

def rows[N: SymVar](x: Array[[N, 3]]) -> Array[[N]]: ...
def identity[S: SizeTuple](x: Array[S]) -> Array[S]: ...
```

The package uses native generic shapes, not jaxtyping string annotations, and is intended
for eventual upstream contribution to Pyrefly alongside its NumPy and PyTorch stub packages.

## Versioning

Releases use `<jax-version>.<stub-revision>`. Version `0.10.2.0` targets JAX 0.10.2;
`0.10.2.1` would be a stub-only correction against the same JAX surface. The package pins its
JAX dependency exactly so an upstream API change cannot silently invalidate the overlay.
