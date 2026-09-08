# pyrefly-jax-stubs

Native Pyrefly shape types for JAX arrays.

This PEP 561 partial stub package preserves JAX's shipped typing and specializes array-bearing APIs with Pyrefly's `IntTuple`, `IntVar`, and shape syntax:

```python
from jax import Array
from shape_extensions import IntTuple, IntVar

def rows[N: IntVar](x: Array[[N, 3]]) -> Array[[N]]: ...
def identity[S: IntTuple](x: Array[S]) -> Array[S]: ...
```

Pyrefly discovers the installed stubs automatically. APIs not included here continue to use the typing shipped with JAX.

## Installation

This checkout pins Pyrefly and its shape extension to `1.3.0.dev4`, which provides the composable shape API. Both are available as wheels; building Pyrefly from source is unnecessary:

```bash
uv sync
```

The package requires Python 3.13 or newer because its signatures use type-parameter defaults. Version `0.10.2.2` targets JAX 0.10.2. Pyrefly's shape API is still under development, so upgrade the checker and extension together and rerun the shape-contract tests.

## Development

A local checkout can be checked and built with:

```bash
uv sync
just lint
just test
just build
```

The copied and adapted JAX declarations retain their original copyright notices and remain under the Apache License 2.0.

## Versioning

Releases use `<jax-version>.<stub-revision>`. The package pins JAX exactly so an upstream API change cannot silently invalidate the overlay.
