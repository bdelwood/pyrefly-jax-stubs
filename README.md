# pyrefly-jax-stubs

Native Pyrefly shape types for JAX arrays.

This PEP 561 partial stub package preserves JAX's shipped typing and specializes array-bearing APIs with Pyrefly's `SizeTuple`, `SymVar`, and shape syntax:

```python
from jax import Array
from shape_extensions import SizeTuple, SymVar

def rows[N: SymVar](x: Array[[N, 3]]) -> Array[[N]]: ...
def identity[S: SizeTuple](x: Array[S]) -> Array[S]: ...
```

Pyrefly discovers the installed stubs automatically. APIs not included here continue to use the typing shipped with JAX.

## Installation

Install the stubs and Pyrefly as development dependencies:

```bash
uv add --dev --prerelease=allow pyrefly==1.2.0.dev2 pyrefly-jax-stubs==0.10.2.1
```

The prerelease flag is required while Pyrefly 1.2 and its shape extension remain prereleases. The package requires Python 3.13 or newer because its signatures use type-parameter defaults. Release `0.10.2.1` targets JAX 0.10.2 and is tested with Pyrefly 1.2.0.dev2.

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
