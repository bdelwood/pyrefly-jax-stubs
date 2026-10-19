# pyrefly-jax-stubs

This repository is retired. JAX shape stubs are now maintained in [Pyrefly's repository](https://github.com/facebook/pyrefly/tree/main/tensor-shapes/jax-pyrefly-stubs) and published as [`jax-pyrefly-stubs`](https://pypi.org/project/jax-pyrefly-stubs/).

Please contribute missing annotations and report issues [upstream](https://github.com/facebook/pyrefly/issues).

## Migration

Replace `pyrefly-jax-stubs` with `jax-pyrefly-stubs`; do not install both, since both provide the `jax-stubs` directory. Follow the upstream installation instructions and use matching Pyrefly and shape-extension versions.

The annotations differ, so run your project's type checks before switching. Existing tags and source remain available for projects pinned to this implementation.

The copied and adapted JAX declarations retain their original copyright notices and remain under the Apache License 2.0.
