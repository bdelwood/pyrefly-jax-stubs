from jax import Array
from shape_extensions import SymVar


def wrong_width[N: SymVar](x: Array[[N, 3]]) -> Array[[N, 2]]:
    # E: Returned type `Array[[N, 3]]` is not assignable
    #    to declared return type `Array[[N, 2]]`
    return x


def wrong_rank(x: Array[[2, 3]]) -> Array[[2, 3, 4]]:
    # E: Returned type `Array[[2, 3]]` is not assignable
    #    to declared return type `Array[[2, 3, 4]]`
    return x


def needs_vectors[N: SymVar](x: Array[[N, 3]]) -> Array[[N, 3]]:
    return x


def wrong_composition[N: SymVar](x: Array[[N, 2]]) -> Array[[N, 3]]:
    return needs_vectors(x)
