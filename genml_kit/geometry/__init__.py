"""Geometry primitives shared across tasks.

Currently: 2-D similarity-transform algebra for the drone visual-odometry
front-end.  All functions are batched, differentiable where meaningful, and
free of any I/O so they are trivially unit-testable.
"""

from genml_kit.geometry.similarity import (
    SimilarityParams,
    compose_similarity,
    corner_residual,
    params_from_matrix,
    params_to_matrix,
    umeyama_similarity,
    warp_similarity,
    wrap_angle,
)

__all__ = [
    "SimilarityParams",
    "compose_similarity",
    "corner_residual",
    "params_to_matrix",
    "params_from_matrix",
    "umeyama_similarity",
    "warp_similarity",
    "wrap_angle",
]
