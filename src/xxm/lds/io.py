"""Canonical LDS parameter schema."""

from typing import NotRequired, TypedDict

import jax

from xxm.core.dists.io import LinearGaussianParams, LinearPoissonParams


class LDSParams(TypedDict):
    initial_mean: jax.Array
    initial_covariance: jax.Array
    dynamics_coefficients: jax.Array
    dynamics_bias: jax.Array
    dynamics_covariance: jax.Array
    dynamics_input_coefficients: NotRequired[jax.Array]
    emissions: tuple[LinearGaussianParams | LinearPoissonParams, ...]
