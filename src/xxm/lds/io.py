"""Canonical LDS parameter schema."""

from typing import TypedDict

import jax


class LDSParams(TypedDict):
    initial_mean: jax.Array
    initial_covariance: jax.Array
    dynamics_coefficients: jax.Array
    dynamics_bias: jax.Array
    dynamics_covariance: jax.Array
    dynamics_input_coefficients: jax.Array
    emissions: tuple[dict[str, jax.Array], ...]
