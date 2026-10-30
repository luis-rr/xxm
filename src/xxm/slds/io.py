"""Canonical SLDS parameter schema."""

from typing import NotRequired, TypedDict

import jax

from xxm.core.dists.io import LinearGaussianParams, LinearPoissonParams


class SLDSParams(TypedDict):
    initial_probs: jax.Array
    transition_probs: jax.Array
    latent_initial_means: jax.Array
    latent_initial_covariances: jax.Array
    dynamics_coefficients: jax.Array
    dynamics_bias: jax.Array
    dynamics_covariances: jax.Array
    dynamics_input_coefficients: NotRequired[jax.Array]
    emissions: tuple[LinearGaussianParams | LinearPoissonParams, ...]
