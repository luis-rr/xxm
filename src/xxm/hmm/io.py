"""Canonical HMM parameter schema, also shared by AR-HMM facades."""

from typing import TypedDict

import jax

from xxm.core.dists.io import (
    GaussianParams,
    LinearGaussianParams,
    LinearPoissonParams,
    PoissonParams,
)


class HMMParams(TypedDict):
    initial_probs: jax.Array
    transition_probs: jax.Array
    emissions: tuple[
        GaussianParams | PoissonParams | LinearGaussianParams | LinearPoissonParams, ...
    ]
