"""Canonical HMM parameter schema, also shared by AR-HMM facades."""

from typing import TypedDict

import jax


class HMMParams(TypedDict):
    initial_probs: jax.Array
    transition_probs: jax.Array
    emissions: tuple[dict[str, jax.Array], ...]
