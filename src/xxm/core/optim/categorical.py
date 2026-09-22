"""Categorical fitting utilities."""

import jax
import jax.numpy as jnp

from xxm.core.dists.categorical import Categorical

DEFAULT_PSEUDOCOUNT = 1e-8


def from_counts(
    counts: jax.Array,
    *,
    pseudocount: float,
) -> Categorical:
    """
    Fit a categorical distribution from counts with additive stabilization.

    `pseudocount` is added to every category before normalization. Positive
    pseudocounts keep fitted probabilities away from the boundary of the
    probability simplex, preventing accidental zero-locking.
    """
    dtype = jnp.result_type(
        counts,
        jnp.float32,
    )

    counts = counts.astype(dtype)

    return _from_counts(
        counts
        + jnp.asarray(
            pseudocount,
            dtype=dtype,
        )
    )


def _from_counts(
    counts: jax.Array,  # (..., K)
) -> Categorical:
    """Construct from category counts, handling zero-count cases with uniform."""
    total = counts.sum(axis=-1, keepdims=True)
    valid = total > 0

    # Handle zero counts with the uniform distribution.
    valid_total = jnp.where(valid, total, counts.shape[-1])
    valid_counts = jnp.where(valid, counts, 1)

    probs = valid_counts / valid_total

    return Categorical(probs=probs)
