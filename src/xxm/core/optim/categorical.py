"""Optimization utilities for categorical state weights."""

import typing

import jax
import jax.numpy as jnp

from xxm.core.dists.categorical import Categorical

PyTreeT = typing.TypeVar('PyTreeT')


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


def filter_valid_states(
    fitted: PyTreeT,
    current: PyTreeT,
    weights: jax.Array,
    *,
    min_expected_count: float = 1.0,
) -> PyTreeT:
    """
    Keep current parameters for state-batched fits with insufficient weight.

    PyTree leaves are expected to have state as their leading batch dimension.
    """
    expected_counts = jnp.sum(
        weights,
        axis=0,
    )

    valid = expected_counts >= min_expected_count

    def select(
        fitted_leaf: jax.Array,
        current_leaf: jax.Array,
    ) -> jax.Array:
        mask = valid.reshape((valid.shape[0],) + (1,) * (fitted_leaf.ndim - 1))

        return jnp.where(
            mask,
            fitted_leaf,
            current_leaf,
        )

    return typing.cast(
        PyTreeT,
        jax.tree.map(
            select,
            fitted,
            current,
        ),
    )
