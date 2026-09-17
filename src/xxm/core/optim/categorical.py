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
    Keep current parameters for fitted batches with insufficient weight.

    Weights have shape `(*B, S)` and all PyTree leaves begin with `*B`.
    """
    if weights.ndim < 1:
        raise ValueError('weights must have shape (*B, S)')

    expected_counts = jnp.sum(
        weights,
        axis=-1,
    )

    valid = expected_counts >= min_expected_count

    def select(
        fitted_leaf: jax.Array,
        current_leaf: jax.Array,
    ) -> jax.Array:
        if (
            fitted_leaf.shape[: valid.ndim] != valid.shape
            or current_leaf.shape[: valid.ndim] != valid.shape
        ):
            raise ValueError(
                f'parameter leaves must begin with fitted batch shape {valid.shape}'
            )
        if fitted_leaf.shape != current_leaf.shape:
            raise ValueError(
                'fitted and current parameter leaves must have equal shapes'
            )

        mask = valid.reshape(valid.shape + (1,) * (fitted_leaf.ndim - valid.ndim))

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
