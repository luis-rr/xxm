"""Optimization utilities for categorical state weights."""

import typing

import jax
import jax.numpy as jnp

PyTreeT = typing.TypeVar('PyTreeT')


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
