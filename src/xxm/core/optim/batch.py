"""Shared selection of fitted parameter batches."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch

PyTreeT = typing.TypeVar('PyTreeT')


def filter_valid_batches(
    fitted: PyTreeT,
    current: PyTreeT,
    weights: jax.Array,
    *,
    min_expected_count: float | None = None,
) -> PyTreeT:
    """
    Keep current parameters for fitted batches with insufficient weight.

    Weights have shape `(*B, S)` and all PyTree leaves begin with `*B`.
    By default, only strictly positive total weights are valid. An explicit
    minimum uses an inclusive comparison, for minimum-occupancy policies.
    """
    if weights.ndim < 1:
        raise ValueError('weights must have shape (*B, S)')

    expected_counts = jnp.sum(
        weights,
        axis=-1,
    )

    valid = (
        expected_counts > 0
        if min_expected_count is None
        else expected_counts >= min_expected_count
    )

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

        mask = batch.expand_trailing(valid, fitted_leaf.ndim)

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
