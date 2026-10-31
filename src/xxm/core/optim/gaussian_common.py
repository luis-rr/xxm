"""Fitting helpers shared by Gaussian models of any covariance structure.

Sample weighting, full moment fits of regression inputs, and the affine
least-squares map.
"""

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import FullGaussian


def normalize_weights(
    num_samples: int,
    weights: jax.Array | None,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    """Normalize optional weights along the sample axis."""
    if weights is None:
        return jnp.full(
            (num_samples,),
            1.0 / num_samples,
            dtype=dtype,
        )

    if weights.shape[-1] != num_samples:
        raise ValueError(
            f'weights must have sample axis of length {num_samples}; '
            f'got shape {weights.shape}'
        )

    weights = weights.astype(dtype)

    totals = jnp.sum(
        weights,
        axis=-1,
    )

    safe_totals = jnp.where(
        totals > 0.0,
        totals,
        1.0,
    )

    return weights / safe_totals[..., None]


def center_samples(
    values: jax.Array,  # (S, N)
    mean: jax.Array,  # (..., N)
) -> jax.Array:  # (..., S, N)
    """Center samples around one or more means."""
    batch_ndim = mean.ndim - 1

    values = values.reshape((1,) * batch_ndim + values.shape)

    return values - mean[..., None, :]


def from_samples_normalized(
    values: jax.Array,  # (S, N)
    weights: jax.Array,  # (..., S)
) -> FullGaussian:
    """Fit a Gaussian using already-normalized weights."""
    mean = jnp.einsum(
        '...t,ti->...i',
        weights,
        values,
    )

    residuals = center_samples(
        values,
        mean,
    )

    covariance = jnp.einsum(
        '...t,...ti,...tj->...ij',
        weights,
        residuals,
        residuals,
    )

    covariance = 0.5 * (
        covariance
        + jnp.swapaxes(
            covariance,
            -2,
            -1,
        )
    )

    return FullGaussian.from_moments(mean, covariance)


def group_weights(
    assignments: jax.Array,  # (S,)
    num_groups: int,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:  # (K, S)
    """One-hot sample weights, one row per group."""
    return jnp.moveaxis(jax.nn.one_hot(assignments, num_groups, dtype=dtype), -1, 0)


def from_moment_match_weighted(
    distributions: FullGaussian,
    normalized_weights: jax.Array,
) -> FullGaussian:
    """Moment-match Gaussian distributions using normalized weights."""
    batch_shape = distributions.batch_shape[:-1]
    num_samples = distributions.batch_shape[-1]
    query_ndim = normalized_weights.ndim - len(batch_shape) - 1
    shape = batch_shape + (1,) * query_ndim + (num_samples,)
    mean = distributions.mean.reshape(shape + (distributions.variable_dim,))
    covariance = distributions.covariance.reshape(
        shape + (distributions.variable_dim, distributions.variable_dim)
    )
    matched_mean = jnp.sum(normalized_weights[..., :, None] * mean, axis=-2)
    residuals = mean - matched_mean[..., None, :]
    covariance = jnp.sum(
        normalized_weights[..., :, None, None] * covariance, axis=-3
    ) + jnp.einsum('...s,...si,...sj->...ij', normalized_weights, residuals, residuals)
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -2, -1))
    return FullGaussian.from_moments(matched_mean, covariance)


def _add_relative_ridge(
    covariance: jax.Array,
    ridge: float,
) -> jax.Array:
    """
    Add isotropic ridge relative to the average marginal variance.

    If the covariance has no positive variance scale, a unit reference scale
    is used so that degenerate regression systems remain well defined.
    """

    scale = jnp.mean(
        jnp.diagonal(
            covariance,
            axis1=-2,
            axis2=-1,
        ),
        axis=-1,
    )

    scale = jnp.where(
        scale > 0.0,
        scale,
        1.0,
    )

    identity = jnp.eye(
        covariance.shape[-1],
        dtype=covariance.dtype,
    )

    return covariance + ridge * scale[..., None, None] * identity


def affine_from_moments(
    left: FullGaussian,
    cross_covariance: jax.Array,  # (..., R, L)
    right_mean: jax.Array,  # (..., R)
    *,
    ridge: float,
) -> Affine:
    r"""Return the affine least-squares map from ``left`` to a right variable.

    ``cross_covariance`` stores Cov(right, left).

    ``ridge`` adds isotropic regularization to the left covariance relative
    to its average marginal variance.
    """
    input_covariance = _add_relative_ridge(
        left.covariance,
        ridge,
    )

    coefficients = jnp.linalg.solve(
        input_covariance,
        jnp.swapaxes(
            cross_covariance,
            -2,
            -1,
        ),
    )

    coefficients = jnp.swapaxes(
        coefficients,
        -2,
        -1,
    )

    bias = right_mean - jnp.einsum(
        '...oi,...i->...o',
        coefficients,
        left.mean,
    )

    return Affine(
        coefficients=coefficients,
        bias=bias,
    )
