"""Moment-matching and weighted least-squares fitting with diagonal covariances.

Diagonal counterparts of `xxm.core.optim.gaussian_full`, in the same order and
with the same names. Output covariances are never formed: right-hand
variables keep only their means and marginal variances, so memory grows with
the output dimension rather than its square.
"""

import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import (
    DiagonalGaussian,
    DiagonalGaussianNoise,
    DiagonalLinearGaussian,
    FullGaussian,
)
from xxm.core.optim import gaussian_common as gaussian_common_fit


class DiagonalPairedGaussian(typing.NamedTuple):
    """Paired moments with a full left and a diagonal right covariance."""

    left: FullGaussian  # (..., L)
    right: DiagonalGaussian  # (..., R)
    cross_covariance: jax.Array  # (..., R, L), Cov(right, left)


def _from_samples_normalized(
    values: jax.Array,  # (S, N)
    weights: jax.Array,  # (..., S)
) -> DiagonalGaussian:
    """Fit a diagonal Gaussian using already-normalized weights."""
    mean = jnp.einsum(
        '...t,ti->...i',
        weights,
        values,
    )

    residuals = gaussian_common_fit.center_samples(
        values,
        mean,
    )

    variances = jnp.einsum(
        '...t,...ti->...i',
        weights,
        residuals**2,
    )

    return DiagonalGaussian.from_moments(mean, variances)


def from_samples_weighted(
    values: jax.Array,  # (S, N)
    weights: jax.Array,  # (..., S)
    *,
    covariance_floor: float,
) -> DiagonalGaussian:
    """Fit one weighted diagonal Gaussian for each batch entry of ``weights``."""
    dtype = jnp.result_type(
        values,
        weights,
        jnp.float32,
    )
    values = values.astype(dtype)

    weights = gaussian_common_fit.normalize_weights(
        num_samples=values.shape[0],
        weights=weights,
        dtype=dtype,
    )

    fitted = _from_samples_normalized(
        values=values,
        weights=weights,
    )

    return fitted._replace(noise=fitted.noise.with_floor(covariance_floor))


def from_samples_grouped(
    values: jax.Array,  # (S, N)
    assignments: jax.Array,  # (S,)
    num_groups: int,
    *,
    covariance_floor: float,
    sample_weights: jax.Array | None = None,  # (S,)
) -> DiagonalGaussian:
    """Fit one diagonal Gaussian to each weighted group of assigned values."""
    weights = gaussian_common_fit.group_weights(
        assignments, num_groups, jnp.result_type(values, jnp.float32)
    )  # (K, S)

    if sample_weights is not None:
        weights = weights * sample_weights[None, :]

    return from_samples_weighted(
        values=values,
        weights=weights,
        covariance_floor=covariance_floor,
    )


def paired_from_samples(
    left: jax.Array,  # (S, L)
    right: jax.Array,  # (S, R)
    *,
    weights: jax.Array | None = None,  # (..., S)
) -> DiagonalPairedGaussian:
    """Fit paired moments from aligned left and right samples."""
    assert left.shape[0] == right.shape[0]

    num_samples = left.shape[0]

    dtype = jnp.result_type(
        left,
        right,
        weights if weights is not None else jnp.float32,
        jnp.float32,
    )

    left = left.astype(dtype)
    right = right.astype(dtype)

    normalized_weights = gaussian_common_fit.normalize_weights(
        num_samples=num_samples,
        weights=weights,
        dtype=dtype,
    )

    left_distribution = gaussian_common_fit.from_samples_normalized(
        values=left,
        weights=normalized_weights,
    )

    right_distribution = _from_samples_normalized(
        values=right,
        weights=normalized_weights,
    )

    left_residuals = gaussian_common_fit.center_samples(
        left,
        left_distribution.mean,
    )

    right_residuals = gaussian_common_fit.center_samples(
        right,
        right_distribution.mean,
    )

    cross_covariance = jnp.einsum(
        '...t,...ti,...tj->...ij',
        normalized_weights,
        right_residuals,
        left_residuals,
    )

    return DiagonalPairedGaussian(
        left=left_distribution,
        right=right_distribution,
        cross_covariance=cross_covariance,
    )


def paired_from_left_marginals(
    left: FullGaussian,
    right: jax.Array,
    *,
    weights: jax.Array | None = None,
) -> DiagonalPairedGaussian:
    """
    Moment-match uncertain left variables with deterministic right samples.
    """
    assert len(left.batch_shape) == 1
    assert left.batch_shape[0] == right.shape[0]

    dtype = jnp.result_type(
        left.dtype,
        right,
        weights if weights is not None else jnp.float32,
        jnp.float32,
    )

    left = left.astype(dtype)
    right = right.astype(dtype)

    normalized_weights = gaussian_common_fit.normalize_weights(
        num_samples=left.batch_shape[0],
        weights=weights,
        dtype=dtype,
    )

    aggregated_left = gaussian_common_fit.from_moment_match_weighted(
        distributions=left,
        normalized_weights=normalized_weights,
    )

    aggregated_right = _from_samples_normalized(
        values=right,
        weights=normalized_weights,
    )

    left_residuals = gaussian_common_fit.center_samples(
        left.mean,
        aggregated_left.mean,
    )

    right_residuals = gaussian_common_fit.center_samples(
        right,
        aggregated_right.mean,
    )

    cross_covariance = jnp.einsum(
        '...t,...ti,...tj->...ij',
        normalized_weights,
        right_residuals,
        left_residuals,
    )

    return DiagonalPairedGaussian(
        left=aggregated_left,
        right=aggregated_right,
        cross_covariance=cross_covariance,
    )


def noise_from_paired(
    paired: DiagonalPairedGaussian,
    affine: Affine,
    *,
    covariance_floor: float,
) -> DiagonalGaussianNoise:
    r"""
    Return the residual variances of ``right | left`` under a fitted ``affine``.

    These are the marginal variances of $v - Wu - b$ under the paired moments,
    with an isotropic floor relative to the marginal output variances.
    """
    coefficients = affine.coefficients

    variances = (
        paired.right.variances
        - 2.0 * jnp.einsum('...oi,...oi->...o', coefficients, paired.cross_covariance)
        + jnp.einsum(
            '...oi,...ij,...oj->...o',
            coefficients,
            paired.left.covariance,
            coefficients,
        )
    )

    return DiagonalGaussianNoise(variances).with_floor(
        covariance_floor, reference=paired.right.noise
    )


def linear_from_paired(
    paired: DiagonalPairedGaussian,
    *,
    ridge: float,
    covariance_floor: float,
) -> DiagonalLinearGaussian:
    r"""
    Return the diagonal linear-Gaussian distribution of ``right | left``.

    ``ridge`` adds isotropic regularization to the left covariance relative
    to its average marginal variance.
    """
    affine = gaussian_common_fit.affine_from_moments(
        paired.left,
        paired.cross_covariance,
        paired.right.mean,
        ridge=ridge,
    )

    return DiagonalLinearGaussian(
        affine=affine,
        noise=noise_from_paired(paired, affine, covariance_floor=covariance_floor),
    )


def linear_from_marginals(
    inputs: FullGaussian,
    outputs: jax.Array,
    *,
    weights: jax.Array | None = None,
    ridge: float,
    covariance_floor: float,
) -> DiagonalLinearGaussian:
    paired = paired_from_left_marginals(
        left=inputs,
        right=outputs,
        weights=weights,
    )

    return linear_from_paired(
        paired,
        ridge=ridge,
        covariance_floor=covariance_floor,
    )


def linear_from_samples(
    inputs: jax.Array,  # (S, *input_shape)
    outputs: jax.Array,  # (S, O)
    *,
    ridge: float,
    covariance_floor: float,
) -> DiagonalLinearGaussian:
    """Fit a diagonal linear Gaussian model from samples."""

    if inputs.ndim < 2:
        raise ValueError('inputs must have shape (S, *input_shape)')

    input_shape = inputs.shape[1:]

    paired = paired_from_samples(
        left=inputs.reshape(
            inputs.shape[0],
            -1,
        ),
        right=outputs,
    )

    return linear_from_paired(
        paired,
        ridge=ridge,
        covariance_floor=covariance_floor,
    ).reshape_input(input_shape)


def linear_from_samples_weighted(
    inputs: jax.Array,  # (S, *input_shape)
    outputs: jax.Array,  # (S, O)
    weights: jax.Array,  # (..., S)
    *,
    ridge: float,
    covariance_floor: float,
) -> DiagonalLinearGaussian:
    """Fit one weighted diagonal model for each batch entry of ``weights``."""

    if inputs.ndim < 2:
        raise ValueError('inputs must have shape (S, *input_shape)')

    input_shape = inputs.shape[1:]

    paired = paired_from_samples(
        left=inputs.reshape(
            inputs.shape[0],
            -1,
        ),
        right=outputs,
        weights=weights,
    )

    return linear_from_paired(
        paired,
        ridge=ridge,
        covariance_floor=covariance_floor,
    ).reshape_input(input_shape)


def linear_from_samples_grouped(
    inputs: jax.Array,  # (S, *input_shape)
    outputs: jax.Array,  # (S, O)
    assignments: jax.Array,  # (S,)
    num_groups: int,
    *,
    ridge: float,
    covariance_floor: float,
) -> DiagonalLinearGaussian:
    """Fit one diagonal linear Gaussian to each assigned group."""

    num_samples = inputs.shape[0]
    if assignments.ndim != 1:
        raise ValueError('assignments must have shape (S,)')
    if assignments.shape[0] != num_samples:
        raise ValueError(
            'assignments must contain one entry per sample; '
            f'expected {num_samples}, got {assignments.shape[0]}'
        )

    weights = gaussian_common_fit.group_weights(
        assignments, num_groups, jnp.result_type(inputs, outputs, jnp.float32)
    )  # (K, S)

    return linear_from_samples_weighted(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        ridge=ridge,
        covariance_floor=covariance_floor,
    )
