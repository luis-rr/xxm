"""Moment-matching and weighted least-squares fitting with full covariances."""

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import (
    FullGaussian,
    FullGaussianNoise,
    FullLinearGaussian,
    PairedGaussian,
)
from xxm.core.optim import gaussian_common as gaussian_common_fit


def from_samples(
    values: jax.Array,  # (S, N)
    *,
    covariance_floor: float,
) -> FullGaussian:
    """Fit a Gaussian from samples along the first axis."""
    dtype = jnp.result_type(
        values,
        jnp.float32,
    )

    values = values.astype(dtype)
    weights = gaussian_common_fit.normalize_weights(
        num_samples=values.shape[0],
        weights=None,
        dtype=dtype,
    )

    fitted = gaussian_common_fit.from_samples_normalized(
        values=values,
        weights=weights,
    )

    return fitted._replace(noise=fitted.noise.with_floor(covariance_floor))


def from_samples_weighted(
    values: jax.Array,  # (S, N)
    weights: jax.Array,  # (..., S)
    *,
    covariance_floor: float,
) -> FullGaussian:
    """Fit one weighted Gaussian for each batch entry of ``weights``."""
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

    fitted = gaussian_common_fit.from_samples_normalized(
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
) -> FullGaussian:
    """Fit one Gaussian to each weighted group of assigned values."""
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


def _normalize_moment_weights(
    batch_shape: tuple[int, ...],
    weights: jax.Array | None,
    dtype: jax.typing.DTypeLike,
) -> jax.Array:
    """Align weights with structural batches and normalize the final sample axis."""
    assert batch_shape
    structural_shape = batch_shape[:-1]
    if weights is None:
        weights = jnp.ones(batch_shape, dtype=dtype)
    if (
        weights.ndim < len(batch_shape)
        or weights.shape[: len(structural_shape)] != structural_shape
    ):
        raise ValueError(
            'weights must have shape (*B, *Q, S) aligned with distributions'
        )
    return gaussian_common_fit.normalize_weights(batch_shape[-1], weights, dtype)


def from_moment_match(
    distributions: FullGaussian,
    weights: jax.Array | None = None,
) -> FullGaussian:
    """
    Moment-match Gaussian distributions along the final batch axis.

    The returned covariance combines average within-distribution
    covariance with covariance across the distribution means.
    Distributions have batch shape ``(*B, S)``. Weights have shape
    ``(*B, *Q, S)`` and the result has batch shape ``(*B, *Q)``.
    With no weights, the result batch shape is ``*B``.
    """
    assert distributions.batch_shape

    dtype = jnp.result_type(
        distributions.dtype,
        weights if weights is not None else jnp.float32,
        jnp.float32,
    )

    distributions = distributions.astype(dtype)

    normalized_weights = _normalize_moment_weights(
        batch_shape=distributions.batch_shape,
        weights=weights,
        dtype=dtype,
    )

    return gaussian_common_fit.from_moment_match_weighted(
        distributions=distributions,
        normalized_weights=normalized_weights,
    )


def paired_from_paired_covariates(
    paired: PairedGaussian,
    covariates: jax.Array,
    *,
    weights: jax.Array | None = None,
) -> PairedGaussian:
    """
    Moment-match a Gaussian pair with deterministic left-side covariates.

    The returned left variable is `[paired.left, covariates]` and the right
    variable is `paired.right`.
    """
    matched = paired_from_moment_match(
        paired,
        weights=weights,
    )

    left_covariates = paired_from_left_marginals(
        paired.left,
        covariates,
        weights=weights,
    )

    right_covariates = paired_from_left_marginals(
        paired.right,
        covariates,
        weights=weights,
    )

    left = FullGaussian.from_moments(left_covariates.mean, left_covariates.covariance)

    cross_covariance = jnp.concatenate(
        [
            matched.cross_covariance,
            jnp.swapaxes(right_covariates.cross_covariance, -1, -2),
        ],
        axis=-1,
    )

    return PairedGaussian(
        left=left,
        right=matched.right,
        cross_covariance=cross_covariance,
    )


def linear_from_paired_covariates(
    paired: PairedGaussian,
    covariates: jax.Array,
    *,
    weights: jax.Array | None = None,
    ridge: float,
    covariance_floor: float,
) -> FullLinearGaussian:
    """
    Fit a linear Gaussian using uncertain inputs and deterministic covariates.
    """
    paired = paired_from_paired_covariates(
        paired,
        covariates,
        weights=weights,
    )

    return linear_from_paired(
        paired,
        ridge=ridge,
        covariance_floor=covariance_floor,
    )


def paired_from_moment_match(
    distributions: PairedGaussian,
    weights: jax.Array | None = None,
) -> PairedGaussian:
    """
    Moment-match paired Gaussian distributions along the final batch axis.

    Distributions have batch shape ``(*B, S)``; weights may have shape
    ``(*B, *Q, S)``, producing result batch ``(*B, *Q)``. With no weights,
    the result batch shape is ``*B``.

    The pair is moment-matched jointly to preserve the covariance structure.
    Eigenvalues below floating-point resolution are floored to keep the
    resulting covariance numerically positive definite.
    """
    assert distributions.batch_shape

    dtype = jnp.result_type(
        distributions.dtype,
        weights if weights is not None else jnp.float32,
        jnp.float32,
    )

    distributions = distributions.astype(dtype)

    normalized_weights = _normalize_moment_weights(
        batch_shape=distributions.batch_shape,
        weights=weights,
        dtype=dtype,
    )

    joint = gaussian_common_fit.from_moment_match_weighted(
        distributions=FullGaussian.from_moments(
            distributions.mean, distributions.covariance
        ),
        normalized_weights=normalized_weights,
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(
        joint.covariance,
    )

    scale = jnp.max(
        jnp.abs(eigenvalues),
        axis=-1,
    )

    floor = distributions.variable_dim * jnp.finfo(dtype).eps * scale

    eigenvalues = jnp.maximum(
        eigenvalues,
        floor[..., None],
    )

    covariance = (eigenvectors * eigenvalues[..., None, :]) @ jnp.swapaxes(
        eigenvectors,
        -2,
        -1,
    )

    covariance = 0.5 * (
        covariance
        + jnp.swapaxes(
            covariance,
            -2,
            -1,
        )
    )

    left_dim = distributions.left_dim

    return PairedGaussian(
        left=FullGaussian.from_moments(
            joint.mean[..., :left_dim],
            covariance[
                ...,
                :left_dim,
                :left_dim,
            ],
        ),
        right=FullGaussian.from_moments(
            joint.mean[..., left_dim:],
            covariance[
                ...,
                left_dim:,
                left_dim:,
            ],
        ),
        cross_covariance=covariance[
            ...,
            left_dim:,
            :left_dim,
        ],
    )


def paired_from_samples(
    left: jax.Array,  # (S, L)
    right: jax.Array,  # (S, R)
    *,
    weights: jax.Array | None = None,  # (..., S)
) -> PairedGaussian:
    """Fit a paired Gaussian from aligned left and right samples."""
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

    right_distribution = gaussian_common_fit.from_samples_normalized(
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

    return PairedGaussian(
        left=left_distribution,
        right=right_distribution,
        cross_covariance=cross_covariance,
    )


def paired_from_left_marginals(
    left: FullGaussian,
    right: jax.Array,
    *,
    weights: jax.Array | None = None,
) -> PairedGaussian:
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

    aggregated_right = gaussian_common_fit.from_samples_normalized(
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

    return PairedGaussian(
        left=aggregated_left,
        right=aggregated_right,
        cross_covariance=cross_covariance,
    )


def affine_from_paired(
    paired: PairedGaussian,
    *,
    ridge: float,
) -> Affine:
    r"""Return the affine least-squares map from ``left`` to ``right``.

    ``ridge`` adds isotropic regularization to the left covariance relative
    to its average marginal variance.
    """
    return gaussian_common_fit.affine_from_moments(
        paired.left,
        paired.cross_covariance,
        paired.right.mean,
        ridge=ridge,
    )


def noise_from_paired(
    paired: PairedGaussian,
    affine: Affine,
    *,
    covariance_floor: float,
) -> FullGaussianNoise:
    r"""
    Return the residual noise of ``right | left`` under a fitted ``affine``.

    The covariance is that of $v - Wu - b$ under the paired moments, with an
    isotropic floor relative to the marginal output covariance.
    """
    coefficients = affine.coefficients

    coefficients_t = jnp.swapaxes(
        coefficients,
        -2,
        -1,
    )

    cross_covariance_t = jnp.swapaxes(
        paired.cross_covariance,
        -2,
        -1,
    )

    covariance = (
        paired.right.covariance
        - coefficients @ cross_covariance_t
        - paired.cross_covariance @ coefficients_t
        + coefficients @ paired.left.covariance @ coefficients_t
    )

    return FullGaussianNoise(covariance).with_floor(
        covariance_floor, reference=paired.right.noise
    )


def linear_from_paired(
    paired: PairedGaussian,
    *,
    ridge: float,
    covariance_floor: float,
) -> FullLinearGaussian:
    r"""
    Return the linear-Gaussian distribution of ``right | left``.

    ``ridge`` adds isotropic regularization to the left covariance relative
    to its average marginal variance. With nonzero ridge, the result is a
    regularized approximation to the exact Gaussian conditional.
    """
    affine = affine_from_paired(
        paired,
        ridge=ridge,
    )

    return FullLinearGaussian(
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
) -> FullLinearGaussian:
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
) -> FullLinearGaussian:
    """Fit a linear Gaussian model from samples."""

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
) -> FullLinearGaussian:
    """Fit one weighted model for each batch entry of ``weights``."""

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
) -> FullLinearGaussian:
    """Fit one linear Gaussian to each assigned group."""

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
