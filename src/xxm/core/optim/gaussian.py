"""Moment-matching and weighted least-squares fitting for Gaussian models."""

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import (
    Gaussian,
    LinearGaussian,
    PairedGaussian,
)

DEFAULT_RIDGE = 1e-6  # package-wide default for relative ridge regularization

# package-wide default for relative covariance floor regularization at fitting
DEFAULT_COV_FLOOR = 1e-6

# package-wide default for relative covariance floor regularization at
#  initialization, which is more conservative than the default for fitting
DEFAULT_COV_FLOOR_INIT = 1e-2


def add_covariance_floor(
    covariance: jax.Array,
    *,
    reference_covariance: jax.Array,  # might be batched
    covariance_floor: float,
) -> jax.Array:
    """
    Add an isotropic covariance floor relative to a reference variance scale.

    `covariance_floor` is a dimensionless fraction of the average marginal
    variance of `reference_covariance`.
    """
    covariance = 0.5 * (
        covariance
        + jnp.swapaxes(
            covariance,
            -2,
            -1,
        )
    )

    scale = jnp.mean(
        jnp.diagonal(
            reference_covariance,
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

    return covariance + (covariance_floor * scale[..., None, None] * identity)


def _normalize_weights(
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


def _center_samples(
    values: jax.Array,  # (S, N)
    mean: jax.Array,  # (..., N)
) -> jax.Array:  # (..., S, N)
    """Center samples around one or more means."""
    batch_ndim = mean.ndim - 1

    values = values.reshape((1,) * batch_ndim + values.shape)

    return values - mean[..., None, :]


def _from_samples_normalized(
    values: jax.Array,  # (S, N)
    weights: jax.Array,  # (..., S)
) -> Gaussian:
    """Fit a Gaussian using already-normalized weights."""
    mean = jnp.einsum(
        '...t,ti->...i',
        weights,
        values,
    )

    residuals = _center_samples(
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

    return Gaussian(
        mean=mean,
        covariance=covariance,
    )


def from_samples(
    values: jax.Array,  # (S, N)
    *,
    covariance_floor: float,
) -> Gaussian:
    """Fit a Gaussian from samples along the first axis."""
    dtype = jnp.result_type(
        values,
        jnp.float32,
    )

    values = values.astype(dtype)
    weights = _normalize_weights(
        num_samples=values.shape[0],
        weights=None,
        dtype=dtype,
    )

    fitted = _from_samples_normalized(
        values=values,
        weights=weights,
    )

    return fitted._replace(
        covariance=add_covariance_floor(
            fitted.covariance,
            reference_covariance=fitted.covariance,
            covariance_floor=covariance_floor,
        ),
    )


def from_samples_weighted(
    values: jax.Array,  # (S, N)
    weights: jax.Array,  # (..., S)
    *,
    covariance_floor: float,
) -> Gaussian:
    """Fit one weighted Gaussian for each batch entry of ``weights``."""
    dtype = jnp.result_type(
        values,
        weights,
        jnp.float32,
    )
    values = values.astype(dtype)

    weights = _normalize_weights(
        num_samples=values.shape[0],
        weights=weights,
        dtype=dtype,
    )

    fitted = _from_samples_normalized(
        values=values,
        weights=weights,
    )

    return fitted._replace(
        covariance=add_covariance_floor(
            fitted.covariance,
            reference_covariance=fitted.covariance,
            covariance_floor=covariance_floor,
        ),
    )


def from_samples_grouped(
    values: jax.Array,  # (S, N)
    assignments: jax.Array,  # (S,)
    num_groups: int,
    *,
    covariance_floor: float,
) -> Gaussian:
    """Fit one Gaussian to each group of assigned values."""
    num_samples = values.shape[0]
    if assignments.ndim != 1:
        raise ValueError('assignments must have shape (S,)')
    if assignments.shape[0] != num_samples:
        raise ValueError(
            'assignments must contain one entry per sample; '
            f'expected {num_samples}, got {assignments.shape[0]}'
        )

    weights = jnp.moveaxis(
        jax.nn.one_hot(
            assignments,
            num_groups,
            dtype=jnp.result_type(
                values,
                jnp.float32,
            ),
        ),
        -1,
        0,
    )

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
    return _normalize_weights(batch_shape[-1], weights, dtype)


def _from_moment_match_weighted(
    distributions: Gaussian,
    normalized_weights: jax.Array,
) -> Gaussian:
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
    return Gaussian(mean=matched_mean, covariance=covariance)


def from_moment_match(
    distributions: Gaussian,
    weights: jax.Array | None = None,
) -> Gaussian:
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

    return _from_moment_match_weighted(
        distributions=distributions,
        normalized_weights=normalized_weights,
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

    joint = _from_moment_match_weighted(
        distributions=Gaussian(
            mean=distributions.mean,
            covariance=distributions.covariance,
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
        left=Gaussian(
            mean=joint.mean[..., :left_dim],
            covariance=covariance[
                ...,
                :left_dim,
                :left_dim,
            ],
        ),
        right=Gaussian(
            mean=joint.mean[..., left_dim:],
            covariance=covariance[
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

    normalized_weights = _normalize_weights(
        num_samples=num_samples,
        weights=weights,
        dtype=dtype,
    )

    left_distribution = _from_samples_normalized(
        values=left,
        weights=normalized_weights,
    )

    right_distribution = _from_samples_normalized(
        values=right,
        weights=normalized_weights,
    )

    left_residuals = _center_samples(
        left,
        left_distribution.mean,
    )

    right_residuals = _center_samples(
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
    left: Gaussian,
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

    normalized_weights = _normalize_weights(
        num_samples=left.batch_shape[0],
        weights=weights,
        dtype=dtype,
    )

    aggregated_left = _from_moment_match_weighted(
        distributions=left,
        normalized_weights=normalized_weights,
    )

    aggregated_right = _from_samples_normalized(
        values=right,
        weights=normalized_weights,
    )

    left_residuals = _center_samples(
        left.mean,
        aggregated_left.mean,
    )

    right_residuals = _center_samples(
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


def affine_from_paired(
    paired: PairedGaussian,
    *,
    ridge: float,
) -> Affine:
    r"""Return the affine least-squares map from ``left`` to ``right``.

    ``ridge`` adds isotropic regularization to the left covariance relative
    to its average marginal variance.
    """
    input_covariance = _add_relative_ridge(
        paired.left.covariance,
        ridge,
    )

    coefficients = jnp.linalg.solve(
        input_covariance,
        jnp.swapaxes(
            paired.cross_covariance,
            -2,
            -1,
        ),
    )

    coefficients = jnp.swapaxes(
        coefficients,
        -2,
        -1,
    )

    bias = paired.right.mean - jnp.einsum(
        '...oi,...i->...o',
        coefficients,
        paired.left.mean,
    )

    return Affine(
        coefficients=coefficients,
        bias=bias,
    )


def linear_from_paired(
    paired: PairedGaussian,
    *,
    ridge: float,
    covariance_floor: float,
) -> LinearGaussian:
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

    covariance = add_covariance_floor(
        covariance,
        reference_covariance=paired.right.covariance,
        covariance_floor=covariance_floor,
    )

    covariance = 0.5 * (
        covariance
        + jnp.swapaxes(
            covariance,
            -2,
            -1,
        )
    )

    return LinearGaussian(
        affine=affine,
        covariance=covariance,
    )


def linear_from_marginals(
    inputs: Gaussian,
    outputs: jax.Array,
    *,
    weights: jax.Array | None = None,
    ridge: float,
    covariance_floor: float,
) -> LinearGaussian:
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
) -> LinearGaussian:
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
) -> LinearGaussian:
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
) -> LinearGaussian:
    """Fit one linear Gaussian to each assigned group."""

    num_samples = inputs.shape[0]
    if assignments.ndim != 1:
        raise ValueError('assignments must have shape (S,)')
    if assignments.shape[0] != num_samples:
        raise ValueError(
            'assignments must contain one entry per sample; '
            f'expected {num_samples}, got {assignments.shape[0]}'
        )

    weights = jnp.moveaxis(
        jax.nn.one_hot(
            assignments,
            num_groups,
            dtype=jnp.result_type(
                inputs,
                outputs,
                jnp.float32,
            ),
        ),
        -1,
        0,
    )  # (K, S)

    return linear_from_samples_weighted(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        ridge=ridge,
        covariance_floor=covariance_floor,
    )
