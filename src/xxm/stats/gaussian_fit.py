import jax
import jax.numpy as jnp

from xxm.stats.gaussian import Affine, Gaussian, LinearGaussian

EPS: float = 1e-8


def gaussian_from_pairs(
    observations: jax.Array,  # (T, N)
) -> Gaussian:
    """Fit a Gaussian from samples along the first axis."""
    mean = jnp.mean(observations, axis=0)
    residuals = observations - mean
    covariance = (
        jnp.einsum(
            't...i,t...j->...ij',
            residuals,
            residuals,
        )
        / observations.shape[0]
    )

    return Gaussian(mean=mean, covariance=covariance)


def gaussian_from_pairs_weighted(
    observations: jax.Array,  # (T, N)
    weights: jax.Array,  # (T, ...)
) -> Gaussian:
    """Fit one weighted Gaussian for each batch entry of ``weights``."""
    total = jnp.sum(weights, axis=0)
    counts = jnp.where(total > 0, total, EPS)
    normalized = weights / counts[None, ...]

    mean = jnp.einsum(
        't...,tn->...n',
        normalized,
        observations,
    )
    second_moment = jnp.einsum(
        't...,ti,tj->...ij',
        normalized,
        observations,
        observations,
    )

    covariance = second_moment - mean[..., :, None] * mean[..., None, :]
    covariance = 0.5 * (covariance + jnp.swapaxes(covariance, -2, -1))

    return Gaussian(mean=mean, covariance=covariance)


def gaussian_from_pairs_grouped(
    observations: jax.Array,  # (T, N)
    assignments: jax.Array,  # (T,)
    num_groups: int,
) -> Gaussian:  # K-batched
    """Fit one Gaussian to each group of assigned observations."""
    weights = jax.nn.one_hot(
        assignments,
        num_groups,
        dtype=observations.dtype,
    )  # (T, K)

    return gaussian_from_pairs_weighted(
        observations=observations,
        weights=weights,
    )


def linear_from_moments(
    input_mean: jax.Array,
    output_mean: jax.Array,
    input_second_moment: jax.Array,
    output_second_moment: jax.Array,
    output_input_moment: jax.Array,
    ridge: float = 0.0,
) -> LinearGaussian:
    r"""Fit from E[x], E[y], E[xxᵀ], E[yyᵀ], and E[yxᵀ].

    Expectations may include both posterior uncertainty and averaging
    over samples. Leading batch dimensions are supported.
    """
    input_covariance = input_second_moment - input_mean[..., :, None] * input_mean[..., None, :]
    output_covariance = output_second_moment - output_mean[..., :, None] * output_mean[..., None, :]
    output_input_covariance = (
        output_input_moment - output_mean[..., :, None] * input_mean[..., None, :]
    )

    identity = jnp.eye(
        input_covariance.shape[-1],
        dtype=input_covariance.dtype,
    )
    regularized_input_covariance = input_covariance + ridge * identity

    coefficients = jnp.linalg.solve(
        regularized_input_covariance,
        jnp.swapaxes(output_input_covariance, -2, -1),
    )
    coefficients = jnp.swapaxes(coefficients, -2, -1)

    bias = output_mean - jnp.einsum(
        '...oi,...i->...o',
        coefficients,
        input_mean,
    )

    noise_covariance = (
        output_covariance
        - coefficients @ jnp.swapaxes(output_input_covariance, -2, -1)
        - output_input_covariance @ jnp.swapaxes(coefficients, -2, -1)
        + coefficients @ input_covariance @ jnp.swapaxes(coefficients, -2, -1)
    )
    noise_covariance = 0.5 * (noise_covariance + jnp.swapaxes(noise_covariance, -2, -1))

    return LinearGaussian(
        affine=Affine(
            coefficients=coefficients,
            bias=bias,
        ),
        covariance=noise_covariance,
    )


def linear_from_pairs(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    ridge: float = 0.0,
) -> 'LinearGaussian':
    """Fit ``y = A x + b + noise`` from paired samples."""
    n = inputs.shape[0]

    return linear_from_moments(
        input_mean=jnp.mean(inputs, axis=0),
        output_mean=jnp.mean(outputs, axis=0),
        input_second_moment=jnp.einsum('t...i,t...j->...ij', inputs, inputs) / n,
        output_second_moment=jnp.einsum('t...o,t...p->...op', outputs, outputs) / n,
        output_input_moment=jnp.einsum('t...o,t...i->...oi', outputs, inputs) / n,
        ridge=ridge,
    )


def linear_from_pairs_weighted(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    weights: jax.Array,  # (T, ...)
    ridge: float = 0.0,
) -> 'LinearGaussian':
    """Fit one weighted model for each batch entry of ``weights``."""
    total = jnp.sum(weights, axis=0)
    counts = jnp.where(total > 0, total, EPS)
    normalized = weights / counts[None, ...]

    return linear_from_moments(
        input_mean=jnp.einsum('t...,ti->...i', normalized, inputs),
        output_mean=jnp.einsum('t...,to->...o', normalized, outputs),
        input_second_moment=jnp.einsum('t...,ti,tj->...ij', normalized, inputs, inputs),
        output_second_moment=jnp.einsum('t...,to,tp->...op', normalized, outputs, outputs),
        output_input_moment=jnp.einsum('t...,to,ti->...oi', normalized, outputs, inputs),
        ridge=ridge,
    )


def linear_from_pairs_grouped(
    inputs: jax.Array,  # (T, I)
    outputs: jax.Array,  # (T, O)
    assignments: jax.Array,  # (T,)
    num_groups: int,
    ridge: float = 0.0,
) -> 'LinearGaussian':
    """Fit one model to each assigned group."""
    weights = jax.nn.one_hot(
        assignments,
        num_groups,
        dtype=jnp.result_type(inputs, outputs, jnp.float32),
    )  # (T, K)

    return linear_from_pairs_weighted(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        ridge=ridge,
    )
