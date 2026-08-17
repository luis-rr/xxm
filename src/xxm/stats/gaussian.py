import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp


class LinearGaussianFit(typing.NamedTuple):
    """Parameters of a fitted linear Gaussian model."""

    coefficients: jax.Array
    bias: jax.Array
    covariance: jax.Array


def log_likelihoods(
    observations: jax.Array,  # (T, N)
    means: jax.Array,  # (1 or T, K, N)
    covariances: jax.Array,  # (K, N, N)
) -> jax.Array:
    """Gaussian log likelihood for each time and state."""
    residuals = observations[:, None, :] - means  # (T, K, N)

    chol = jnp.linalg.cholesky(covariances)  # (K, N, N)

    solved = jsp.linalg.solve_triangular(
        chol[None, :, :, :],
        residuals[..., None],
        lower=True,
    )[..., 0]  # (T, K, N)

    mahalanobis = jnp.sum(solved**2, axis=-1)  # (T, K)

    log_det = 2 * jnp.sum(
        jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)),
        axis=-1,
    )  # (K,)

    n_dims = observations.shape[-1]

    return -0.5 * (n_dims * jnp.log(2 * jnp.pi) + log_det[None, :] + mahalanobis)


def fit_weighted(
    observations: jax.Array,  # (T, N)
    weights: jax.Array,  # (T, K)
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array]:
    """Fit weighted Gaussian distributions."""
    counts = jnp.maximum(weights.sum(axis=0), eps)  # (K,)

    means = weights.T @ observations / counts[:, None]  # (K, N)

    residuals = observations[:, None, :] - means[None, :, :]  # (T, K, N)

    covariances = (
        jnp.einsum(
            'tk,tki,tkj->kij',
            weights,
            residuals,
            residuals,
        )
        / counts[:, None, None]
    )  # (K, N, N)

    return means, covariances


def fit_linear_from_moments(
    input_mean: jax.Array,
    output_mean: jax.Array,
    input_second_moment: jax.Array,
    output_second_moment: jax.Array,
    output_input_moment: jax.Array,
    ridge: float = 0.0,
) -> LinearGaussianFit:
    r"""Fit the linear Gaussian model

        y = A x + b + \epsilon,    \epsilon ~ N(0, \Sigma),

    from raw moments averaged over samples.

    Parameters are E[x], E[y], E[xxᵀ], E[yyᵀ], and E[yxᵀ],
    where the expectation includes both posterior uncertainty and
    averaging over samples. Leading batch dimensions are supported.
    """
    input_covariance = input_second_moment - input_mean[..., :, None] * input_mean[..., None, :]

    output_input_covariance = (
        output_input_moment - output_mean[..., :, None] * input_mean[..., None, :]
    )

    output_covariance = output_second_moment - output_mean[..., :, None] * output_mean[..., None, :]

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

    bias = output_mean - jnp.einsum('...np,...p->...n', coefficients, input_mean)

    noise_covariance = (
        output_covariance
        - coefficients @ jnp.swapaxes(output_input_covariance, -2, -1)
        - output_input_covariance @ jnp.swapaxes(coefficients, -2, -1)
        + coefficients @ input_covariance @ jnp.swapaxes(coefficients, -2, -1)
    )
    noise_covariance = 0.5 * (noise_covariance + jnp.swapaxes(noise_covariance, -2, -1))

    return LinearGaussianFit(
        coefficients=coefficients,
        bias=bias,
        covariance=noise_covariance,
    )


def fit_linear(
    inputs: jax.Array,
    outputs: jax.Array,
    ridge: float = 0.0,
) -> LinearGaussianFit:
    """Fit y = A x + b + noise from paired samples."""
    n = inputs.shape[0]

    return fit_linear_from_moments(
        input_mean=jnp.mean(inputs, axis=0),
        output_mean=jnp.mean(outputs, axis=0),
        input_second_moment=inputs.T @ inputs / n,
        output_second_moment=outputs.T @ outputs / n,
        output_input_moment=outputs.T @ inputs / n,
        ridge=ridge,
    )


def fit_weighted_linear(
    inputs: jax.Array,  # (T, P)
    outputs: jax.Array,  # (T, N)
    weights: jax.Array,  # (T, K)
    eps: float = 1e-8,
    ridge: float = 0.0,
) -> LinearGaussianFit:
    """Fit weighted linear Gaussian models y = A x + b + noise."""
    counts = jnp.maximum(weights.sum(axis=0), eps)  # (K,)

    input_means = jnp.einsum('tk,tp->kp', weights, inputs) / counts[:, None]
    output_means = jnp.einsum('tk,tn->kn', weights, outputs) / counts[:, None]

    input_second_moments = (
        jnp.einsum('tk,tp,tq->kpq', weights, inputs, inputs) / counts[:, None, None]
    )
    output_second_moments = (
        jnp.einsum('tk,tn,tm->knm', weights, outputs, outputs) / counts[:, None, None]
    )
    output_input_moments = (
        jnp.einsum('tk,tn,tp->knp', weights, outputs, inputs) / counts[:, None, None]
    )

    return fit_linear_from_moments(
        input_mean=input_means,
        output_mean=output_means,
        input_second_moment=input_second_moments,
        output_second_moment=output_second_moments,
        output_input_moment=output_input_moments,
        ridge=ridge,
    )
