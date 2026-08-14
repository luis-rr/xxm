import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg


def log_likelihoods(
    observations: jax.Array,  # (T, N)
    means: jax.Array,  # (1 or T, K, N)
    covariances: jax.Array,  # (K, N, N)
) -> jax.Array:
    residuals = observations[:, None, :] - means  # (T, K, N)

    chol = jnp.linalg.cholesky(covariances)  # (K, N, N)

    solved = jsp_linalg.solve_triangular(
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""Fit the linear Gaussian model

        y = A x + b + \epsilon,    \epsilon ~ N(0, \Sigma),

    from raw moments averaged over samples.

    Parameters are E[x], E[y], E[xxᵀ], E[yyᵀ], and E[yxᵀ],
    where the expectation includes both posterior uncertainty and
    averaging over samples.

    Returns A, b, and \Sigma.
    """
    covariance_xx = input_second_moment - jnp.outer(input_mean, input_mean)
    covariance_yx = output_input_moment - jnp.outer(output_mean, input_mean)
    covariance_yy = output_second_moment - jnp.outer(output_mean, output_mean)

    covariance_xx = covariance_xx + ridge * jnp.eye(
        covariance_xx.shape[-1],
        dtype=covariance_xx.dtype,
    )

    coefficients = jnp.linalg.solve(
        covariance_xx,
        covariance_yx.T,
    ).T

    bias = output_mean - coefficients @ input_mean

    noise_covariance = (
        covariance_yy
        - coefficients @ covariance_yx.T
        - covariance_yx @ coefficients.T
        + coefficients @ covariance_xx @ coefficients.T
    )
    noise_covariance = 0.5 * (noise_covariance + noise_covariance.T)

    return coefficients, bias, noise_covariance


def fit_linear(
    inputs: jax.Array,
    outputs: jax.Array,
    ridge: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit weighted linear Gaussian models y = A x + b + noise."""
    counts = jnp.maximum(weights.sum(axis=0), eps)  # (K,)

    input_means = jnp.einsum('tk,tp->kp', weights, inputs) / counts[:, None]  # (K, P)

    output_means = jnp.einsum('tk,tn->kn', weights, outputs) / counts[:, None]  # (K, N)

    input_second_moments = (
        jnp.einsum('tk,tp,tq->kpq', weights, inputs, inputs) / counts[:, None, None]
    )  # (K, P, P)

    output_second_moments = (
        jnp.einsum('tk,tn,tm->knm', weights, outputs, outputs) / counts[:, None, None]
    )  # (K, N, N)

    output_input_moments = (
        jnp.einsum('tk,tn,tp->knp', weights, outputs, inputs) / counts[:, None, None]
    )  # (K, N, P)

    return jax.vmap(
        fit_linear_from_moments,
        in_axes=(0, 0, 0, 0, 0, None),
    )(
        input_means,
        output_means,
        input_second_moments,
        output_second_moments,
        output_input_moments,
        ridge,
    )
