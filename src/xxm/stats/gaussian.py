import jax
import jax.numpy as jnp


def fit_linear_from_moments(
    input_mean: jax.Array,
    output_mean: jax.Array,
    input_second_moment: jax.Array,
    output_second_moment: jax.Array,
    output_input_moment: jax.Array,
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

    matrix = jnp.linalg.solve(
        covariance_xx,
        covariance_yx.T,
    ).T

    bias = output_mean - matrix @ input_mean

    noise_covariance = covariance_yy - matrix @ covariance_yx.T
    noise_covariance = 0.5 * (noise_covariance + noise_covariance.T)

    return matrix, bias, noise_covariance


def fit_linear(
    inputs: jax.Array,
    outputs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit y = A x + b + noise from paired samples."""
    n = inputs.shape[0]

    return fit_linear_from_moments(
        input_mean=jnp.mean(inputs, axis=0),
        output_mean=jnp.mean(outputs, axis=0),
        input_second_moment=inputs.T @ inputs / n,
        output_second_moment=outputs.T @ outputs / n,
        output_input_moment=outputs.T @ inputs / n,
    )
