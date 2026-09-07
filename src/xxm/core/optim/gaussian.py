import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian

EPS: float = 1e-8


def from_samples(
    values: jax.Array,  # (T, N)
) -> Gaussian:
    """Fit a Gaussian from samples along the first axis."""
    mean = jnp.mean(values, axis=0)
    residuals = values - mean
    covariance = (
        jnp.einsum(
            't...i,t...j->...ij',
            residuals,
            residuals,
        )
        / values.shape[0]
    )

    return Gaussian(mean=mean, covariance=covariance)


def from_samples_weighted(
    values: jax.Array,
    weights: jax.Array,
) -> Gaussian:
    """Fit one weighted Gaussian for each batch entry of ``weights``."""
    total = jnp.sum(weights, axis=0)
    counts = jnp.where(total > 0, total, EPS)
    normalized = weights / counts[None, ...]

    mean = jnp.einsum(
        't...,tn->...n',
        normalized,
        values,
    )

    batch_ndim = weights.ndim - 1

    expanded_values = values.reshape(
        (values.shape[0],) + (1,) * batch_ndim + (values.shape[-1],)
    )

    residuals = expanded_values - mean[None, ...]

    covariance = jnp.einsum(
        't...,t...i,t...j->...ij',
        normalized,
        residuals,
        residuals,
    )

    return Gaussian(
        mean=mean,
        covariance=covariance,
    )


def from_samples_grouped(
    values: jax.Array,  # (T, N)
    assignments: jax.Array,  # (T,)
    num_groups: int,
) -> Gaussian:  # K-batched
    """Fit one Gaussian to each group of assigned values."""
    weights = jax.nn.one_hot(
        assignments,
        num_groups,
        dtype=values.dtype,
    )  # (T, K)

    return from_samples_weighted(
        values=values,
        weights=weights,
    )


def linear_from_centered_moments(
    input_mean: jax.Array,
    output_mean: jax.Array,
    input_covariance: jax.Array,
    output_covariance: jax.Array,
    output_input_covariance: jax.Array,
    ridge: float = 0.0,
) -> LinearGaussian:
    r"""
    Fit a linear Gaussian from means and centered second moments.

    Fits

        y | x ~ N(A x + b, Q)

    from E[x], E[y], Cov(x), Cov(y), and Cov(y, x).
    Leading batch dimensions are supported.
    """
    identity = jnp.eye(
        input_covariance.shape[-1],
        dtype=input_covariance.dtype,
    )

    regularized_input_covariance = input_covariance + ridge * identity

    coefficients = jnp.linalg.solve(
        regularized_input_covariance,
        jnp.swapaxes(
            output_input_covariance,
            -2,
            -1,
        ),
    )
    coefficients = jnp.swapaxes(
        coefficients,
        -2,
        -1,
    )

    bias = output_mean - jnp.einsum(
        '...oi,...i->...o',
        coefficients,
        input_mean,
    )

    noise_covariance = (
        output_covariance
        - coefficients
        @ jnp.swapaxes(
            output_input_covariance,
            -2,
            -1,
        )
        - output_input_covariance
        @ jnp.swapaxes(
            coefficients,
            -2,
            -1,
        )
        + coefficients
        @ input_covariance
        @ jnp.swapaxes(
            coefficients,
            -2,
            -1,
        )
    )

    noise_covariance = 0.5 * (
        noise_covariance
        + jnp.swapaxes(
            noise_covariance,
            -2,
            -1,
        )
    )

    return LinearGaussian(
        affine=Affine(
            coefficients=coefficients,
            bias=bias,
        ),
        covariance=noise_covariance,
    )


def linear_from_samples(
    inputs: jax.Array,
    outputs: jax.Array,
    ridge: float = 0.0,
) -> LinearGaussian:
    """Fit a linear Gaussian model from paired samples."""
    if inputs.ndim < 2:
        raise ValueError('inputs must have shape (T, *input_shape)')

    input_shape = inputs.shape[1:]
    flat_inputs = inputs.reshape(
        inputs.shape[0],
        -1,
    )

    input_mean = jnp.mean(flat_inputs, axis=0)
    output_mean = jnp.mean(outputs, axis=0)

    input_residuals = flat_inputs - input_mean
    output_residuals = outputs - output_mean

    num_samples = inputs.shape[0]

    model = linear_from_centered_moments(
        input_mean=input_mean,
        output_mean=output_mean,
        input_covariance=(input_residuals.T @ input_residuals / num_samples),
        output_covariance=(output_residuals.T @ output_residuals / num_samples),
        output_input_covariance=(output_residuals.T @ input_residuals / num_samples),
        ridge=ridge,
    )

    return model.reshape_input(input_shape)


def linear_from_samples_weighted(
    inputs: jax.Array,
    outputs: jax.Array,
    weights: jax.Array,
    ridge: float = 0.0,
) -> LinearGaussian:
    """Fit one weighted model for each batch entry of ``weights``."""
    if inputs.ndim < 2:
        raise ValueError('inputs must have shape (T, *input_shape)')

    input_shape = inputs.shape[1:]
    flat_inputs = inputs.reshape(
        inputs.shape[0],
        -1,
    )

    total = jnp.sum(weights, axis=0)
    counts = jnp.where(total > 0, total, EPS)
    normalized = weights / counts[None, ...]

    input_mean = jnp.einsum(
        't...,ti->...i',
        normalized,
        flat_inputs,
    )
    output_mean = jnp.einsum(
        't...,to->...o',
        normalized,
        outputs,
    )

    batch_ndim = weights.ndim - 1

    expanded_inputs = flat_inputs.reshape(
        (flat_inputs.shape[0],) + (1,) * batch_ndim + (flat_inputs.shape[-1],)
    )

    expanded_outputs = outputs.reshape(
        (outputs.shape[0],) + (1,) * batch_ndim + (outputs.shape[-1],)
    )

    input_residuals = expanded_inputs - input_mean[None, ...]
    output_residuals = expanded_outputs - output_mean[None, ...]

    input_covariance = jnp.einsum(
        't...,t...i,t...j->...ij',
        normalized,
        input_residuals,
        input_residuals,
    )

    output_covariance = jnp.einsum(
        't...,t...o,t...p->...op',
        normalized,
        output_residuals,
        output_residuals,
    )

    output_input_covariance = jnp.einsum(
        't...,t...o,t...i->...oi',
        normalized,
        output_residuals,
        input_residuals,
    )

    model = linear_from_centered_moments(
        input_mean=input_mean,
        output_mean=output_mean,
        input_covariance=input_covariance,
        output_covariance=output_covariance,
        output_input_covariance=output_input_covariance,
        ridge=ridge,
    )

    return model.reshape_input(input_shape)


def linear_from_samples_grouped(
    inputs: jax.Array,  # (T, *input_shape)
    outputs: jax.Array,  # (T, O)
    assignments: jax.Array,  # (T,)
    num_groups: int,
    ridge: float = 0.0,
) -> LinearGaussian:
    """Fit one model to each assigned group."""
    weights = jax.nn.one_hot(
        assignments,
        num_groups,
        dtype=jnp.result_type(inputs, outputs, jnp.float32),
    )  # (T, K)

    return linear_from_samples_weighted(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        ridge=ridge,
    )
