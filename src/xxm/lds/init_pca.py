import jax
from jax import numpy as jnp

from ..gaussian_chain import fit_linear_gaussian
from .core import GaussianEmissions, Model


def _pca(
    observations: jax.Array,
    num_components: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit PCA and return scores, mean, and components."""
    mean = jnp.mean(observations, axis=0)
    centered = observations - mean

    _, _, vt = jnp.linalg.svd(centered, full_matrices=False)

    components = vt[:num_components]  # (D, N)
    scores = centered @ components.T  # (T, D)

    return scores, mean, components


def _residual_covariance(
    residuals: jax.Array,
    covariance_floor: float,
    reference: jax.Array | None = None,
) -> jax.Array:
    """Estimate residual covariance with a scale-aware isotropic floor."""
    num_samples, dim = residuals.shape

    covariance = residuals.T @ residuals / num_samples

    if reference is None:
        scale = jnp.trace(covariance) / dim
    else:
        scale = jnp.mean(jnp.var(reference, axis=0))

    return covariance + covariance_floor * scale * jnp.eye(
        dim,
        dtype=residuals.dtype,
    )


def _fit_linear_dynamics(
    states: jax.Array,
    covariance_floor: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fit x[t+1] = A x[t] + b + noise by least squares."""
    inputs = states[:-1]
    outputs = states[1:]
    num_samples = inputs.shape[0]

    matrix, bias, covariance = fit_linear_gaussian_from_moments(
        input_mean=jnp.mean(inputs, axis=0),
        output_mean=jnp.mean(outputs, axis=0),
        input_second_moment=inputs.T @ inputs / num_samples,
        output_second_moment=outputs.T @ outputs / num_samples,
        output_input_moment=outputs.T @ inputs / num_samples,
    )

    covariance += covariance_floor * jnp.eye(
        covariance.shape[0],
        dtype=covariance.dtype,
    )

    return matrix, bias, covariance


def initialize_lds(
    observations: jax.Array,
    state_dim: int,
    covariance_floor: float = 1e-2,
) -> Model:
    """Initialize an LDS using PCA followed by linear dynamics regression."""
    if observations.ndim != 2:
        raise ValueError('observations must have shape (T, N)')

    if observations.shape[0] < 2:
        raise ValueError('At least two time steps are required')

    if state_dim < 1 or state_dim > min(observations.shape):
        raise ValueError('state_dim must be between 1 and min(T, N)')

    states, observation_mean, components = _pca(
        observations,
        state_dim,
    )

    # y_t = C x_t + d + noise
    emission_matrix = components.T
    emission_bias = observation_mean

    reconstructed_observations = states @ emission_matrix.T + emission_bias
    observation_residuals = observations - reconstructed_observations

    emission_covariance = _residual_covariance(
        observation_residuals,
        covariance_floor,
        reference=observations,
    )

    emissions = GaussianEmissions(
        readout=emission_matrix,
        bias=emission_bias,
        noise_covariance=emission_covariance,
    )

    dynamics_matrix, dynamics_bias, dynamics_covariance = fit_linear_gaussian(
        states[:-1],
        states[1:],
    )

    dynamics_covariance += covariance_floor * jnp.eye(state_dim)

    # There is only one observed initial state, so use the overall
    # latent scale as a reasonable initial covariance.
    initial_mean = states[0]

    centered_states = states - jnp.mean(states, axis=0)
    initial_covariance = _residual_covariance(
        centered_states,
        covariance_floor,
    )

    return Model(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_matrix=dynamics_matrix,
        dynamics_bias=dynamics_bias,
        dynamics_noise_covariance=dynamics_covariance,
        emissions=emissions,
    )
