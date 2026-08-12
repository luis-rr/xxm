import jax
from jax import numpy as jnp

from ..fit import unstack_models
from ..gaussian_chain import fit_linear_gaussian
from .core import GaussianEmissions, Model


def _covariance(
    values: jax.Array,
) -> jax.Array:
    centered = values - jnp.mean(values, axis=0)
    return centered.T @ centered / values.shape[0]


def _add_covariance_floor(
    covariance: jax.Array,
    covariance_floor: float,
    reference: jax.Array,
) -> jax.Array:
    """Add an isotropic floor relative to the typical variance of a reference."""
    scale = jnp.mean(jnp.var(reference, axis=0))

    return covariance + covariance_floor * scale * jnp.eye(
        covariance.shape[0],
        dtype=covariance.dtype,
    )


def _init_from_states(
    observations: jax.Array,
    states: jax.Array,
    covariance_floor: float,
) -> Model:
    """Construct an LDS from an initial latent trajectory."""

    # y_t = C x_t + d + noise
    emission_matrix, emission_bias, emission_covariance = fit_linear_gaussian(
        states,
        observations,
    )
    emission_covariance = _add_covariance_floor(
        emission_covariance,
        covariance_floor,
        reference=observations,
    )

    emissions = GaussianEmissions(
        readout=emission_matrix,
        bias=emission_bias,
        noise_covariance=emission_covariance,
    )

    # x[t+1] = A x[t] + b + noise
    dynamics_matrix, dynamics_bias, dynamics_covariance = fit_linear_gaussian(
        states[:-1],
        states[1:],
    )
    dynamics_covariance = _add_covariance_floor(
        dynamics_covariance,
        covariance_floor,
        reference=states,
    )

    # There is only one initial state estimate, so use the overall
    # latent covariance as a reasonable scale for its uncertainty.
    initial_mean = states[0]
    initial_covariance = _add_covariance_floor(
        _covariance(states),
        covariance_floor,
        reference=states,
    )

    return Model(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_matrix=dynamics_matrix,
        dynamics_bias=dynamics_bias,
        dynamics_noise_covariance=dynamics_covariance,
        emissions=emissions,
    )


def _validate_initialization(
    observations: jax.Array,
    state_dim: int,
) -> None:
    if observations.ndim != 2:
        raise ValueError('observations must have shape (T, N)')

    time_steps, observation_dim = observations.shape

    if state_dim < 1 or state_dim > observation_dim:
        raise ValueError('state_dim must be between 1 and the observation dimension')

    if time_steps < state_dim + 2:
        raise ValueError(
            'At least state_dim + 2 time steps are required to initialize the dynamics'
        )


def _pca_states(
    observations: jax.Array,
    state_dim: int,
) -> jax.Array:
    centered = observations - jnp.mean(observations, axis=0)

    _, _, vt = jnp.linalg.svd(
        centered,
        full_matrices=False,
    )

    return centered @ vt[:state_dim].T


def init_pca(
    observations: jax.Array,
    state_dim: int,
    covariance_floor: float = 1e-2,
) -> Model:
    _validate_initialization(observations, state_dim)

    states = _pca_states(
        observations,
        state_dim,
    )

    return _init_from_states(
        observations,
        states,
        covariance_floor,
    )


def init_pca_many(
    observations: jax.Array,
    state_dim: int,
    covariance_floors: jax.Array | None = None,
) -> tuple[Model, ...]:
    """Initialize multiple LDS models from PCA latent projections."""

    stacked = jax.vmap(
        lambda covariance_floor: init_pca(
            observations,
            state_dim,
            covariance_floor,
        )
    )(covariance_floors)

    return unstack_models(stacked)


def _random_states(
    observations: jax.Array,
    state_dim: int,
    key: jax.Array,
) -> jax.Array:
    centered = observations - jnp.mean(observations, axis=0)

    projection = jax.random.normal(
        key,
        shape=(observations.shape[1], state_dim),
        dtype=observations.dtype,
    )

    basis, _ = jnp.linalg.qr(projection)

    return centered @ basis


def init_random(
    observations: jax.Array,
    state_dim: int,
    key: jax.Array,
    covariance_floor: float = 1e-2,
) -> Model:
    _validate_initialization(observations, state_dim)

    states = _random_states(
        observations,
        state_dim,
        key,
    )

    return _init_from_states(
        observations,
        states,
        covariance_floor,
    )


def init_random_many(
    observations: jax.Array,
    state_dim: int,
    num_models: int,
    key: jax.Array,
    covariance_floor: float = 1e-2,
) -> tuple[Model, ...]:
    """Initialize multiple LDS models from random latent projections."""
    keys = jax.random.split(key, num_models)

    stacked = jax.vmap(
        lambda key: init_random(
            observations,
            state_dim,
            key,
            covariance_floor,
        )
    )(keys)

    return unstack_models(stacked)
