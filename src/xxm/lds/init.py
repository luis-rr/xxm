import jax
from jax import numpy as jnp

from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.models.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim.loop import unstack_models

from .core import Model


def _validate_initialization(
    observations: jax.Array,
    latent_dim: int,
) -> None:
    if observations.ndim != 2:
        raise ValueError('observations must have shape (T, N)')

    time_steps, observation_dim = observations.shape

    if latent_dim < 1 or latent_dim > observation_dim:
        raise ValueError('latent_dim must be between 1 and the observation dimension')

    if time_steps < latent_dim + 2:
        raise ValueError(
            'At least latent_dim + 2 time steps are required to initialize the dynamics'
        )


def pca_latents(
    observations: jax.Array,
    latent_dim: int,
) -> jax.Array:
    centered = observations - jnp.mean(observations, axis=0)

    _, _, vt = jnp.linalg.svd(
        centered,
        full_matrices=False,
    )

    return centered @ vt[:latent_dim].T


def init_pca_gaussian(
    observations: jax.Array,
    latent_dim: int,
    covariance_floor: float = 1e-2,
) -> Model[GaussianEmissions]:
    _validate_initialization(observations, latent_dim)

    latents = pca_latents(
        observations,
        latent_dim,
    )

    return Model(
        initial=GaussianInitial.from_latents(latents, covariance_floor),
        dynamics=GaussianLinearDynamics.from_latents(latents, covariance_floor),
        emissions=GaussianEmissions.from_latents(
            latents=latents,
            observations=observations,
            covariance_floor=covariance_floor,
        ),
    )


def init_pca_gaussian_many(
    observations: jax.Array,
    latent_dim: int,
    covariance_floors: jax.Array | None = None,
) -> tuple[Model[GaussianEmissions], ...]:
    """Initialize multiple LDS models from PCA latent projections."""

    stacked = jax.vmap(
        lambda covariance_floor: init_pca_gaussian(
            observations,
            latent_dim,
            covariance_floor,
        )
    )(covariance_floors)

    return unstack_models(stacked)


def init_pca_poisson(
    observations: jax.Array,
    latent_dim: int,
    covariance_floor: float = 1e-2,
) -> Model[PoissonEmissions]:
    _validate_initialization(observations, latent_dim)

    latents = pca_latents(
        observations,
        latent_dim,
    )

    return Model(
        initial=GaussianInitial.from_latents(
            latents,
            covariance_floor,
        ),
        dynamics=GaussianLinearDynamics.from_latents(
            latents,
            covariance_floor,
        ),
        emissions=PoissonEmissions.from_latents(
            latents=latents,
            observations=observations,
        ),
    )


def init_pca_poisson_many(
    observations: jax.Array,
    latent_dim: int,
    covariance_floors: jax.Array | None = None,
) -> tuple[Model[PoissonEmissions], ...]:
    """Initialize multiple LDS models from PCA latent projections."""

    stacked = jax.vmap(
        lambda covariance_floor: init_pca_poisson(
            observations,
            latent_dim,
            covariance_floor,
        )
    )(covariance_floors)

    return unstack_models(stacked)
