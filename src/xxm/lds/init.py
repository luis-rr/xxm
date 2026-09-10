"""Initialize linear dynamical systems from principal-component latent trajectories."""

import jax
from jax import numpy as jnp

from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.loop import unstack_states

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
    """Project centered observations onto their leading `latent_dim` components."""
    centered = observations - jnp.mean(observations, axis=0)

    _, _, vt = jnp.linalg.svd(
        centered,
        full_matrices=False,
    )

    return centered @ vt[:latent_dim].T


def pca_latents_poisson(
    observations: jax.Array,
    latent_dim: int,
    count_floor: float,
) -> jax.Array:
    """
    Project inverse-log-link-transformed Poisson observations onto PCA latents.

    Counts are floored before taking logs so zero observations remain finite.
    The transformation is used only to initialize the latent trajectory.
    """

    transformed = poisson_fit.inverse_exp_link(
        observations,
        count_floor=count_floor,
    )

    return pca_latents(
        transformed,
        latent_dim,
    )


def init_pca_gaussian(
    observations: jax.Array,
    latent_dim: int,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[GaussianEmissions]:
    """Initialize a Gaussian LDS from PCA latents and a covariance floor."""
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

    return unstack_states(stacked)


def init_pca_poisson(
    observations: jax.Array,
    latent_dim: int,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> Model[PoissonEmissions]:
    """Initialize a Poisson LDS from inverse-link PCA latents."""
    _validate_initialization(
        observations,
        latent_dim,
    )

    latents = pca_latents_poisson(
        observations,
        latent_dim,
        count_floor=count_floor,
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
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> tuple[Model[PoissonEmissions], ...]:
    """Initialize multiple LDS models from PCA latent projections."""

    stacked = jax.vmap(
        lambda covariance_floor: init_pca_poisson(
            observations,
            latent_dim,
            covariance_floor,
            count_floor=count_floor,
        )
    )(covariance_floors)

    return unstack_states(stacked)
