import jax
from jax import numpy as jnp

from xxm.core.gaussian.emissions import GaussianEmissions, PoissonEmissions
from xxm.stats import gaussian_fit, poisson_fit
from xxm.stats.gaussian import Affine, Gaussian

from ..fit import unstack_models
from .core import GaussianInitialModel, LinearGaussianDynamicsModel, Model


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


def poisson_emissions_from_latents(
    observations: jax.Array,
    latents: jax.Array,
) -> PoissonEmissions:
    """Fit Poisson emissions to a known latent trajectory."""

    observation_dim = observations.shape[1]
    latent_dim = latents.shape[1]

    # Sensible intercept-only starting point.
    mean_rates = jnp.maximum(
        jnp.mean(observations, axis=0),
        1e-6,
    )

    initial_affine = Affine(
        coefficients=jnp.zeros(
            (observation_dim, latent_dim),
            dtype=latents.dtype,
        ),
        bias=jnp.log(mean_rates),
    )

    model = poisson_fit.linear_from_pairs(
        outputs=observations,
        inputs=latents,
        initial_affine=initial_affine,
    )

    return PoissonEmissions(model)


def gaussian_emissions_from_latents(
    observations: jax.Array,
    latents: jax.Array,
    covariance_floor: float,
) -> GaussianEmissions:
    """Fit Gaussian emissions to a known latent trajectory."""
    model = gaussian_fit.linear_from_pairs(
        latents,
        observations,
    )

    model = model.add_covariance_jitter(covariance_floor)

    return GaussianEmissions(model)


def dynamics_from_latents(
    latents: jax.Array,
    covariance_floor: float,
) -> LinearGaussianDynamicsModel:
    """Fit linear dynamics to a known latent trajectory."""
    model = gaussian_fit.linear_from_pairs(
        latents[:-1],
        latents[1:],
    )

    model = model.add_covariance_jitter(covariance_floor)

    return LinearGaussianDynamicsModel(model)


def initial_from_latents(
    observations: jax.Array,
    latents: jax.Array,
    covariance_floor: float = 1e-2,
) -> GaussianInitialModel:
    """Construct an LDS from a known latent trajectory."""

    # There is only one initial state estimate, so use the overall
    # latent covariance as a reasonable scale for its uncertainty.
    return GaussianInitialModel(
        Gaussian(
            mean=latents[0],
            covariance=_add_covariance_floor(
                _covariance(latents),
                covariance_floor,
                reference=latents,
            ),
        )
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


def init_pca_gaussian(
    observations: jax.Array,
    state_dim: int,
    covariance_floor: float = 1e-2,
) -> Model[GaussianEmissions]:
    _validate_initialization(observations, state_dim)

    states = _pca_states(
        observations,
        state_dim,
    )

    return Model(
        initial=initial_from_latents(observations, states, covariance_floor),
        dynamics=dynamics_from_latents(states, covariance_floor),
        emissions=gaussian_emissions_from_latents(observations, states, covariance_floor),
    )


def init_pca_gaussian_many(
    observations: jax.Array,
    state_dim: int,
    covariance_floors: jax.Array | None = None,
) -> tuple[Model[GaussianEmissions], ...]:
    """Initialize multiple LDS models from PCA latent projections."""

    stacked = jax.vmap(
        lambda covariance_floor: init_pca_gaussian(
            observations,
            state_dim,
            covariance_floor,
        )
    )(covariance_floors)

    return unstack_models(stacked)


def init_pca_poisson(
    observations: jax.Array,
    state_dim: int,
    covariance_floor: float = 1e-2,
) -> Model[PoissonEmissions]:
    _validate_initialization(observations, state_dim)

    states = _pca_states(
        observations,
        state_dim,
    )

    return Model(
        initial=initial_from_latents(observations, states, covariance_floor),
        dynamics=dynamics_from_latents(states, covariance_floor),
        emissions=poisson_emissions_from_latents(observations, states),
    )


def init_pca_poisson_many(
    observations: jax.Array,
    state_dim: int,
    covariance_floors: jax.Array | None = None,
) -> tuple[Model[PoissonEmissions], ...]:
    """Initialize multiple LDS models from PCA latent projections."""

    stacked = jax.vmap(
        lambda covariance_floor: init_pca_poisson(
            observations,
            state_dim,
            covariance_floor,
        )
    )(covariance_floors)

    return unstack_models(stacked)
