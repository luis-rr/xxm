"""Initialize linear dynamical systems from principal-component latent trajectories."""

import logging

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.data import Sequences, WeightedObservations
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit

from .core import Model

logger = logging.getLogger(__name__)


def _validate_initialization(
    data: Sequences,
    latent_dim: int,
) -> None:
    observation_dim = data.observation_dim()

    if latent_dim < 1 or latent_dim > observation_dim:
        raise ValueError('latent_dim must be between 1 and the observation dimension')

    num_observed = int(jnp.sum(data.observation_weights()))
    if num_observed < latent_dim + 1:
        raise ValueError(
            'At least latent_dim + 1 observed timesteps are required for PCA initialization'
        )


def _pca_projection(
    rows: jax.Array,
    latent_dim: int,
) -> Affine:
    """Fit a centered principal-component projection from complete rows."""
    mean = jnp.mean(rows, axis=0)

    _, _, vt = jnp.linalg.svd(
        rows - mean,
        full_matrices=False,
    )

    coefficients = vt[:latent_dim]
    return Affine(coefficients=coefficients, bias=-(coefficients @ mean))


def pca_latents(
    observations: jax.Array,
    latent_dim: int,
) -> jax.Array:
    """Project one complete sequence onto its leading `latent_dim` components."""
    observations = jnp.asarray(observations)

    if observations.ndim != 2:
        raise ValueError('observations must have shape (T, D_y)')

    return _pca_projection(observations, latent_dim).apply(observations)


def pca_latents_poisson(
    observations: jax.Array,
    latent_dim: int,
    count_floor: float,
) -> jax.Array:
    """Project stabilized log-count observations onto PCA latents."""
    transformed = poisson_fit.inverse_exp_link(
        observations,
        count_floor=count_floor,
    )

    return pca_latents(
        transformed,
        latent_dim,
    )


def _masked_pca_latents(
    values: jax.Array,
    data: Sequences,
    latent_dim: int,
) -> jax.Array:
    """Estimate one PCA space from all observed valid rows and project the dataset."""
    observations = WeightedObservations(
        values=values,
        weights=data.observation_weights(),
    )
    safe_values = observations.safe_values()
    projection = _pca_projection(safe_values[observations.weights], latent_dim)

    # Centering gives absent rows a nonzero offset; remove it after projection.
    return observations.safe_aligned(projection.apply(safe_values))


def _init_initial(
    latents: jax.Array,
    data: Sequences,
    *,
    covariance_floor: float,
) -> GaussianInitial:
    """Fit the shared initial distribution from observable sequence starts."""
    observed = data.observation_weights()
    start_weights = observed[:, 0]
    num_starts = int(jnp.sum(start_weights))

    if num_starts == 0:
        raise ValueError(
            'PCA initialization requires at least one observed sequence start'
        )

    latent_dim = latents.shape[-1]

    # A single observed start is the normal single-sequence case. For multiple
    # sequences, warn when there are too few observed starts for their empirical
    # covariance to be potentially full rank.
    if data.num_sequences() > 1 and num_starts < latent_dim + 1:
        logger.warning(
            f'PCA initial-state initialization has only {num_starts} observed sequence '
            f'starts for latent_dim={latent_dim}. Initial-state parameters are weakly '
            'informed by sequence starts: with one usable start, the covariance '
            'falls back to the trajectory-wide latent covariance; with fewer '
            f'than {latent_dim + 1} starts, the empirical start covariance cannot '
            'be full rank. Initialization will rely on covariance regularization '
            f'(covariance_floor={covariance_floor}).',
        )

    initial = gaussian_fit.from_samples_weighted(
        values=latents[:, 0, :],
        weights=start_weights,
        covariance_floor=0.0,
    )

    reference = gaussian_fit.from_samples(
        latents[observed],
        covariance_floor=0.0,
    )

    # With only one usable start, its empirical covariance is undefined.
    # Reuse the trajectory-wide latent covariance, matching the historical
    # single-sequence initializer while still estimating the mean from x[0].
    base_covariance = jnp.where(
        num_starts > 1,
        initial.covariance,
        reference.covariance,
    )

    covariance = gaussian_fit.add_covariance_floor(
        base_covariance,
        reference_covariance=reference.covariance,
        covariance_floor=covariance_floor,
    )

    return GaussianInitial(
        dist=initial._replace(covariance=covariance),
    )


def _init_dynamics(
    latents: jax.Array,
    data: Sequences,
    *,
    ridge: float,
    covariance_floor: float,
) -> GaussianLinearDynamics:
    """Fit controlled dynamics where both PCA endpoint observations are available."""
    observed = data.observation_weights()
    usable = data.valid_transitions() & observed[:, :-1] & observed[:, 1:]

    latent_dim = latents.shape[-1]
    input_dim = data.input_dim()
    predictor_dim = latent_dim + input_dim
    num_usable = int(jnp.sum(usable))

    if num_usable < 2:
        raise ValueError(
            'PCA dynamics initialization requires at least 2 observed '
            'adjacent transition pairs; '
            f'got {num_usable}'
        )

    if num_usable < predictor_dim + 1:
        logger.warning(
            f'PCA dynamics initialization has only {num_usable} observed adjacent '
            f'transition pairs for a {predictor_dim}-dimensional predictor '
            f'(latent_dim={latent_dim}, input_dim={input_dim}). The regression is '
            'underdetermined from the observed pairs and will rely on '
            f'regularization (ridge={ridge}, covariance_floor={covariance_floor}). Consider '
            'providing more adjacent observations or adjusting these '
            'regularization parameters if initialization is unstable.',
        )

    predictors = jnp.concatenate(
        [
            latents[:, :-1, :],
            data.transition_inputs(),
        ],
        axis=-1,
    )

    dist = gaussian_fit.linear_from_samples(
        inputs=predictors[usable],
        outputs=latents[:, 1:, :][usable],
        ridge=ridge,
        covariance_floor=covariance_floor,
    )

    return GaussianLinearDynamics(dist=dist)


def _init_gaussian_emissions(
    latents: jax.Array,
    data: Sequences,
    covariance_floor: float,
) -> GaussianEmissions:
    observed = data.observation_weights()

    return GaussianEmissions.from_latents(
        latents=latents[observed],
        observations=data.observations[observed],
        covariance_floor=covariance_floor,
    )


def _init_poisson_emissions(
    latents: jax.Array,
    data: Sequences,
) -> PoissonEmissions:
    observed = data.observation_weights()

    return PoissonEmissions.from_latents(
        latents=latents[observed],
        observations=data.observations[observed],
    )


def init_gaussian_via_pca(
    data: Sequences,
    latent_dim: int,
    *,
    ridge=gaussian_fit.DEFAULT_RIDGE,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[GaussianEmissions]:
    """Initialize one shared Gaussian LDS from independent sequences."""
    _validate_initialization(data, latent_dim)

    latents = _masked_pca_latents(
        data.observations,
        data,
        latent_dim,
    )

    return Model(
        initial=_init_initial(
            latents,
            data,
            covariance_floor=covariance_floor,
        ),
        dynamics=_init_dynamics(
            latents,
            data,
            ridge=ridge,
            covariance_floor=covariance_floor,
        ),
        emissions=_init_gaussian_emissions(
            latents,
            data,
            covariance_floor=covariance_floor,
        ),
    )


def init_poisson_via_pca(
    data: Sequences,
    latent_dim: int,
    *,
    ridge=gaussian_fit.DEFAULT_RIDGE,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> Model[PoissonEmissions]:
    """Initialize one shared Poisson LDS from independent sequences."""
    _validate_initialization(data, latent_dim)

    transformed = poisson_fit.inverse_exp_link(
        data.weighted_observations().safe_values(),
        count_floor=count_floor,
    )

    latents = _masked_pca_latents(
        transformed,
        data,
        latent_dim,
    )

    return Model(
        initial=_init_initial(
            latents,
            data,
            covariance_floor=covariance_floor,
        ),
        dynamics=_init_dynamics(
            latents,
            data,
            ridge=ridge,
            covariance_floor=covariance_floor,
        ),
        emissions=_init_poisson_emissions(
            latents,
            data,
        ),
    )
