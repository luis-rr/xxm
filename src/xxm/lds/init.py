"""Initialize linear dynamical systems from principal-component latent trajectories."""

import logging
import math

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.data import Dataset, WeightedObservations
from xxm.core.dists.gaussian import GaussianNoise
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import (
    EmissionsT,
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit

from .core import Model

logger = logging.getLogger(__name__)


def _validate_initialization(
    data: Dataset,
    latent_dim: int,
) -> None:
    observation_dim = data.observation_dim

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
    data: Dataset,
    latent_dim: int,
) -> jax.Array:
    """Estimate one PCA space from all observed valid rows and project the dataset."""
    observations = WeightedObservations(
        values=values,
        weights=data.observation_weights(),
        lengths=data.lengths,
    )
    safe_values = observations.safe_values()
    projection = _pca_projection(safe_values[observations.weights], latent_dim)

    # Centering gives absent rows a nonzero offset; remove it after projection.
    return observations.safe_aligned(projection.apply(safe_values))


def _init_initial(
    latents: jax.Array,
    data: Dataset,
    *,
    covariance_floor: float,
) -> GaussianInitial:
    """Fit the shared initial distribution from observable sequence starts."""
    observed = data.observation_weights()
    start_weights = observed[..., 0]
    num_starts = int(jnp.sum(start_weights))

    if num_starts == 0:
        raise ValueError(
            'PCA initialization requires at least one observed sequence start'
        )

    latent_dim = latents.shape[-1]

    # A single observed start is the normal single-sequence case. For multiple
    # sequences, warn when there are too few observed starts for their empirical
    # covariance to be potentially full rank.
    if math.prod(data.batch_shape) > 1 and num_starts < latent_dim + 1:
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
        values=latents[..., 0, :].reshape((-1, latent_dim)),
        weights=start_weights.reshape(-1),
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

    noise = GaussianNoise(base_covariance).with_floor(
        covariance_floor, reference=reference.noise
    )

    return GaussianInitial(dist=initial._replace(noise=noise))


def _init_dynamics(
    latents: jax.Array,
    data: Dataset,
    *,
    ridge: float,
    covariance_floor: float,
) -> GaussianLinearDynamics:
    """Fit controlled dynamics where both PCA endpoint observations are available."""
    observed = data.observation_weights()

    usable = data.lengths.transitions().clear(observed[..., :-1] & observed[..., 1:])

    latent_dim = latents.shape[-1]
    input_dim = data.input_dim
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
            latents[..., :-1, :],
            data.transition_inputs(),
        ],
        axis=-1,
    )

    dist = gaussian_fit.linear_from_samples(
        inputs=predictors[usable],
        outputs=latents[..., 1:, :][usable],
        ridge=ridge,
        covariance_floor=covariance_floor,
    )

    return GaussianLinearDynamics(dist=dist)


def masked_pca_latents(
    values: jax.Array,  # (*B, T, D_y)
    data: Dataset,
    latent_dim: int,
) -> jax.Array:  # (*B, T, D_x)
    """Validate PCA initialization, then project `values` onto shared PCA latents.

    `values` are the observations in the space PCA should operate on, e.g. the
    raw observations or a link-transformed version of them.
    """
    _validate_initialization(data, latent_dim)
    return _masked_pca_latents(values, data, latent_dim)


def observed_rows(
    latents: jax.Array,  # (*B, T, D_x)
    data: Dataset,
) -> tuple[jax.Array, jax.Array]:  # (N, D_x), (N, D_y)
    """Return latent and observation rows at observed timesteps."""
    observed = data.observation_weights()
    return latents[observed], data.observations.values[observed]


def init_from_latents(
    latents: jax.Array,  # (*B, T, D_x)
    data: Dataset,
    emissions: ContinuousEmissionsCollection[EmissionsT],
    *,
    ridge: float,
    covariance_floor: float,
) -> Model[EmissionsT]:
    """Fit the initial state and dynamics from latents and attach `emissions`."""
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
        emissions=emissions,
    )


def init_gaussian_via_pca(
    data: Dataset,
    latent_dim: int,
    *,
    ridge: float = gaussian_fit.DEFAULT_RIDGE,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[GaussianEmissions]:
    """Initialize a Gaussian LDS from PCA latents of the observations."""
    latents = masked_pca_latents(data.observations.values, data, latent_dim)
    latent_rows, observation_rows = observed_rows(latents, data)
    emissions = GaussianEmissions.from_latents(
        latent_rows,
        observation_rows,
        covariance_floor=covariance_floor,
    )
    return init_from_latents(
        latents,
        data,
        ContinuousEmissionsCollection.from_single(emissions),
        ridge=ridge,
        covariance_floor=covariance_floor,
    )


def init_poisson_via_pca(
    data: Dataset,
    latent_dim: int,
    *,
    ridge: float = gaussian_fit.DEFAULT_RIDGE,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
) -> Model[PoissonEmissions]:
    """Initialize a Poisson LDS from PCA latents of stabilized log counts."""
    log_counts = poisson_fit.inverse_exp_link(
        data.weighted_observations().safe_values(),
        count_floor=count_floor,
    )
    latents = masked_pca_latents(log_counts, data, latent_dim)
    latent_rows, observation_rows = observed_rows(latents, data)
    emissions = PoissonEmissions.from_latents(latent_rows, observation_rows)
    return init_from_latents(
        latents,
        data,
        ContinuousEmissionsCollection.from_single(emissions),
        ridge=ridge,
        covariance_floor=covariance_floor,
    )
