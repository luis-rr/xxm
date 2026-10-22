"""HMM initialization from data via clustering."""

import typing

import jax
from jax import numpy as jnp

from xxm.core.data import Dataset
from xxm.core.dists.discrete import Categorical
from xxm.core.emissions.collection import DiscreteEmissionsCollection
from xxm.core.emissions.discrete import (
    Emissions,
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.kmeans import kmeans_assignments
from xxm.hmm.core import Model

DEFAULT_SELF_TRANSITION_PROB = 0.9


EmissionT = typing.TypeVar(
    'EmissionT',
    bound=Emissions,
)


def _validate_inputs(data: Dataset, num_states: int):
    if data.input_dim != 0:
        raise ValueError('stationary HMMs require data.input_dim == 0')

    if num_states < 1:
        raise ValueError('num_states must be positive')


def _clustering_observations(
    data: Dataset,
) -> tuple[jax.Array, jax.Array]:
    """Flatten observations and their visibility-aware weights."""
    observations = data.weighted_observations()

    return (
        # Fixed number of rows
        observations.safe_values().reshape(-1, data.observation_dim),
        # Zero for invisible/padded rows
        observations.weights.reshape(-1),
    )


def _init(
    emissions: DiscreteEmissionsCollection[EmissionT],
    num_states: int,
    dtype: jnp.dtype,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
) -> Model[EmissionT]:
    if num_states == 1:
        transition_probs = jnp.array(
            [[1.0]],
            dtype=dtype,
        )
    else:
        off_diagonal_prob = (1.0 - self_transition_prob) / (num_states - 1)

        transition_probs = jnp.full(
            (num_states, num_states),
            off_diagonal_prob,
            dtype=dtype,
        )
        transition_probs = transition_probs.at[jnp.diag_indices(num_states)].set(
            self_transition_prob
        )

    initial_probs = (
        jnp.ones(
            num_states,
            dtype=dtype,
        )
        / num_states
    )

    return Model(
        initial=CategoricalInitial(Categorical(probs=initial_probs)),
        transitions=CategoricalTransitions(Categorical(probs=transition_probs)),
        emissions=emissions,
    )


def _init_gaussian_emissions(
    key: jax.Array,
    observations: jax.Array,  # (T, N)
    sample_weights: jax.Array,
    num_states: int,
    covariance_floor: float,
) -> GaussianEmissions:
    assignments = kmeans_assignments(
        key,
        observations,
        num_states,
        sample_weights=sample_weights,
    )  # (T,)

    gaussian = gaussian_fit.from_samples_grouped(
        values=observations,
        assignments=assignments,
        num_groups=num_states,
        sample_weights=sample_weights,
        covariance_floor=covariance_floor,
    )

    counts = jnp.bincount(
        assignments,
        weights=sample_weights.astype(jnp.result_type(observations, jnp.float32)),
        length=num_states,
    )

    pooled = gaussian_fit.from_samples_weighted(
        observations,
        sample_weights,
        covariance_floor=covariance_floor,
    )

    empty = counts <= 0

    gaussian = gaussian._replace(
        mean=jnp.where(
            empty[:, None],
            pooled.mean[None, :],
            gaussian.mean,
        ),
        covariance=jnp.where(
            empty[:, None, None],
            pooled.covariance[None, :, :],
            gaussian.covariance,
        ),
    )

    return GaussianEmissions(dist=gaussian)


def init_gaussian_via_kmeans(
    key: jax.Array,
    dataset: Dataset,
    num_states: int,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[GaussianEmissions]:
    """Initialize a Gaussian HMM with weighted K-means."""
    _validate_inputs(dataset, num_states)

    observations, weights = _clustering_observations(dataset)

    emissions = _init_gaussian_emissions(
        key=key,
        observations=observations,
        sample_weights=weights,
        num_states=num_states,
        covariance_floor=covariance_floor,
    )

    return _init(
        emissions=DiscreteEmissionsCollection((emissions,)),
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=observations.dtype,
    )


def _init_poisson_emissions(
    key: jax.Array,
    observations: jax.Array,
    sample_weights: jax.Array,
    num_states: int,
) -> PoissonEmissions:
    assignments = kmeans_assignments(
        key=key,
        observations=observations,
        num_groups=num_states,
        sample_weights=sample_weights,
    )  # (T,)

    poisson = poisson_fit.from_samples_grouped(
        values=observations,
        assignments=assignments,
        num_groups=num_states,
        sample_weights=sample_weights,
    )

    counts = jnp.bincount(
        assignments,
        weights=sample_weights.astype(jnp.result_type(observations, jnp.float32)),
        length=num_states,
    )

    pooled = poisson_fit.from_samples_weighted(
        observations,
        sample_weights,
    )

    poisson = poisson._replace(
        log_rates=jnp.where(
            (counts <= 0)[:, None],
            pooled.log_rates[None, :],
            poisson.log_rates,
        )
    )

    return PoissonEmissions(dist=poisson)


def init_poisson_via_kmeans(
    key: jax.Array,
    dataset: Dataset,
    num_states: int,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    dtype=jnp.float32,
) -> Model[PoissonEmissions]:
    """Initialize a Poisson HMM with weighted K-means."""

    _validate_inputs(dataset, num_states)

    observations, weights = _clustering_observations(dataset)

    emissions = _init_poisson_emissions(
        key=key,
        observations=observations,
        sample_weights=weights,
        num_states=num_states,
    )

    return _init(
        emissions=DiscreteEmissionsCollection((emissions,)),
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=jnp.result_type(observations.dtype, dtype),
    )
