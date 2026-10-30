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


def cluster_observations(
    key: jax.Array,
    dataset: Dataset,
    num_states: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:  # (T, D_y), (T,), (T,)
    """Flatten visible observations and assign them to states by weighted K-means.

    Returns the flattened observations, their visibility-aware weights, and
    one state assignment per row.
    """
    _validate_inputs(dataset, num_states)

    observations, weights = _clustering_observations(dataset)

    assignments = kmeans_assignments(
        key,
        observations,
        num_states,
        sample_weights=weights,
    )

    return observations, weights, assignments


def init_from_emissions(
    emissions: DiscreteEmissionsCollection[EmissionT],
    num_states: int,
    dtype: jnp.dtype,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
) -> Model[EmissionT]:
    """Attach emissions to uniform initial and sticky transition probabilities."""
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


def init_gaussian_via_kmeans(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[GaussianEmissions]:
    """Initialize a Gaussian HMM with weighted K-means."""
    observations, weights, assignments = cluster_observations(key, data, num_states)
    emissions = GaussianEmissions.from_assignments(
        observations,
        assignments,
        weights,
        num_states,
        covariance_floor=covariance_floor,
    )
    return init_from_emissions(
        DiscreteEmissionsCollection.from_single(emissions),
        num_states,
        jnp.result_type(observations, jnp.float32),
        self_transition_prob=self_transition_prob,
    )


def init_poisson_via_kmeans(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
) -> Model[PoissonEmissions]:
    """Initialize a Poisson HMM with weighted K-means."""
    observations, weights, assignments = cluster_observations(key, data, num_states)
    emissions = PoissonEmissions.from_assignments(
        observations,
        assignments,
        weights,
        num_states,
    )
    return init_from_emissions(
        DiscreteEmissionsCollection.from_single(emissions),
        num_states,
        jnp.result_type(observations, jnp.float32),
        self_transition_prob=self_transition_prob,
    )
