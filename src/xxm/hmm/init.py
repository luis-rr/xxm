"""HMM initialization from data via clustering."""

import jax
from jax import numpy as jnp

from xxm.core.data import Dataset
from xxm.core.dists.categorical import Categorical
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


def _validate_inputs(data: Dataset, num_states: int):
    if data.input_dim != 0:
        raise ValueError('stationary HMMs require data.input_dim == 0')

    if num_states < 1:
        raise ValueError('num_states must be positive')


def _visible_observations(data: Dataset, num_states: int) -> jax.Array:
    """Collect visible valid rows on the host for clustering."""

    observations = data.observations.values[data.observation_weights()]

    if observations.shape[0] < num_states:
        raise ValueError(
            'HMM initialization requires at least num_states visible valid rows'
        )

    return observations


def _init(
    emissions: Emissions,
    num_states: int,
    dtype: jnp.dtype,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
) -> Model:
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
    num_states: int,
    covariance_floor,
) -> GaussianEmissions:
    assignments = kmeans_assignments(
        key=key,
        observations=observations,
        num_groups=num_states,
    )  # (T,)

    gaussian = gaussian_fit.from_samples_grouped(
        values=observations,
        assignments=assignments,
        num_groups=num_states,
        covariance_floor=covariance_floor,
    )

    return GaussianEmissions(
        dist=gaussian,
    )


def init_gaussian_via_kmeans(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model:
    """Initialize a shared Gaussian HMM from visible valid Dataset rows."""
    _validate_inputs(data, num_states)

    observations = _visible_observations(data, num_states)
    emissions = _init_gaussian_emissions(
        observations=observations,
        num_states=num_states,
        key=key,
        covariance_floor=covariance_floor,
    )
    return _init(
        emissions=emissions,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=observations.dtype,
    )


def _init_poisson_emissions(
    key: jax.Array,
    observations: jax.Array,  # (T, N)
    num_states: int,
) -> PoissonEmissions:
    assignments = kmeans_assignments(
        key=key,
        observations=observations,
        num_groups=num_states,
    )  # (T,)

    poisson = poisson_fit.from_samples_grouped(
        values=observations,
        assignments=assignments,
        num_groups=num_states,
    )

    return PoissonEmissions(
        dist=poisson,
    )


def init_poisson_via_kmeans(
    key: jax.Array,
    data: Dataset,
    num_states: int,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
) -> Model:
    """Initialize a shared Poisson HMM from visible valid Dataset rows."""
    _validate_inputs(data, num_states)

    observations = _visible_observations(data, num_states)
    emissions = _init_poisson_emissions(
        key=key,
        observations=observations,
        num_states=num_states,
    )
    return _init(
        emissions=emissions,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=observations.dtype,
    )
