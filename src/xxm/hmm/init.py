import jax
from jax import numpy as jnp

from xxm.core.discrete.emissions import GaussianEmissions, PoissonEmissions

from .core import Emissions, LatentInitialModel, LatentTransitionModel, Model


def _kmeans(
    observations: jax.Array,
    num_states: int,
    key: jax.Array,
    num_iters: int = 20,
) -> jax.Array:
    """Return hard K-means assignments with shape (T,)."""
    initial_indices = jax.random.choice(
        key,
        observations.shape[0],
        shape=(num_states,),
        replace=False,
    )
    initial_centers = observations[initial_indices]

    def step(_, centers):
        distances = jnp.sum(
            (observations[:, None, :] - centers[None, :, :]) ** 2,
            axis=-1,
        )
        assignments = jnp.argmin(distances, axis=1)

        weights = jax.nn.one_hot(assignments, num_states)
        counts = weights.sum(axis=0)

        new_centers = weights.T @ observations / jnp.maximum(counts[:, None], 1)

        # Keep the old center if a cluster is empty.
        return jnp.where(
            (counts > 0)[:, None],
            new_centers,
            centers,
        )

    centers = jax.lax.fori_loop(
        0,
        num_iters,
        step,
        initial_centers,
    )

    distances = jnp.sum(
        (observations[:, None, :] - centers[None, :, :]) ** 2,
        axis=-1,
    )
    return jnp.argmin(distances, axis=1)


def _initialize(
    num_states: int,
    emissions: Emissions,
    self_transition_prob: float = 0.9,
) -> Model:
    initial_probs = jnp.ones(num_states) / num_states

    off_diagonal_prob = (1.0 - self_transition_prob) / (num_states - 1)

    transition_probs = jnp.full(
        (num_states, num_states),
        off_diagonal_prob,
    )
    transition_probs = transition_probs.at[jnp.diag_indices(num_states)].set(self_transition_prob)

    return Model(
        initial=LatentInitialModel(initial_probs=initial_probs),
        transitions=LatentTransitionModel(transition_probs=transition_probs),
        emissions=emissions,
    )


def _initialize_gaussian_emissions(
    observations: jax.Array,
    num_states: int,
    key: jax.Array,
) -> 'GaussianEmissions':
    assignments = _kmeans(
        observations,
        num_states,
        key,
    )

    weights = jax.nn.one_hot(assignments, num_states)
    counts = weights.sum(axis=0)

    means = weights.T @ observations / jnp.maximum(counts[:, None], 1)

    residuals = observations[:, None, :] - means[None, :, :]

    covariances = jnp.einsum(
        'tk,tki,tkj->kij',
        weights,
        residuals,
        residuals,
    ) / jnp.maximum(counts[:, None, None], 1)

    covariances += 1e-6 * jnp.eye(observations.shape[-1])[None, :, :]

    return GaussianEmissions(
        means=means,
        covariances=covariances,
    )


def initialize_hmm_gaussian(
    num_states: int,
    observations: jax.Array,
    key: jax.Array,
    self_transition_prob: float = 0.9,
) -> Model:
    emissions = _initialize_gaussian_emissions(observations, num_states, key=key)
    return _initialize(num_states, emissions, self_transition_prob)


def _initialize_poisson_emissions(
    observations: jax.Array,
    num_states: int,
    key: jax.Array,
) -> 'PoissonEmissions':
    assignments = _kmeans(
        observations,
        num_states,
        key,
    )

    weights = jax.nn.one_hot(assignments, num_states)
    counts = weights.sum(axis=0)

    rates = weights.T @ observations / jnp.maximum(counts[:, None], 1)

    rates = jnp.maximum(rates, 1e-8)

    return PoissonEmissions(rates=rates)


def initialize_hmm_poisson(
    num_states: int,
    observations: jax.Array,
    key: jax.Array,
    self_transition_prob: float = 0.9,
) -> Model:
    emissions = _initialize_poisson_emissions(observations, num_states, key=key)
    return _initialize(num_states, emissions, self_transition_prob)
