import jax
from jax import numpy as jnp

from xxm.core.emissions.discrete import (
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.emissions.discrete_ar import (
    AREmissions,
    lagged_observations,
)
from xxm.optim import gaussian as gaussian_fit, poisson as poisson_fit
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.dists.poisson import LinearPoisson

from .core import DiscreteInitialModel, DiscreteTransitionModel, Emissions, Model


def _kmeans(
    observations: jax.Array,
    num_states: int,
    key: jax.Array,
    num_iters: int = 20,
) -> jax.Array:
    """Return hard K-means assignments with shape (T,)."""
    observations = jnp.asarray(
        observations,
        dtype=jnp.result_type(observations, jnp.float32),
    )

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

        weights = jax.nn.one_hot(assignments, num_states, dtype=observations.dtype)
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
    dtype: jnp.dtype,
    self_transition_prob: float = 0.9,
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
        initial=DiscreteInitialModel(Categorical(probs=initial_probs)),
        transitions=DiscreteTransitionModel(Categorical(probs=transition_probs)),
        emissions=emissions,
    )


def _initialize_gaussian_emissions(
    observations: jax.Array,  # (T, N)
    num_states: int,
    key: jax.Array,
) -> GaussianEmissions:
    assignments = _kmeans(
        observations,
        num_states,
        key,
    )  # (T,)

    gaussian = gaussian_fit.gaussian_from_samples_grouped(
        observations=observations,
        assignments=assignments,
        num_groups=num_states,
    )

    covariance = (
        gaussian.covariance
        + 1e-6
        * jnp.eye(
            observations.shape[-1],
            dtype=gaussian.covariance.dtype,
        )[None]
    )  # (K, N, N)

    return GaussianEmissions(
        model=gaussian._replace(
            covariance=covariance,
        )
    )


def initialize_hmm_gaussian(
    num_states: int,
    observations: jax.Array,
    key: jax.Array,
    self_transition_prob: float = 0.9,
) -> Model:
    emissions = _initialize_gaussian_emissions(observations, num_states, key=key)
    return _initialize(
        num_states,
        emissions,
        self_transition_prob=self_transition_prob,
        dtype=observations.dtype,
    )


def _initialize_ar_state_assignments(
    predictors: jax.Array,  # (T-L, L*N)
    current: jax.Array,  # (T-L, N)
    num_states: int,
    key: jax.Array,
) -> jax.Array:  # (T-L,)
    """Initialize AR states by clustering predictors and current observations."""
    features = jnp.concatenate(
        [predictors, current],
        axis=-1,
    )  # (T-L, (L+1)*N)

    return _kmeans(
        features,
        num_states,
        key,
    )


def _initialize_ar_gaussian_emissions(
    observations: jax.Array,  # (T, N)
    num_states: int,
    max_lag: int,
    key: jax.Array,
) -> AREmissions[LinearGaussian]:
    history = lagged_observations(
        observations,
        max_lag=max_lag,
    )  # (T-L, L, N)

    current = observations[max_lag:]  # (T-L, N)

    num_samples, _, num_dims = history.shape

    predictors = history.reshape(
        num_samples,
        max_lag * num_dims,
    )  # (T-L, L*N)

    assignments = _initialize_ar_state_assignments(
        predictors,
        current,
        num_states,
        key,
    )  # (T-L,)

    model = gaussian_fit.linear_from_pairs_grouped(
        inputs=predictors,
        outputs=current,
        assignments=assignments,
        num_groups=num_states,
        ridge=1e-6,
    )

    model = model.add_covariance_jitter(1e-6)

    return AREmissions(model)


def initialize_arhmm_gaussian(
    num_states: int,
    observations: jax.Array,
    lag: int,
    key: jax.Array,
    self_transition_prob: float = 0.9,
) -> Model:
    emissions = _initialize_ar_gaussian_emissions(
        observations,
        num_states,
        lag,
        key,
    )

    return _initialize(
        num_states,
        emissions,
        dtype=observations.dtype,
        self_transition_prob=self_transition_prob,
    )


def _initialize_poisson_emissions(
    observations: jax.Array,  # (T, N)
    num_states: int,
    key: jax.Array,
) -> PoissonEmissions:
    assignments = _kmeans(
        observations,
        num_states,
        key,
    )  # (T,)

    poisson = poisson_fit.poisson_from_samples_grouped(
        observations=observations,
        assignments=assignments,
        num_groups=num_states,
    )

    return PoissonEmissions(
        model=poisson,
    )


def initialize_hmm_poisson(
    num_states: int,
    observations: jax.Array,
    key: jax.Array,
    self_transition_prob: float = 0.9,
) -> Model:
    emissions = _initialize_poisson_emissions(observations, num_states, key=key)
    return _initialize(
        num_states,
        emissions,
        self_transition_prob=self_transition_prob,
        dtype=observations.dtype,
    )


def _initialize_ar_poisson_emissions(
    observations: jax.Array,  # (T, N)
    num_states: int,
    max_lag: int,
    key: jax.Array,
) -> AREmissions[LinearPoisson]:
    history = lagged_observations(
        observations,
        max_lag=max_lag,
    )  # (T-L, L, N)

    current = observations[max_lag:]  # (T-L, N)

    num_samples, _, num_dims = history.shape

    predictors = history.reshape(
        num_samples,
        max_lag * num_dims,
    )  # (T-L, L*N)

    assignments = _initialize_ar_state_assignments(
        predictors,
        current,
        num_states,
        key,
    )  # (T-L,)

    model = poisson_fit.linear_from_pairs_grouped(
        inputs=predictors,
        outputs=current,
        assignments=assignments,
        num_groups=num_states,
    )

    return AREmissions(model)


def initialize_arhmm_poisson(
    num_states: int,
    observations: jax.Array,
    max_lag: int,
    key: jax.Array,
    self_transition_prob: float = 0.9,
) -> Model:

    emissions = _initialize_ar_poisson_emissions(observations, num_states, max_lag=max_lag, key=key)
    return _initialize(
        num_states,
        emissions,
        self_transition_prob=self_transition_prob,
        dtype=jnp.result_type(observations, jnp.float32),
    )
