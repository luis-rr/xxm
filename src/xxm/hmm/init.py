import jax
from jax import numpy as jnp

from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.discrete import (
    Emissions,
    GaussianEmissions,
    PoissonEmissions,
)
from xxm.core.emissions.discrete_ar import (
    AREmissions,
    lagged_observations,
)
from xxm.core.models.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.hmm.core import Model


def _kmeans(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
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


def _init(
    emissions: Emissions,
    num_states: int,
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
        initial=CategoricalInitial(Categorical(probs=initial_probs)),
        transitions=CategoricalTransitions(Categorical(probs=transition_probs)),
        emissions=emissions,
    )


def _init_gaussian_emissions(
    key: jax.Array,
    observations: jax.Array,  # (T, N)
    num_states: int,
) -> GaussianEmissions:
    assignments = _kmeans(
        key=key,
        observations=observations,
        num_states=num_states,
    )  # (T,)

    gaussian = gaussian_fit.from_samples_grouped(
        values=observations,
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


def init_gaussian(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    self_transition_prob: float = 0.9,
) -> Model:
    emissions = _init_gaussian_emissions(
        observations=observations,
        num_states=num_states,
        key=key,
    )
    return _init(
        emissions=emissions,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=observations.dtype,
    )


def _init_ar_state_assignments(
    key: jax.Array,
    predictors: jax.Array,  # (T-L, L, N)
    current: jax.Array,  # (T-L, N)
    num_states: int,
) -> jax.Array:  # (T-L,)
    """Initialize AR states by clustering predictors and current observations."""
    flat_predictors = predictors.reshape(
        predictors.shape[0],
        -1,
    )  # (T-L, L*N)

    features = jnp.concatenate(
        [
            flat_predictors,
            current,
        ],
        axis=-1,
    )  # (T-L, (L+1)*N)

    return _kmeans(
        key=key,
        observations=features,
        num_states=num_states,
    )


def _init_ar_gaussian_emissions(
    key: jax.Array,
    observations: jax.Array,  # (T, N)
    num_states: int,
    num_lags: int,
) -> AREmissions[LinearGaussian]:
    predictors = lagged_observations(
        observations,
        num_lags=num_lags,
    )  # (T-L, L, N)

    current = observations[num_lags:]  # (T-L, N)

    assignments = _init_ar_state_assignments(
        key=key,
        predictors=predictors,
        current=current,
        num_states=num_states,
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


def init_gaussian_ar(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    num_lags: int,
    self_transition_prob: float = 0.9,
) -> Model:
    """
    Initialize a Gaussian AR-HMM conditional on the first ``num_lags`` values.

    ``observations[:num_lags]`` provide the fixed autoregressive history.
    Latent states correspond only to ``observations[num_lags:]``.
    """
    emissions = _init_ar_gaussian_emissions(
        key=key,
        observations=observations,
        num_states=num_states,
        num_lags=num_lags,
    )

    return _init(
        emissions=emissions,
        num_states=num_states,
        dtype=observations.dtype,
        self_transition_prob=self_transition_prob,
    )


def _init_poisson_emissions(
    key: jax.Array,
    observations: jax.Array,  # (T, N)
    num_states: int,
) -> PoissonEmissions:
    assignments = _kmeans(
        key=key,
        observations=observations,
        num_states=num_states,
    )  # (T,)

    poisson = poisson_fit.from_samples_grouped(
        values=observations,
        assignments=assignments,
        num_groups=num_states,
    )

    return PoissonEmissions(
        model=poisson,
    )


def init_poisson(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    self_transition_prob: float = 0.9,
) -> Model:
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


def _init_ar_poisson_emissions(
    key: jax.Array,
    observations: jax.Array,  # (T, N)
    num_states: int,
    num_lags: int,
) -> AREmissions[LinearPoisson]:
    predictors = lagged_observations(
        observations,
        num_lags=num_lags,
    )  # (T-L, L, N)

    current = observations[num_lags:]  # (T-L, N)

    assignments = _init_ar_state_assignments(
        key=key,
        predictors=predictors,
        current=current,
        num_states=num_states,
    )  # (T-L,)

    model = poisson_fit.linear_from_pairs_grouped(
        inputs=predictors,
        outputs=current,
        assignments=assignments,
        num_groups=num_states,
    )

    return AREmissions(model)


def init_poisson_ar(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    num_lags: int,
    self_transition_prob: float = 0.9,
) -> Model:
    """
    Initialize a Poisson AR-HMM conditional on the first ``num_lags`` values.

    ``observations[:num_lags]`` provide the fixed autoregressive history.
    Latent states correspond only to ``observations[num_lags:]``.
    """
    emissions = _init_ar_poisson_emissions(
        key=key,
        observations=observations,
        num_states=num_states,
        num_lags=num_lags,
    )

    return _init(
        emissions=emissions,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=jnp.result_type(observations, jnp.float32),
    )
