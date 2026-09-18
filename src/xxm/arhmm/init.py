"""AR-HMM initialization from aligned regression rows."""

import jax
import jax.numpy as jnp

from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.kmeans import kmeans_assignments

from .core import Model
from .data import ARObservations
from .emissions import AREmissions, ConditionalDistT

DEFAULT_SELF_TRANSITION_PROB = 0.9


def _init(
    emissions: AREmissions[ConditionalDistT],
    num_states: int,
    dtype: jnp.dtype,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
) -> Model[ConditionalDistT]:
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


def _init_state_assignments(
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

    return kmeans_assignments(
        key=key,
        observations=features,
        num_groups=num_states,
    )


def _init_gaussian_emissions(
    key: jax.Array,
    data: ARObservations,
    num_states: int,
    ridge=gaussian_fit.DEFAULT_RIDGE,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> AREmissions[LinearGaussian]:
    predictors = data.predictors
    current = data.targets

    assignments = _init_state_assignments(
        key=key,
        predictors=predictors,
        current=current,
        num_states=num_states,
    )  # (T-L,)

    model = gaussian_fit.linear_from_samples_grouped(
        inputs=predictors,
        outputs=current,
        assignments=assignments,
        num_groups=num_states,
        ridge=ridge,
        covariance_floor=covariance_floor,
    )

    return AREmissions(model)


def init_gaussian_via_kmeans(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    num_lags: int,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
    *,
    covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[LinearGaussian]:
    r"""Initialize Gaussian AR-HMM with first $L$ observations as fixed history."""
    data = ARObservations.from_observations(observations, num_lags)
    emissions = _init_gaussian_emissions(
        key=key,
        data=data,
        num_states=num_states,
        covariance_floor=covariance_floor,
    )

    return _init(
        emissions=emissions,
        num_states=num_states,
        dtype=observations.dtype,
        self_transition_prob=self_transition_prob,
    )


def _init_poisson_emissions(
    key: jax.Array,
    data: ARObservations,
    num_states: int,
    ridge=poisson_fit.DEFAULT_RIDGE,
) -> AREmissions[LinearPoisson]:
    predictors = data.predictors
    current = data.targets

    assignments = _init_state_assignments(
        key=key,
        predictors=predictors,
        current=current,
        num_states=num_states,
    )  # (T-L,)

    model = poisson_fit.linear_from_samples_grouped(
        inputs=predictors,
        outputs=current,
        assignments=assignments,
        num_groups=num_states,
        ridge=ridge,
    )

    return AREmissions(model)


def init_poisson_via_kmeans(
    key: jax.Array,
    observations: jax.Array,
    num_states: int,
    num_lags: int,
    self_transition_prob=DEFAULT_SELF_TRANSITION_PROB,
) -> Model[LinearPoisson]:
    """
    Initialize a Poisson AR-HMM conditional on the first ``num_lags`` values.

    ``observations[:num_lags]`` provide the fixed autoregressive history.
    Latent states correspond only to ``observations[num_lags:]``.
    """
    data = ARObservations.from_observations(observations, num_lags)
    emissions = _init_poisson_emissions(
        key=key,
        data=data,
        num_states=num_states,
    )

    return _init(
        emissions=emissions,
        num_states=num_states,
        self_transition_prob=self_transition_prob,
        dtype=jnp.result_type(observations, jnp.float32),
    )
