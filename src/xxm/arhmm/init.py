"""AR-HMM initialization from aligned regression rows."""

import jax
import jax.numpy as jnp

from xxm.core.dists.discrete import Categorical
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.kmeans import kmeans_assignments

from .core import Model
from .data import ARDataset
from .emissions import AREmissions, AREmissionsCollection, ConditionalDistT

DEFAULT_SELF_TRANSITION_PROB = 0.9


def _visible_rows(
    data: ARDataset,
) -> tuple[jax.Array, jax.Array]:
    """Pool visible complete AR windows across independent sequences."""

    observations = data.weighted_targets()

    active = observations.weights.reshape(-1)

    predictors = data.safe_predictors().reshape(
        (
            -1,
            data.num_lags,
            data.observation_dim,
        )
    )

    targets = observations.safe_values().reshape(
        (
            -1,
            data.observation_dim,
        )
    )

    return (
        predictors[active],
        targets[active],
    )


def init_from_emissions(
    emissions: AREmissionsCollection[ConditionalDistT],
    num_states: int,
    dtype: jnp.dtype,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
) -> Model[ConditionalDistT]:
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


def cluster_rows(
    key: jax.Array,
    data: ARDataset,
    num_states: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:  # (R, L, D_y), (R, D_y), (R,)
    """Pool visible complete AR windows and assign them to states by K-means.

    Returns the lagged predictors, the current targets, and one state
    assignment per visible regression row.
    """
    predictors, targets = _visible_rows(data)

    if predictors.shape[0] < num_states:
        raise ValueError(
            f'AR-HMM initialization requires at least {num_states} '
            f'visible complete regression rows; got {predictors.shape[0]}'
        )

    assignments = _init_state_assignments(
        key=key,
        predictors=predictors,
        current=targets,
        num_states=num_states,
    )

    return predictors, targets, assignments


def init_gaussian_via_kmeans(
    key: jax.Array,
    data: ARDataset,
    num_states: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
    covariance_floor: float = gaussian_fit.DEFAULT_COV_FLOOR_INIT,
) -> Model[LinearGaussian]:
    r"""Initialize a Gaussian AR-HMM by K-means over complete AR windows.

    The first $L$ values of each sequence are fixed conditioning history.
    """
    predictors, targets, assignments = cluster_rows(key, data, num_states)
    emissions = AREmissions(
        gaussian_fit.linear_from_samples_grouped(
            inputs=predictors,
            outputs=targets,
            assignments=assignments,
            num_groups=num_states,
            ridge=gaussian_fit.DEFAULT_RIDGE,
            covariance_floor=covariance_floor,
        )
    )
    return init_from_emissions(
        AREmissionsCollection.from_single(emissions),
        num_states,
        jnp.result_type(targets, jnp.float32),
        self_transition_prob=self_transition_prob,
    )


def init_poisson_via_kmeans(
    key: jax.Array,
    data: ARDataset,
    num_states: int,
    *,
    self_transition_prob: float = DEFAULT_SELF_TRANSITION_PROB,
) -> Model[LinearPoisson]:
    r"""Initialize a Poisson AR-HMM by K-means over complete AR windows.

    The first $L$ values of each sequence are fixed conditioning history.
    """
    predictors, targets, assignments = cluster_rows(key, data, num_states)
    emissions = AREmissions(
        poisson_fit.linear_from_samples_grouped(
            inputs=predictors,
            outputs=targets,
            assignments=assignments,
            num_groups=num_states,
            ridge=poisson_fit.DEFAULT_RIDGE,
        )
    )
    return init_from_emissions(
        AREmissionsCollection.from_single(emissions),
        num_states,
        jnp.result_type(targets, jnp.float32),
        self_transition_prob=self_transition_prob,
    )
