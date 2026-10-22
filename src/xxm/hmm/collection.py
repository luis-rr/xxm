"""Multi-session HMM initialization through independent fitting and alignment."""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core.align import (
    PermutationEval,
    maximize_permutation_exhaustive,
    maximize_permutation_random,
)
from xxm.core.data import DatasetCollection
from xxm.core.emissions.collection import DiscreteEmissionsCollection
from xxm.core.inference import InferenceState
from xxm.core.optim import categorical as categorical_fit
from xxm.core.optim.coordinate import coordinate_ascent
from xxm.hmm.core import EmissionsT, Model, Posterior

InferenceT = InferenceState[Model, Posterior]


PermutationSearch = typing.Callable[
    [jax.Array, PermutationEval, jax.Array],
    jax.Array,
]


class MarkovAlignmentState(typing.NamedTuple):
    """Pooled Markov sufficient statistics under session permutations."""

    initial: jax.Array  # (S, K)
    transitions: jax.Array  # (S, K, K)
    permutations: jax.Array  # (S, K)
    pooled_initial: jax.Array  # (K,)
    pooled_transitions: jax.Array  # (K, K)

    @classmethod
    def from_counts(
        cls,
        initial: jax.Array,
        transitions: jax.Array,
        permutations: jax.Array,
    ) -> typing.Self:
        aligned_initial = jnp.take_along_axis(
            initial,
            permutations,
            axis=1,
        )

        aligned_transitions = jax.vmap(
            lambda counts, perm: counts[
                perm[:, None],
                perm[None, :],
            ]
        )(transitions, permutations)

        return cls(
            initial=initial,
            transitions=transitions,
            permutations=permutations,
            pooled_initial=aligned_initial.sum(axis=0),
            pooled_transitions=aligned_transitions.sum(axis=0),
        )

    def contribution(
        self,
        session: jax.Array,
        permutation: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Aligned sufficient statistics contributed by one session."""
        return (
            self.initial[session, permutation],
            self.transitions[session][
                permutation[:, None],
                permutation[None, :],
            ],
        )

    def update(
        self,
        session: jax.Array,
        permutation: jax.Array,
    ) -> typing.Self:
        """Replace one session's permutation."""
        old_initial, old_transitions = self.contribution(
            session,
            self.permutations[session],
        )

        new_initial, new_transitions = self.contribution(
            session,
            permutation,
        )

        return self._replace(
            permutations=self.permutations.at[session].set(permutation),
            pooled_initial=(self.pooled_initial + (new_initial - old_initial)),
            pooled_transitions=(
                self.pooled_transitions + (new_transitions - old_transitions)
            ),
        )

    def score(self) -> jax.Array:
        """Profile expected log-likelihood of the shared Markov chain."""
        pseudocount = categorical_fit.DEFAULT_PSEUDOCOUNT

        initial_probs = categorical_fit.from_counts(
            self.pooled_initial,
            pseudocount=pseudocount,
        ).probs

        transition_probs = categorical_fit.from_counts(
            self.pooled_transitions,
            pseudocount=pseudocount,
        ).probs

        return (
            jsp.special.xlogy(
                self.pooled_initial,
                initial_probs,
            ).sum()
            + jsp.special.xlogy(
                self.pooled_transitions,
                transition_probs,
            ).sum()
        )


@functools.partial(
    jax.jit,
    static_argnames=('search_permutation', 'max_sweeps'),
)
def align_markov_counts(
    key: jax.Array,
    initial: jax.Array,
    transitions: jax.Array,
    search_permutation: PermutationSearch,
    *,
    max_sweeps: int = 50,
) -> jax.Array:
    """Align independent Markov chains by pooled likelihood maximization."""
    num_sessions, num_states = initial.shape

    identity = jnp.arange(num_states, dtype=jnp.int32)

    permutations = jnp.broadcast_to(
        identity,
        (num_sessions, num_states),
    )

    state = MarkovAlignmentState.from_counts(
        initial,
        transitions,
        permutations,
    )

    reference = jnp.argmax(transitions.sum(axis=(-2, -1)))

    state = coordinate_ascent(
        key,
        state,
        num_coordinates=num_sessions,
        objective=MarkovAlignmentState.score,
        coordinate=lambda state, session: state.permutations[session],
        update=MarkovAlignmentState.update,
        propose=search_permutation,
        active=(jnp.arange(num_sessions) != reference),
        max_sweeps=max_sweeps,
    )

    return state.permutations


def _search_exhaustive(
    _: jax.Array,
    objective: PermutationEval,
    current: jax.Array,
) -> jax.Array:
    """Facade adapter for deterministic exhaustive search."""
    return maximize_permutation_exhaustive(
        objective,
        current.shape[0],
    )


def _search_random(
    key: jax.Array,
    objective: PermutationEval,
    current: jax.Array,
    *,
    num_restarts: int,
) -> jax.Array:
    """Facade adapter for randomized search."""
    return maximize_permutation_random(
        objective,
        current,
        key,
        num_restarts=num_restarts,
    )


@functools.cache
def _random_policy(num_restarts: int) -> PermutationSearch:
    return functools.partial(
        _search_random,
        num_restarts=num_restarts,
    )


InitSingle = typing.Callable[
    [jax.Array, DatasetCollection],
    Model,
]


def collection_align(
    key: jax.Array,
    posteriors: tuple[Posterior, ...],
    *,
    num_restarts: int | None = 10,
) -> jax.Array:
    """Align independently inferred HMM sessions."""

    if num_restarts is None:
        search_permutation = _search_exhaustive
    else:
        search_permutation = _random_policy(num_restarts)

    initial = jnp.stack(
        [posterior.expected_initial_counts() for posterior in posteriors]
    )

    transitions = jnp.stack(
        [posterior.expected_transition_counts() for posterior in posteriors]
    )

    return align_markov_counts(
        key,
        initial,
        transitions,
        search_permutation,
    )


def collection_compose(
    models: tuple[Model[EmissionsT], ...],
    posteriors: tuple[Posterior, ...],
    alignment: jax.Array,
) -> Model[EmissionsT]:
    """Compose aligned HMMs by pooling Markov sufficient statistics."""

    aligned_models = tuple(
        model.permute_states(permutation)
        for model, permutation in zip(models, alignment, strict=True)
    )

    aligned_posteriors = tuple(
        posterior.permute_states(permutation)
        for posterior, permutation in zip(posteriors, alignment, strict=True)
    )

    initial_counts = jnp.stack(
        [p.expected_initial_counts() for p in aligned_posteriors]
    ).sum(axis=0)

    transition_counts = jnp.stack(
        [p.expected_transition_counts() for p in aligned_posteriors]
    ).sum(axis=0)

    initial = aligned_models[0].initial
    transitions = aligned_models[0].transitions

    pseudocount = categorical_fit.DEFAULT_PSEUDOCOUNT

    fitted_initial = categorical_fit.from_counts(
        initial_counts,
        pseudocount=pseudocount,
    )

    fitted_transitions = categorical_fit.from_counts(
        transition_counts,
        pseudocount=pseudocount,
    )

    # Preserve the existing fit_params policy for unsupported
    # sufficient-statistic batches.
    initial_probs = jnp.where(
        initial_counts.sum() > 0,
        fitted_initial.probs,
        initial.dist.probs,
    )

    transition_probs = jnp.where(
        transition_counts.sum(axis=-1, keepdims=True) > 0,
        fitted_transitions.probs,
        transitions.dist.probs,
    )

    return Model(
        initial=initial._replace(dist=fitted_initial._replace(probs=initial_probs)),
        transitions=transitions._replace(
            dist=fitted_transitions._replace(probs=transition_probs)
        ),
        emissions=DiscreteEmissionsCollection(
            tuple(model.emissions.get() for model in aligned_models)
        ),
    )


def combine(
    key: jax.Array,
    models: tuple[Model[EmissionsT], ...],
    posteriors: tuple[Posterior, ...],
    *,
    num_restarts: int | None = 10,
) -> Model[EmissionsT]:
    """Align and compose existing HMM models without additional fitting."""

    alignment = collection_align(
        key,
        posteriors,
        num_restarts=num_restarts,
    )

    return collection_compose(
        models,
        posteriors,
        alignment,
    )
