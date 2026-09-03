r"""Single-sequence homogeneous Hidden Markov Model."""

from __future__ import annotations

import typing

import jax

from xxm.core.chains.discrete import DiscreteChainMarginals as Posterior
from xxm.core.emissions.discrete import ContinuationEmissions, Emissions
from xxm.core.models.discrete import CategoricalInitial, CategoricalTransitions

EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)

ContinuationEmissionsT = typing.TypeVar(
    'ContinuationEmissionsT',
    bound=ContinuationEmissions,
)


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Container for HMM model parameters.

    ``initial`` defines the prior over the first modeled state,
    ``transitions`` defines the state Markov chain, and ``emissions`` defines
    the observation model associated with each state.

    Ordinary HMMs associate one state with every observation. Autoregressive
    emissions may condition on an initial observation history, so their first
    modeled state corresponds to the first observation following that history.
    """

    initial: CategoricalInitial
    transitions: CategoricalTransitions
    emissions: EmissionsT

    @property
    def num_states(self) -> int:
        return self.initial.num_states

    def permute(
        self,
        permutation: jax.Array,
    ) -> Model:
        return Model(
            initial=self.initial.permute(permutation),
            transitions=self.transitions.permute(permutation),
            emissions=self.emissions.permute(permutation),
        )

    def sample_states(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample a complete discrete state trajectory."""
        key_initial, key_transitions = jax.random.split(key)

        initial_state = self.initial.sample(
            key_initial,
        )

        return self.transitions.sample(
            key_transitions,
            initial_state,
            num_steps,
        )

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample latent states and observations autonomously."""
        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            num_steps,
        )

        observations = self.emissions.sample(key_observations, states)

        return states, observations

    def sample_continuation(
        self: Model[ContinuationEmissionsT],
        key: jax.Array,
        num_steps: int,
        initial_history: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample states and observations conditional on an explicit history."""
        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(key_states, num_steps)

        observations = self.emissions.sample_continuation(
            key_observations, states, initial_history
        )

        return states, observations

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> Model[EmissionsT]:
        """Fit model parameters to observations and a state posterior."""
        return Model(
            initial=self.initial.fit_params(posterior),
            transitions=self.transitions.fit_params(posterior),
            emissions=self.emissions.fit_params(
                observations,
                posterior,
            ),
        )
