r"""Single-sequence autoregressive Hidden Markov Model."""

from __future__ import annotations

import typing

import jax

from xxm.core.chains.discrete import DiscreteChainMarginals as Posterior
from xxm.core.latents.discrete import CategoricalInitial, CategoricalTransitions

from .data import ARObservations
from .emissions import AREmissions, ConditionalDistT


class Model(typing.NamedTuple, typing.Generic[ConditionalDistT]):
    r"""
    HMM whose state selects an autoregressive conditional distribution.

    For history length $L$, posterior row $r$ selects the regression producing
    $y_{L+r}$ from $(y_{L+r-1}, ..., y_r)$. The first $L$ observations are
    fixed conditioning history; the chain has $T-L$ states.

    Sampling uses zero prehistory or explicit chronological continuation history.
    """

    initial: CategoricalInitial
    transitions: CategoricalTransitions
    emissions: AREmissions[ConditionalDistT]

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.initial.num_states

    @property
    def num_lags(self) -> int:
        """Number of autoregressive lags $L$."""
        return self.emissions.num_lags

    @property
    def output_dim(self) -> int:
        """Observation dimension $D_y$."""
        return self.emissions.output_dim

    def permute_states(
        self,
        permutation: jax.Array,
    ) -> Model[ConditionalDistT]:
        """Relabel discrete states by permutation."""
        return Model(
            initial=self.initial.permute_states(permutation),
            transitions=self.transitions.permute_states(permutation),
            emissions=self.emissions.permute_states(permutation),
        )

    def sample_states(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample state trajectory $z_{0:T-1}$ from the prior."""
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
        r"""Sample complete trajectory $(z_{0:T-1}, y_{0:T-1})$ autonomously."""
        key_states, key_observations = jax.random.split(key)

        states = self.sample_states(
            key_states,
            num_steps,
        )

        observations = self.emissions.sample(key_observations, states)

        return states, observations

    def sample_continuation(
        self,
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
        data: ARObservations,
        posterior: Posterior,
    ) -> Model[ConditionalDistT]:
        """
        Perform the EM M-step from posterior state and pair marginals.
        """
        return Model(
            initial=self.initial.fit_params(posterior),
            transitions=self.transitions.fit_params(posterior),
            emissions=self.emissions.fit_params(
                data,
                posterior,
            ),
        )
