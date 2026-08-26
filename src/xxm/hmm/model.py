r"""Single-sequence homogeneous Hidden Markov Model."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from xxm.core.chains.discrete import DiscreteChainMarginals as Posterior
from xxm.core.emissions.discrete import Emissions
from xxm.core.models.discrete import CategoricalInitial, CategoricalTransitions

EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Container for HMM model parameters.

    * ``initial_probs[k] = p(z_0=k)``
    * ``transition_probs[i, j] = p(z_{t+1}=j \mid z_t=i)``
    * ``emission_log_likelihoods[t, k] = \log p(y_t \mid z_t=k)``
    """

    initial: CategoricalInitial
    transitions: CategoricalTransitions
    emissions: EmissionsT

    @property
    def num_states(self) -> int:
        return self.initial.num_states

    def permute(self, permutation: jax.Array) -> Model:

        return Model(
            initial=self.initial.permute(permutation),
            transitions=self.transitions.permute(permutation),
            emissions=self.emissions.permute(permutation),
        )

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample latent states and observations from an HMM."""

        key_initial, key_scan, key_observation = jax.random.split(key, 3)

        initial_state = self.initial.sample(key_initial)

        def step(carry, _):
            state, key = carry
            key, key_transition = jax.random.split(key)

            state = self.transitions.sample(
                key_transition,
                state,
            )

            return (state, key), state

        _, subsequent_states = jax.lax.scan(
            step,
            (initial_state, key_scan),
            xs=None,
            length=num_steps - 1,
        )

        states = jnp.concatenate([initial_state[None], subsequent_states])

        observations = self.emissions.sample(
            key_observation,
            states,
        )

        return states, observations

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> Model[EmissionsT]:
        """Fit the model parameters to the given observations and posterior."""

        return Model(
            initial=self.initial.fit_params(posterior),
            transitions=self.transitions.fit_params(posterior),
            emissions=self.emissions.fit_params(observations, posterior),
        )
