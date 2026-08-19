r"""Single-sequence homogeneous Hidden Markov Model."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from xxm.core.discrete.chain import DiscreteChainMarginals as Posterior
from xxm.stats.categorical import Categorical


class Emissions(typing.Protocol):
    def log_likelihoods(self, observations: jax.Array) -> jax.Array: ...

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self: ...

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array: ...

    def permute(
        self,
        permutation: jax.Array,
    ) -> typing.Self: ...


EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


class DiscreteInitialModel(typing.NamedTuple):
    model: Categorical  # no batch

    @property
    def num_states(self) -> int:
        return self.model.num_categories

    def sample(
        self,
        key: jax.Array,
    ) -> jax.Array:  # ()
        return self.model.sample(key)

    def permute(
        self,
        permutation: jax.Array,  # (K,)
    ) -> DiscreteInitialModel:
        return self._replace(
            model=self.model.permute(permutation),
        )

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        """Maximum-likelihood update from expected initial-state counts."""
        return self._replace(model=Categorical.from_counts(posterior.state_marginals[0]))


class DiscreteTransitionModel(typing.NamedTuple):
    model: Categorical  # K-batched

    @property
    def num_states(self) -> int:
        return self.model.num_categories

    def conditional(
        self,
        previous: jax.Array,  # (...)
    ) -> Categorical:
        """Conditional distribution of the next state."""
        return self.model.select(previous)

    def sample(
        self,
        key: jax.Array,
        previous: jax.Array,  # (...)
    ) -> jax.Array:  # (...)
        """Sample the next state conditional on the previous state."""
        return self.conditional(previous).sample(key)

    def permute(
        self,
        permutation: jax.Array,  # (K,)
    ) -> DiscreteTransitionModel:
        return self._replace(model=self.model.select(permutation).permute(permutation))

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        """Maximum-likelihood update from expected transition counts."""
        expected_transitions = posterior.pair_marginals.sum(axis=0)  # (K, K)

        return self._replace(model=Categorical.from_counts(expected_transitions))


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Container for HMM model parameters.

    * ``initial_probs[k] = p(z_0=k)``
    * ``transition_probs[i, j] = p(z_{t+1}=j \mid z_t=i)``
    * ``emission_log_likelihoods[t, k] = \log p(y_t \mid z_t=k)``
    """

    initial: DiscreteInitialModel
    transitions: DiscreteTransitionModel
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
        num_steps: int,
        key: jax.Array,
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
