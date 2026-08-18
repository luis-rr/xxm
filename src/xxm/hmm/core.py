r"""Single-sequence homogeneous Hidden Markov Model."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from xxm.core.discrete.chain import DiscreteChainMarginals as Posterior


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
    initial_probs: jax.Array

    @property
    def num_states(self) -> int:
        return self.initial_probs.shape[0]

    def sample(self, key: jax.Array) -> jax.Array:
        return jax.random.categorical(
            key,
            jnp.log(self.initial_probs),
        )

    def permute(self, permutation: jax.Array) -> DiscreteInitialModel:
        return DiscreteInitialModel(
            initial_probs=self.initial_probs[permutation],
        )

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        r"""
        Maximum-likelihood update of the initial-state probabilities.

        .. math::

            \pi_k^{new}
            = p(z_0 = k \mid y)
            = \gamma_0(k)

        where ``gamma`` is the smoothed state posterior.
        """
        return self.__class__(initial_probs=posterior.state_marginals[0])


class DiscreteTransitionModel(typing.NamedTuple):
    transition_probs: jax.Array

    @property
    def num_states(self) -> int:
        return self.transition_probs.shape[0]

    def sample(
        self,
        key: jax.Array,
        previous: jax.Array,
    ) -> jax.Array:
        return jax.random.categorical(
            key,
            jnp.log(self.transition_probs[previous]),
        )

    def permute(self, permutation: jax.Array) -> DiscreteTransitionModel:
        return DiscreteTransitionModel(
            transition_probs=self.transition_probs[permutation][:, permutation],
        )

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        r"""
        Maximum-likelihood update of the transition probabilities.

        .. math::

            A_{ij}^{new}
            =
            \frac{\sum_t \xi_t(i,j)}
                {\sum_t \sum_j \xi_t(i,j)}

        where ``xi[t, i, j]`` is the posterior probability of the
        transition ``i -> j`` at time ``t``.
        """
        expected_transitions = posterior.pair_marginals.sum(axis=0)

        return self.__class__(
            transition_probs=expected_transitions / expected_transitions.sum(axis=-1, keepdims=True)
        )

    def broadcast(
        self,
        batch_shape: tuple[int, ...],
    ) -> jax.Array:
        return jnp.broadcast_to(
            self.transition_probs,
            (
                *batch_shape,
                self.num_states,
                self.num_states,
            ),
        )


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
        return self.initial.initial_probs.shape[0]

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
