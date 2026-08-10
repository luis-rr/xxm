r"""Single-sequence homogeneous Hidden Markov Model."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp


class Emissions(typing.Protocol):
    def log_likelihoods(self, observations: jax.Array) -> jax.Array: ...

    @classmethod
    def initialize(
        cls,
        observations: jax.Array,
        num_states: int,
        key: jax.Array,
    ) -> typing.Self: ...

    def m_step(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self: ...

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array: ...

    def permute(
        self,
        permutation: jax.Array,
    ) -> typing.Self: ...


class Model(typing.NamedTuple):
    r"""
    Container for HMM model parameters.

    * ``initial_probs[k] = p(z_0=k)``
    * ``transition_probs[i, j] = p(z_{t+1}=j \mid z_t=i)``
    * ``emission_log_likelihoods[t, k] = \log p(y_t \mid z_t=k)``
    """

    initial_probs: jax.Array
    transition_probs: jax.Array
    emissions: Emissions

    @property
    def num_states(self) -> int:
        return self.initial_probs.shape[0]

    @classmethod
    def initialize(
        cls,
        observations: jax.Array,
        num_states: int,
        emissions_type: type[Emissions],
        key: jax.Array,
        self_transition_prob: float = 0.9,
    ) -> Model:
        initial_probs = jnp.ones(num_states) / num_states

        off_diagonal_prob = (1.0 - self_transition_prob) / (num_states - 1)

        transition_probs = jnp.full(
            (num_states, num_states),
            off_diagonal_prob,
        )
        transition_probs = transition_probs.at[jnp.diag_indices(num_states)].set(
            self_transition_prob
        )

        emissions = emissions_type.initialize(
            observations,
            num_states,
            key,
        )

        return cls(
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            emissions=emissions,
        )

    def permute(self, permutation: jax.Array) -> Model:

        return Model(
            initial_probs=self.initial_probs[permutation],
            transition_probs=self.transition_probs[
                permutation[:, None],
                permutation[None, :],
            ],
            emissions=self.emissions.permute(permutation),
        )

    def sample(
        self,
        num_steps: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample latent states and observations from an HMM."""

        key_initial, key_scan = jax.random.split(key)

        initial_state = jax.random.categorical(
            key_initial,
            jnp.log(self.initial_probs),
        )

        def step(carry, _):
            state, key = carry

            key, key_observation, key_transition = jax.random.split(
                key,
                3,
            )

            observation = self.emissions.sample(
                key_observation,
                state,
            )

            next_state = jax.random.categorical(
                key_transition,
                jnp.log(self.transition_probs[state]),
            )

            return (next_state, key), (state, observation)

        _, (states, observations) = jax.lax.scan(
            step,
            (initial_state, key_scan),
            xs=None,
            length=num_steps,
        )

        return states, observations


class Posterior(typing.NamedTuple):
    r"""Container for forward-backward inference outputs.

    * ``forward_probs[t]`` is the filtered state distribution
    :math:`p(z_t \mid y_{0:t})`.

    * ``backward_probs[t]`` is the scaled backward message used with
    ``forward_probs[t]`` to compute the smoothed posterior.

    * ``log_scaling_factors[t]`` is the predictive log likelihood
    :math:`\log p(y_t \mid y_{0:t-1})`.

    * ``state_posterior_probs[t]`` is the smoothed state distribution
    :math:`p(z_t \mid y_{0:T-1})`.

    * ``pair_posterior_probs[t, i, j]`` is the smoothed pair distribution
    :math:`p(z_t=i, z_{t+1}=j \mid y_{0:T-1})`.

    * ``log_marginal_likelihood`` is the log likelihood of the full
    observation sequence,
    :math:`\log p(y_{0:T-1}) = \sum_t \log p(y_t \mid y_{0:t-1})`.
    """

    forward_probs: jax.Array
    backward_probs: jax.Array
    log_scaling_factors: jax.Array

    state_marginals: jax.Array
    pair_marginals: jax.Array

    def log_likelihood(self) -> jax.Array:
        return self.log_scaling_factors.sum()
