"""Forward-backward inference for a finite-state discrete chain."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp


class DiscretePotential(typing.NamedTuple):
    """Unary log potentials over a discrete variable."""

    log_values: jax.Array  # (..., K)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.log_values.shape[:-1]

    @property
    def num_states(self) -> int:
        return self.log_values.shape[-1]


class DiscreteChain(typing.NamedTuple):
    r"""Parameters defining a finite-state chain with local state potentials.

    For fixed observations, the unnormalized distribution over states is

        f(z_{0:T-1})
        = p(z_0)
        \prod_{t=0}^{T-2} p(z_{t+1} | z_t)
        \prod_{t=0}^{T-1} f_t(z_t).

    where

    * ``initial_probs[k] = p(z_0=k)``
    * ``transition_probs[t, i, j] = p(z_{t+1}=j | z_t=i)``
    * ``state_log_potentials[t, k] = log f_t(z_t=k)``

    A ``DiscreteChain`` represents a single chain. Batch dimensions are
    intentionally not supported; use ``jax.vmap`` over chains instead.

    Transitions are always represented explicitly for each transition,
    with shape ``(T - 1, K, K)``.
    """

    initial_probs: jax.Array  # (K,)
    transition_probs: jax.Array  # (T - 1, K, K)
    state_log_potentials: jax.Array  # (T, K)

    def validate(self) -> None:
        if self.initial_probs.ndim != 1:
            raise ValueError(
                f'initial_probs must have shape (K,). Got shape {self.initial_probs.shape}'
            )

        if self.transition_probs.ndim != 3:
            raise ValueError(
                f'transition_probs must have shape (T - 1, K, K). '
                f'Got shape {self.transition_probs.shape}'
            )

        if self.state_log_potentials.ndim != 2:
            raise ValueError(
                f'state_log_potentials must have shape (T, K). '
                f'Got shape {self.state_log_potentials.shape}'
            )

        k = self.initial_probs.shape[0]
        t = self.state_log_potentials.shape[0]

        if t < 1:
            raise ValueError('Chain must contain at least one time step')

        if k < 1:
            raise ValueError('Chain must contain at least one state')

        if self.state_log_potentials.shape != (t, k):
            raise ValueError(
                f'state_log_potentials must have shape (T, K). '
                f'Got shape {self.state_log_potentials.shape}'
            )

        if self.transition_probs.shape != (t - 1, k, k):
            raise ValueError(
                f'transition_probs must have shape (T - 1, K, K). '
                f'Got shape {self.transition_probs.shape}'
            )

    @property
    def num_states(self) -> int:
        return self.initial_probs.shape[0]

    @property
    def num_time_steps(self) -> int:
        return self.state_log_potentials.shape[0]

    def forward_backward(self) -> DiscreteChainMarginals:
        """Run full forward-backward inference for one chain."""
        self.validate()

        messages = _forward_backward(self)

        return messages.calculate_marginals(self)

    def add_local_potential(
        self,
        potential: DiscretePotential,
    ) -> DiscreteChain:
        if potential.batch_shape != (self.num_time_steps,):
            raise ValueError(...)

        if potential.num_states != self.num_states:
            raise ValueError(...)

        return self._replace(
            state_log_potentials=(self.state_log_potentials + potential.log_values)
        )


class DiscreteChainMarginals(typing.NamedTuple):
    r"""Marginals and log normalizer of a discrete chain.

    * ``state_marginals[t, k] = p_f(z_t=k)``.
    * ``pair_marginals[t, i, j] = p_f(z_t=i, z_{t+1}=j)``.
    * ``log_normalizer = log Z``.

    Represents the marginals of one chain. Batch dimensions can be introduced
    externally by applying ``jax.vmap`` to chain inference.
    """

    state_marginals: jax.Array  # (T, K)
    pair_marginals: jax.Array  # (T - 1, K, K)
    log_normalizer: jax.Array  # scalar


class _DiscreteChainMessages(typing.NamedTuple):
    r"""Normalized messages and per-step log normalizers for one chain."""

    forward_messages: jax.Array  # (T, K)
    backward_messages: jax.Array  # (T, K)
    log_scaling_factors: jax.Array  # (T,)

    def calculate_marginals(
        self,
        chain: DiscreteChain,
    ) -> DiscreteChainMarginals:
        return DiscreteChainMarginals(
            state_marginals=self.calculate_state_marginals(),
            pair_marginals=self.calculate_pair_marginals(chain),
            log_normalizer=self.calculate_log_normalizer(),
        )

    def calculate_state_marginals(self) -> jax.Array:
        """Compute state probabilities gamma[t, k]."""
        if (
            self.forward_messages.ndim != 2
            or self.backward_messages.ndim != 2
            or self.forward_messages.shape != self.backward_messages.shape
        ):
            raise ValueError(
                'forward_messages and backward_messages must both have shape (T, K) and match.'
            )

        unnormalized_state_marginals = self.forward_messages * self.backward_messages

        min_normalizer = jnp.finfo(self.forward_messages.dtype).tiny

        state_marginal_normalizers = jnp.maximum(
            jnp.sum(
                unnormalized_state_marginals,
                axis=1,
                keepdims=True,
            ),
            min_normalizer,
        )

        return unnormalized_state_marginals / state_marginal_normalizers

    def calculate_pair_marginals(
        self,
        chain: DiscreteChain,
    ) -> jax.Array:
        """Compute pair marginal probabilities xi[t, i, j]."""
        if (
            self.forward_messages.ndim != 2
            or self.backward_messages.ndim != 2
            or self.forward_messages.shape != self.backward_messages.shape
        ):
            raise ValueError(
                'forward_messages and backward_messages must both have shape (T, K) and match.'
            )

        t = chain.num_time_steps
        k = chain.num_states

        if t == 1:
            return jnp.zeros(
                (0, k, k),
                dtype=self.forward_messages.dtype,
            )

        def pair_step(
            current_forward_messages: jax.Array,
            transition_probs: jax.Array,
            next_backward_messages: jax.Array,
            next_state_log_potential: jax.Array,
            next_log_normalizer: jax.Array,
        ) -> jax.Array:
            observation_offset = jnp.max(next_state_log_potential)

            next_observation_weights = jnp.exp(next_state_log_potential - observation_offset)

            future_weights = next_observation_weights * next_backward_messages

            unnormalized_pair_probs = (
                current_forward_messages[:, None] * transition_probs * future_weights[None, :]
            )

            normalization_scale = jnp.exp(observation_offset - next_log_normalizer)

            return unnormalized_pair_probs * normalization_scale

        return jax.vmap(pair_step)(
            self.forward_messages[:-1],
            chain.transition_probs,
            self.backward_messages[1:],
            chain.state_log_potentials[1:],
            self.log_scaling_factors[1:],
        )

    def calculate_log_normalizer(self) -> jax.Array:
        """Compute the log normalizer of the chain distribution."""
        return jnp.sum(self.log_scaling_factors)


def _forward_pass(
    chain: DiscreteChain,
) -> tuple[jax.Array, jax.Array]:
    """Run a normalized probability-space forward recursion."""
    t = chain.num_time_steps

    min_normalizer = jnp.finfo(chain.state_log_potentials.dtype).tiny

    def _normalized_observation_weights(
        state_log_potential: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        observation_offset = jnp.max(state_log_potential)
        observation_weights = jnp.exp(state_log_potential - observation_offset)

        return observation_weights, observation_offset

    (
        first_observation_weights,
        first_observation_offset,
    ) = _normalized_observation_weights(chain.state_log_potentials[0])

    first_unnormalized = chain.initial_probs * first_observation_weights

    first_normalizer = jnp.maximum(
        jnp.sum(first_unnormalized),
        min_normalizer,
    )

    first_forward_messages = first_unnormalized / first_normalizer

    first_log_normalizer = jnp.log(first_normalizer) + first_observation_offset

    def forward_step(
        previous_forward_messages: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[
        jax.Array,
        tuple[jax.Array, jax.Array],
    ]:
        transition_probs, state_log_potential = inputs

        predictive_probs = previous_forward_messages @ transition_probs

        (
            observation_weights,
            observation_offset,
        ) = _normalized_observation_weights(state_log_potential)

        unnormalized_forward_messages = predictive_probs * observation_weights

        forward_normalizer = jnp.maximum(
            jnp.sum(unnormalized_forward_messages),
            min_normalizer,
        )

        current_forward_messages = unnormalized_forward_messages / forward_normalizer

        current_log_normalizer = jnp.log(forward_normalizer) + observation_offset

        return current_forward_messages, (
            current_forward_messages,
            current_log_normalizer,
        )

    if t == 1:
        return (
            first_forward_messages[None, :],
            first_log_normalizer[None],
        )

    (
        _,
        (
            remaining_forward_messages,
            remaining_log_scaling_factors,
        ),
    ) = jax.lax.scan(
        forward_step,
        first_forward_messages,
        (
            chain.transition_probs,
            chain.state_log_potentials[1:],
        ),
    )

    forward_messages = jnp.concatenate(
        [
            first_forward_messages[None, :],
            remaining_forward_messages,
        ],
        axis=0,
    )

    log_scaling_factors = jnp.concatenate(
        [
            first_log_normalizer[None],
            remaining_log_scaling_factors,
        ],
        axis=0,
    )

    return forward_messages, log_scaling_factors


def _backward_pass(
    chain: DiscreteChain,
    log_scaling_factors: jax.Array,
) -> jax.Array:
    """Run the backward recursion consistent with forward normalizers."""
    k = chain.num_states
    t = chain.num_time_steps

    if log_scaling_factors.shape != (t,):
        raise ValueError('log_scaling_factors must have shape (T,)')

    terminal_backward_messages = jnp.ones(
        (k,),
        dtype=chain.state_log_potentials.dtype,
    )

    if t == 1:
        return terminal_backward_messages[None, :]

    def backward_step(
        backward_messages_at_next_time: jax.Array,
        inputs: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[jax.Array, jax.Array]:
        (
            transition_probs,
            next_state_log_potential,
            next_log_normalizer,
        ) = inputs

        observation_offset = jnp.max(next_state_log_potential)

        next_observation_weights = jnp.exp(next_state_log_potential - observation_offset)

        weighted_future_probs = next_observation_weights * backward_messages_at_next_time

        propagated_backward_messages = transition_probs @ weighted_future_probs

        normalization_scale = jnp.exp(observation_offset - next_log_normalizer)

        backward_messages_at_current_time = propagated_backward_messages * normalization_scale

        return (
            backward_messages_at_current_time,
            backward_messages_at_current_time,
        )

    _, reverse_backward_messages = jax.lax.scan(
        backward_step,
        terminal_backward_messages,
        (
            chain.transition_probs[::-1],
            chain.state_log_potentials[1:][::-1],
            log_scaling_factors[1:][::-1],
        ),
    )

    return jnp.concatenate(
        [
            reverse_backward_messages[::-1],
            terminal_backward_messages[None, :],
        ],
        axis=0,
    )


def _forward_backward(
    chain: DiscreteChain,
) -> _DiscreteChainMessages:
    """Run full forward-backward inference for one chain."""
    (
        forward_messages,
        log_scaling_factors,
    ) = _forward_pass(chain)

    backward_messages = _backward_pass(
        chain,
        log_scaling_factors=log_scaling_factors,
    )

    return _DiscreteChainMessages(
        forward_messages=forward_messages,
        backward_messages=backward_messages,
        log_scaling_factors=log_scaling_factors,
    )
