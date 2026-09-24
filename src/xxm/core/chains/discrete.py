"""Finite-state chain potentials, marginals, and forward-backward inference."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core import batch
from xxm.core.mask import NO_MASK, ArbitraryMask, Mask


class DiscretePotential(typing.NamedTuple):
    r"""Unary log potential.

    $$\ell_t(k) = \log \phi_t(z_t=k).$$

    `log_values` stores $\ell$. Leading dimensions are batch dimensions.
    """

    log_values: jax.Array  # (..., K)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.log_values.shape[:-1]

    @property
    def num_states(self) -> int:
        """Number of states."""
        return self.log_values.shape[-1]

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)


class DiscreteChain(typing.NamedTuple):
    r"""Finite-state chain with local state potentials.

    The unnormalized joint distribution is

    $$f(z_{0:T-1}) = p(z_0) \prod_{t=0}^{T-2} p(z_{t+1} \mid z_t) \prod_{t=0}^{T-1} \phi_t(z_t).$$

    Where:
    - `initial_probs` stores $\pi_k = p(z_0=k)$
    - `transition_probs[t,i,j]` stores $P_t(i,j) = p(z_{t+1}=j \mid z_t=i)$
    - `state_log_potentials[t,k]` stores $\ell_t(k) = \log \phi_t(z_t=k)$

    Leading batch dimensions index independent chains. Transitions have shape
    `(*B, T - 1, K, K)`.
    """

    initial_probs: jax.Array  # (*B, K)
    transition_probs: jax.Array  # (*B, T - 1, K, K)
    state_log_potentials: jax.Array  # (*B, T, K)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.initial_probs.shape[-1]

    @property
    def num_steps(self) -> int:
        """Number of time steps $T$."""
        return self.state_log_potentials.shape[-2]

    @classmethod
    def from_markov_prior(
        cls,
        initial_probs: jax.Array,
        transition_probs: jax.Array,
    ) -> typing.Self:
        """Construct chain from Markov model with uniform potentials."""

        if initial_probs.ndim < 1 or transition_probs.ndim < 3:
            raise ValueError(
                'initial and transition probabilities need intrinsic state/time axes'
            )
        batch = initial_probs.shape[:-1]
        num_steps = transition_probs.shape[-3] + 1
        num_states = initial_probs.shape[-1]
        if transition_probs.shape != batch + (num_steps - 1, num_states, num_states):
            raise ValueError(
                'transition_probs must have shape (*B, T - 1, K, K) matching initial_probs'
            )

        return cls(
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            state_log_potentials=jnp.zeros(
                batch + (num_steps, num_states),
                dtype=initial_probs.dtype,
            ),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.initial_probs.shape[:-1]

    def forward_backward(
        self,
        valid: Mask = NO_MASK,
    ) -> tuple[DiscreteChainMarginals, jax.Array]:
        """Infer chains with neutral padded factors and exclude invalid suffixes.

        Callers supply neutral local potentials in padded suffixes. Validity
        denotes a contiguous sequence prefix, never observation visibility.
        """
        if self.num_steps < 1:
            raise ValueError('Discrete chain inference requires at least one time step')
        batch_shape = self.batch_shape
        t, k = self.num_steps, self.num_states
        if self.transition_probs.shape != batch_shape + (
            t - 1,
            k,
            k,
        ) or self.state_log_potentials.shape != batch_shape + (t, k):
            raise ValueError(
                'Discrete chain fields must have aligned batch, time, and state dimensions'
            )
        valid = valid.align_batch(batch_shape)
        valid.validate(self.state_log_potentials)

        if not batch_shape:
            return _forward_backward(self).compute_marginals(self, valid)

        return batch.vmap_batch(
            lambda chain, mask: _forward_backward(chain).compute_marginals(chain, mask),
            self,
            valid,
            batch_shape=self.batch_shape,
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def add_local_potential(
        self,
        potential: DiscretePotential,
    ) -> DiscreteChain:
        """Add unary log potentials to each state."""
        if potential.batch_shape != self.batch_shape + (self.num_steps,):
            raise ValueError(
                'Potential must have batch shape '
                f'{self.batch_shape + (self.num_steps,)}; '
                f'got {potential.batch_shape}'
            )

        if potential.num_states != self.num_states:
            raise ValueError(
                f'Potential must have the same number of states as the chain. '
                f'Got {potential.num_states}, expected {self.num_states}'
            )

        return self._replace(
            state_log_potentials=(self.state_log_potentials + potential.log_values)
        )


def _masked_initial_probs(chain: DiscreteChain, valid: Mask) -> jax.Array:
    """Return initial probabilities `(*B, K)`, replacing absent starts with ones.

    Ones give a neutral log potential for empty valid prefixes. A temporary
    length-one time axis lets the temporal mask apply to the initial state;
    that axis is removed before returning. The result is for evaluating log
    potentials, not a normalized distribution for absent starts.
    """
    first_step_valid = valid.materialize(chain.num_steps)[..., :1]

    initial_valid = ArbitraryMask(first_step_valid)

    initial_probs = initial_valid.apply(
        chain.initial_probs[..., None, :],
        fill=1,
    )

    return initial_probs[..., 0, :]


class DiscreteChainMarginals(typing.NamedTuple):
    r"""
    Marginals of the normalized finite-state chain distribution.

    For $q(z) = f(z) / Z$,

    - `state_probs[t,k]` stores $\gamma_t(k) = q(z_t=k)$
    - `pair_probs[t,i,j]` stores $\xi_t(i,j) = q(z_t=i,z_{t+1}=j)$

    Leading batch dimensions index independent chains.
    """

    state_probs: jax.Array  # (*B, T, K)
    pair_probs: jax.Array  # (*B, T - 1, K, K)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading independent-chain dimensions."""
        return self.state_probs.shape[:-2]

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return self.state_probs.shape[-2]

    @property
    def num_states(self) -> int:
        """Number of discrete states."""
        return self.state_probs.shape[-1]

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def incoming_state_probs(
        self,
    ) -> jax.Array:
        """State probabilities associated with incoming transitions."""
        return self.state_probs[..., 1:, :]

    def entropy(self, valid: Mask = NO_MASK) -> jax.Array:
        """
        Entropy of the normalized chain distribution.
        """

        valid = valid.align_batch(self.batch_shape)

        states = valid.apply(self.state_probs)

        initial_entropy = -jnp.sum(
            jsp.special.xlogy(
                states[..., 0, :],
                states[..., 0, :],
            ),
            axis=-1,
        )

        transitions = valid.adjacent_pairs()
        pairs = transitions.apply(self.pair_probs)

        pair_entropy = -jnp.sum(
            jsp.special.xlogy(
                pairs,
                pairs,
            ),
            axis=(-2, -1),
        )

        conditioning_entropy = jnp.sum(
            jsp.special.xlogy(
                states[..., :-1, :],
                states[..., :-1, :],
            ),
            axis=-1,
        )

        return initial_entropy + jnp.sum(
            transitions.apply(pair_entropy + conditioning_entropy),
            axis=-1,
        )

    def expected_log_potential(
        self,
        chain: DiscreteChain,
        valid: Mask = NO_MASK,
    ) -> jax.Array:
        r"""Expected chain log potential $\mathbb{E}_q[\log f(z)]$."""

        batch.require_same_shape(self.batch_shape, chain.batch_shape)

        if self.state_probs.shape != chain.state_log_potentials.shape:
            raise ValueError('chain and posterior time/state dimensions must match')

        if self.pair_probs.shape != chain.transition_probs.shape:
            raise ValueError('chain and posterior transition dimensions must match')

        valid = valid.align_batch(self.batch_shape)

        states = valid.apply(self.state_probs)

        initial_probs = _masked_initial_probs(chain, valid)
        expected_initial = jnp.sum(
            jsp.special.xlogy(
                states[..., 0, :],
                initial_probs,
            ),
            axis=-1,
        )

        transitions = valid.adjacent_pairs()
        pairs = transitions.apply(self.pair_probs)
        transition_probs = transitions.apply(chain.transition_probs, fill=1)

        expected_transitions = jnp.sum(
            jsp.special.xlogy(
                pairs,
                transition_probs,
            ),
            axis=(-2, -1),
        )

        # Zero-probability states contribute zero even for a -inf log potential.
        local = jnp.where(states != 0, valid.apply(chain.state_log_potentials), 0)
        expected_local = jnp.sum(
            states * local,
            axis=-1,
        )

        return (
            expected_initial
            + jnp.sum(transitions.apply(expected_transitions), axis=-1)
            + jnp.sum(valid.apply(expected_local), axis=-1)
        )

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel discrete states by permutation."""
        permutation = batch.permutation_indices(permutation, self.num_states)
        return self._replace(
            state_probs=self.state_probs[..., permutation],
            pair_probs=self.pair_probs[..., permutation, :][..., permutation],
        )


class _DiscreteChainMessages(typing.NamedTuple):
    """
    Normalized forward messages and scaled backward messages with optional leading batch dimensions.

    - `forward_messages[t,k]` are normalized probability-space messages
    - `backward_messages[t,k]` use forward scaling factors and terminate at
      ones; they are not probability distributions over states
    - `log_scaling_factors[t]` are per-step log normalization constants
    """

    forward_messages: jax.Array  # (*B, T, K)
    backward_messages: jax.Array  # (*B, T, K)
    log_scaling_factors: jax.Array  # (*B, T,)

    def compute_marginals(
        self,
        chain: DiscreteChain,
        valid: Mask = NO_MASK,
    ) -> tuple[DiscreteChainMarginals, jax.Array]:
        """Compute normalized chain marginals and the log normalizer."""

        posterior = DiscreteChainMarginals(
            state_probs=self.compute_state_marginals(),
            pair_probs=self.compute_pair_marginals(chain),
        )

        log_normalizer = self.compute_log_normalizer(valid)

        return posterior, log_normalizer

    def compute_state_marginals(self) -> jax.Array:
        r"""
        Compute state marginals $\gamma_t(k)$.
        """
        if (
            self.forward_messages.ndim < 2
            or self.backward_messages.ndim < 2
            or self.forward_messages.shape != self.backward_messages.shape
        ):
            raise ValueError(
                'forward_messages and backward_messages must both have shape (*B, T, K) and match.'
            )

        unnormalized_state_marginals = self.forward_messages * self.backward_messages

        min_normalizer = jnp.finfo(self.forward_messages.dtype).tiny

        state_marginal_normalizers = jnp.maximum(
            jnp.sum(
                unnormalized_state_marginals,
                axis=-1,
                keepdims=True,
            ),
            min_normalizer,
        )

        return unnormalized_state_marginals / state_marginal_normalizers

    def compute_pair_marginals(
        self,
        chain: DiscreteChain,
    ) -> jax.Array:
        r"""
        Compute adjacent-state marginals $\xi_t(i,j)$.
        """
        if (
            self.forward_messages.ndim < 2
            or self.backward_messages.ndim < 2
            or self.forward_messages.shape != self.backward_messages.shape
        ):
            raise ValueError(
                'forward_messages and backward_messages must both have shape (*B, T, K) and match.'
            )

        if self.forward_messages.shape != chain.state_log_potentials.shape:
            raise ValueError('message and chain shapes must match')
        next_potentials = chain.state_log_potentials[..., 1:, :]
        offset = jnp.max(next_potentials, axis=-1, keepdims=True)
        future = jnp.exp(next_potentials - offset) * self.backward_messages[..., 1:, :]
        scale = jnp.exp(offset[..., 0] - self.log_scaling_factors[..., 1:])
        return (
            self.forward_messages[..., :-1, :, None]
            * chain.transition_probs
            * future[..., None, :]
            * scale[..., None, None]
        )

    def compute_log_normalizer(self, valid: Mask = NO_MASK) -> jax.Array:
        """Compute the chain log normalizer from forward scaling factors."""
        valid = valid.align_batch(self.log_scaling_factors.shape[:-1])
        return jnp.sum(valid.apply(self.log_scaling_factors), axis=-1)


def _forward_pass(
    chain: DiscreteChain,
) -> tuple[jax.Array, jax.Array]:
    """Run a normalized probability-space forward recursion."""
    t = chain.num_steps

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

    # TODO: Explore parallel-prefix inference with jax.lax.associative_scan
    # for long sequences and accelerators before considering it as an
    # alternative to the current sequential scan.
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
    t = chain.num_steps

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

        next_observation_weights = jnp.exp(
            next_state_log_potential - observation_offset
        )

        weighted_future_probs = (
            next_observation_weights * backward_messages_at_next_time
        )

        propagated_backward_messages = transition_probs @ weighted_future_probs

        normalization_scale = jnp.exp(observation_offset - next_log_normalizer)

        backward_messages_at_current_time = (
            propagated_backward_messages * normalization_scale
        )

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
