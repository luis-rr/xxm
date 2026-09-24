"""Discrete model components: initial distribution and transition probabilities."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.chains.discrete import DiscreteChain
from xxm.core.dists.categorical import Categorical
from xxm.core.optim import categorical as categorical_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.core.posteriors import DiscretePosterior


def homogeneous_chain(
    initial: 'CategoricalInitial',
    transitions: 'CategoricalTransitions',
    num_steps: int,
) -> DiscreteChain:
    """Construct a homogeneous Markov-chain prior over num_steps states."""
    num_states = initial.num_states
    return DiscreteChain(
        initial_probs=initial.dist.probs,
        transition_probs=batch.broadcast_array(
            transitions.dist.probs,
            (num_steps - 1,),
            axis=len(transitions.batch_shape),
        ),
        state_log_potentials=jnp.zeros(
            initial.batch_shape + (num_steps, num_states),
            dtype=initial.dist.probs.dtype,
        ),
    )


class CategoricalInitial(typing.NamedTuple):
    r"""Initial distribution $p(z_0)$ for discrete latent state."""

    dist: Categorical  # structural batch *B

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.dist.batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.dist.num_categories

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample initial discrete state."""
        return self.dist.sample(key)

    def permute_states(self, permutation: jax.Array) -> 'CategoricalInitial':
        """Relabel states by permutation."""
        return self._replace(
            dist=self.dist.permute_categories(permutation),
        )

    def fit_params(
        self,
        posterior: DiscretePosterior,
        pseudocount=categorical_fit.DEFAULT_PSEUDOCOUNT,
    ) -> typing.Self:
        r"""Fit initial distribution from posterior marginals $\gamma_0(k)$."""

        if posterior.num_states != self.num_states:
            raise ValueError('posterior and initial distribution must share states')

        replicate_shape = batch.split_prefix(
            posterior.batch_shape, self.batch_shape, name='posterior batch shape'
        )

        starts = batch.pool_samples(
            posterior.state_probs[..., 0, :], self.batch_shape, replicate_shape
        )
        counts = jnp.sum(starts, axis=-2)
        fitted = categorical_fit.from_counts(counts, pseudocount=pseudocount)

        return self._replace(dist=filter_valid_batches(fitted, self.dist, counts))


class CategoricalTransitions(typing.NamedTuple):
    r"""Stationary transition probabilities for discrete latent states.

    `dist.probs[...,i,j]` stores $P(i,j)=p(z_{t+1}=j\mid z_t=i)$.
    """

    dist: Categorical  # underlying batch (*B, K); K is the previous state axis

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        assert self.dist.batch_shape[-1] == self.dist.num_categories
        return self.dist.num_categories

    def conditional(self, previous: jax.Array) -> Categorical:
        """Conditional distribution $p(z_t|z_{t-1})$ for next state."""
        return batch.take_along_last_batch(self.dist, self.dist.batch_shape, previous)

    def sample_next(self, key: jax.Array, previous: jax.Array) -> jax.Array:
        """Sample next state conditional on previous state."""
        return self.conditional(previous).sample(key)

    def sample(
        self, key: jax.Array, initial_state: jax.Array, num_steps: int
    ) -> jax.Array:
        """Sample state sequence conditional on initial state."""

        def step(carry, _):
            state, key = carry

            key, sample_key = jax.random.split(key)
            state = self.sample_next(
                sample_key,
                state,
            )

            return (state, key), state

        _, subsequent_states = jax.lax.scan(
            step,
            (initial_state, key),
            xs=None,
            length=num_steps - 1,
        )

        subsequent_states = jnp.moveaxis(subsequent_states, 0, -1)
        return jnp.concatenate(
            [
                initial_state[..., None],
                subsequent_states,
            ],
            axis=-1,
        )

    def permute_states(self, permutation: jax.Array) -> 'CategoricalTransitions':
        """Relabel states by permutation."""
        return self._replace(
            dist=self.dist.permute(permutation, axis=-1).permute_categories(permutation)
        )

    def fit_params(
        self,
        posterior: DiscretePosterior,
        pseudocount=categorical_fit.DEFAULT_PSEUDOCOUNT,
        *,
        weights: jax.Array | None = None,
    ) -> typing.Self:
        r"""Fit transition probabilities from posterior pair marginals $\xi_t(i,j)$."""

        if posterior.num_states != self.num_states:
            raise ValueError('posterior and transitions must share states')

        pairs = posterior.pair_probs
        expected_shape = (*posterior.batch_shape, posterior.num_steps - 1)

        if pairs.shape != (*expected_shape, self.num_states, self.num_states):
            raise ValueError('posterior pair probabilities must match time and states')

        if weights is not None:
            if weights.shape != expected_shape:
                raise ValueError(f'transition weights must have shape {expected_shape}')

            expanded_weights = batch.expand_trailing(weights, pairs.ndim)
            pairs = jnp.where(expanded_weights != 0, pairs, 0) * expanded_weights

        replicate_shape = batch.split_prefix(
            posterior.batch_shape, self.batch_shape, name='posterior batch shape'
        )

        samples = batch.pool_samples(
            pairs,
            self.batch_shape,
            (*replicate_shape, posterior.num_steps - 1),
        )
        expected_transitions = samples.sum(axis=-3)

        fitted = categorical_fit.from_counts(
            expected_transitions,
            pseudocount=pseudocount,
        )

        return self._replace(
            dist=filter_valid_batches(
                fitted,
                self.dist,
                expected_transitions,
            )
        )
