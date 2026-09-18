"""Discrete model components: initial distribution and transition probabilities."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import _batch
from xxm.core.chains.discrete import DiscreteChain
from xxm.core.dists.categorical import Categorical
from xxm.core.optim import categorical as categorical_fit
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
        transition_probs=_batch.broadcast_array(
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

    def select(self, index) -> typing.Self:
        return self._replace(dist=self.dist.select(index))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        return self._replace(dist=self.dist.squeeze(axis))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return self._replace(dist=self.dist.move_axis(source, destination))

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

        return self._replace(
            dist=categorical_fit.from_counts(
                posterior.state_probs[..., 0, :],
                pseudocount=pseudocount,
            )
        )


class CategoricalTransitions(typing.NamedTuple):
    r"""Stationary transition probabilities for discrete latent states.

    `dist.probs[...,i,j]` stores $P(i,j)=p(z_{t+1}=j\mid z_t=i)$.
    """

    dist: Categorical  # underlying batch (*B, K); K is the previous state axis

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

    def select(self, index) -> typing.Self:
        index = _batch.selection(index, len(self.batch_shape))
        return self._replace(dist=self.dist.select(index + (slice(None),)))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self._replace(dist=self.dist.squeeze(axes))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        axis = _batch.axis_index(axis, len(self.batch_shape))
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self._replace(dist=self.dist.move_axis(source, destination))

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        assert self.dist.batch_shape[-1] == self.dist.num_categories
        return self.dist.num_categories

    def conditional(self, previous: jax.Array) -> Categorical:
        """Conditional distribution $p(z_t|z_{t-1})$ for next state."""
        return _batch.take_along_last_batch(self.dist, self.dist.batch_shape, previous)

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
    ) -> typing.Self:
        r"""Fit transition probabilities from posterior pair marginals $\xi_t(i,j)$."""

        expected_transitions = posterior.pair_probs.sum(axis=-3)  # (*B, K, K)

        return self._replace(
            dist=categorical_fit.from_counts(
                expected_transitions,
                pseudocount=pseudocount,
            )
        )
