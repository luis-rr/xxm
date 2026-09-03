import typing

import jax
import jax.numpy as jnp

from xxm.core.dists.categorical import Categorical
from xxm.core.posteriors import DiscretePosterior


class CategoricalInitial(typing.NamedTuple):
    model: Categorical  # no batch

    @property
    def num_states(self) -> int:
        return self.model.num_categories

    def sample(self, key: jax.Array) -> jax.Array:
        return self.model.sample(key)

    def permute(self, permutation: jax.Array) -> 'CategoricalInitial':
        return self._replace(
            model=self.model.permute(permutation),
        )

    def fit_params(self, posterior: DiscretePosterior) -> typing.Self:
        """Maximum-likelihood update from expected initial-state counts."""
        return self._replace(model=Categorical.from_counts(posterior.state_probs[0]))


class CategoricalTransitions(typing.NamedTuple):
    model: Categorical  # K-batched

    @property
    def num_states(self) -> int:
        return self.model.num_categories

    def conditional(self, previous: jax.Array) -> Categorical:
        """Conditional distribution of the next state."""
        return self.model.select(previous)

    def sample_next(self, key: jax.Array, previous: jax.Array) -> jax.Array:
        """Sample the next state conditional on the previous state."""
        return self.conditional(previous).sample(key)

    def sample(
        self, key: jax.Array, initial_state: jax.Array, num_steps: int
    ) -> jax.Array:
        """Sample a state sequence conditional on its initial state."""

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

        return jnp.concatenate(
            [
                initial_state[None],
                subsequent_states,
            ],
            axis=0,
        )

    def permute(self, permutation: jax.Array) -> 'CategoricalTransitions':
        return self._replace(model=self.model.select(permutation).permute(permutation))

    def fit_params(self, posterior: DiscretePosterior) -> typing.Self:
        """Maximum-likelihood update from expected transition counts."""
        expected_transitions = posterior.pair_probs.sum(axis=0)  # (K, K)

        return self._replace(model=Categorical.from_counts(expected_transitions))
