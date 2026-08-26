import typing

import jax

from xxm.core.dists.categorical import Categorical
from xxm.core.emissions.discrete import DiscretePosterior


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


class CategoricalTransition(typing.NamedTuple):
    model: Categorical  # K-batched

    @property
    def num_states(self) -> int:
        return self.model.num_categories

    def conditional(self, previous: jax.Array) -> Categorical:
        """Conditional distribution of the next state."""
        return self.model.select(previous)

    def sample(self, key: jax.Array, previous: jax.Array) -> jax.Array:  # (...)
        """Sample the next state conditional on the previous state."""
        return self.conditional(previous).sample(key)

    def permute(self, permutation: jax.Array) -> 'CategoricalTransition':
        return self._replace(model=self.model.select(permutation).permute(permutation))

    def fit_params(self, posterior: DiscretePosterior) -> typing.Self:
        """Maximum-likelihood update from expected transition counts."""
        expected_transitions = posterior.pair_probs.sum(axis=0)  # (K, K)

        return self._replace(model=Categorical.from_counts(expected_transitions))
