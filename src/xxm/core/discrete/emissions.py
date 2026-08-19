import typing

import jax
import jax.numpy as jnp

from xxm.core.discrete.chain import DiscretePotential
from xxm.hmm.core import Posterior
from xxm.stats import gaussian_fit
from xxm.stats.gaussian import Gaussian
from xxm.stats.poisson import Poisson


class GaussianEmissions(typing.NamedTuple):
    model: Gaussian  # K-batched

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:

        return self.model.log_prob_broadcast(observations)

    def get_potential(self, observations: jax.Array) -> DiscretePotential:
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        gaussian = gaussian_fit.gaussian_from_pairs_weighted(
            observations,
            posterior.state_marginals,
        )

        return self._replace(
            model=gaussian,
        )

    def sample(self, key: jax.Array, states: jax.Array) -> jax.Array:
        return self.model.select(states).sample(key)

    def permute(self, permutation: jax.Array) -> 'GaussianEmissions':
        return self._replace(
            model=self.model.select(permutation),
        )


class PoissonEmissions(typing.NamedTuple):
    model: Poisson  # K-batched

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        return self.model.log_prob_broadcast(observations)

    def get_potential(self, observations: jax.Array) -> DiscretePotential:
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(self, observations: jax.Array, posterior: Posterior) -> 'PoissonEmissions':
        rates = posterior.weighted_means(observations)
        rates = jnp.maximum(rates, 1e-8)
        return self._replace(model=Poisson(log_rates=jnp.log(rates)))

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'PoissonEmissions':
        return self._replace(
            model=self.model.select(permutation),
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        return self.model.select(states).sample(key)
