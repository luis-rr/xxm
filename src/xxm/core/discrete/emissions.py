import typing

import jax
import jax.numpy as jnp

from xxm.hmm.core import Posterior
from xxm.stats import gaussian, poisson


class GaussianEmissions(typing.NamedTuple):
    means: jax.Array  # (K, N)
    covariances: jax.Array  # (K, N, N)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:

        return gaussian.log_likelihoods(
            observations=observations,
            means=self.means[None, :, :],
            covariances=self.covariances,
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        means, covariances = gaussian.fit_weighted(
            observations,
            posterior.state_marginals,
        )

        return self._replace(
            means=means,
            covariances=covariances,
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        return jax.random.multivariate_normal(
            key,
            self.means[states],
            self.covariances[states],
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'GaussianEmissions':
        return GaussianEmissions(
            means=self.means[permutation],
            covariances=self.covariances[permutation],
        )


class PoissonEmissions(typing.NamedTuple):
    rates: jax.Array  # (K, N)

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        return poisson.log_likelihoods(
            observations,
            jnp.log(self.rates)[None, :, :],
        )

    def fit_params(self, observations: jax.Array, posterior: Posterior) -> 'PoissonEmissions':
        rates = posterior.weighted_means(observations)
        rates = jnp.maximum(rates, 1e-8)
        return self.__class__(rates=rates)

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'PoissonEmissions':
        return PoissonEmissions(
            rates=self.rates[permutation],
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        return jax.random.poisson(
            key,
            self.rates[states],
        )
