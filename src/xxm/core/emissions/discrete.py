import typing

import jax

from xxm.core.chains.discrete import DiscretePotential
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.posteriors import DiscretePosterior


class Emissions(typing.Protocol):
    def log_likelihoods(self, observations: jax.Array) -> jax.Array: ...

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
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


class GaussianEmissions(typing.NamedTuple):
    model: Gaussian  # K-batched

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:

        return self.model.log_prob_broadcast(observations)

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        gaussian = gaussian_fit.from_samples_weighted(
            observations,
            posterior.state_probs,
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

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self, observations: jax.Array, posterior: DiscretePosterior
    ) -> 'PoissonEmissions':
        return self._replace(
            model=poisson_fit.from_samples_weighted(
                values=observations,
                weights=posterior.state_probs,
            ),
        )

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
