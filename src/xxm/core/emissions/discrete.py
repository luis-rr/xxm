"""Emission models for discrete latent variables (HMM emissions)."""

import typing

import jax

from xxm.core.chains.discrete import DiscretePotential
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.posteriors import DiscretePosterior


class Emissions(typing.Protocol):
    """Protocol for emission models over discrete latent states."""

    def log_likelihoods(
        self,
        observations: jax.Array,
    ) -> jax.Array:
        """Evaluate state-conditional log likelihoods over time."""
        ...

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> DiscretePotential:
        """Construct discrete state potentials from observation likelihoods."""
        ...

    def fit_params(
        self,
        observations: jax.Array,
        posterior: DiscretePosterior,
    ) -> typing.Self:
        """Fit emission parameters from observations and posterior state weights."""
        ...

    def permute(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel state-specific emission parameters."""
        ...

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        """Sample observations conditional on discrete states."""
        ...


class ContinuationEmissions(Emissions, typing.Protocol):
    """Emissions that can conditionally sample given history."""

    def sample_continuation(
        self,
        key: jax.Array,
        states: jax.Array,
        initial_history: jax.Array,
    ) -> jax.Array: ...


class GaussianEmissions(typing.NamedTuple):
    r"""State-dependent Gaussian emissions.

    $$y_t|z_t=k \sim \mathcal{N}(\mu_k, \Sigma_k).$$
    """

    dist: Gaussian  # K-batched

    @property
    def num_states(self) -> int:
        return self.dist.batch_shape[0]

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        r"""Evaluate log probabilities $\log p(y_t|z_t=k)$ for all states and time."""
        return self.dist.log_prob_broadcast(observations)

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        """Construct one state potential from each observation log likelihood."""
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
            dist=gaussian,
        )

    def sample(self, key: jax.Array, states: jax.Array) -> jax.Array:
        """Sample observations conditional on discrete state indices."""
        return self.dist.select(states).sample(key)

    def permute(self, permutation: jax.Array) -> 'GaussianEmissions':
        return self._replace(
            dist=self.dist.select(permutation),
        )


class PoissonEmissions(typing.NamedTuple):
    r"""State-dependent Poisson emissions parameterized by log rates $\eta_k=\log\lambda_k$."""

    dist: Poisson  # K-batched

    @property
    def num_states(self) -> int:
        return self.dist.batch_shape[0]

    def log_likelihoods(self, observations: jax.Array) -> jax.Array:
        r"""Evaluate $\log p(y_t\mid z_t=k)$ for all states and time."""
        return self.dist.log_prob_broadcast(observations)

    def compute_potential(self, observations: jax.Array) -> DiscretePotential:
        """Construct one state potential from each observation log likelihood."""
        return DiscretePotential(
            log_values=self.log_likelihoods(observations),
        )

    def fit_params(
        self, observations: jax.Array, posterior: DiscretePosterior
    ) -> 'PoissonEmissions':
        """Fit state-specific Poisson log rates from posterior state weights."""
        return self._replace(
            dist=poisson_fit.from_samples_weighted(
                values=observations,
                weights=posterior.state_probs,
            ),
        )

    def permute(
        self,
        permutation: jax.Array,
    ) -> 'PoissonEmissions':
        """Relabel state-specific emission parameters."""
        return self._replace(
            dist=self.dist.select(permutation),
        )

    def sample(
        self,
        key: jax.Array,
        states: jax.Array,
    ) -> jax.Array:
        """Sample observations conditional on discrete state indices."""
        return self.dist.select(states).sample(key)
