from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianPotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.posteriors import ContinuousPosterior


class Emissions(typing.Protocol):
    def sample(self, key, latents) -> jax.Array: ...

    def fit_params(
        self,
        observations: jax.Array,
        posterior: ContinuousPosterior,
    ) -> typing.Self: ...

    def log_likelihood(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array: ...

    def compose_input(
        self,
        alignment: Affine,
    ) -> typing.Self: ...


class QuadraticEmissions(Emissions, typing.Protocol):
    """Emissions with a quadratic log-likelihood, so that the posterior is Gaussian."""

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential: ...


QuadraticEmissionsT = typing.TypeVar(
    'QuadraticEmissionsT',
    bound=QuadraticEmissions,
)


class LaplaceEmissions(Emissions, typing.Protocol):
    """Emissions that provide a local quadratic approximation for a latent."""

    def compute_local_potential(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> GaussianPotential: ...


LaplaceEmissionsT = typing.TypeVar(
    'LaplaceEmissionsT',
    bound=LaplaceEmissions,
)


class GaussianEmissions(typing.NamedTuple):
    """Linear Gaussian emissions for continuous latent variables."""

    model: LinearGaussian  # no batch

    def conditional(
        self,
        latents: jax.Array,  # (..., D)
    ) -> Gaussian:
        """Conditional observation distribution given latent values."""
        return self.model.conditional(latents)

    def log_likelihood(
        self,
        observations: jax.Array,  # (T, N)
        latents: jax.Array,  # (T, D)
    ) -> jax.Array:  # ()
        """Compute the total conditional log likelihood."""
        return jnp.sum(self.conditional(latents).log_prob(observations))

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential:
        """Convert the Gaussian likelihood into a potential over latents."""
        return GaussianPotential.from_linear_likelihood(
            self.model,
            observations,
        )

    def sample(
        self,
        key: jax.Array,
        latents: jax.Array,  # (..., D)
    ) -> jax.Array:  # (..., N)
        """Sample observations conditional on latent values."""
        return self.conditional(latents).sample(key)

    def fit_params(
        self,
        observations: jax.Array,  # (T, N)
        posterior: ContinuousPosterior,  # (T, D)
    ) -> typing.Self:
        """Fit the emission parameters from Gaussian latent marginals."""
        means = posterior.means  # (T, D)
        second_moments = posterior.raw_second_moments()  # (T, D, D)

        num_samples = observations.shape[0]

        model = gaussian_fit.linear_from_moments(
            input_mean=jnp.mean(means, axis=0),
            output_mean=jnp.mean(observations, axis=0),
            input_second_moment=jnp.mean(second_moments, axis=0),
            output_second_moment=(observations.T @ observations / num_samples),
            output_input_moment=(observations.T @ means / num_samples),
        )

        return self._replace(
            model=model,
        )

    def observation_mean(self, posterior: ContinuousPosterior) -> jax.Array:
        return self.model.conditional_mean(
            posterior.means,
        )

    def compose_input(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Express the emissions in aligned latent coordinates."""
        return self._replace(
            model=self.model.compose_input(
                alignment.inverse(),
            ),
        )


class PoissonEmissions(typing.NamedTuple):
    """Linear Poisson emissions for continuous latent variables."""

    model: LinearPoisson  # no batch

    def conditional(
        self,
        latents: jax.Array,  # (..., D)
    ) -> Poisson:
        """Conditional observation distribution given latent values."""
        return self.model.conditional(latents)

    def rates(
        self,
        latents: jax.Array,  # (..., D)
    ) -> jax.Array:  # (..., N)
        """Compute conditional Poisson rates."""
        return self.conditional(latents).rates

    def log_likelihood(
        self,
        observations: jax.Array,  # (T, N)
        latents: jax.Array,  # (T, D)
    ) -> jax.Array:  # ()
        """Compute the total conditional log likelihood."""
        return jnp.sum(self.conditional(latents).log_prob(observations))

    def compute_local_potential(
        self,
        observations: jax.Array,  # (T, N)
        latents: jax.Array,  # (T, D)
    ) -> GaussianPotential:
        """Quadratic approximation of the likelihood around ``latents``."""
        coefficients = self.model.affine.coefficients  # (N, D)

        conditional = self.conditional(latents)
        rates = conditional.rates  # (T, N)

        gradient = (observations - rates) @ coefficients  # (T, D)

        precision = jnp.einsum(
            'tn,ni,nj->tij',
            rates,
            coefficients,
            coefficients,
        )  # (T, D, D)

        return GaussianPotential.from_local_quadratic(
            point=latents,
            log_value=conditional.log_prob(observations),
            gradient=gradient,
            precision=precision,
        )

    def sample(
        self,
        key: jax.Array,
        latents: jax.Array,  # (..., D)
    ) -> jax.Array:  # (..., N)
        """Sample observations conditional on latent values."""
        return self.conditional(latents).sample(key)

    def fit_params(
        self,
        observations: jax.Array,  # (T, N)
        posterior: ContinuousPosterior,  # (T, D)
    ) -> typing.Self:
        """Fit the emission parameters from Gaussian latent marginals."""
        model = poisson_fit.linear_from_marginals(
            outputs=observations,
            inputs=Gaussian(
                mean=posterior.means,
                covariance=posterior.covariances,
            ),
            initial_affine=self.model.affine,
        )

        return self._replace(
            model=model,
        )

    def observation_mean(self, posterior: ContinuousPosterior) -> jax.Array:
        return self.model.expected_rates(
            Gaussian(
                mean=posterior.means,
                covariance=posterior.covariances,
            )
        )

    def compose_input(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Express the emissions in aligned latent coordinates."""
        return self._replace(
            model=self.model.compose_input(
                alignment.inverse(),
            ),
        )
