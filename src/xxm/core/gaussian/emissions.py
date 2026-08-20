from __future__ import annotations

import typing

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jsp_linalg

from xxm.stats import gaussian_fit, poisson_fit
from xxm.stats.gaussian import Gaussian, LinearGaussian
from xxm.stats.poisson import LinearPoisson, Poisson

from .chain import GaussianChainMarginals, GaussianPotential


class Emissions(typing.Protocol):
    def sample(self, key, latents) -> jax.Array: ...

    def fit_params(self, observations, posterior) -> typing.Self: ...

    def log_likelihood(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array: ...


class QuadraticEmissions(Emissions, typing.Protocol):
    """Emissions with a quadratic log-likelihood, so that the posterior is Gaussian."""

    def get_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential: ...


QuadraticEmissionsT = typing.TypeVar(
    'QuadraticEmissionsT',
    bound=QuadraticEmissions,
)


class LaplaceEmissions(Emissions, typing.Protocol):
    """Emissions that provide a local quadratic approximation for a latent."""

    def get_local_potential(
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

    def get_potential(
        self,
        observations: jax.Array,  # (T, N)
    ) -> GaussianPotential:
        """Convert the Gaussian likelihood into a potential over latents."""
        coefficients = self.model.affine.coefficients  # (N, D)
        bias = self.model.affine.bias  # (N,)
        covariance = self.model.covariance  # (N, N)

        num_samples = observations.shape[0]
        latent_dim = self.model.affine.input_dim

        cholesky = jnp.linalg.cholesky(covariance)  # (N, N)

        precision = jsp_linalg.cho_solve(
            (cholesky, True),
            jnp.eye(
                self.model.affine.output_dim,
                dtype=covariance.dtype,
            ),
        )  # (N, N)

        centered = observations - bias  # (T, N)

        precision_coefficients = precision @ coefficients  # (N, D)

        precision_block = coefficients.T @ precision_coefficients  # (D, D)
        precision_blocks = jnp.broadcast_to(
            precision_block,
            (num_samples, latent_dim, latent_dim),
        )  # (T, D, D)

        information_vectors = centered @ precision_coefficients  # (T, D)

        quadratic_terms = jnp.einsum(
            'tn,nm,tm->t',
            centered,
            precision,
            centered,
        )  # (T,)

        log_det_covariance = 2.0 * jnp.sum(jnp.log(jnp.diagonal(cholesky)))  # ()

        log_constant = -0.5 * (
            quadratic_terms
            + log_det_covariance
            + self.model.affine.output_dim * jnp.log(2.0 * jnp.pi)
        )  # (T,)

        return GaussianPotential(
            precision_blocks=precision_blocks,
            information_vectors=information_vectors,
            log_constant=log_constant,
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
        posterior: GaussianChainMarginals,  # (T, D)
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

    def get_local_potential(
        self,
        observations: jax.Array,  # (T, N)
        latents: jax.Array,  # (T, D)
    ) -> GaussianPotential:
        """Quadratic approximation of the likelihood around ``latents``."""
        coefficients = self.model.affine.coefficients  # (N, D)

        conditional = self.conditional(latents)
        rates = conditional.rates  # (T, N)

        gradients = (observations - rates) @ coefficients  # (T, D)

        precision_blocks = jnp.einsum(
            'tn,ni,nj->tij',
            rates,
            coefficients,
            coefficients,
        )  # (T, D, D)

        information_vectors = gradients + jnp.einsum(
            'tij,tj->ti',
            precision_blocks,
            latents,
        )  # (T, D)

        log_likelihoods = conditional.log_prob(observations)  # (T,)

        gradient_terms = jnp.einsum(
            'ti,ti->t',
            gradients,
            latents,
        )  # (T,)

        quadratic_terms = jnp.einsum(
            'ti,tij,tj->t',
            latents,
            precision_blocks,
            latents,
        )  # (T,)

        log_constant = log_likelihoods - gradient_terms - 0.5 * quadratic_terms  # (T,)

        return GaussianPotential(
            precision_blocks=precision_blocks,
            information_vectors=information_vectors,
            log_constant=log_constant,
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
        posterior: GaussianChainMarginals,  # (T, D)
    ) -> typing.Self:
        """Fit the emission parameters from Gaussian latent marginals."""
        model = poisson_fit.linear_from_marginals(
            outputs=observations,
            inputs=Gaussian(mean=posterior.means, covariance=posterior.covariances),
            initial_affine=self.model.affine,
        )

        return self._replace(
            model=model,
        )
