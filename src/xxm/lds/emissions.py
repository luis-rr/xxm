from __future__ import annotations

import typing

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jsp_linalg
from jax.scipy import special as jsp_special

from ..gaussian_chain import GaussianPotential
from ..stats import gaussian, poisson


class Emissions(typing.Protocol):
    def sample(self, key, latent) -> jax.Array: ...

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
    readout: jax.Array  # C, shape (N, D)
    bias: jax.Array  # d, shape (N,)
    noise_covariance: jax.Array  # R, shape (N, N)

    def mean(self, latents: jax.Array) -> jax.Array:
        return latents @ self.readout.T + self.bias

    def get_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential:
        # observations: (T, N)
        t = observations.shape[0]
        n = observations.shape[1]

        cholesky = jnp.linalg.cholesky(self.noise_covariance)

        precision = jsp_linalg.cho_solve(
            (cholesky, True),
            jnp.eye(n, dtype=self.noise_covariance.dtype),
        )

        centered_observations = observations - self.bias

        precision_matrix = precision @ self.readout

        precision_blocks = jnp.broadcast_to(
            self.readout.T @ precision_matrix,
            (t, self.readout.shape[1], self.readout.shape[1]),
        )

        information_vectors = centered_observations @ precision_matrix

        log_det_covariance = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))

        quadratic_terms = jnp.einsum(
            'ti,ij,tj->t',
            centered_observations,
            precision,
            centered_observations,
        )

        log_constant = jnp.sum(
            -0.5 * quadratic_terms - 0.5 * log_det_covariance - 0.5 * n * jnp.log(2.0 * jnp.pi)
        )

        return GaussianPotential(
            precision_blocks=precision_blocks,
            information_vectors=information_vectors,
            log_constant=log_constant,
        )

    def sample(
        self,
        key: jax.Array,
        latent: jax.Array,
    ) -> jax.Array:
        """Sample an observation conditional on a latent."""
        mean = self.readout @ latent + self.bias

        return jax.random.multivariate_normal(
            key,
            mean=mean,
            cov=self.noise_covariance,
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: typing.Any,
    ) -> GaussianEmissions:
        """Fit the parameters of the emissions model given a posterior over latents."""

        means = posterior.means
        second = posterior.raw_second_moments()

        num_samples = observations.shape[0]

        readout, bias, noise_covariance = gaussian.fit_linear_from_moments(
            input_mean=jnp.mean(means, axis=0),
            output_mean=jnp.mean(observations, axis=0),
            input_second_moment=jnp.mean(second, axis=0),
            output_second_moment=observations.T @ observations / num_samples,
            output_input_moment=observations.T @ means / num_samples,
        )

        return GaussianEmissions(
            readout=readout,
            bias=bias,
            noise_covariance=noise_covariance,
        )

    def log_likelihood(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array:
        """Compute log p(observations | latents)."""
        means = latents @ self.readout.T + self.bias
        residuals = observations - means

        cholesky = jsp_linalg.cholesky(
            self.noise_covariance,
            lower=True,
        )

        whitened = jsp_linalg.solve_triangular(
            cholesky,
            residuals.T,
            lower=True,
        ).T

        quadratic = jnp.sum(whitened**2, axis=1)

        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))

        observation_dim = observations.shape[1]

        return jnp.sum(-0.5 * (quadratic + log_det + observation_dim * jnp.log(2.0 * jnp.pi)))


class PoissonEmissions(typing.NamedTuple):
    readout: jax.Array  # C, shape (N, D)
    bias: jax.Array  # d, shape (N,)

    def rates(
        self,
        latents: jax.Array,
    ) -> jax.Array:
        """Compute Poisson rates for a latent trajectory."""
        linear_predictors = latents @ self.readout.T + self.bias
        return jnp.exp(linear_predictors)

    def log_likelihood(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array:
        """Compute log p(observations | latents)."""
        linear_predictors = latents @ self.readout.T + self.bias
        rates = jnp.exp(linear_predictors)

        return jnp.sum(
            observations * linear_predictors - rates - jsp_special.gammaln(observations + 1.0)
        )

    def get_local_potential(self, observations: jax.Array, latents: jax.Array) -> GaussianPotential:
        r"""
        Quadratic Taylor approximation of the Poisson log likelihood.

        For

            y_t ~ Poisson(exp(C x_t + d)),

        approximate log p(y_t | x_t) around ``reference_latents[t]`` as

            -1/2 x_t.T @ J_t @ x_t + h_t.T @ x_t + c_t.

        The approximation matches the value, gradient, and Hessian of the
        true Poisson log likelihood at each reference state.
        """
        linear_predictors = latents @ self.readout.T + self.bias
        rates = jnp.exp(linear_predictors)

        # Gradient of log p(y_t | x_t) at the reference state:
        #
        #     g_t = C.T @ (y_t - lambda_t)
        #
        gradients = (observations - rates) @ self.readout

        # Negative Hessian:
        #
        #     J_t = C.T @ diag(lambda_t) @ C
        #
        # shape: (T, D, D)
        precision_blocks = jnp.einsum(
            'tn,ni,nj->tij',
            rates,
            self.readout,
            self.readout,
        )

        # Expanding
        #
        #   l(x) ~= l(x0)
        #            + g.T (x - x0)
        #            - 1/2 (x - x0).T J (x - x0)
        #
        # into canonical Gaussian-potential form gives
        #
        #   h = g + J x0.
        #
        information_vectors = gradients + jnp.einsum(
            'tij,tj->ti',
            precision_blocks,
            latents,
        )

        log_likelihoods = jnp.sum(
            observations * linear_predictors - rates - jsp_special.gammaln(observations + 1.0),
            axis=1,
        )

        gradient_terms = jnp.einsum(
            'ti,ti->t',
            gradients,
            latents,
        )

        quadratic_terms = jnp.einsum(
            'ti,tij,tj->t',
            latents,
            precision_blocks,
            latents,
        )

        # Constant chosen so that the quadratic approximation has exactly
        # the true likelihood value at the reference state.
        log_constant = jnp.sum(log_likelihoods - gradient_terms - 0.5 * quadratic_terms)

        return GaussianPotential(
            precision_blocks=precision_blocks,
            information_vectors=information_vectors,
            log_constant=log_constant,
        )

    def sample(
        self,
        key: jax.Array,
        latent: jax.Array,
    ) -> jax.Array:
        """Sample an observation conditional on a latent."""
        rate = jnp.exp(self.readout @ latent + self.bias)

        return jax.random.poisson(
            key,
            lam=rate,
        )

    def fit_params(
        self,
        observations: jax.Array,
        posterior: typing.Any,
    ) -> PoissonEmissions:
        """Fit the parameters of the emissions model given a posterior over latents."""
        readout, bias = poisson.fit_from_moments(
            observations=observations,
            means=posterior.means,
            covariances=posterior.covariances,
            readout=self.readout,
            bias=self.bias,
        )

        return self.__class__(
            readout=readout,
            bias=bias,
        )
