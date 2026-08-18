r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jsp_linalg

from xxm.core.gaussian.chain import GaussianChainMarginals as Posterior
from xxm.core.gaussian.chain import (
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.gaussian.emissions import Emissions

from ..stats import gaussian

EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


def _gaussian_log_prob_residuals(
    residuals: jax.Array,
    covariance: jax.Array,
) -> jax.Array:
    """Log probability of zero-mean Gaussian residuals with shared covariance."""
    dimension = covariance.shape[0]

    cholesky = jsp_linalg.cholesky(
        covariance,
        lower=True,
    )

    whitened = jsp_linalg.solve_triangular(
        cholesky,
        residuals.T,
        lower=True,
    ).T

    quadratic = jnp.sum(whitened**2, axis=1)

    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))

    return jnp.sum(-0.5 * (quadratic + log_det + dimension * jnp.log(2.0 * jnp.pi)))


class GaussianInitialModel(typing.NamedTuple):
    mean: jax.Array
    covariance: jax.Array

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        r"""
        Maximum-likelihood update of the latent model.
        """
        mean = posterior.means[0]
        covariance = posterior.covariances[0]

        return self.__class__(
            mean=mean,
            covariance=covariance,
        )

    def sample(
        self,
        key: jax.Array,
    ) -> jax.Array:

        return jax.random.multivariate_normal(
            key,
            mean=self.mean,
            cov=self.covariance,
        )


class LinearGaussianDynamicsModel(typing.NamedTuple):
    matrix: jax.Array
    bias: jax.Array
    noise_covariance: jax.Array

    def fit_params(
        self,
        posterior: Posterior,
    ) -> typing.Self:
        r"""
        Maximum-likelihood update of the latent model.
        """
        means = posterior.means
        second = posterior.raw_second_moments()
        cross = posterior.raw_cross_moments()

        matrix, bias, noise_covariance = gaussian.fit_linear_from_moments(
            input_mean=jnp.mean(means[:-1], axis=0),
            output_mean=jnp.mean(means[1:], axis=0),
            input_second_moment=jnp.mean(second[:-1], axis=0),
            output_second_moment=jnp.mean(second[1:], axis=0),
            output_input_moment=jnp.mean(cross, axis=0).T,
        )

        return self.__class__(
            matrix=matrix,
            bias=bias,
            noise_covariance=noise_covariance,
        )

    def next_mean(
        self,
        latent: jax.Array,
    ) -> jax.Array:

        return latent @ self.matrix.T + self.bias

    def sample(
        self,
        key: jax.Array,
        previous: jax.Array,
    ) -> jax.Array:
        latent_mean = self.next_mean(previous)

        latent = jax.random.multivariate_normal(
            key,
            mean=latent_mean,
            cov=self.noise_covariance,
        )

        return latent


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Container for LDS model parameters.
    """

    initial: GaussianInitialModel
    dynamics: LinearGaussianDynamicsModel
    emissions: EmissionsT

    def get_initial_potential(self) -> GaussianPotential:
        return GaussianPotential.from_moments(
            self.initial.mean,
            self.initial.covariance,
        )

    def get_pair_potentials(
        self,
        num_time_steps: int,
    ) -> GaussianPairPotential:
        potential = GaussianPairPotential.from_linear_conditional(
            self.dynamics.matrix,
            self.dynamics.bias,
            self.dynamics.noise_covariance,
        )

        return potential.broadcast(batch_shape=(num_time_steps - 1,))

    def sample(
        self,
        num_time_steps: int,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample latent and observations from the LDS."""
        key, initial_key, observation_key = jax.random.split(key, 3)

        initial_latent = self.initial.sample(initial_key)

        def sample_step(
            carry: tuple[jax.Array, jax.Array],
            _: None,
        ) -> tuple[
            tuple[jax.Array, jax.Array],
            jax.Array,
        ]:
            previous_latent, key = carry

            key, latent_key = jax.random.split(key)

            latent = self.dynamics.sample(latent_key, previous_latent)

            return (latent, key), latent

        _, remaining_latents = jax.lax.scan(
            sample_step,
            (initial_latent, key),
            None,
            length=num_time_steps - 1,
        )

        latents = jnp.concatenate(
            [initial_latent[None], remaining_latents],
            axis=0,
        )

        observations = self.emissions.sample(
            observation_key,
            latents,
        )

        return latents, observations

    def get_prior_mean_latents(self, num_time_steps: int) -> jax.Array:
        """Compute the mean latent trajectory under the model's prior."""

        def step(
            latent: jax.Array,
            _: None,
        ) -> tuple[jax.Array, jax.Array]:

            next_state = self.dynamics.next_mean(latent)

            return next_state, next_state

        _, remaining_latents = jax.lax.scan(
            step,
            self.initial.mean,
            None,
            length=num_time_steps - 1,
        )

        return jnp.concatenate(
            [
                self.initial.mean[None],
                remaining_latents,
            ],
            axis=0,
        )

    def log_joint(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array:
        """Compute log p(x, y) for a latentsstrajectory."""

        initial_residual = (latents[0] - self.initial.mean)[None]

        initial_log_prob = _gaussian_log_prob_residuals(
            initial_residual,
            self.initial.covariance,
        )

        dynamics_means = self.dynamics.next_mean(latents[:-1])

        dynamics_residuals = latents[1:] - dynamics_means

        dynamics_log_prob = _gaussian_log_prob_residuals(
            dynamics_residuals,
            self.dynamics.noise_covariance,
        )

        emission_log_prob = self.emissions.log_likelihood(
            observations,
            latents,
        )

        return initial_log_prob + dynamics_log_prob + emission_log_prob

    def fit_params(
        self,
        observations: jax.Array,
        posterior: Posterior,
    ) -> typing.Self:
        """Fit the parameters of the LDS given a posterior over latents."""

        initial = self.initial.fit_params(posterior)
        dynamics = self.dynamics.fit_params(posterior)
        emissions = self.emissions.fit_params(observations, posterior)

        return self.__class__(
            initial=initial,
            dynamics=dynamics,
            emissions=emissions,
        )
