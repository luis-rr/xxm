r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jsp_linalg

from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import (
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.emissions.continuous import Emissions
from xxm.core.models.gaussian import GaussianInitial
from xxm.core.models.gaussian import GaussianLinearDynamics

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


class Model(typing.NamedTuple, typing.Generic[EmissionsT]):
    r"""
    Container for LDS model parameters.
    """

    initial: GaussianInitial
    dynamics: LinearDynamics
    emissions: EmissionsT

    def get_initial_potential(self) -> GaussianPotential:
        return GaussianPotential.from_moments(self.initial.model)

    def get_pair_potentials(
        self,
        num_time_steps: int,
    ) -> GaussianPairPotential:
        potential = GaussianPairPotential.from_linear_conditional(self.dynamics.model)

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

            next_state = self.dynamics.model.conditional(latent).mean

            return next_state, next_state

        _, remaining_latents = jax.lax.scan(
            step,
            self.initial.model.mean,
            None,
            length=num_time_steps - 1,
        )

        return jnp.concatenate(
            [
                self.initial.model.mean[None],
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

        initial_residual = (latents[0] - self.initial.model.mean)[None]

        initial_log_prob = _gaussian_log_prob_residuals(
            initial_residual,
            self.initial.model.covariance,
        )

        dynamics_means = self.dynamics.model.conditional(latents[:-1]).mean

        dynamics_residuals = latents[1:] - dynamics_means

        dynamics_log_prob = _gaussian_log_prob_residuals(
            dynamics_residuals,
            self.dynamics.model.covariance,
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
