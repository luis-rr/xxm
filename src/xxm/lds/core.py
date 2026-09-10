r"""Linear Dynamical Systems."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp
from jax.scipy import linalg as jsp_linalg

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import (
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.emissions.continuous import EmissionsT
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics


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
    r"""Linear Dynamical System model.

    $$x_0 \sim p(x_0), \quad x_t|x_{t-1} \sim \mathcal{N}(Ax_{t-1}+b, Q), \quad y_t|x_t \sim p(y_t|x_t).$$

    `initial` stores $p(x_0)$, `dynamics` stores the linear-Gaussian state evolution,
    and `emissions` stores $p(y_t|x_t)$.
    """

    initial: GaussianInitial
    dynamics: GaussianLinearDynamics
    emissions: EmissionsT

    def compute_initial_potential(self) -> GaussianPotential:
        """Return the canonical potential for the initial latent distribution."""
        return GaussianPotential.from_moments(self.initial.dist)

    def compute_pair_potentials(
        self,
        num_steps: int,
    ) -> GaussianPairPotential:
        """Return the $T-1$ repeated pair potentials for latent dynamics."""
        potential = GaussianPairPotential.from_linear_conditional(self.dynamics.dist)

        return potential.broadcast(batch_shape=(num_steps - 1,))

    def sample(
        self,
        key: jax.Array,
        num_steps: int,
    ) -> tuple[jax.Array, jax.Array]:
        r"""Sample complete trajectory $(x_{0:T-1}, y_{0:T-1})$ from the prior."""
        key_initial, key_latents, key_observations = jax.random.split(key, 3)

        initial_latent = self.initial.sample(key_initial)

        latents = self.dynamics.sample(key_latents, initial_latent, num_steps)

        observations = self.emissions.sample(key_observations, latents)

        return latents, observations

    def compute_prior_means(self, num_steps: int) -> jax.Array:
        r"""Compute prior latent means $\mathbb{E}[x_t]$ under the generative model."""

        def step(
            latent: jax.Array,
            _: None,
        ) -> tuple[jax.Array, jax.Array]:

            next_latent = self.dynamics.dist.conditional(latent).mean

            return next_latent, next_latent

        _, remaining_latents = jax.lax.scan(
            step,
            self.initial.dist.mean,
            None,
            length=num_steps - 1,
        )

        return jnp.concatenate(
            [
                self.initial.dist.mean[None],
                remaining_latents,
            ],
            axis=0,
        )

    def log_joint(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array:
        r"""Compute $\log p(x_{0:T-1}, y_{0:T-1})$ for a latent trajectory."""

        initial_residual = (latents[0] - self.initial.dist.mean)[None]

        initial_log_prob = _gaussian_log_prob_residuals(
            initial_residual,
            self.initial.dist.covariance,
        )

        dynamics_means = self.dynamics.dist.conditional(latents[:-1]).mean

        dynamics_residuals = latents[1:] - dynamics_means

        dynamics_log_prob = _gaussian_log_prob_residuals(
            dynamics_residuals,
            self.dynamics.dist.covariance,
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

    def align(self, alignment: Affine) -> typing.Self:
        """Express the LDS in aligned continuous latent coordinates."""
        return self._replace(
            initial=self.initial.align(alignment),
            dynamics=self.dynamics.align(alignment),
            emissions=self.emissions.compose_input(alignment),
        )

    def whiten(self) -> typing.Self:
        r"""Express an LDS in latent coordinates where dynamics noise is identity.

        For $Q = LL^\top$, use $\bar x = L^{-1}x$.
        """
        covariance = self.dynamics.dist.covariance
        latent_dim = self.dynamics.dist.output_dim

        cholesky = jnp.linalg.cholesky(
            covariance,
        )

        identity = jnp.eye(
            latent_dim,
            dtype=covariance.dtype,
        )

        whitening = jnp.linalg.solve(
            cholesky,
            identity,
        )

        alignment = Affine(
            coefficients=whitening,
            bias=jnp.zeros(
                latent_dim,
                dtype=covariance.dtype,
            ),
        )

        return self.align(alignment)
