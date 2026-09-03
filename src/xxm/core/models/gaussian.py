import typing

import jax
from jax import numpy as jnp

from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.posteriors import ContinuousPosterior


class GaussianInitial(typing.NamedTuple):
    model: Gaussian  # no batch

    def fit_params(self, posterior: ContinuousPosterior) -> typing.Self:
        r"""
        Maximum-likelihood update of the latent model.
        """
        mean = posterior.means[0]
        covariance = posterior.covariances[0]

        return self._replace(model=Gaussian(mean=mean, covariance=covariance))

    def sample(self, key: jax.Array) -> jax.Array:
        return self.model.sample(key)


class GaussianLinearDynamics(typing.NamedTuple):
    model: LinearGaussian  # no batch

    def fit_params(self, posterior: ContinuousPosterior) -> typing.Self:
        r"""
        Maximum-likelihood update of the latent model.
        """
        means = posterior.means
        second = posterior.raw_second_moments()
        cross = posterior.raw_cross_moments()

        model = gaussian_fit.linear_from_moments(
            input_mean=jnp.mean(means[:-1], axis=0),
            output_mean=jnp.mean(means[1:], axis=0),
            input_second_moment=jnp.mean(second[:-1], axis=0),
            output_second_moment=jnp.mean(second[1:], axis=0),
            output_input_moment=jnp.mean(cross, axis=0).T,
        )

        return self._replace(model=model)

    def sample_next(
        self,
        key: jax.Array,
        previous: jax.Array,
    ) -> jax.Array:
        """Sample the next latent conditional on the previous latent."""
        return self.model.sample(
            key,
            previous,
        )

    def sample(
        self,
        key: jax.Array,
        initial_latent: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample a latent trajectory conditional on its initial value."""

        def step(carry, _):
            latent, key = carry

            key, sample_key = jax.random.split(key)
            latent = self.sample_next(
                sample_key,
                latent,
            )

            return (latent, key), latent

        _, subsequent_latents = jax.lax.scan(
            step,
            (initial_latent, key),
            xs=None,
            length=num_steps - 1,
        )

        return jnp.concatenate(
            [
                initial_latent[None],
                subsequent_latents,
            ],
            axis=0,
        )
