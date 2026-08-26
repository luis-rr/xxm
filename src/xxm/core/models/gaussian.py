from jax import numpy as jnp

from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.dists.gaussian import Gaussian, LinearGaussian


import jax


import typing

from xxm.core.optim import gaussian as gaussian_fit


class GaussianInitial(typing.NamedTuple):
    model: Gaussian  # no batch

    def fit_params(self, posterior: Posterior) -> typing.Self:
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

    def fit_params(self, posterior: Posterior) -> typing.Self:
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

    def sample(self, key: jax.Array, previous: jax.Array) -> jax.Array:
        return self.model.conditional(previous).sample(key=key)
