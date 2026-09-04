from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals, GaussianPotential
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


EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


class QuadraticEmissions(Emissions, typing.Protocol):
    r"""
    Emissions with a likelihood that is quadratic in the latent variables.

    The observation log likelihood can be represented exactly as a Gaussian
    potential in ``x``, so continuous latent inference remains conjugate.
    """

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential:
        r"""
        Construct the exact Gaussian likelihood potential.

        Returns a potential representing

        .. math::

            \log p(y \mid x)

        as a quadratic function of the latent variables. Unlike Laplace
        emissions, no expansion point or local approximation is required.
        """
        ...


QuadraticEmissionsT = typing.TypeVar(
    'QuadraticEmissionsT',
    bound=QuadraticEmissions,
)


class LaplaceEmissions(Emissions, typing.Protocol):
    r"""
    Emissions supporting Laplace updates and ELBO evaluation.

    The likelihood need not be conjugate to a Gaussian latent posterior, but it
    must admit a local quadratic approximation and an expectation under
    Gaussian latent marginals.
    """

    def compute_local_potential(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> GaussianPotential:
        r"""
        Construct the local Gaussian likelihood potential at ``latents``.

        Returns the quadratic approximation to

        .. math::

            \log p(y \mid x)

        obtained from its value, gradient, and Hessian at the supplied latent
        trajectory. This potential is used to form the Gaussian Laplace update.
        """
        ...

    def expected_log_likelihood(
        self,
        observations: jax.Array,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        r"""
        Evaluate the expected log likelihood under Gaussian latent marginals.

        Computes

        .. math::

            \mathbb{E}_{q(x)}[\log p(y \mid x)],

        where ``q(x)`` is represented by ``posterior``. The expectation may be
        evaluated analytically or numerically by the emission implementation.
        """
        ...


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

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        observations: jax.Array,
        covariance_floor: float,
    ) -> typing.Self:
        """Fit Gaussian emissions to a known latent trajectory."""
        model = gaussian_fit.linear_from_pairs(
            latents,
            observations,
        )

        model = model.add_covariance_jitter(covariance_floor)

        return cls(model)


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

    def expected_log_likelihood(
        self,
        observations: jax.Array,
        posterior: ContinuousPosterior,
    ) -> jax.Array:
        """Expected conditional log likelihood under Gaussian latent marginals."""

        return self.model.expected_log_prob(
            values=observations,
            inputs=Gaussian(
                mean=posterior.means,
                covariance=posterior.covariances,
            ),
        )

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

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        observations: jax.Array,
    ) -> typing.Self:
        """Fit Poisson emissions to a known latent trajectory."""

        observation_dim = observations.shape[1]
        latent_dim = latents.shape[1]

        # Sensible intercept-only starting point.
        mean_rates = jnp.maximum(
            jnp.mean(observations, axis=0),
            1e-6,
        )

        initial_affine = Affine(
            coefficients=jnp.zeros(
                (observation_dim, latent_dim),
                dtype=latents.dtype,
            ),
            bias=jnp.log(mean_rates),
        )

        model = poisson_fit.linear_from_pairs(
            outputs=observations,
            inputs=latents,
            initial_affine=initial_affine,
        )

        return cls(model)
