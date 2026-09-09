"""Emission models for continuous latent variables."""

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
    """Protocol for emission models over continuous latents."""

    def sample(self, key, latents) -> jax.Array:
        """Sample observations conditional on latent values."""
        ...

    def fit_params(
        self,
        observations: jax.Array,
        posterior: ContinuousPosterior,
    ) -> typing.Self:
        """Fit emission parameters from observations and posterior moments."""
        ...

    def log_likelihood(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> jax.Array:
        """Evaluate the total conditional log likelihood."""
        ...

    def compose_input(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Transform the latent input coordinates of the emission model."""
        ...


EmissionsT = typing.TypeVar('EmissionsT', bound=Emissions)


class QuadraticEmissions(Emissions, typing.Protocol):
    r"""Emissions with quadratic log likelihood in latent variables.

    The observation log likelihood is exactly a Gaussian potential in $x$,
    so continuous latent inference remains conjugate.
    """

    def compute_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential:
        r"""Construct the exact Gaussian potential for $\log p(y\mid x)$.

        No expansion point or local approximation is required.
        """
        ...


QuadraticEmissionsT = typing.TypeVar(
    'QuadraticEmissionsT',
    bound=QuadraticEmissions,
)


class LaplaceEmissions(Emissions, typing.Protocol):
    r"""Emissions with Laplace and ELBO support.

    The likelihood need not be conjugate to Gaussian latents but must admit
    a local quadratic approximation and expectations under Gaussian marginals.
    """

    def compute_local_potential(
        self,
        observations: jax.Array,
        latents: jax.Array,
    ) -> GaussianPotential:
        r"""Construct a local Gaussian approximation to $\log p(y\mid x)$ at `latents`."""
        ...

    def expected_log_likelihood(
        self,
        observations: jax.Array,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        r"""Evaluate $\mathbb{E}_{q(x)}[\log p(y\mid x)]$ under Gaussian marginals."""
        ...


LaplaceEmissionsT = typing.TypeVar(
    'LaplaceEmissionsT',
    bound=LaplaceEmissions,
)


class GaussianEmissions(typing.NamedTuple):
    r"""Linear-Gaussian emissions.

    $$y_t|x_t \sim \mathcal{N}(Cx_t + d, R).$$

    `dist.affine.coefficients` stores $C$, `dist.affine.bias` stores $d$,
    and `dist.covariance` stores $R$.
    """

    dist: LinearGaussian  # no batch

    def conditional(
        self,
        latents: jax.Array,  # (..., D)
    ) -> Gaussian:
        """Conditional distribution $p(y|x)$ at deterministic latent."""
        return self.dist.conditional(latents)

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
        """Construct exact Gaussian likelihood potential over latents."""
        return GaussianPotential.from_linear_likelihood(
            self.dist,
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
        observations: jax.Array,
        posterior: ContinuousPosterior,
        ridge=gaussian_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        """Fit emission parameters from Gaussian latent marginals."""

        return self._replace(
            dist=gaussian_fit.linear_from_marginals(
                inputs=Gaussian(
                    mean=posterior.means,
                    covariance=posterior.covariances,
                ),
                outputs=observations,
                ridge=ridge,
            ),
        )

    def observation_mean(self, posterior: ContinuousPosterior) -> jax.Array:
        """Expected observations under posterior marginals."""
        return self.dist.conditional_mean(
            posterior.means,
        )

    def compose_input(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Express the emissions in aligned latent coordinates."""
        return self._replace(
            dist=self.dist.compose_input(
                alignment.inverse(),
            ),
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        observations: jax.Array,
        covariance_floor: float,
        ridge=gaussian_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        """Fit Gaussian emissions to a known latent trajectory."""
        model = gaussian_fit.linear_from_samples(
            latents,
            observations,
            ridge=ridge,
        )

        model = model.add_covariance_jitter(covariance_floor)

        return cls(model)


class PoissonEmissions(typing.NamedTuple):
    r"""Linear Poisson emissions for continuous latent variables.

    $$y_t\mid x_t \sim \operatorname{Poisson}(\lambda_t),
    \qquad \log\lambda_t=Cx_t+d.$$

    `dist.affine.coefficients` stores $C$ and `dist.affine.bias` stores $d$.
    """

    dist: LinearPoisson  # no batch

    def conditional(
        self,
        latents: jax.Array,  # (..., D)
    ) -> Poisson:
        """Conditional observation distribution given latent values."""
        return self.dist.conditional(latents)

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

        return self.dist.expected_log_prob(
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
        """Construct a quadratic approximation of the log likelihood around `latents`."""
        coefficients = self.dist.affine.coefficients  # (N, D)

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
        ridge=poisson_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        """Fit the emission parameters from Gaussian latent marginals."""
        model = poisson_fit.linear_from_marginals(
            outputs=observations,
            inputs=Gaussian(
                mean=posterior.means,
                covariance=posterior.covariances,
            ),
            initial_affine=self.dist.affine,
            ridge=ridge,
        )

        return self._replace(
            dist=model,
        )

    def observation_mean(self, posterior: ContinuousPosterior) -> jax.Array:
        """Compute expected observations under posterior latent marginals."""
        return self.dist.expected_rates(
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
            dist=self.dist.compose_input(
                alignment.inverse(),
            ),
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        observations: jax.Array,
        ridge=poisson_fit.DEFAULT_RIDGE,
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

        model = poisson_fit.linear_from_samples(
            outputs=observations,
            inputs=latents,
            initial_affine=initial_affine,
            ridge=ridge,
        )

        return cls(model)
