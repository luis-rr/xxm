"""Emission models for continuous latent variables."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core import _batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChainMarginals, GaussianPotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.posteriors import ContinuousPosterior


class Emissions(typing.Protocol):
    """Protocol for emission models over continuous latents."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def select(self, index) -> typing.Self: ...

    def broadcast(self, shape, axis: int = 0) -> typing.Self: ...

    def squeeze(self, axis=None) -> typing.Self: ...

    def permute(self, permutation, axis: int = 0) -> typing.Self: ...

    def move_axis(self, source: int, destination: int) -> typing.Self: ...

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

    dist: LinearGaussian  # structural batch *B

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.dist.batch_shape

    def select(self, index) -> typing.Self:
        return self._replace(dist=self.dist.select(index))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        return self._replace(dist=self.dist.squeeze(axis))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return self._replace(dist=self.dist.move_axis(source, destination))

    def conditional(
        self,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> Gaussian:
        """Conditional distribution $p(y|x)$ at deterministic latent."""
        return self.dist.conditional(latents)

    def log_likelihood(
        self,
        observations: jax.Array,  # (*B, T, O)
        latents: jax.Array,  # (*B, T, D)
    ) -> jax.Array:  # (*B,)
        """Compute the total conditional log likelihood."""
        return jnp.sum(self.conditional(latents).log_prob(observations), axis=-1)

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
        latents: jax.Array,  # (*B, *Q, D)
    ) -> jax.Array:  # (*B, *Q, O)
        """Sample observations conditional on latent values."""
        return self.conditional(latents).sample(key)

    def fit_params(
        self,
        observations: jax.Array,
        posterior: ContinuousPosterior,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        """Fit emission parameters from Gaussian latent marginals."""

        batch_shape = self.batch_shape

        def fit_one(observations_i, means_i, covariances_i):
            return gaussian_fit.linear_from_marginals(
                inputs=Gaussian(
                    mean=means_i,
                    covariance=covariances_i,
                ),
                outputs=observations_i,
                ridge=ridge,
                covariance_floor=covariance_floor,
            )

        flat_fitted = jax.vmap(fit_one)(
            _batch.flatten_batch(observations, batch_shape),
            _batch.flatten_batch(posterior.means, batch_shape),
            _batch.flatten_batch(posterior.covariances, batch_shape),
        )
        return self._replace(
            dist=_batch.unflatten_batch(flat_fitted, batch_shape),
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
        if alignment.batch_shape == ():
            alignment = alignment.broadcast(self.batch_shape)
        else:
            assert alignment.batch_shape == self.batch_shape
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
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Fit Gaussian emissions to a known latent trajectory."""
        batch_shape = latents.shape[:-2]
        flat_model = jax.vmap(
            lambda latents_i, observations_i: gaussian_fit.linear_from_samples(
                latents_i,
                observations_i,
                ridge=ridge,
                covariance_floor=covariance_floor,
            )
        )(
            _batch.flatten_batch(latents, batch_shape),
            _batch.flatten_batch(observations, batch_shape),
        )

        return cls(_batch.unflatten_batch(flat_model, batch_shape))

    def invert(
        self,
        observations: jax.Array,
    ) -> jax.Array:
        """Construct a latent estimate by pseudoinverting the emission mean map."""
        return self.dist.affine.pseudoinverse().apply(
            observations,
        )


class PoissonEmissions(typing.NamedTuple):
    r"""Linear Poisson emissions for continuous latent variables.

    $$y_t\mid x_t \sim \operatorname{Poisson}(\lambda_t),
    \qquad \log\lambda_t=Cx_t+d.$$

    `dist.affine.coefficients` stores $C$ and `dist.affine.bias` stores $d$.
    """

    dist: LinearPoisson  # structural batch *B

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.dist.batch_shape

    def select(self, index) -> typing.Self:
        return self._replace(dist=self.dist.select(index))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        return self._replace(dist=self.dist.squeeze(axis))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return self._replace(dist=self.dist.move_axis(source, destination))

    def conditional(
        self,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> Poisson:
        """Conditional observation distribution given latent values."""
        return self.dist.conditional(latents)

    def rates(
        self,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> jax.Array:  # (*B, *Q, O)
        """Compute conditional Poisson rates."""
        return self.conditional(latents).rates

    def log_likelihood(
        self,
        observations: jax.Array,  # (*B, T, O)
        latents: jax.Array,  # (*B, T, D)
    ) -> jax.Array:  # (*B,)
        """Compute the total conditional log likelihood."""
        return jnp.sum(self.conditional(latents).log_prob(observations), axis=-1)

    def expected_log_likelihood(
        self,
        observations: jax.Array,
        posterior: ContinuousPosterior,
    ) -> jax.Array:
        """Expected conditional log likelihood under Gaussian latent marginals."""

        return jnp.sum(
            self.dist.expected_log_prob(
                values=observations,
                inputs=Gaussian(
                    mean=posterior.means,
                    covariance=posterior.covariances,
                ),
            ),
            axis=-1,
        )

    def compute_local_potential(
        self,
        observations: jax.Array,  # (*B, T, O)
        latents: jax.Array,  # (*B, T, D)
    ) -> GaussianPotential:
        """Construct a quadratic approximation of the log likelihood around `latents`."""
        coefficients = self.dist.affine.coefficients  # (*B, O, D)

        conditional = self.conditional(latents)
        rates = conditional.rates  # (*B, T, O)

        gradient = (observations - rates) @ coefficients  # (*B, T, D)

        precision = jnp.einsum(
            '...tn,...ni,...nj->...tij',
            rates,
            coefficients,
            coefficients,
        )  # (*B, T, D, D)

        return GaussianPotential.from_local_quadratic(
            point=latents,
            log_value=conditional.log_prob(observations),
            gradient=gradient,
            precision=precision,
        )

    def sample(
        self,
        key: jax.Array,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> jax.Array:  # (*B, *Q, O)
        """Sample observations conditional on latent values."""
        return self.conditional(latents).sample(key)

    def fit_params(
        self,
        observations: jax.Array,  # (*B, T, O)
        posterior: ContinuousPosterior,  # (*B, T, D)
        ridge=poisson_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        """Fit the emission parameters from Gaussian latent marginals."""

        batch_shape = self.batch_shape

        def fit_one(observations_i, means_i, covariances_i, initial_affine_i):
            return poisson_fit.linear_from_marginals(
                outputs=observations_i,
                inputs=Gaussian(mean=means_i, covariance=covariances_i),
                initial_affine=initial_affine_i,
                ridge=ridge,
            )

        flat_fitted = jax.vmap(fit_one)(
            _batch.flatten_batch(observations, batch_shape),
            _batch.flatten_batch(posterior.means, batch_shape),
            _batch.flatten_batch(posterior.covariances, batch_shape),
            _batch.flatten_batch(self.dist.affine, batch_shape),
        )

        return self._replace(
            dist=_batch.unflatten_batch(flat_fitted, batch_shape),
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
        if alignment.batch_shape == ():
            alignment = alignment.broadcast(self.batch_shape)
        else:
            assert alignment.batch_shape == self.batch_shape
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

        batch_shape = latents.shape[:-2]

        def fit_one(latents_i, observations_i):
            observation_dim = observations_i.shape[-1]
            latent_dim = latents_i.shape[-1]

            # Intercept-only starting point for each independent sequence.
            mean_rates = jnp.maximum(jnp.mean(observations_i, axis=0), 1e-6)
            initial_affine = Affine(
                coefficients=jnp.zeros(
                    (observation_dim, latent_dim), dtype=latents_i.dtype
                ),
                bias=jnp.log(mean_rates),
            )
            return poisson_fit.linear_from_samples(
                outputs=observations_i,
                inputs=latents_i,
                initial_affine=initial_affine,
                ridge=ridge,
            )

        flat_model = jax.vmap(fit_one)(
            _batch.flatten_batch(latents, batch_shape),
            _batch.flatten_batch(observations, batch_shape),
        )

        return cls(_batch.unflatten_batch(flat_model, batch_shape))

    def invert(
        self,
        observations: jax.Array,
        *,
        count_floor: float = poisson_fit.DEFAULT_COUNT_FLOOR,
    ) -> jax.Array:
        """Construct a latent estimate from stabilized log counts."""
        log_rates = poisson_fit.inverse_exp_link(
            observations,
            count_floor=count_floor,
        )

        return self.dist.affine.pseudoinverse().apply(
            log_rates,
        )
