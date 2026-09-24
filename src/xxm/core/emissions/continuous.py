"""Emission models for continuous latent variables."""

from __future__ import annotations

import typing

import jax
from jax import numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import (
    GaussianChainMarginals,
    GaussianPotential,
)
from xxm.core.data import WeightedObservations
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim import poisson as poisson_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.core.posteriors import ContinuousPosterior


class Emissions(typing.Protocol):
    """Protocol for emission models over continuous latents."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def select(self, index: batch.SelT) -> typing.Self: ...

    def broadcast(self, shape, axis: int = 0) -> typing.Self: ...

    def squeeze(self, axis=None) -> typing.Self: ...

    def permute(self, permutation, axis: int = 0) -> typing.Self: ...

    def move_axis(self, source: int, destination: int) -> typing.Self: ...

    def sample(self, key, latents) -> jax.Array:
        """Sample observations conditional on latent values."""
        ...

    def fit_params(
        self,
        observations: WeightedObservations,
        posterior: ContinuousPosterior,
    ) -> typing.Self:
        """Fit shared emission parameters across independent sequences."""
        ...

    def log_likelihood(
        self,
        observations: WeightedObservations,
        latents: jax.Array,
    ) -> jax.Array:
        """Evaluate the total conditional log likelihood."""
        ...

    def expected_log_likelihood(
        self,
        observations: WeightedObservations,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        r"""Evaluate E_q[log p(y|x)] under Gaussian marginals."""
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
        observations: WeightedObservations,
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
        observations: WeightedObservations,
        latents: jax.Array,
    ) -> GaussianPotential:
        r"""Construct a local Gaussian approximation to $\log p(y\mid x)$ at `latents`."""
        ...

    def expected_log_likelihood(
        self,
        observations: WeightedObservations,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        r"""Evaluate $\mathbb{E}_{q(x)}[\log p(y\mid x)]$ under Gaussian marginals."""
        ...


LaplaceEmissionsT = typing.TypeVar(
    'LaplaceEmissionsT',
    bound=LaplaceEmissions,
)


def _flatten_fit_data(
    observations: WeightedObservations,
    posterior: ContinuousPosterior,
    batch_shape: tuple[int, ...],
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """
    Pool trailing replicate batches and time into one fitting sample axis.

    ``batch_shape`` is the receiver parameter batch ``*B``. Observations and
    posterior batches must begin with ``*B``; any additional trailing batch
    dimensions are independent replicates and are pooled together with time.
    """
    observation_batch = observations.batch_shape
    replicate_shape = batch.split_prefix(
        observation_batch, batch_shape, name='observation batch shape'
    )

    posterior_batch = posterior.batch_shape
    batch.require_same_shape(
        observation_batch,
        posterior_batch,
    )

    if observations.num_steps != posterior.num_steps:
        raise ValueError(
            'posterior and observations must have the same number of timesteps'
        )

    return batch.pool_samples(
        (
            observations.safe_values(),
            observations.weights,
            posterior.means,
            posterior.covariances,
        ),
        batch_shape,
        (*replicate_shape, observations.num_steps),
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

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

    def conditional(
        self,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> Gaussian:
        """Conditional distribution $p(y|x)$ at deterministic latent."""
        return self.dist.conditional(latents)

    def log_likelihood(
        self,
        observations: WeightedObservations,
        latents: jax.Array,
    ) -> jax.Array:
        """Compute the weighted conditional log likelihood."""
        batch.require_same_shape(self.batch_shape, observations.batch_shape)

        latents = observations.safe_aligned(latents)
        log_probs = self.conditional(latents).log_prob(observations.safe_values())

        return observations.weighted_sum(log_probs)

    def expected_log_likelihood(
        self,
        observations: WeightedObservations,
        posterior: ContinuousPosterior,
    ) -> jax.Array:
        """Compute the weighted expected conditional log likelihood."""
        batch.require_same_shape(
            self.batch_shape,
            observations.batch_shape,
            posterior.batch_shape,
        )

        if observations.num_steps != posterior.num_steps:
            raise ValueError(
                'posterior and observations must have the same number of timesteps'
            )

        potential = self.compute_potential(
            observations,
        )

        means = observations.safe_aligned(
            posterior.means,
        )

        second_moments = observations.safe_aligned(
            posterior.raw_second_moments(),
        )

        terms = potential.expected_log_potential(
            means,
            second_moments,
        )

        return jnp.sum(
            terms,
            axis=-1,
        )

    def compute_potential(
        self,
        observations: WeightedObservations,
    ) -> GaussianPotential:
        """Construct exact weighted Gaussian likelihood potentials."""
        batch.require_same_shape(self.batch_shape, observations.batch_shape)

        potential = GaussianPotential.from_linear_likelihood(
            self.dist,
            observations.safe_values(),
        )

        return potential.scale(observations.weights)

    def sample(
        self,
        key: jax.Array,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> jax.Array:  # (*B, *Q, O)
        """Sample observations conditional on latent values."""
        return self.conditional(latents).sample(key)

    def fit_params(
        self,
        observations: WeightedObservations,
        posterior: ContinuousPosterior,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        """
        Fit emissions while preserving the receiver batch prefix.

        Any observation/posterior batch axes following ``self.batch_shape``
        are pooled together with time.
        """
        (
            values,
            weights,
            means,
            covariances,
        ) = _flatten_fit_data(
            observations,
            posterior,
            self.batch_shape,
        )

        def fit_one(
            values_i: jax.Array,
            weights_i: jax.Array,
            means_i: jax.Array,
            covariances_i: jax.Array,
        ) -> LinearGaussian:
            return gaussian_fit.linear_from_marginals(
                inputs=Gaussian(
                    mean=means_i,
                    covariance=covariances_i,
                ),
                outputs=values_i,
                weights=weights_i,
                ridge=ridge,
                covariance_floor=covariance_floor,
            )

        fitted = batch.vmap_batch(
            fit_one,
            values,
            weights,
            means,
            covariances,
            batch_shape=self.batch_shape,
        )

        fitted = filter_valid_batches(fitted, self.dist, weights)

        return self._replace(dist=fitted)

    def observation_mean(self, posterior: ContinuousPosterior) -> jax.Array:
        """Expected observations under posterior marginals."""
        batch.require_same_shape(self.batch_shape, posterior.batch_shape)

        return self.dist.conditional_mean(
            posterior.means,
        )

    def compose_input(
        self,
        alignment: Affine,
    ) -> typing.Self:
        """Express the emissions in aligned latent coordinates."""
        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

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
        fitted = batch.vmap_batch(
            lambda latents_i, observations_i: gaussian_fit.linear_from_samples(
                latents_i,
                observations_i,
                ridge=ridge,
                covariance_floor=covariance_floor,
            ),
            latents,
            observations,
            batch_shape=batch_shape,
        )

        return cls(fitted)

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

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)

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
        observations: WeightedObservations,
        latents: jax.Array,
    ) -> jax.Array:
        """Compute the weighted conditional log likelihood."""
        batch.require_same_shape(
            self.batch_shape,
            observations.batch_shape,
        )

        latents = observations.safe_aligned(latents)
        log_probs = self.conditional(latents).log_prob(observations.safe_values())

        return observations.weighted_sum(log_probs)

    def expected_log_likelihood(
        self,
        observations: WeightedObservations,
        posterior: ContinuousPosterior,
    ) -> jax.Array:
        """Compute the weighted expected conditional log likelihood."""
        batch.require_same_shape(
            self.batch_shape,
            observations.batch_shape,
            posterior.batch_shape,
        )

        dist = self.dist.broadcast(
            (observations.num_steps,),
            axis=len(self.batch_shape),
        )

        safe_mean = observations.safe_aligned(posterior.means)
        safe_covariance = observations.safe_aligned(posterior.covariances)

        log_probs = dist.expected_log_prob(
            observations.safe_values(),
            Gaussian(
                mean=safe_mean,
                covariance=safe_covariance,
            ),
        )

        return observations.weighted_sum(log_probs)

    def compute_local_potential(
        self,
        observations: WeightedObservations,
        latents: jax.Array,
    ) -> GaussianPotential:
        """Construct a weighted quadratic approximation around `latents`."""
        batch.require_same_shape(
            self.batch_shape,
            observations.batch_shape,
        )

        values = observations.safe_values()
        latents = observations.safe_aligned(latents)

        conditional = self.conditional(latents)
        rates = conditional.rates

        coefficients = batch.align_array(
            self.dist.affine.coefficients_flat,
            self.batch_shape,
            rates.shape[:-1],
        )

        gradient = jnp.einsum(
            '...o,...od->...d',
            values - rates,
            coefficients,
        )

        precision = jnp.einsum(
            '...o,...oi,...oj->...ij',
            rates,
            coefficients,
            coefficients,
        )  # (*B, T, D, D)

        potential = GaussianPotential.from_local_quadratic(
            point=latents,
            log_value=conditional.log_prob(values),
            gradient=gradient,
            precision=precision,
        )

        return potential.scale(observations.weights)

    def sample(
        self,
        key: jax.Array,
        latents: jax.Array,  # (*B, *Q, D)
    ) -> jax.Array:  # (*B, *Q, O)
        """Sample observations conditional on latent values."""
        return self.conditional(latents).sample(key)

    def fit_params(
        self,
        observations: WeightedObservations,
        posterior: ContinuousPosterior,
        ridge=poisson_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        """
        Fit emissions while preserving the receiver batch prefix.

        Any observation/posterior batch axes following ``self.batch_shape``
        are pooled together with time.
        """
        (
            values,
            weights,
            means,
            covariances,
        ) = _flatten_fit_data(
            observations,
            posterior,
            self.batch_shape,
        )

        def fit_one(
            values_i: jax.Array,
            weights_i: jax.Array,
            means_i: jax.Array,
            covariances_i: jax.Array,
            dist_i: LinearPoisson,
        ) -> LinearPoisson:
            return poisson_fit.linear_from_marginals(
                inputs=Gaussian(
                    mean=means_i,
                    covariance=covariances_i,
                ),
                outputs=values_i,
                weights=weights_i,
                initial_affine=dist_i.affine,
                ridge=ridge,
            )

        fitted = batch.vmap_batch(
            fit_one,
            values,
            weights,
            means,
            covariances,
            self.dist,
            batch_shape=self.batch_shape,
        )

        fitted = filter_valid_batches(fitted, self.dist, weights)

        return self._replace(dist=fitted)

    def observation_mean(self, posterior: ContinuousPosterior) -> jax.Array:
        """Compute expected observations under posterior latent marginals."""
        batch.require_same_shape(
            self.batch_shape,
            posterior.batch_shape,
        )

        num_steps = posterior.num_steps

        dist = self.dist.broadcast(
            (num_steps,),
            axis=len(self.batch_shape),
        )

        return dist.expected_rates(
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
        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

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

        fitted = batch.vmap_batch(
            fit_one, latents, observations, batch_shape=batch_shape
        )

        return cls(fitted)

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
