"""Continuous model components: initial distribution and linear dynamics."""

import typing

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianPotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, PairedGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.posteriors import ContinuousPosterior


class GaussianInitial(typing.NamedTuple):
    r"""Initial distribution $p(x_0)$ for continuous latent state."""

    dist: Gaussian  # no batch

    def fit_params(self, posterior: ContinuousPosterior) -> typing.Self:
        r"""Fit the initial Gaussian from the posterior moments of $x_0$."""
        mean = posterior.means[0]
        covariance = posterior.covariances[0]

        return self._replace(dist=Gaussian(mean=mean, covariance=covariance))

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample initial latent state."""
        return self.dist.sample(key)

    def align(self, alignment: Affine) -> typing.Self:
        r"""
        Express the initial distribution in coordinates $x' = f(x)$ defined by `alignment`.
        """
        return self._replace(dist=self.dist.affine(alignment))

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        covariance_floor: float = 1e-2,
    ) -> typing.Self:
        """Estimate an initial Gaussian from a known latent trajectory."""

        def _covariance(
            values: jax.Array,
        ) -> jax.Array:
            centered = values - jnp.mean(values, axis=0)
            return centered.T @ centered / values.shape[0]

        def _add_covariance_floor(
            covariance: jax.Array,
            covariance_floor: float,
            reference: jax.Array,
        ) -> jax.Array:
            """Add an isotropic floor relative to the typical variance of a reference."""
            scale = jnp.mean(jnp.var(reference, axis=0))

            return covariance + covariance_floor * scale * jnp.eye(
                covariance.shape[0],
                dtype=covariance.dtype,
            )

        # There is only one initial latent estimate, so use the overall
        # latent covariance as a reasonable scale for its uncertainty.
        return cls(
            Gaussian(
                mean=latents[0],
                covariance=_add_covariance_floor(
                    _covariance(latents),
                    covariance_floor,
                    reference=latents,
                ),
            )
        )


class GaussianLinearDynamics(typing.NamedTuple):
    r"""Linear-Gaussian dynamics $x_t|x_{t-1} \sim \mathcal{N}(Ax_{t-1}+b, Q)$."""

    dist: LinearGaussian  # no batch

    def fit_params(
        self,
        posterior: ContinuousPosterior,
        ridge=gaussian_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        r"""Fit dynamics from posterior pair marginals via moment matching."""
        paired = PairedGaussian(
            left=Gaussian(
                mean=posterior.means[:-1],
                covariance=posterior.covariances[:-1],
            ),
            right=Gaussian(
                mean=posterior.means[1:],
                covariance=posterior.covariances[1:],
            ),
            # posterior stores Cov(x_t, x_{t+1});
            # PairedGaussian stores Cov(right, left).
            cross_covariance=jnp.swapaxes(
                posterior.cross_covariances,
                -2,
                -1,
            ),
        )

        paired = gaussian_fit.paired_from_moment_match(paired)

        return self._replace(
            dist=gaussian_fit.linear_from_paired(
                paired,
                ridge=ridge,
            ),
        )

    def sample_next(
        self,
        key: jax.Array,
        previous: jax.Array,
    ) -> jax.Array:
        """Sample next latent conditional on previous latent."""
        return self.dist.sample(
            key,
            previous,
        )

    def sample(
        self,
        key: jax.Array,
        initial_latent: jax.Array,
        num_steps: int,
    ) -> jax.Array:
        """Sample latent trajectory conditional on initial state."""

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

    def align(self, alignment: Affine) -> typing.Self:
        """
        Express the dynamics in coordinates $x' = f(x)$ defined by `alignment`.
        """
        inverse = alignment.inverse()

        return self._replace(
            dist=(self.dist.compose_input(inverse).compose_output(alignment)),
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        covariance_floor: float,
        ridge=gaussian_fit.DEFAULT_RIDGE,
    ) -> typing.Self:
        """Fit linear dynamics to a known latent trajectory."""
        model = gaussian_fit.linear_from_samples(latents[:-1], latents[1:], ridge=ridge)

        model = model.add_covariance_jitter(covariance_floor)

        return cls(model)


class StateConditionedGaussian(typing.NamedTuple):
    r"""
    Gaussian distribution conditioned on a discrete state.

    $$x \mid z=k \sim \mathcal{N}(\mu_k,\Sigma_k).$$
    """

    dist: Gaussian  # K-batched

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.dist.batch_shape[0]

    def conditional(
        self,
        state: jax.Array,
    ) -> Gaussian:
        """Gaussian distribution conditional on the given state."""
        return self.dist.select(state)

    def compute_potentials(self) -> GaussianPotential:
        """Return one Gaussian potential per discrete state."""
        return GaussianPotential.from_moments(
            self.dist,
        )

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        """Sample conditional on the given discrete state."""
        return self.conditional(state).sample(key)

    def permute(self, permutation: jax.Array) -> typing.Self:
        """
        Relabel state-conditioned Gaussian distributions.
        """
        return self._replace(dist=self.dist.select(permutation))

    def align(self, alignment: Affine) -> typing.Self:
        """
        Express the conditional distributions in coordinates $x' = f(x)$.
        """
        return self._replace(dist=self.dist.affine(alignment))
