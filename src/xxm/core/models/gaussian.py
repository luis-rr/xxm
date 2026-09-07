import typing

import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianPotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, PairedGaussian
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

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent initial distribution in aligned coordinates."""
        return self._replace(model=self.model.affine(alignment))

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        covariance_floor: float = 1e-2,
    ) -> typing.Self:
        """Construct an LDS from a known latent trajectory."""

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
    model: LinearGaussian  # no batch

    def fit_params(self, posterior: ContinuousPosterior) -> typing.Self:
        """Maximum-likelihood update of the latent dynamics."""
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
            model=gaussian_fit.linear_from_paired(paired),
        )

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

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        inverse = alignment.inverse()

        return self._replace(
            model=(self.model.compose_input(inverse).compose_output(alignment)),
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        covariance_floor: float,
    ) -> typing.Self:
        """Fit linear dynamics to a known latent trajectory."""
        model = gaussian_fit.linear_from_samples(
            latents[:-1],
            latents[1:],
        )

        model = model.add_covariance_jitter(covariance_floor)

        return cls(model)


class StateConditionedGaussian(typing.NamedTuple):
    """Gaussian distribution conditioned on a discrete state."""

    model: Gaussian  # K-batched

    @property
    def num_states(self) -> int:
        return self.model.batch_shape[0]

    def conditional(
        self,
        state: jax.Array,
    ) -> Gaussian:
        """Gaussian distribution conditional on the given state."""
        return self.model.select(state)

    def compute_potentials(self) -> GaussianPotential:
        """Return one Gaussian potential per discrete state."""
        return GaussianPotential.from_moments(
            self.model,
        )

    def sample(
        self,
        key: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        """Sample conditional on the given discrete state."""
        return self.conditional(state).sample(key)

    def permute(self, permutation: jax.Array) -> typing.Self:
        return self._replace(model=self.model.select(permutation))

    def align(self, alignment: Affine) -> typing.Self:
        return self._replace(model=self.model.affine(alignment))
