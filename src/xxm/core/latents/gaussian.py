"""Continuous model components: initial distribution and linear dynamics."""

import typing

import jax
from jax import numpy as jnp

from xxm.core import _batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianPotential
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.posteriors import ContinuousPosterior


class GaussianInitial(typing.NamedTuple):
    r"""Initial distribution $p(x_0)$ for continuous latent state."""

    dist: Gaussian  # structural batch *B

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

    def fit_params(
        self,
        posterior: ContinuousPosterior,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        r"""Fit the initial Gaussian from the posterior moments of $x_0$."""

        reference = gaussian_fit.from_moment_match(
            Gaussian(
                mean=posterior.means,
                covariance=posterior.covariances,
            )
        )

        covariance = gaussian_fit.add_covariance_floor(
            posterior.covariances[..., 0, :, :],
            reference_covariance=reference.covariance,
            covariance_floor=covariance_floor,
        )

        return self._replace(
            dist=Gaussian(
                mean=posterior.means[..., 0, :],
                covariance=covariance,
            )
        )

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample initial latent state."""
        return self.dist.sample(key)

    def align(self, alignment: Affine) -> typing.Self:
        r"""
        Express the initial distribution in coordinates $x' = f(x)$ defined by `alignment`.
        """
        assert alignment.batch_shape in ((), self.batch_shape)
        return self._replace(dist=self.dist.affine(alignment))

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Estimate an initial Gaussian from a known latent trajectory."""

        batch_shape = latents.shape[:-2]
        flat = _batch.flatten_batch(latents, batch_shape)
        flat_reference = jax.vmap(
            lambda values: gaussian_fit.from_samples(
                values, covariance_floor=covariance_floor
            )
        )(flat)
        reference = _batch.unflatten_batch(flat_reference, batch_shape)

        return cls(
            Gaussian(
                mean=latents[..., 0, :],
                covariance=reference.covariance,
            )
        )


class GaussianLinearDynamics(typing.NamedTuple):
    r"""Linear-Gaussian dynamics $x_t|x_{t-1} \sim \mathcal{N}(Ax_{t-1}+b, Q)$."""

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

    def fit_params(
        self,
        posterior: ContinuousPosterior,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        r"""Fit dynamics from posterior pair marginals via moment matching."""

        paired = gaussian_fit.paired_from_moment_match(
            posterior.paired_marginals(),
        )

        return self._replace(
            dist=gaussian_fit.linear_from_paired(
                paired,
                ridge=ridge,
                covariance_floor=covariance_floor,
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
        """Sample latent trajectories with shape `(*B, T, D)`."""

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

        subsequent_latents = jnp.moveaxis(
            subsequent_latents,
            0,
            -2,
        )

        return jnp.concatenate(
            [
                initial_latent[..., None, :],
                subsequent_latents,
            ],
            axis=-2,
        )

    def align(self, alignment: Affine) -> typing.Self:
        """
        Express the dynamics in coordinates $x' = f(x)$ defined by `alignment`.
        """
        if alignment.batch_shape == ():
            alignment = alignment.broadcast(self.batch_shape)
        else:
            assert alignment.batch_shape == self.batch_shape
        inverse = alignment.inverse()

        return self._replace(
            dist=(self.dist.compose_input(inverse).compose_output(alignment)),
        )

    @classmethod
    def from_latents(
        cls,
        latents: jax.Array,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR_INIT,
    ) -> typing.Self:
        """Fit linear dynamics to a known latent trajectory."""
        batch_shape = latents.shape[:-2]
        flat = _batch.flatten_batch(latents, batch_shape)
        flat_model = jax.vmap(
            lambda values: gaussian_fit.linear_from_samples(
                values[:-1],
                values[1:],
                ridge=ridge,
                covariance_floor=covariance_floor,
            )
        )(flat)
        return cls(_batch.unflatten_batch(flat_model, batch_shape))


class StateConditionedGaussian(typing.NamedTuple):
    r"""
    Gaussian distribution conditioned on a discrete state.

    $$x \mid z=k \sim \mathcal{N}(\mu_k,\Sigma_k).$$
    """

    dist: Gaussian  # underlying batch (*B, K); K is the intrinsic state axis

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

    def select(self, index) -> typing.Self:
        index = _batch.selection(index, len(self.batch_shape))
        return self._replace(dist=self.dist.select(index + (slice(None),)))

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self._replace(dist=self.dist.broadcast(shape, axis=axis))

    def squeeze(self, axis=None) -> typing.Self:
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self._replace(dist=self.dist.squeeze(axes))

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        axis = _batch.axis_index(axis, len(self.batch_shape))
        return self._replace(dist=self.dist.permute(permutation, axis=axis))

    def move_axis(self, source: int, destination: int) -> typing.Self:
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self._replace(dist=self.dist.move_axis(source, destination))

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    def conditional(
        self,
        state: jax.Array,
    ) -> Gaussian:
        """Gaussian distribution conditional on the given state."""
        return _batch.take_along_last_batch(self.dist, self.dist.batch_shape, state)

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

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """
        Relabel state-conditioned Gaussian distributions.
        """
        return self._replace(dist=self.dist.permute(permutation, axis=-1))

    def align(self, alignment: Affine) -> typing.Self:
        """
        Express the conditional distributions in coordinates $x' = f(x)$.
        """
        assert alignment.batch_shape in ((), self.batch_shape)
        return self._replace(dist=self.dist.affine(alignment))
