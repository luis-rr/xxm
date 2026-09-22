"""Switching model components."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import _batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import (
    GaussianPairPotential,
)
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim._batch import filter_valid_batches
from xxm.core.posteriors import ContinuousPosterior, DiscretePosterior


class GaussianLinearSwitchingDynamics(typing.NamedTuple):
    r"""State-dependent linear-Gaussian dynamics.

    Under SLDS convention, state $z[t]$ indexes dynamics generating $x[t]$ from $x[t-1]$.
    Consequently the first transition uses $z[1]$; $z[0]$ indexes the initial distribution.
    """

    dist: LinearGaussian  # underlying batch (*B, K); K is the intrinsic state axis

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

    def compute_pair_potentials(self) -> GaussianPairPotential:
        """Return one Gaussian transition potential per discrete state."""
        return GaussianPairPotential.from_linear_conditional(
            self.dist,
        )

    def fit_params(
        self,
        discrete: DiscretePosterior,
        continuous: ContinuousPosterior,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        r"""
        Fit dynamics from continuous pair moments weighted by incoming state marginals.

        The pair $(x_{t-1},x_t)$ receives weight $q(z_t=k)$, taken from
        `discrete.state_probs[..., 1:, :]`. States with insufficient posterior mass retain
        their current parameters.
        """

        weights = jnp.moveaxis(discrete.state_probs[..., 1:, :], -1, -2)
        # (*B, K, T - 1)

        paired = gaussian_fit.paired_from_moment_match(
            continuous.paired_marginals(),
            weights=weights,
        )  # batch (*B, K)

        fitted = gaussian_fit.linear_from_paired(
            paired,
            ridge=ridge,
            covariance_floor=covariance_floor,
        )

        return self._replace(
            dist=filter_valid_batches(
                fitted,
                self.dist,
                weights,
                min_expected_count=1.0,
            ),
        )

    def sample_next(
        self, key: jax.Array, previous: jax.Array, state: jax.Array
    ) -> jax.Array:
        """Sample the next latent using the state being entered."""
        conditional = _batch.take_along_last_batch(
            self.dist, self.dist.batch_shape, state
        )
        return conditional.sample(key, previous)

    def sample(
        self, key: jax.Array, initial_latent: jax.Array, states: jax.Array
    ) -> jax.Array:
        """Sample subsequent latents from their incoming switching states."""

        def step(carry, state):
            latent, key = carry

            key, sample_key = jax.random.split(key)

            latent = self.sample_next(
                sample_key,
                latent,
                state,
            )

            return (latent, key), latent

        _, subsequent_latents = jax.lax.scan(
            step,
            (initial_latent, key),
            jnp.moveaxis(states, -1, 0),
        )

        subsequent_latents = jnp.moveaxis(subsequent_latents, 0, -2)
        return jnp.concatenate(
            [
                initial_latent[..., None, :],
                subsequent_latents,
            ],
            axis=-2,
        )

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """
        Relabel discrete-state-indexed dynamics by permutation.
        """
        return self._replace(
            dist=self.dist.permute(permutation, axis=-1),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        _batch.require_same(
            alignment.batch_shape,
            self.batch_shape,
        )

        aligned = alignment.broadcast(
            (self.num_states,),
            axis=len(self.batch_shape),
        )
        inverse = aligned.inverse()

        return self._replace(
            dist=(self.dist.compose_input(inverse).compose_output(aligned)),
        )
