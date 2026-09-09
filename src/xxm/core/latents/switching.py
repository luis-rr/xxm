"""Switching model components."""

import typing

import jax
import jax.numpy as jnp

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import (
    GaussianPairPotential,
)
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, PairedGaussian
from xxm.core.optim import categorical as categorical_fit
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.posteriors import ContinuousPosterior, DiscretePosterior


class GaussianLinearSwitchingDynamics(typing.NamedTuple):
    r"""State-dependent linear-Gaussian dynamics.

    Under SLDS convention, state $z[t]$ indexes dynamics generating $x[t]$ from $x[t-1]$.
    Consequently the first transition uses $z[1]$; $z[0]$ indexes the initial distribution.
    """

    dist: LinearGaussian  # K-batched, input dimension D, output dimension D

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        return self.dist.covariance.shape[0]

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
        `discrete.state_probs[1:]`. States with insufficient posterior mass retain
        their current parameters.
        """

        weights = discrete.state_probs[1:]  # (T-1, K)

        paired = PairedGaussian(
            left=Gaussian(
                mean=continuous.means[:-1],
                covariance=continuous.covariances[:-1],
            ),
            right=Gaussian(
                mean=continuous.means[1:],
                covariance=continuous.covariances[1:],
            ),
            cross_covariance=jnp.swapaxes(
                continuous.cross_covariances,
                -2,
                -1,
            ),
        )  # (T-1)-batched

        paired = gaussian_fit.paired_from_moment_match(
            paired,
            weights=weights,
        )  # K-batched

        fitted = gaussian_fit.linear_from_paired(
            paired,
            ridge=ridge,
            covariance_floor=covariance_floor,
        )

        return self._replace(
            dist=categorical_fit.filter_valid_states(
                fitted,
                self.dist,
                weights,
            ),
        )

    def sample_next(
        self, key: jax.Array, previous: jax.Array, state: jax.Array
    ) -> jax.Array:
        """Sample the next latent using the state being entered."""
        return self.dist.select(state).sample(key, previous)

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
            states,
        )

        return jnp.concatenate(
            [
                initial_latent[None],
                subsequent_latents,
            ],
            axis=0,
        )

    def permute(self, permutation: jax.Array) -> typing.Self:
        """
        Relabel discrete-state-indexed dynamics by permutation.
        """
        return self._replace(
            dist=self.dist.select(permutation),
        )

    def align(self, alignment: Affine) -> typing.Self:
        """Express the latent dynamics in aligned coordinates."""
        inverse = alignment.inverse()

        return self._replace(
            dist=(self.dist.compose_input(inverse).compose_output(alignment)),
        )
