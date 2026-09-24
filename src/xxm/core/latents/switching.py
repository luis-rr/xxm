"""Switching model components."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.gaussian import (
    GaussianPairPotential,
)
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.latents.gaussian import GaussianLinearDynamics
from xxm.core.mask import ArbitraryMask
from xxm.core.optim import gaussian as gaussian_fit
from xxm.core.optim.batch import filter_valid_batches
from xxm.core.posteriors import ContinuousPosterior, DiscretePosterior


class GaussianLinearSwitchingDynamics(typing.NamedTuple):
    r"""State-dependent linear-Gaussian dynamics.

    Under SLDS convention, state $z[t]$ indexes dynamics generating $x[t]$ from $x[t-1]$.
    Consequently the first transition uses $z[1]$; $z[0]$ indexes the initial distribution.
    """

    dist: LinearGaussian  # underlying batch (*B, K); K is the intrinsic state axis

    @classmethod
    def from_params(
        cls,
        *,
        latent_coefficients: jax.Array,
        bias: jax.Array,
        covariance: jax.Array,
        input_coefficients: jax.Array | None = None,
    ) -> typing.Self:
        """Construct state-specific controlled dynamics with predictors [x, u]."""
        dynamics = GaussianLinearDynamics.from_params(
            latent_coefficients=latent_coefficients,
            input_coefficients=input_coefficients,
            bias=bias,
            covariance=covariance,
        )
        return cls(dynamics.dist)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        assert self.dist.batch_shape
        return self.dist.batch_shape[:-1]

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

    @property
    def num_states(self) -> int:
        """Number of discrete states $K$."""
        assert self.dist.batch_shape
        return self.dist.batch_shape[-1]

    @property
    def latent_dim(self) -> int:
        return self.dist.output_dim

    @property
    def input_dim(self) -> int:
        return self.dist.input_size - self.latent_dim

    @property
    def latent_coefficients(self) -> jax.Array:
        return GaussianLinearDynamics(self.dist).latent_coefficients

    @property
    def input_coefficients(self) -> jax.Array:
        return GaussianLinearDynamics(self.dist).input_coefficients

    def conditional(self, inputs: jax.Array) -> LinearGaussian:
        """Condition on inputs `(*B, *Q, D_u)`, returning batch `(*B, *Q, K)`."""
        batch.split_prefix(
            inputs.shape[:-1], self.batch_shape, name='input batch shape'
        )
        state_inputs = batch.broadcast_array(
            inputs, (self.num_states,), axis=len(self.batch_shape)
        )
        conditional = GaussianLinearDynamics(self.dist).conditional(state_inputs)
        return conditional.move_axis(len(self.batch_shape), -1)

    def compute_pair_potentials(self, inputs: jax.Array) -> GaussianPairPotential:
        """Return transition potentials with batch `(*B, *Q, K)`."""
        return GaussianPairPotential.from_linear_conditional(
            self.conditional(inputs),
        )

    def fit_params(
        self,
        discrete: DiscretePosterior,
        continuous: ContinuousPosterior,
        *,
        inputs: jax.Array,
        weights: jax.Array,
        ridge=gaussian_fit.DEFAULT_RIDGE,
        covariance_floor=gaussian_fit.DEFAULT_COV_FLOOR,
    ) -> typing.Self:
        r"""
        Fit dynamics from continuous pair moments weighted by incoming state marginals.

        The pair $(x_{t-1},x_t)$ receives weight $q(z_t=k)$, taken from
        `discrete.state_probs[..., 1:, :]`. States with insufficient posterior mass retain
        their current parameters.
        """

        batch.require_same_shape(discrete.batch_shape, continuous.batch_shape)

        if discrete.num_steps != continuous.num_steps or continuous.num_steps < 1:
            raise ValueError('posteriors must share a nonempty time dimension')
        if (
            discrete.num_states != self.num_states
            or continuous.variable_dim != self.latent_dim
        ):
            raise ValueError(
                'posterior state and latent dimensions must match dynamics'
            )
        transition_shape = (*continuous.batch_shape, continuous.num_steps - 1)
        if inputs.shape != (*transition_shape, self.input_dim):
            raise ValueError(
                'inputs must match posterior transitions and input dimension'
            )
        if weights.shape != transition_shape:
            raise ValueError('weights must match posterior transitions')

        replicate_shape = batch.split_prefix(
            continuous.batch_shape, self.batch_shape, name='posterior batch shape'
        )
        sample_shape = (*replicate_shape, continuous.num_steps - 1)
        if 0 in sample_shape:
            return self

        active = ArbitraryMask(weights != 0)
        state_weights = (
            active.apply(discrete.state_probs[..., 1:, :]) * weights[..., None]
        )
        safe_inputs = active.apply(inputs)
        paired = jax.tree.map(active.apply, continuous.paired_marginals())
        paired, safe_inputs, state_weights = batch.pool_samples(
            (paired, safe_inputs, state_weights),
            self.batch_shape,
            sample_shape,
        )
        state_weights = jnp.moveaxis(state_weights, -1, -2)  # (*B_model, K, S)

        def fit_one(paired_i, inputs_i, weights_i):
            return gaussian_fit.linear_from_paired_covariates(
                paired_i,
                inputs_i,
                weights=weights_i,
                ridge=ridge,
                covariance_floor=covariance_floor,
            )

        fitted = batch.vmap_batch(
            fit_one,
            paired,
            safe_inputs,
            state_weights,
            batch_shape=self.batch_shape,
        )

        return self._replace(
            dist=filter_valid_batches(
                fitted,
                self.dist,
                state_weights,
                min_expected_count=1.0,
            ),
        )

    def sample_next(
        self, key: jax.Array, previous: jax.Array, state: jax.Array, inputs: jax.Array
    ) -> jax.Array:
        """Sample the next latent using the state being entered."""
        dynamics = self.conditional(inputs)
        conditional = batch.take_along_last_batch(dynamics, dynamics.batch_shape, state)
        return conditional.sample(key, previous)

    def sample(
        self,
        key: jax.Array,
        initial_latent: jax.Array,
        states: jax.Array,
        *,
        inputs: jax.Array,
    ) -> jax.Array:
        """Sample from transition-aligned states and inputs, including the start."""
        if states.shape[:-1] != self.batch_shape:
            raise ValueError('states must have shape (*B, T - 1)')
        if inputs.shape != (*states.shape, self.input_dim):
            raise ValueError('inputs must have shape (*B, T - 1, D_u)')
        if initial_latent.shape != (*self.batch_shape, self.latent_dim):
            raise ValueError(
                'initial_latent must match dynamics batch and latent dimension'
            )

        def step(carry, args):
            latent, key = carry
            state, step_inputs = args

            key, sample_key = jax.random.split(key)

            latent = self.sample_next(
                sample_key,
                latent,
                state,
                step_inputs,
            )

            return (latent, key), latent

        _, subsequent_latents = jax.lax.scan(
            step,
            (initial_latent, key),
            (jnp.moveaxis(states, -1, 0), jnp.moveaxis(inputs, -2, 0)),
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
        batch.require_same_shape(
            alignment.batch_shape,
            self.batch_shape,
        )

        aligned = alignment.broadcast(
            (self.num_states,),
            axis=len(self.batch_shape),
        )
        return self._replace(dist=GaussianLinearDynamics(self.dist).align(aligned).dist)
