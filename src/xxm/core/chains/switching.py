"""State-conditioned Gaussian chain potentials."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.chains.discrete import DiscreteChainMarginals, DiscretePotential
from xxm.core.chains.gaussian import (
    GaussianChainMarginals,
    GaussianChainPotentials,
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.lengths import Lengths, LengthsSpec


class ConditionedGaussianChainPotentials(typing.NamedTuple):
    """Gaussian factors conditioned on the incoming discrete state.

    Initial factors have batch (*W, K), pair factors (*W, T-1, K).
    The state axis K is the last Gaussian-potential batch axis.
    """

    initial: GaussianPotential
    pairs: GaussianPairPotential
    lengths: Lengths

    def compute_continuous_potentials(
        self,
        discrete: DiscreteChainMarginals,
    ) -> GaussianChainPotentials:
        states = self.lengths.clear(discrete.state_probs)

        initial = self.initial.weighted_sum(
            states[..., 0, :],
            axis=-1,
        )

        transitions = self.lengths.transitions()
        weights = transitions.clear(states[..., 1:, :])

        pairs = self.pairs.weighted_sum(weights, axis=-1)

        return GaussianChainPotentials(
            initial=initial,
            pairs=pairs,
            lengths=self.lengths,
        )

    def compute_discrete_potential(
        self,
        continuous: GaussianChainMarginals,
    ) -> DiscretePotential:
        """Expected state-specific Gaussian log factors."""

        means = self.lengths.clear(continuous.means)
        covariances = self.lengths.clear(continuous.covariances)

        num_states = self.initial.batch_shape[-1]
        state_axis = len(continuous.batch_shape)

        mean = batch.broadcast_array(
            means[..., 0, :],
            (num_states,),
            axis=state_axis,
        )
        covariance = batch.broadcast_array(
            covariances[..., 0, :, :],
            (num_states,),
            axis=state_axis,
        )

        second = covariance + mean[..., :, None] * mean[..., None, :]

        initial = self.initial.expected_log_potential(mean, second)

        transitions = self.lengths.transitions()
        paired = jax.tree.map(
            transitions.clear,
            continuous.paired_marginals(),
        )
        paired = paired.broadcast((num_states,), axis=-1)

        dynamics = self.pairs.expected_log_potential(paired)
        dynamics = transitions.clear(dynamics)

        values = jnp.concatenate(
            (initial[..., None, :], dynamics),
            axis=-2,
        )

        return DiscretePotential(self.lengths.clear(values))

    def compute_discrete_potential_at(
        self,
        latents: jax.Array,
    ) -> DiscretePotential:
        """State-specific Gaussian factors at deterministic latents."""

        latents = self.lengths.clear(latents)

        num_states = self.initial.batch_shape[-1]
        axis = len(self.lengths.batch_shape)

        initial = self.initial.log_potential(
            batch.broadcast_array(
                latents[..., 0, :],
                (num_states,),
                axis=axis,
            )
        )

        left = batch.broadcast_array(
            latents[..., :-1, :],
            (num_states,),
            axis=axis + 1,
        )
        right = batch.broadcast_array(
            latents[..., 1:, :],
            (num_states,),
            axis=axis + 1,
        )

        dynamics = self.pairs.log_potential(left, right)

        values = jnp.concatenate(
            (initial[..., None, :], dynamics),
            axis=-2,
        )

        return DiscretePotential(self.lengths.clear(values))


class GatedGaussianPotentials(typing.NamedTuple):
    """Initial, active and inactive Gaussian random-walk factors."""

    initial: GaussianPotential
    active: GaussianPairPotential
    inactive: GaussianPairPotential

    def compute_chain_potentials(
        self,
        active_weights: jax.Array,
        lengths_spec: LengthsSpec,
    ) -> GaussianChainPotentials:
        """Average active/inactive log potentials under gate probabilities.

        The gate axis belongs to the independent Gaussian-chain batch.
        Active weights have shape (*B, K, T-1), where T is the
        number of random-walk values.
        """

        lengths = lengths_spec.lengths
        num_pairs = max(lengths_spec.max_steps - 1, 0)

        batch.require_same_shape(
            self.initial.batch_shape,
            lengths.batch_shape,
        )

        expected = lengths.batch_shape + (num_pairs,)
        if active_weights.shape != expected:
            raise ValueError(f'active weights must have shape {expected}')

        inactive = self.inactive.broadcast(
            (num_pairs,),
            axis=len(self.inactive.batch_shape),
        )

        difference = self.active.sub(self.inactive).broadcast(
            (num_pairs,),
            axis=len(self.active.batch_shape),
        )

        pairs = inactive.add(difference.scale(active_weights))

        pairs = pairs.scale(lengths.transitions().to_mask(num_pairs).values)

        return GaussianChainPotentials(
            initial=self.initial.scale(lengths.values > 0),
            pairs=pairs,
            lengths=lengths,
        )

    def expected_gate_log_difference(
        self,
        posterior: GaussianChainMarginals,
    ) -> jax.Array:
        """Expected log active minus log inactive transition factors.

        Returns (*B, K, T-1), aligned to random-walk transitions,
        not the outer switching-model timeline.
        """

        num_pairs = max(posterior.max_steps - 1, 0)

        if num_pairs == 0:
            return jnp.zeros(
                posterior.batch_shape + (0,),
                dtype=posterior.means.dtype,
            )

        difference = self.active.sub(self.inactive).broadcast(
            (num_pairs,),
            axis=len(self.active.batch_shape),
        )

        values = difference.expected_log_potential(posterior.paired_marginals())

        return posterior.lengths.transitions().clear(values)
