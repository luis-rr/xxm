"""Prepared conjugate inference for temporal Gaussian variables."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianChainPotentials,
    GaussianPotential,
)
from xxm.core.dists.gaussian import Gaussian
from xxm.core.latents.hermite import HermiteGaussian
from xxm.core.latents.random_walk import GaussianRandomWalk
from xxm.core.lengths import Lengths, LengthsSpec


class HermiteGaussianInference(typing.NamedTuple):
    """Gaussian coefficient inference with cached temporal basis."""

    drift: HermiteGaussian
    prior: GaussianPotential
    basis_values: jax.Array
    transition_lengths: Lengths

    @classmethod
    def from_aligned(
        cls,
        drift: HermiteGaussian,
        lengths_spec: LengthsSpec,
    ) -> typing.Self:
        batch.require_same_shape(
            drift.batch_shape,
            lengths_spec.batch_shape,
        )

        return cls(
            drift=drift,
            prior=drift.compute_potential(),
            basis_values=drift.temporal_basis(lengths_spec),
            transition_lengths=lengths_spec.lengths,
        )

    @classmethod
    def from_drift(
        cls,
        drift: HermiteGaussian,
        lengths_spec: LengthsSpec,
    ) -> typing.Self:
        """Cartesian-broadcast independent model and sequence batches."""

        drift, lengths = batch.cartesian_broadcast(
            drift,
            lengths_spec.lengths,
        )

        return cls.from_aligned(
            drift,
            LengthsSpec(lengths, lengths_spec.max_steps),
        )

    def prepare_initial(
        self,
        posterior: HermiteGaussian,
    ) -> HermiteGaussian:
        return batch.where(
            self.transition_lengths.values > 0,
            posterior,
            self.drift,
        )

    def infer(
        self,
        local: GaussianPotential,
    ) -> HermiteGaussian:
        coefficient_local = self.drift.to_coefficient_potential(
            local,
            self.basis_values,
        )

        coefficients = Gaussian.from_canonical(
            precision=(
                self.prior.precision_blocks + coefficient_local.precision_blocks
            ),
            information=(
                self.prior.information_vectors + coefficient_local.information_vectors
            ),
        )

        posterior = self.drift.with_coefficients(coefficients)
        return self.prepare_initial(posterior)

    def marginals(self, posterior: HermiteGaussian) -> Gaussian:
        return posterior.marginals_from_basis(self.basis_values)

    def expected_log_prior(self, posterior: HermiteGaussian) -> jax.Array:
        """Expected log density of the coefficient prior."""
        return self.drift.expected_log_prob(posterior)

    def entropy(self, posterior: HermiteGaussian) -> jax.Array:
        return posterior.entropy()


class GaussianChainInference(typing.NamedTuple):
    """Gaussian-chain posterior inference from prepared prior factors."""

    prior: GaussianChainPotentials
    max_steps: int

    @classmethod
    def from_drift(
        cls,
        drift: GaussianRandomWalk,
        lengths_spec: LengthsSpec,
    ) -> typing.Self:
        drift, lengths = batch.cartesian_broadcast(
            drift,
            lengths_spec.lengths,
        )

        aligned = LengthsSpec(
            lengths,
            lengths_spec.max_steps,
        )

        return cls(
            prior=drift.compute_potential(aligned),
            max_steps=aligned.max_steps,
        )

    def prepare_initial(
        self,
        posterior: GaussianChainMarginals,
    ) -> GaussianChainMarginals:
        return posterior

    def infer(
        self,
        local: GaussianPotential,
    ) -> GaussianChainMarginals:
        if self.max_steps == 0:
            shape = self.prior.lengths.batch_shape
            dim = self.prior.initial.variable_dim
            dtype = local.precision_blocks.dtype

            return GaussianChainMarginals(
                means=jnp.zeros(shape + (0, dim), dtype=dtype),
                covariances=jnp.zeros(shape + (0, dim, dim), dtype=dtype),
                cross_covariances=jnp.zeros(shape + (0, dim, dim), dtype=dtype),
                lengths=self.prior.lengths,
            )

        chain = GaussianChain.from_potentials(self.prior)
        return chain.add_local_potential(local).forward_backward().posterior

    def marginals(
        self,
        posterior: GaussianChainMarginals,
    ) -> Gaussian:
        return Gaussian(
            mean=posterior.means,
            covariance=posterior.covariances,
        )

    def expected_log_prior(self, posterior: GaussianChainMarginals) -> jax.Array:
        """Expected log density of the prepared chain prior."""
        return self.prior.expected_log_potential(posterior)

    def entropy(self, posterior: GaussianChainMarginals) -> jax.Array:
        return posterior.entropy()
