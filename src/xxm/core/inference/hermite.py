"""Conjugate Gaussian coefficient inference for a temporal Hermite variable."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.dists.gaussian import FullGaussian
from xxm.core.latents.hermite import HermiteGaussian
from xxm.core.lengths import Lengths
from xxm.core.potentials.gaussian import GaussianPotential


def to_coefficient_potential(
    hermite: HermiteGaussian,
    local: GaussianPotential,
    basis_values: jax.Array,
) -> GaussianPotential:
    """Pull time-local Gaussian factors back to Hermite coefficient space.

    `basis_values` has shape (*B, T, C), where C is the number of
    basis coefficients. `local` contains factors over P temporal
    variables, with batch shape (*B, T).

    The result is one potential over the flattened P * C coefficients
    for each batch element.
    """
    coefficient_dim = hermite.coefficients.variable_dim

    precision = jnp.einsum(
        '...tc,...tpq,...td->...pcqd',
        basis_values,
        local.precision_blocks,
        basis_values,
    )

    information = jnp.einsum(
        '...tc,...tp->...pc',
        basis_values,
        local.information_vectors,
    )

    bshape = hermite.batch_shape

    precision = precision.reshape((*bshape, coefficient_dim, coefficient_dim))
    information = information.reshape((*bshape, coefficient_dim))

    return GaussianPotential(
        precision_blocks=precision,
        information_vectors=information,
        log_constant=jnp.sum(local.log_constant, axis=-1),
    )


def infer(
    prior: HermiteGaussian,
    local: GaussianPotential,
    *,
    basis_values: jax.Array,
    transition_lengths: Lengths,
    prior_potential: GaussianPotential | None = None,
) -> HermiteGaussian:
    """Update Gaussian coefficients from aligned temporal log potentials.

    A precomputed coefficient prior potential and temporal basis can be shared
    across variational iterations. Chains with no drift values retain the prior.
    """
    if prior_potential is None:
        prior_potential = prior.compute_potential()

    local_coefficients = to_coefficient_potential(
        prior,
        local,
        basis_values,
    )

    coefficients: FullGaussian = prior_potential.add(local_coefficients).normalize()

    posterior = prior.with_coefficients(coefficients)

    return batch.where(
        transition_lengths.values > 0,
        posterior,
        prior,
    )
