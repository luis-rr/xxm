"""Gaussian-chain inference from canonical log potentials."""

from __future__ import annotations

import jax.numpy as jnp

from xxm.core.dists.gaussian import GaussianChainMarginals
from xxm.core.lengths import Lengths
from xxm.core.potentials.gaussian import GaussianChainPotentials, GaussianPotential

from .chains.gaussian import GaussianChain
from .results import InferredPosterior


def _infer_dummy(
    lengths: Lengths,
    dim: int,
    dtype: jnp.dtype,
) -> InferredPosterior[GaussianChainMarginals]:

    shape = lengths.batch_shape

    return InferredPosterior(
        posterior=GaussianChainMarginals(
            means=jnp.zeros(shape + (0, dim), dtype=dtype),
            covariances=jnp.zeros(shape + (0, dim, dim), dtype=dtype),
            cross_covariances=jnp.zeros(shape + (0, dim, dim), dtype=dtype),
            lengths=lengths,
        ),
        log_normalizer=jnp.zeros(
            shape,
            dtype=dtype,
        ),
    )


def infer(
    potentials: GaussianChainPotentials,
    local: GaussianPotential,
) -> InferredPosterior[GaussianChainMarginals]:
    """Normalize a Gaussian chain, including an empty (zero-step) chain."""

    if local.precision_blocks.shape[-3] == 0:
        return _infer_dummy(
            lengths=potentials.lengths,
            dim=potentials.initial.variable_dim,
            dtype=local.precision_blocks.dtype,
        )

    chain = GaussianChain.from_potentials(potentials)
    return chain.add_local_potential(local).forward_backward()
