"""LDS inference via Gaussian message passing and Laplace approximation."""

import jax
from jax import numpy as jnp

from xxm.core.chains.gaussian import GaussianChain as Chain
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import GaussianPairPotential, GaussianPotential
from xxm.core.data import Sequences
from xxm.core.dists.gaussian import Gaussian
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import Inferred
from xxm.core.optim.laplace import laplace_inference
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model, _align_data, _AlignedData


def _to_chain(
    aligned: _AlignedData,
) -> Chain:
    """Construct variable-length latent-prior chains from aligned LDS data."""
    initial_potential = GaussianPotential.from_moments(
        aligned.initial.dist,
    )

    safe_inputs = jnp.where(
        aligned.valid_transitions[..., None],
        aligned.transition_inputs,
        jnp.zeros((), dtype=aligned.transition_inputs.dtype),
    )

    transition_dist = aligned.dynamics.conditional(safe_inputs)
    pair_potential = GaussianPairPotential.from_linear_conditional(
        transition_dist,
    ).scale(
        aligned.valid_transitions,
    )

    chain = Chain.from_pair_potentials(
        initial_potential,
        pair_potential,
    )

    latent_dim = aligned.dynamics.latent_dim
    dummy_mean = jnp.zeros(
        (*aligned.valid.shape, latent_dim),
        dtype=aligned.initial.dist.mean.dtype,
    )
    dummy_covariance = jnp.broadcast_to(
        jnp.eye(
            latent_dim,
            dtype=aligned.initial.dist.covariance.dtype,
        ),
        (*aligned.valid.shape, latent_dim, latent_dim),
    )

    dummy_potential = GaussianPotential.from_moments(
        Gaussian(
            mean=dummy_mean,
            covariance=dummy_covariance,
        )
    ).scale(~aligned.valid)

    return chain.add_local_potential(dummy_potential)


def to_chain(
    model: Model[EmissionsT],
    data: Sequences,
) -> Chain:
    """Construct the aligned variable-length latent-prior chains for a dataset."""
    return _to_chain(
        _align_data(model, data),
    )


def infer_exact(
    model: Model[QuadraticEmissionsT],
    data: Sequences,
) -> Inferred[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Run exact Gaussian inference independently across all sequences."""
    aligned = _align_data(model, data)
    latent_chain = _to_chain(aligned)

    observation_potential = aligned.emissions.compute_potential(
        aligned.observations,
    )

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward()
    objective = jnp.sum(log_normalizer, axis=-1)

    return Inferred(
        model=model,
        posterior=posterior,
        objective=objective,
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    data: Sequences,
    initial_latents: jax.Array | None = None,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Inferred[
    Model[LaplaceEmissionsT],
    Posterior,
]:
    """
    Approximate each sequence posterior and sum their Laplace objectives.

    The model × sequence Cartesian batch is constructed explicitly before the
    generic Laplace stack is called.
    """
    aligned = _align_data(model, data)
    chain = _to_chain(aligned)

    expected_latent_shape = (
        *aligned.valid.shape,
        aligned.dynamics.latent_dim,
    )

    if initial_latents is None:
        prior, _ = chain.forward_backward()
        latents = prior.means
    else:
        if initial_latents.shape != expected_latent_shape:
            raise ValueError(
                f'initial_latents must have shape {expected_latent_shape}; '
                f'got {initial_latents.shape}'
            )
        latents = initial_latents

    posterior, log_normalizer = laplace_inference(
        chain=chain,
        emissions=aligned.emissions,
        observations=aligned.observations,
        initial_latents=latents,
        search_params=params,
    )

    objective = jnp.sum(log_normalizer, axis=-1)

    return Inferred(
        model=model,
        posterior=posterior,
        objective=objective,
    )
