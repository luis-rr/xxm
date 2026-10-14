"""LDS inference via Gaussian message passing and Laplace approximation."""

import typing

import jax
from jax import numpy as jnp

from xxm.core import batch
from xxm.core.chains.gaussian import GaussianChain as Chain
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import (
    GaussianChainPotentials,
    GaussianPotential,
)
from xxm.core.data import DatasetCollection, Sequences
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import InferenceState
from xxm.core.optim.laplace import laplace_inference
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model


class Potentials(typing.NamedTuple):
    latent: GaussianChainPotentials
    observation: GaussianPotential


def compute_latent_potentials(
    model: Model[EmissionsT],
    inputs: Sequences,
) -> GaussianChainPotentials:

    initial = model.initial.compute_potential()

    # Broadcast from (*M) initial and (N,) inputs to (*M, N) both.
    initial, inputs = batch.cartesian_broadcast(initial, inputs)

    # Broadcasting is handled internally.
    pairs = model.dynamics.compute_pair_potentials(inputs)

    return GaussianChainPotentials(
        initial=initial,
        pairs=pairs,
        lengths=inputs.lengths,
    )


def compute_potential(
    model: Model[QuadraticEmissionsT],
    data: DatasetCollection,
) -> Potentials:
    inputs = data.pool_inputs()
    latent_potentials = compute_latent_potentials(model, inputs)

    observations = data.weighted_observations()
    observation_potential = model.emissions.compute_potential(observations)

    return Potentials(
        latent=latent_potentials,
        observation=observation_potential,
    )


def infer_exact(
    model: Model[QuadraticEmissionsT],
    data: DatasetCollection,
) -> InferenceState[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Run exact inference across all sessions."""

    potentials = compute_potential(model, data)

    latent_chain = Chain.from_potentials(potentials.latent)
    latent_chain = latent_chain.add_local_potential(potentials.observation)
    inferred = latent_chain.forward_backward()

    return InferenceState(
        model=model,
        posterior=inferred.posterior,
        objective=inferred.log_normalizer_total(),
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    data: DatasetCollection,
    initial_latents: jax.Array | None = None,
    params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> InferenceState[
    Model[LaplaceEmissionsT],
    Posterior,
]:
    """
    Approximate each sequence posterior and sum their Laplace objectives.

    The model × dataset Cartesian batch is constructed explicitly before the
    generic Laplace stack is called.
    """
    inputs = data.pool_inputs()
    latent_potentials = compute_latent_potentials(model, inputs)

    observations = data.weighted_observations()

    latent_chain = Chain.from_potentials(latent_potentials)

    if initial_latents is None:
        initial_latents = latent_chain.forward_backward().posterior.means
    initial_latents = latent_chain.lengths.clear(initial_latents)

    inferred = laplace_inference(
        chain=latent_chain,
        emissions=model.emissions,
        observations=observations,
        initial_latents=initial_latents,
        search_params=params,
    )

    return InferenceState(
        model=model,
        posterior=inferred.posterior,
        objective=inferred.log_normalizer_total(),
    )


def elbo(
    model: Model[EmissionsT],
    data: DatasetCollection,
    posterior: Posterior,
) -> jax.Array:
    """Compute the dataset-summed ELBO, preserving model batch."""

    inputs = data.pool_inputs()
    observations = data.weighted_observations()

    latent = compute_latent_potentials(model, inputs)

    expected_latent = latent.expected_log_potential(posterior)

    expected_observation = model.emissions.expected_log_likelihood(
        observations,
        posterior,
    )

    return jnp.sum(
        expected_latent + expected_observation + posterior.entropy(),
        axis=-1,
    )
