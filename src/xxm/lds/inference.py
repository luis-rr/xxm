"""LDS inference via Gaussian message passing and Laplace approximation."""

import jax
from jax import numpy as jnp

from xxm.core import batch
from xxm.core.chains.gaussian import GaussianChain as Chain
from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.chains.gaussian import GaussianPairPotential, GaussianPotential
from xxm.core.data import Dataset
from xxm.core.dists.gaussian import Gaussian
from xxm.core.emissions.continuous import (
    EmissionsT,
    LaplaceEmissionsT,
    QuadraticEmissionsT,
)
from xxm.core.inference import InferenceState
from xxm.core.optim.laplace import laplace_inference
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model


def _prior_potentials(
    working_model: Model[EmissionsT],
    working_data: Dataset,
) -> tuple[
    GaussianPotential,
    GaussianPairPotential,
]:
    """Construct the genuine initial and transition factors."""
    initial_potential = GaussianPotential.from_moments(
        working_model.initial.dist,
    )

    valid_transitions = working_data.valid_transitions()
    transition_inputs = working_data.transition_inputs()

    safe_inputs = jnp.where(
        valid_transitions[..., None],
        transition_inputs,
        jnp.zeros(
            (),
            dtype=transition_inputs.dtype,
        ),
    )

    transition_dist = working_model.dynamics.conditional(
        safe_inputs,
    )

    pair_potential = GaussianPairPotential.from_linear_conditional(
        transition_dist,
    ).scale(valid_transitions)

    return initial_potential, pair_potential


def to_chain(
    model: Model[EmissionsT],
    data: Dataset,
) -> Chain:
    """Construct proper latent chains for numerical inference."""
    initial_potential, pair_potential = _prior_potentials(
        model,
        data,
    )

    chain = Chain.from_pair_potentials(
        initial_potential,
        pair_potential,
    )

    valid = data.valid()
    latent_dim = model.dynamics.latent_dim

    dummy_mean = jnp.zeros(
        (*valid.shape, latent_dim),
        dtype=model.initial.dist.mean.dtype,
    )

    dummy_covariance = jnp.broadcast_to(
        jnp.eye(
            latent_dim,
            dtype=model.initial.dist.covariance.dtype,
        ),
        (*valid.shape, latent_dim, latent_dim),
    )

    dummy_potential = GaussianPotential.from_moments(
        Gaussian(
            mean=dummy_mean,
            covariance=dummy_covariance,
        )
    ).scale(~valid)

    return chain.add_local_potential(
        dummy_potential,
    )


def infer_exact(
    model: Model[QuadraticEmissionsT],
    data: Dataset,
) -> InferenceState[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Run exact Gaussian inference independently across all sequences."""
    model_batch_shape = model.batch_shape
    data_batch_shape = data.batch_shape
    working_model, working_data = batch.cartesian_broadcast(model, data)
    latent_chain = to_chain(working_model, working_data)

    observation_potential = working_model.emissions.compute_potential(
        working_data.weighted_observations(),
    )

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward()
    data_axes = tuple(
        range(
            len(model_batch_shape),
            len(model_batch_shape) + len(data_batch_shape),
        )
    )
    objective = jnp.sum(log_normalizer, axis=data_axes) if data_axes else log_normalizer

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=objective,
    )


def infer_laplace(
    model: Model[LaplaceEmissionsT],
    data: Dataset,
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
    model_batch_shape = model.batch_shape
    data_batch_shape = data.batch_shape
    working_model, working_data = batch.cartesian_broadcast(model, data)
    chain = to_chain(working_model, working_data)

    expected_latent_shape = (
        *working_data.batch_shape,
        working_data.num_steps,
        working_model.dynamics.latent_dim,
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
        emissions=working_model.emissions,
        observations=working_data.weighted_observations(),
        initial_latents=latents,
        search_params=params,
    )

    data_axes = tuple(
        range(
            len(model_batch_shape),
            len(model_batch_shape) + len(data_batch_shape),
        )
    )
    objective = jnp.sum(log_normalizer, axis=data_axes) if data_axes else log_normalizer

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=objective,
    )


def elbo_terms(
    model: Model[EmissionsT],
    data: Dataset,
    posterior: Posterior,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Compute per-dataset ELBO contributions."""
    model_batch_shape = model.batch_shape
    data_batch_shape = data.batch_shape

    working_model, working_data = batch.cartesian_broadcast(
        model,
        data,
    )

    expected_batch_shape = (
        *model_batch_shape,
        *data_batch_shape,
    )

    if posterior.batch_shape != expected_batch_shape:
        raise ValueError(
            f'posterior must have batch shape {expected_batch_shape}; '
            f'got {posterior.batch_shape}'
        )

    if posterior.num_steps != working_data.num_steps:
        raise ValueError('posterior and dataset must have the same number of timesteps')

    if posterior.variable_dim != working_model.dynamics.latent_dim:
        raise ValueError(
            'posterior variable dimension must match model latent dimension'
        )

    initial_potential, pair_potential = _prior_potentials(
        working_model,
        working_data,
    )

    second_moments = posterior.raw_second_moments()

    initial = initial_potential.expected_log_potential(
        posterior.means[..., 0, :],
        second_moments[..., 0, :, :],
    )

    dynamics = jnp.sum(
        pair_potential.expected_log_potential(
            posterior.paired_marginals(),
        ),
        axis=-1,
    )

    emissions = working_model.emissions.expected_log_likelihood(
        working_data.weighted_observations(),
        posterior,
    )

    entropy = posterior.entropy(
        valid=working_data.valid(),
    )

    return (
        initial,
        dynamics,
        emissions,
        entropy,
    )


def elbo(
    model: Model[EmissionsT],
    data: Dataset,
    posterior: Posterior,
) -> jax.Array:
    """Compute the dataset-summed ELBO, preserving model batch."""

    (initial, dynamics, emissions, entropy) = elbo_terms(model, data, posterior)

    value = initial + dynamics + emissions + entropy

    data_axes = tuple(
        range(
            len(model.batch_shape),
            len(model.batch_shape) + len(data.batch_shape),
        )
    )

    if data_axes:
        value = jnp.sum(
            value,
            axis=data_axes,
        )

    return value
