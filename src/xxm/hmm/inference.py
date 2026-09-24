"""
Exact inference for Hidden Markov Models.
"""

import jax.numpy as jnp

from xxm.core import batch
from xxm.core.chains.discrete import DiscreteChain as Chain
from xxm.core.data import Dataset
from xxm.core.inference import InferenceState
from xxm.core.latents.discrete import homogeneous_chain

from .core import Model, Posterior


def to_chain(
    model: Model,
    num_steps: int,
) -> Chain:
    """Construct discrete chain from model structure and time horizon."""

    return homogeneous_chain(model.initial, model.transitions, num_steps)


def infer_exact(
    model: Model,
    data: Dataset,
) -> InferenceState[Model, Posterior]:
    """Infer the model × data Cartesian batch and sum only data objectives."""
    if data.input_dim != 0:
        raise ValueError('stationary HMMs require data.input_dim == 0')

    working_model, working_data = batch.cartesian_broadcast(model, data)

    observation_potential = working_model.emissions.compute_potential(
        working_data.weighted_observations(),
    )

    latent_chain = to_chain(
        working_model,
        num_steps=working_data.num_steps,
    )

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward(
        valid=working_data.valid(),
    )
    data_axes = tuple(
        range(
            len(model.batch_shape),
            len(model.batch_shape) + len(data.batch_shape),
        )
    )
    objective = jnp.sum(log_normalizer, axis=data_axes) if data_axes else log_normalizer

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=objective,
    )
