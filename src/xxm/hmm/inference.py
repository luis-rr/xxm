"""
Exact inference for Hidden Markov Models.
"""

import jax.numpy as jnp

from xxm.core import batch
from xxm.core.data import DatasetCollection
from xxm.core.inference import InferenceState
from xxm.core.latents.discrete import homogeneous_chain

from .core import Model, Posterior


def infer_exact(
    model: Model,
    data: DatasetCollection,
) -> InferenceState[Model, Posterior]:
    """Infer the model x data Cartesian batch and sum only data objectives."""
    dataset = data.single()
    if dataset.input_dim != 0:
        raise ValueError('stationary HMMs require dataset.input_dim == 0')

    working_model, working_data = batch.cartesian_broadcast(model, dataset)

    observation_potential = working_model.emissions.single().compute_potential(
        working_data.weighted_observations(),
    )

    latent_chain = homogeneous_chain(
        working_model.initial,
        working_model.transitions,
        ls=working_data.lengths_spec,
    )

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward()
    data_axes = tuple(
        range(
            len(model.batch_shape),
            len(model.batch_shape) + len(dataset.batch_shape),
        )
    )
    objective = jnp.sum(log_normalizer, axis=data_axes) if data_axes else log_normalizer

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=objective,
    )
