"""Exact inference conditional on fixed autoregressive history."""

import jax.numpy as jnp

from xxm.core import batch
from xxm.core.inference import InferenceState
from xxm.core.latents.discrete import homogeneous_chain

from .core import Model, Posterior
from .data import ARDataset
from .emissions import ConditionalDistT


def _infer_prepared(
    model: Model[ConditionalDistT],
    data: ARDataset,
) -> InferenceState[
    Model[ConditionalDistT],
    Posterior,
]:
    """Infer compact AR states independently across all source sequences."""
    model_batch_shape = model.batch_shape
    data_batch_shape = data.batch_shape

    working_model, working_data = batch.cartesian_broadcast(
        model,
        data,
    )

    chain = homogeneous_chain(
        working_model.initial,
        working_model.transitions,
        working_data.num_steps,
    )

    posterior_chain = chain.add_local_potential(
        working_model.emissions.compute_potential(
            working_data,
        )
    )

    posterior, log_normalizer = posterior_chain.forward_backward(
        valid=working_data.valid(),
    )

    data_axes = tuple(
        range(
            len(model_batch_shape),
            len(model_batch_shape) + len(data_batch_shape),
        )
    )

    objective = (
        jnp.sum(
            log_normalizer,
            axis=data_axes,
        )
        if data_axes
        else log_normalizer
    )

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=objective,
    )


def infer_exact(
    model: Model[ConditionalDistT],
    data: ARDataset,
) -> InferenceState[
    Model[ConditionalDistT],
    Posterior,
]:
    """Prepare AR rows and infer on the compact post-history timeline."""
    if data.input_dim != 0:
        raise ValueError('AR-HMM does not currently support known external inputs')

    if data.num_lags != model.num_lags:
        raise ValueError('Mismatch between data.num_lags and model.num_lags')

    return _infer_prepared(
        model,
        data,
    )
