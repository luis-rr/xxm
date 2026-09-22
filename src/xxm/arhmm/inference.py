"""Exact inference conditional on fixed autoregressive history."""

import jax

from xxm.core.inference import InferenceState
from xxm.core.latents.discrete import homogeneous_chain

from .core import Model, Posterior
from .data import ARObservations
from .emissions import ConditionalDistT


def _infer_prepared(
    model: Model[ConditionalDistT],
    data: ARObservations,
) -> InferenceState[Model[ConditionalDistT], Posterior]:
    """Infer the state selecting each prepared conditional regression row."""
    chain = homogeneous_chain(model.initial, model.transitions, data.num_steps)
    posterior_chain = chain.add_local_potential(
        model.emissions.compute_potential(data),
    )
    posterior, log_normalizer = posterior_chain.forward_backward()
    return InferenceState(
        model=model,
        posterior=posterior,
        objective=log_normalizer,
    )


def infer_exact(
    model: Model[ConditionalDistT],
    observations: jax.Array,
) -> InferenceState[Model[ConditionalDistT], Posterior]:
    """Infer T-L states; posterior row r corresponds to observations[L+r]."""
    data = ARObservations.from_observations(observations, model.num_lags)
    return _infer_prepared(model, data)
