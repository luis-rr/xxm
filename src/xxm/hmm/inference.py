"""
Exact inference for Hidden Markov Models.
"""

import jax

from xxm.core.chains.discrete import DiscreteChain as Chain
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
    observations: jax.Array,
) -> InferenceState[Model, Posterior]:
    """Run forward-backward inference: T observations have T latent states."""

    observation_potential = model.emissions.compute_potential(
        observations,
    )

    latent_chain = to_chain(
        model,
        num_steps=observations.shape[0],
    )

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward()

    return InferenceState(
        model=model,
        posterior=posterior,
        objective=log_normalizer,
    )
