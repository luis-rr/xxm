"""
Exact inference for Hidden Markov Models.
"""

import jax
from jax import numpy as jnp

from xxm.core.chains.discrete import DiscreteChain as Chain
from xxm.core.inference import Inferred

from .core import Model, Posterior


def to_chain(
    model: Model,
    num_steps: int,
) -> Chain:
    """Construct discrete chain from model structure and time horizon."""

    transition_probs = jnp.broadcast_to(
        model.transitions.dist.probs,
        (
            num_steps - 1,
            model.num_states,
            model.num_states,
        ),
    )

    return Chain(
        initial_probs=model.initial.dist.probs,
        transition_probs=transition_probs,
        state_log_potentials=jnp.zeros(
            (num_steps, model.num_states),
            dtype=model.initial.dist.probs.dtype,
        ),
    )


def infer_exact(
    model: Model,
    observations: jax.Array,
) -> Inferred[Model, Posterior]:
    """
    Run forward-backward inference.

    For autoregressive emissions, the likelihood is conditional on the fixed
    initial observation history.
    """

    observation_potential = model.emissions.compute_potential(
        observations,
    )

    latent_chain = to_chain(
        model,
        num_steps=observation_potential.log_values.shape[0],
    )

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    posterior, log_normalizer = posterior_chain.forward_backward()

    return Inferred(
        model=model,
        posterior=posterior,
        objective=log_normalizer,
    )
