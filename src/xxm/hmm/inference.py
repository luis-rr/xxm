import jax
from jax import numpy as jnp

from xxm.core.discrete.chain import DiscreteChain

from .core import Model, Posterior


def _to_chain(model: Model, observations: jax.Array) -> DiscreteChain:
    state_log_potentials = model.emissions.log_likelihoods(observations)

    num_time_steps = observations.shape[0]

    transition_probs = jnp.broadcast_to(
        model.transitions.transition_probs,
        (
            num_time_steps - 1,
            model.num_states,
            model.num_states,
        ),
    )

    return DiscreteChain(
        initial_probs=model.initial.initial_probs,
        transition_probs=transition_probs,
        state_log_potentials=state_log_potentials,
    )


def inference_exact(model: Model, observations: jax.Array) -> Posterior:
    chain = _to_chain(model, observations)
    return chain.forward_backward()
