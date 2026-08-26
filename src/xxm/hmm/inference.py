import jax
from jax import numpy as jnp

from xxm.core.chains.discrete import DiscreteChain as Chain
from xxm.hmm.core import Model, Posterior


def to_chain(
    model: Model,
    num_time_steps: int,
) -> Chain:
    transition_probs = jnp.broadcast_to(
        model.transitions.model.probs,
        (
            num_time_steps - 1,
            model.num_states,
            model.num_states,
        ),
    )

    return Chain(
        initial_probs=model.initial.model.probs,
        transition_probs=transition_probs,
        state_log_potentials=jnp.zeros(
            (num_time_steps, model.num_states),
            dtype=model.initial.model.probs.dtype,
        ),
    )


def inference_exact(
    model: Model,
    observations: jax.Array,
) -> tuple[Posterior, jax.Array]:
    """Compute the exact posterior over latents."""

    latent_chain = to_chain(
        model,
        num_time_steps=observations.shape[0],
    )

    observation_potential = model.emissions.get_potential(observations)

    posterior_chain = latent_chain.add_local_potential(
        observation_potential,
    )

    return posterior_chain.forward_backward()
