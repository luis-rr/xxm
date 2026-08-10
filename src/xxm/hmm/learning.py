from __future__ import annotations

import jax
from jax import numpy as jnp

from .core import Model, Posterior
from .inference import forward_backward


def m_step_initial_probs(
    posterior: Posterior,
) -> jax.Array:
    r"""
    Maximum-likelihood update of the initial-state probabilities.

    .. math::

        \pi_k^{new}
        = p(z_0 = k \mid y)
        = \gamma_0(k)

    where ``gamma`` is the smoothed state posterior.
    """
    return posterior.state_marginals[0]


def m_step_transition_probs(
    posterior: Posterior,
) -> jax.Array:
    r"""
    Maximum-likelihood update of the transition probabilities.

    .. math::

        A_{ij}^{new}
        =
        \frac{\sum_t \xi_t(i,j)}
             {\sum_t \sum_j \xi_t(i,j)}

    where ``xi[t, i, j]`` is the posterior probability of the
    transition ``i -> j`` at time ``t``.
    """
    expected_transitions = posterior.pair_marginals.sum(axis=0)
    return expected_transitions / expected_transitions.sum(axis=-1, keepdims=True)


def e_step(
    model: Model,
    observations: jax.Array,
) -> Posterior:

    emission_log_likelihoods = model.emissions.log_likelihoods(observations)

    return forward_backward(model, emission_log_likelihoods)


def m_step(
    model: Model,
    observations: jax.Array,
    posterior: Posterior,
) -> Model:

    return Model(
        initial_probs=m_step_initial_probs(posterior),
        transition_probs=m_step_transition_probs(posterior),
        emissions=model.emissions.m_step(observations, posterior),
    )


def em_step(
    model: Model,
    observations: jax.Array,
) -> tuple[Model, Posterior]:
    posterior = e_step(model, observations)
    new_params = m_step(model, observations, posterior)
    return new_params, posterior


def fit_em(
    model: Model,
    observations: jax.Array,
    num_iters: int,
) -> tuple[Model, jax.Array]:

    def step(model, _):
        new_model, posterior = em_step(model, observations)
        return new_model, posterior.log_likelihood()

    model, log_likelihoods = jax.lax.scan(
        step,
        model,
        xs=None,
        length=num_iters,
    )

    return model, log_likelihoods


fit_em_jit = jax.jit(fit_em, static_argnames='num_iters')
