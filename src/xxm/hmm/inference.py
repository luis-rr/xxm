r"""Forward-backward inference for a single-sequence hidden Markov model."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from xxm.hmm.core import Model, Posterior


def _validate_forward_inputs(model: Model, emission_log_likelihoods: jax.Array) -> tuple[int, int]:

    if model.initial_probs.ndim != 1:
        raise ValueError('initial_probs must have shape (K,)')

    if emission_log_likelihoods.ndim != 2:
        raise ValueError('emission_log_likelihoods must have shape (T, K)')

    if model.transition_probs.ndim != 2:
        raise ValueError('transition_probs must have shape (K, K)')

    k = model.initial_probs.shape[0]
    t = emission_log_likelihoods.shape[0]

    if emission_log_likelihoods.shape[1] != k:
        raise ValueError('emission_log_likelihoods must have shape (T, K) with matching K')

    if model.transition_probs.shape != (k, k):
        raise ValueError('transition_probs must have shape (K, K)')

    return t, k


def _validate_backward_inputs(
    model: Model,
    emission_log_likelihoods: jax.Array,
    log_scaling_factors: jax.Array,
) -> tuple[int, int]:

    if emission_log_likelihoods.ndim != 2:
        raise ValueError('emission_log_likelihoods must have shape (T, K)')

    if model.transition_probs.ndim != 2:
        raise ValueError('transition_probs must have shape (K, K)')

    t, k = emission_log_likelihoods.shape

    if model.transition_probs.shape != (k, k):
        raise ValueError('transition_probs must have shape (K, K)')

    if log_scaling_factors.ndim != 1:
        raise ValueError('log_scaling_factors must have shape (T,)')

    if log_scaling_factors.shape != (t,):
        raise ValueError('log_scaling_factors must have shape (T,)')

    return t, k


def forward_pass(
    model: Model,
    emission_log_likelihoods: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run a normalized probability-space forward recursion."""
    t, k = _validate_forward_inputs(model, emission_log_likelihoods)

    min_normalizer = jnp.finfo(emission_log_likelihoods.dtype).tiny

    def _normalized_emission_weights(
        emission_log_likelihood: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        emission_offset = jnp.max(emission_log_likelihood)
        emission_weights = jnp.exp(emission_log_likelihood - emission_offset)
        return emission_weights, emission_offset

    first_emission_weights, first_emission_offset = _normalized_emission_weights(
        emission_log_likelihoods[0]
    )
    first_unnormalized = model.initial_probs * first_emission_weights
    first_normalizer = jnp.maximum(jnp.sum(first_unnormalized), min_normalizer)
    first_forward_probs = first_unnormalized / first_normalizer
    first_log_normalizer = jnp.log(first_normalizer) + first_emission_offset

    def forward_step(
        previous_forward_probs: jax.Array,
        emission_log_likelihood: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        predictive_probs = previous_forward_probs @ model.transition_probs

        emission_weights, emission_offset = _normalized_emission_weights(emission_log_likelihood)

        unnormalized_forward_probs = predictive_probs * emission_weights
        forward_normalizer = jnp.maximum(
            jnp.sum(unnormalized_forward_probs),
            min_normalizer,
        )

        current_forward_probs = unnormalized_forward_probs / forward_normalizer
        current_log_normalizer = jnp.log(forward_normalizer) + emission_offset

        return current_forward_probs, (
            current_forward_probs,
            current_log_normalizer,
        )

    if t == 1:
        return first_forward_probs[None, :], first_log_normalizer[None]

    _, (remaining_forward_probs, remaining_log_scaling_factors) = jax.lax.scan(
        forward_step,
        first_forward_probs,
        emission_log_likelihoods[1:],
    )

    forward_probs = jnp.concatenate(
        [first_forward_probs[None, :], remaining_forward_probs],
        axis=0,
    )
    log_scaling_factors = jnp.concatenate(
        [first_log_normalizer[None], remaining_log_scaling_factors],
        axis=0,
    )

    return forward_probs, log_scaling_factors


def backward_pass(
    model: Model,
    emission_log_likelihoods: jax.Array,
    log_scaling_factors: jax.Array,
) -> jax.Array:
    """Run the backward recursion consistent with forward normalizers."""
    t, k = _validate_backward_inputs(
        model,
        emission_log_likelihoods,
        log_scaling_factors,
    )

    terminal_backward_probs = jnp.ones(
        (k,),
        dtype=emission_log_likelihoods.dtype,
    )

    if t == 1:
        return terminal_backward_probs[None, :]

    def backward_step(
        backward_probs_at_next_time: jax.Array,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        next_emission_log_likelihood, next_log_normalizer = inputs

        emission_offset = jnp.max(next_emission_log_likelihood)
        next_emission_weights = jnp.exp(next_emission_log_likelihood - emission_offset)

        weighted_future_probs = next_emission_weights * backward_probs_at_next_time

        propagated_backward_probs = model.transition_probs @ weighted_future_probs

        normalization_scale = jnp.exp(emission_offset - next_log_normalizer)

        backward_probs_at_current_time = propagated_backward_probs * normalization_scale

        return (
            backward_probs_at_current_time,
            backward_probs_at_current_time,
        )

    _, reverse_backward_probs = jax.lax.scan(
        backward_step,
        terminal_backward_probs,
        (
            emission_log_likelihoods[1:][::-1],
            log_scaling_factors[1:][::-1],
        ),
    )

    return jnp.concatenate(
        [reverse_backward_probs[::-1], terminal_backward_probs[None, :]],
        axis=0,
    )


def posterior_marginals(
    forward_probs: jax.Array,
    backward_probs: jax.Array,
) -> jax.Array:
    """Compute posterior state probabilities gamma[t, k]."""
    if (
        forward_probs.ndim != 2
        or backward_probs.ndim != 2
        or forward_probs.shape != backward_probs.shape
    ):
        raise ValueError('forward_probs and backward_probs must both have shape (T, K) and match.')

    unnormalized_state_posterior_probs = forward_probs * backward_probs
    min_normalizer = jnp.finfo(forward_probs.dtype).tiny
    state_posterior_normalizers = jnp.maximum(
        jnp.sum(unnormalized_state_posterior_probs, axis=1, keepdims=True),
        min_normalizer,
    )
    return unnormalized_state_posterior_probs / state_posterior_normalizers


def _validate_posterior_pair_marginals_inputs(
    model: Model,
    emission_log_likelihoods: jax.Array,
    log_scaling_factors: jax.Array,
    t: int,
    k: int,
):
    if emission_log_likelihoods.shape != (t, k):
        raise ValueError('emission_log_likelihoods must have shape (T, K).')

    if model.transition_probs.shape != (k, k):
        raise ValueError('transition_probs must have shape (K, K).')

    if log_scaling_factors.shape != (t,):
        raise ValueError('log_scaling_factors must have shape (T,).')


def posterior_pair_marginals(
    forward_probs: jax.Array,
    backward_probs: jax.Array,
    model: Model,
    emission_log_likelihoods: jax.Array,
    log_scaling_factors: jax.Array,
) -> jax.Array:
    """Compute posterior pair probabilities xi[t, i, j]."""
    if (
        forward_probs.ndim != 2
        or backward_probs.ndim != 2
        or forward_probs.shape != backward_probs.shape
    ):
        raise ValueError('forward_probs and backward_probs must both have shape (T, K) and match.')

    t, k = forward_probs.shape

    _validate_posterior_pair_marginals_inputs(
        model,
        emission_log_likelihoods,
        log_scaling_factors,
        t,
        k,
    )

    if t == 1:
        return jnp.zeros((0, k, k), dtype=forward_probs.dtype)

    def pair_step(
        current_forward_probs: jax.Array,
        next_backward_probs: jax.Array,
        next_emission_log_likelihood: jax.Array,
        next_log_normalizer: jax.Array,
    ) -> jax.Array:
        emission_offset = jnp.max(next_emission_log_likelihood)

        next_emission_weights = jnp.exp(next_emission_log_likelihood - emission_offset)

        future_weights = next_emission_weights * next_backward_probs

        unnormalized_pair_probs = (
            current_forward_probs[:, None] * model.transition_probs * future_weights[None, :]
        )

        normalization_scale = jnp.exp(emission_offset - next_log_normalizer)

        return unnormalized_pair_probs * normalization_scale

    return jax.vmap(pair_step)(
        forward_probs[:-1],
        backward_probs[1:],
        emission_log_likelihoods[1:],
        log_scaling_factors[1:],
    )


def forward_backward(model: Model, emission_log_likelihoods: jax.Array) -> Posterior:
    """Run full forward-backward inference for one sequence."""

    forward_probs, log_scaling_factors = forward_pass(model, emission_log_likelihoods)

    backward_probs = backward_pass(
        model=model,
        log_scaling_factors=log_scaling_factors,
        emission_log_likelihoods=emission_log_likelihoods,
    )

    return Posterior(
        forward_probs=forward_probs,
        backward_probs=backward_probs,
        log_scaling_factors=log_scaling_factors,
        state_marginals=posterior_marginals(
            forward_probs,
            backward_probs,
        ),
        pair_marginals=posterior_pair_marginals(
            forward_probs,
            backward_probs,
            model,
            emission_log_likelihoods,
            log_scaling_factors,
        ),
    )
