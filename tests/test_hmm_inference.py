from __future__ import annotations

import itertools
import typing

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xxm.hmm.inference


ATOL = 1e-5
RTOL = 1e-4


class EmissionsDummy(typing.NamedTuple):
    def log_likelihoods(
        self,
        observations: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.log(observations)

    def m_step(
        self,
        observations: jnp.ndarray,
        posterior: xxm.hmm.inference.Posterior,
    ) -> typing.Self:
        return self

    def sample(
        self,
        key: jax.Array,
        state: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.zeros(state.shape[0], dtype=jnp.float32)


def _logsumexp(values: np.ndarray) -> float:
    maximum = np.max(values)
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def _enumerate_exact_posterior(
    initial_probs: np.ndarray,
    transition_probs: np.ndarray,
    emission_log_likelihoods: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    t, k = emission_log_likelihoods.shape
    state_sequences = list(itertools.product(range(k), repeat=t))

    log_weights = np.empty(len(state_sequences), dtype=float)

    for sequence_index, sequence in enumerate(state_sequences):
        log_weight = np.log(initial_probs[sequence[0]]) + emission_log_likelihoods[0, sequence[0]]

        for time_index in range(1, t):
            prev_state = sequence[time_index - 1]
            state = sequence[time_index]

            log_weight += np.log(transition_probs[prev_state, state])
            log_weight += emission_log_likelihoods[
                time_index,
                state,
            ]

        log_weights[sequence_index] = log_weight

    log_z = _logsumexp(log_weights)
    posterior_weights = np.exp(log_weights - log_z)

    state_posterior_probs = np.zeros((t, k), dtype=float)
    pair_posterior_probs = np.zeros((t - 1, k, k), dtype=float)

    for sequence, sequence_weight in zip(
        state_sequences,
        posterior_weights,
        strict=True,
    ):
        for time_index, state in enumerate(sequence):
            state_posterior_probs[time_index, state] += sequence_weight

        for time_index in range(t - 1):
            pair_posterior_probs[
                time_index,
                sequence[time_index],
                sequence[time_index + 1],
            ] += sequence_weight

    return state_posterior_probs, pair_posterior_probs, log_z


def test_forward_backward_matches_exact_enumeration() -> None:
    model = xxm.hmm.inference.Model(
        initial_probs=jnp.array([0.6, 0.4]),
        transition_probs=jnp.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ]
        ),
        emissions=EmissionsDummy(),
    )

    emission_log_likelihoods = jnp.log(
        jnp.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )
    )

    result = xxm.hmm.inference.forward_backward(
        model,
        emission_log_likelihoods,
    )
    exact_state_posterior_probs, exact_pair_posterior_probs, exact_log_marginal_likelihood = (
        _enumerate_exact_posterior(
            initial_probs=np.asarray(model.initial_probs),
            transition_probs=np.asarray(model.transition_probs),
            emission_log_likelihoods=np.asarray(emission_log_likelihoods),
        )
    )

    np.testing.assert_allclose(
        np.asarray(result.state_marginals), exact_state_posterior_probs, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_marginals), exact_pair_posterior_probs, atol=ATOL
    )
    np.testing.assert_allclose(
        float(result.log_likelihood()),
        exact_log_marginal_likelihood,
        atol=ATOL,
    )


def test_normalization_and_finiteness_invariants() -> None:
    initial_probs = jnp.array([0.55, 0.45])

    transition_probs = jnp.array(
        [
            [0.8, 0.2],
            [0.15, 0.85],
        ]
    )

    emission_log_likelihoods = jnp.log(
        jnp.array(
            [
                [0.8, 0.2],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        )
    )

    model = xxm.hmm.inference.Model(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emissions=EmissionsDummy(),
    )

    result = xxm.hmm.inference.forward_backward(
        model,
        emission_log_likelihoods,
    )

    np.testing.assert_allclose(
        np.asarray(result.forward_probs.sum(axis=1)),
        np.ones(4),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.state_marginals.sum(axis=1)),
        np.ones(4),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_marginals.sum(axis=(1, 2))),
        np.ones(3),
        atol=ATOL,
    )

    assert np.isfinite(np.asarray(result.forward_probs)).all()
    assert np.isfinite(np.asarray(result.backward_probs)).all()
    assert np.isfinite(np.asarray(result.state_marginals)).all()
    assert np.isfinite(np.asarray(result.pair_marginals)).all()
    assert np.isfinite(np.asarray(result.log_scaling_factors)).all()
    assert np.isfinite(float(result.log_likelihood()))

    assert (np.asarray(result.forward_probs) >= 0.0).all()
    assert (np.asarray(result.state_marginals) >= 0.0).all()
    assert (np.asarray(result.pair_marginals) >= 0.0).all()


def test_marginal_consistency_between_state_and_pair_posteriors() -> None:
    initial_probs = jnp.array([0.4, 0.6])
    transition_probs = jnp.array(
        [
            [0.5, 0.5],
            [0.2, 0.8],
        ]
    )
    emission_log_likelihoods = jnp.log(
        jnp.array(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.35, 0.65],
            ]
        )
    )

    model = xxm.hmm.inference.Model(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emissions=EmissionsDummy(),
    )
    result = xxm.hmm.inference.forward_backward(model, emission_log_likelihoods)

    for time_index in range(emission_log_likelihoods.shape[0] - 1):
        np.testing.assert_allclose(
            np.asarray(result.state_marginals[time_index]),
            np.asarray(result.pair_marginals[time_index].sum(axis=1)),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.state_marginals[time_index + 1]),
            np.asarray(result.pair_marginals[time_index].sum(axis=0)),
            atol=ATOL,
        )


def test_log_likelihood_matches_normalizers_and_exact_enumeration() -> None:
    initial_probs = jnp.array([0.5, 0.5])
    transition_probs = jnp.array(
        [
            [0.85, 0.15],
            [0.3, 0.7],
        ]
    )
    emission_log_likelihoods = jnp.array(
        [
            [1000.0, 998.0],
            [1001.5, 999.0],
            [1002.0, 1000.5],
        ]
    )

    model = xxm.hmm.inference.Model(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emissions=EmissionsDummy(),
    )
    result = xxm.hmm.inference.forward_backward(model, emission_log_likelihoods)
    exact_state_posterior_probs, exact_pair_posterior_probs, exact_log_marginal_likelihood = (
        _enumerate_exact_posterior(
            initial_probs=np.asarray(model.initial_probs),
            transition_probs=np.asarray(model.transition_probs),
            emission_log_likelihoods=np.asarray(emission_log_likelihoods),
        )
    )

    np.testing.assert_allclose(
        float(result.log_likelihood()),
        float(np.sum(result.log_scaling_factors)),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        float(result.log_likelihood()),
        exact_log_marginal_likelihood,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.state_marginals),
        exact_state_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_marginals),
        exact_pair_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )


def test_time_varying_transitions_match_exact_enumeration() -> None:
    initial_probs = jnp.array([0.8, 0.2])
    transition_probs = jnp.array(
        [
            [0.9, 0.1],
            [0.4, 0.6],
        ]
    )
    emission_log_likelihoods = jnp.log(
        jnp.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.5, 0.5],
                [0.3, 0.7],
            ]
        )
    )

    model = xxm.hmm.inference.Model(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        emissions=EmissionsDummy(),
    )
    result = xxm.hmm.inference.forward_backward(model, emission_log_likelihoods)
    exact_state_posterior_probs, exact_pair_posterior_probs, exact_log_marginal_likelihood = (
        _enumerate_exact_posterior(
            initial_probs=np.asarray(model.initial_probs),
            transition_probs=np.asarray(model.transition_probs),
            emission_log_likelihoods=np.asarray(emission_log_likelihoods),
        )
    )

    np.testing.assert_allclose(
        np.asarray(result.state_marginals),
        exact_state_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_marginals),
        exact_pair_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        float(result.log_likelihood()),
        exact_log_marginal_likelihood,
        atol=ATOL,
        rtol=RTOL,
    )


def test_shape_validation_raises_value_error() -> None:
    model = xxm.hmm.inference.Model(
        initial_probs=jnp.array([0.6, 0.4]),
        transition_probs=jnp.array(
            [
                [0.7, 0.3],
                [0.2, 0.8],
            ],
        ),
        emissions=EmissionsDummy(),
    )
    emission_log_likelihoods = jnp.log(
        jnp.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )
    )

    with pytest.raises(ValueError):
        xxm.hmm.inference.forward_pass(
            model=xxm.hmm.inference.Model(
                initial_probs=jnp.array([[0.6, 0.4]]),
                transition_probs=model.transition_probs,
                emissions=EmissionsDummy(),
            ),
            emission_log_likelihoods=emission_log_likelihoods,
        )

    with pytest.raises(ValueError):
        xxm.hmm.inference.forward_pass(
            model=xxm.hmm.inference.Model(
                initial_probs=model.initial_probs,
                transition_probs=model.transition_probs[:, 0],
                emissions=EmissionsDummy(),
            ),
            emission_log_likelihoods=emission_log_likelihoods,
        )

    with pytest.raises(ValueError):
        xxm.hmm.inference.backward_pass(
            model=xxm.hmm.inference.Model(
                initial_probs=model.initial_probs,
                transition_probs=model.transition_probs,
                emissions=EmissionsDummy(),
            ),
            emission_log_likelihoods=emission_log_likelihoods,
            log_scaling_factors=jnp.ones((2,)),
        )

    forward_probs, log_scaling_factors = xxm.hmm.inference.forward_pass(
        model,
        emission_log_likelihoods,
    )
    backward_probs = xxm.hmm.inference.backward_pass(
        model=model,
        emission_log_likelihoods=emission_log_likelihoods,
        log_scaling_factors=log_scaling_factors,
    )

    with pytest.raises(ValueError):
        xxm.hmm.inference.posterior_marginals(
            forward_probs=forward_probs[:-1],
            backward_probs=backward_probs,
        )

    with pytest.raises(ValueError):
        xxm.hmm.inference.posterior_pair_marginals(
            forward_probs=forward_probs,
            backward_probs=backward_probs,
            model=model,
            log_scaling_factors=log_scaling_factors[:-1],
            emission_log_likelihoods=emission_log_likelihoods,
        )


def test_deterministic_transitions_concentrate_posterior_path() -> None:

    model = xxm.hmm.inference.Model(
        initial_probs=jnp.array([1.0, 0.0]),
        transition_probs=jnp.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        emissions=EmissionsDummy(),
    )
    emission_log_likelihoods = jnp.zeros((4, 2))

    result = xxm.hmm.inference.forward_backward(model, emission_log_likelihoods)

    expected_state_posterior_probs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    expected_pair_posterior_probs = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(result.state_marginals),
        expected_state_posterior_probs,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_marginals),
        expected_pair_posterior_probs,
        atol=ATOL,
    )


def test_forward_backward_is_jit_compatible() -> None:
    model = xxm.hmm.inference.Model(
        initial_probs=jnp.array([0.6, 0.4]),
        transition_probs=jnp.array(
            [
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        ),
        emissions=EmissionsDummy(),
    )
    emission_log_likelihoods = jnp.log(
        jnp.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )
    )

    eager_result = xxm.hmm.inference.forward_backward(model, emission_log_likelihoods)
    jitted_forward_backward = jax.jit(xxm.hmm.inference.forward_backward)
    jitted_result = jitted_forward_backward(model, emission_log_likelihoods)

    np.testing.assert_allclose(
        np.asarray(jitted_result.forward_probs),
        np.asarray(eager_result.forward_probs),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(jitted_result.backward_probs),
        np.asarray(eager_result.backward_probs),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(jitted_result.state_marginals),
        np.asarray(eager_result.state_marginals),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(jitted_result.pair_marginals),
        np.asarray(eager_result.pair_marginals),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        float(jitted_result.log_likelihood()),
        float(eager_result.log_likelihood()),
        atol=ATOL,
    )
