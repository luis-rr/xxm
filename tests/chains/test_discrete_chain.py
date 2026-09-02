from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

import xxm.core.chains.discrete

jax.config.update('jax_enable_x64', True)

RTOL = 1e-7
ATOL = 1e-6


def _logsumexp(values: np.ndarray) -> float:
    maximum = np.max(values)
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def _enumerate_exact_posterior(
    initial_probs: np.ndarray,
    transition_probs: np.ndarray,
    state_log_potentials: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    t, k = state_log_potentials.shape
    state_sequences = list(itertools.product(range(k), repeat=t))

    log_weights = np.empty(len(state_sequences), dtype=float)

    for sequence_index, sequence in enumerate(state_sequences):
        log_weight = (
            np.log(initial_probs[sequence[0]]) + state_log_potentials[0, sequence[0]]
        )

        for time_index in range(1, t):
            prev_state = sequence[time_index - 1]
            state = sequence[time_index]

            log_weight += np.log(transition_probs[time_index - 1, prev_state, state])
            log_weight += state_log_potentials[
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


def _stack_static_transition(transition_probs: jax.Array, num_steps: int) -> jax.Array:
    return jnp.repeat(
        transition_probs[None, :, :], repeats=max(0, num_steps - 1), axis=0
    )


def test_forward_backward_matches_exact_enumeration() -> None:
    num_steps = 3

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=jnp.array([0.6, 0.4]),
        transition_probs=_stack_static_transition(
            jnp.array(
                [
                    [0.7, 0.3],
                    [0.2, 0.8],
                ]
            ),
            num_steps,
        ),
        state_log_potentials=jnp.log(
            jnp.array(
                [
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.6, 0.4],
                ]
            )
        ),
    )

    result, log_normalizer = chain.forward_backward()
    (
        exact_state_posterior_probs,
        exact_pair_posterior_probs,
        exact_log_marginal_likelihood,
    ) = _enumerate_exact_posterior(
        initial_probs=np.asarray(chain.initial_probs),
        transition_probs=np.asarray(chain.transition_probs),
        state_log_potentials=np.asarray(chain.state_log_potentials),
    )

    np.testing.assert_allclose(
        np.asarray(result.state_probs), exact_state_posterior_probs, atol=ATOL
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_probs), exact_pair_posterior_probs, atol=ATOL
    )
    np.testing.assert_allclose(
        float(log_normalizer),
        exact_log_marginal_likelihood,
        atol=ATOL,
    )


def test_normalization_and_finiteness_invariants() -> None:
    initial_probs = jnp.array([0.55, 0.45])
    num_steps = 4

    transition_probs = _stack_static_transition(
        jnp.array(
            [
                [0.8, 0.2],
                [0.15, 0.85],
            ]
        ),
        num_steps,
    )

    state_log_potentials = jnp.log(
        jnp.array(
            [
                [0.8, 0.2],
                [0.5, 0.5],
                [0.3, 0.7],
                [0.6, 0.4],
            ]
        )
    )

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        state_log_potentials=state_log_potentials,
    )

    messages = xxm.core.chains.discrete._forward_backward(chain)

    np.testing.assert_allclose(
        np.asarray(messages.forward_messages.sum(axis=1)),
        np.ones(4),
        atol=ATOL,
    )

    assert np.isfinite(np.asarray(messages.forward_messages)).all()
    assert np.isfinite(np.asarray(messages.backward_messages)).all()
    assert np.isfinite(np.asarray(messages.log_scaling_factors)).all()
    assert (np.asarray(messages.forward_messages) >= 0.0).all()

    marginals, log_normalizer = messages.compute_marginals(chain)

    np.testing.assert_allclose(
        np.asarray(marginals.state_probs.sum(axis=1)),
        np.ones(4),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(marginals.pair_probs.sum(axis=(1, 2))),
        np.ones(3),
        atol=ATOL,
    )
    assert np.isfinite(np.asarray(marginals.state_probs)).all()
    assert np.isfinite(np.asarray(marginals.pair_probs)).all()
    assert np.isfinite(float(log_normalizer))

    assert (np.asarray(marginals.state_probs) >= 0.0).all()
    assert (np.asarray(marginals.pair_probs) >= 0.0).all()


def test_marginal_consistency_between_state_and_pair_posteriors() -> None:
    initial_probs = jnp.array([0.4, 0.6])
    transition_probs = _stack_static_transition(
        jnp.array(
            [
                [0.5, 0.5],
                [0.2, 0.8],
            ]
        ),
        3,
    )
    state_log_potentials = jnp.log(
        jnp.array(
            [
                [0.7, 0.3],
                [0.6, 0.4],
                [0.35, 0.65],
            ]
        )
    )

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        state_log_potentials=state_log_potentials,
    )
    result, _log_normalizer = chain.forward_backward()

    for time_index in range(chain.state_log_potentials.shape[0] - 1):
        np.testing.assert_allclose(
            np.asarray(result.state_probs[time_index]),
            np.asarray(result.pair_probs[time_index].sum(axis=1)),
            atol=ATOL,
        )
        np.testing.assert_allclose(
            np.asarray(result.state_probs[time_index + 1]),
            np.asarray(result.pair_probs[time_index].sum(axis=0)),
            atol=ATOL,
        )


def test_log_normalizer_matches_scaling_and_exact_enumeration() -> None:
    initial_probs = jnp.array([0.5, 0.5])
    transition_probs = _stack_static_transition(
        jnp.array(
            [
                [0.85, 0.15],
                [0.3, 0.7],
            ]
        ),
        3,
    )
    state_log_potentials = jnp.array(
        [
            [1000.0, 998.0],
            [1001.5, 999.0],
            [1002.0, 1000.5],
        ]
    )

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        state_log_potentials=state_log_potentials,
    )

    messages = xxm.core.chains.discrete._forward_backward(chain)
    result, log_normalizer = chain.forward_backward()
    (
        exact_state_posterior_probs,
        exact_pair_posterior_probs,
        exact_log_marginal_likelihood,
    ) = _enumerate_exact_posterior(
        initial_probs=np.asarray(chain.initial_probs),
        transition_probs=np.asarray(chain.transition_probs),
        state_log_potentials=np.asarray(chain.state_log_potentials),
    )

    np.testing.assert_allclose(
        float(log_normalizer),
        float(np.sum(messages.log_scaling_factors)),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        float(log_normalizer),
        exact_log_marginal_likelihood,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.state_probs),
        exact_state_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_probs),
        exact_pair_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )


def test_time_varying_transitions_match_exact_enumeration() -> None:
    initial_probs = jnp.array([0.8, 0.2])
    transition_probs = jnp.array(
        [
            [[0.9, 0.1], [0.4, 0.6]],
            [[0.75, 0.25], [0.2, 0.8]],
            [[0.6, 0.4], [0.3, 0.7]],
        ]
    )
    state_log_potentials = jnp.log(
        jnp.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.5, 0.5],
                [0.3, 0.7],
            ]
        )
    )

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=initial_probs,
        transition_probs=transition_probs,
        state_log_potentials=state_log_potentials,
    )
    result, log_normalizer = chain.forward_backward()
    (
        exact_state_posterior_probs,
        exact_pair_posterior_probs,
        exact_log_marginal_likelihood,
    ) = _enumerate_exact_posterior(
        initial_probs=np.asarray(chain.initial_probs),
        transition_probs=np.asarray(chain.transition_probs),
        state_log_potentials=np.asarray(chain.state_log_potentials),
    )

    np.testing.assert_allclose(
        np.asarray(result.state_probs),
        exact_state_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_probs),
        exact_pair_posterior_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        float(log_normalizer),
        exact_log_marginal_likelihood,
        atol=ATOL,
        rtol=RTOL,
    )


def test_deterministic_transitions_concentrate_posterior_path() -> None:
    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=jnp.array([1.0, 0.0]),
        transition_probs=jnp.array(
            [
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ),
        state_log_potentials=jnp.zeros((4, 2)),
    )

    result, _log_normalizer = chain.forward_backward()

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
        np.asarray(result.state_probs),
        expected_state_posterior_probs,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_probs),
        expected_pair_posterior_probs,
        atol=ATOL,
    )


def test_forward_backward_is_jit_compatible() -> None:
    initial_probs = jnp.array([0.6, 0.4])
    transition_probs = _stack_static_transition(
        jnp.array(
            [
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        ),
        3,
    )
    state_log_potentials = jnp.log(
        jnp.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
            ]
        )
    )

    def run_forward_backward(
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        state_log_potentials: jax.Array,
    ) -> tuple[xxm.core.chains.discrete.DiscreteChainMarginals, jax.Array]:
        return xxm.core.chains.discrete.DiscreteChain(
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            state_log_potentials=state_log_potentials,
        ).forward_backward()

    eager_result, eager_log_normalizer = run_forward_backward(
        initial_probs, transition_probs, state_log_potentials
    )
    jitted_forward_backward = jax.jit(run_forward_backward)
    jitted_result, jitted_log_normalizer = jitted_forward_backward(
        initial_probs, transition_probs, state_log_potentials
    )

    np.testing.assert_allclose(
        np.asarray(jitted_result.state_probs),
        np.asarray(eager_result.state_probs),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(jitted_result.pair_probs),
        np.asarray(eager_result.pair_probs),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        float(jitted_log_normalizer),
        float(eager_log_normalizer),
        atol=ATOL,
    )


def test_forward_backward_supports_batching_with_vmap() -> None:
    """Independent chains can be batched externally with vmap."""
    initial_probs = jnp.array(
        [
            [0.6, 0.4],
            [0.25, 0.75],
        ]
    )

    transition_probs = jnp.array(
        [
            [
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.6, 0.4], [0.1, 0.9]],
            ],
            [
                [[0.5, 0.5], [0.2, 0.8]],
                [[0.9, 0.1], [0.4, 0.6]],
            ],
        ]
    )

    state_log_potentials = jnp.log(
        jnp.array(
            [
                [
                    [0.9, 0.1],
                    [0.4, 0.6],
                    [0.7, 0.3],
                ],
                [
                    [0.2, 0.8],
                    [0.6, 0.4],
                    [0.3, 0.7],
                ],
            ]
        )
    )

    def run(
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        state_log_potentials: jax.Array,
    ) -> tuple[xxm.core.chains.discrete.DiscreteChainMarginals, jax.Array]:
        return xxm.core.chains.discrete.DiscreteChain(
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            state_log_potentials=state_log_potentials,
        ).forward_backward()

    batched_result, batched_log_normalizer = jax.vmap(run)(
        initial_probs,
        transition_probs,
        state_log_potentials,
    )

    for batch_index in range(2):
        expected, log_normalizer = run(
            initial_probs[batch_index],
            transition_probs[batch_index],
            state_log_potentials[batch_index],
        )

        np.testing.assert_allclose(
            batched_result.state_probs[batch_index],
            expected.state_probs,
            atol=ATOL,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            batched_result.pair_probs[batch_index],
            expected.pair_probs,
            atol=ATOL,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            batched_log_normalizer[batch_index],
            log_normalizer,
            atol=ATOL,
            rtol=RTOL,
        )

    assert batched_result.state_probs.shape == (2, 3, 2)
    assert batched_result.pair_probs.shape == (2, 2, 2, 2)
    assert batched_log_normalizer.shape == (2,)


def test_vmapped_forward_backward_is_jit_compatible() -> None:
    initial_probs = jnp.array(
        [
            [0.6, 0.4],
            [0.3, 0.7],
        ]
    )

    transition_probs = jnp.array(
        [
            [
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.6, 0.4], [0.1, 0.9]],
            ],
            [
                [[0.5, 0.5], [0.2, 0.8]],
                [[0.9, 0.1], [0.4, 0.6]],
            ],
        ]
    )

    state_log_potentials = jnp.log(
        jnp.array(
            [
                [
                    [0.9, 0.1],
                    [0.4, 0.6],
                    [0.7, 0.3],
                ],
                [
                    [0.2, 0.8],
                    [0.6, 0.4],
                    [0.3, 0.7],
                ],
            ]
        )
    )

    def run(
        initial_probs: jax.Array,
        transition_probs: jax.Array,
        state_log_potentials: jax.Array,
    ) -> tuple[xxm.core.chains.discrete.DiscreteChainMarginals, jax.Array]:
        return xxm.core.chains.discrete.DiscreteChain(
            initial_probs=initial_probs,
            transition_probs=transition_probs,
            state_log_potentials=state_log_potentials,
        ).forward_backward()

    batched_run = jax.vmap(run)

    eager, eager_log_normalizer = batched_run(
        initial_probs,
        transition_probs,
        state_log_potentials,
    )
    jitted, jitted_log_normalizer = jax.jit(batched_run)(
        initial_probs,
        transition_probs,
        state_log_potentials,
    )

    np.testing.assert_allclose(
        jitted.state_probs,
        eager.state_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        jitted.pair_probs,
        eager.pair_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        jitted_log_normalizer,
        eager_log_normalizer,
        atol=ATOL,
        rtol=RTOL,
    )


def test_single_time_step_matches_direct_normalization() -> None:
    initial_probs = jnp.array([0.6, 0.4])
    state_log_potentials = jnp.log(
        jnp.array(
            [
                [0.2, 0.8],
            ]
        )
    )

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=initial_probs,
        transition_probs=jnp.zeros((0, 2, 2)),
        state_log_potentials=state_log_potentials,
    )

    result, log_normalizer = chain.forward_backward()

    unnormalized = np.array([0.6 * 0.2, 0.4 * 0.8])
    normalizer = unnormalized.sum()
    expected = unnormalized / normalizer

    np.testing.assert_allclose(
        result.state_probs[0],
        expected,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        log_normalizer,
        np.log(normalizer),
        atol=ATOL,
        rtol=RTOL,
    )

    assert result.pair_probs.shape == (0, 2, 2)


def test_single_state_chain() -> None:
    state_log_potentials = jnp.array(
        [
            [0.2],
            [-0.4],
            [0.7],
        ]
    )

    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=jnp.ones(1),
        transition_probs=jnp.ones((2, 1, 1)),
        state_log_potentials=state_log_potentials,
    )

    result, log_normalizer = chain.forward_backward()

    np.testing.assert_allclose(
        result.state_probs,
        np.ones((3, 1)),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        result.pair_probs,
        np.ones((2, 1, 1)),
        atol=ATOL,
    )
    np.testing.assert_allclose(
        log_normalizer,
        np.sum(state_log_potentials),
        atol=ATOL,
    )


def test_state_log_potential_offsets_only_shift_log_normalizer() -> None:
    chain = xxm.core.chains.discrete.DiscreteChain(
        initial_probs=jnp.array([0.6, 0.4]),
        transition_probs=jnp.array(
            [
                [[0.8, 0.2], [0.3, 0.7]],
                [[0.6, 0.4], [0.1, 0.9]],
            ]
        ),
        state_log_potentials=jnp.log(
            jnp.array(
                [
                    [0.9, 0.1],
                    [0.4, 0.6],
                    [0.7, 0.3],
                ]
            )
        ),
    )

    offsets = jnp.array([100.0, -20.0, 7.5])

    shifted_chain = chain._replace(
        state_log_potentials=(chain.state_log_potentials + offsets[:, None])
    )

    original, original_log_normalizer = chain.forward_backward()
    shifted, shifted_log_normalizer = shifted_chain.forward_backward()

    np.testing.assert_allclose(
        shifted.state_probs,
        original.state_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        shifted.pair_probs,
        original.pair_probs,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        shifted_log_normalizer,
        original_log_normalizer + offsets.sum(),
        atol=ATOL,
        rtol=RTOL,
    )
