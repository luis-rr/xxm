"""Native batch prefixes preserve independent chain inference and reductions."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.core.inference.chains.batch_helpers import (
    discrete_chain,
    expected_log_potential,
)
from xxm.core.inference.chains.discrete import DiscreteChain
from xxm.core.lengths import Lengths


@pytest.mark.parametrize(
    ('batch', 'steps'),
    [
        ((), 1),
        ((2, 3), 3),
        ((0, 2), 3),
    ],
)
def test_discrete_batch_matches_enumerated_paths(batch, steps):
    chain = discrete_chain(batch, steps)
    posterior, logz = jax.jit(DiscreteChain.forward_backward)(chain)
    assert logz.shape == posterior.entropy().shape == batch
    assert posterior.incoming_state_probs().shape == batch + (steps - 1, 3)
    np.testing.assert_allclose(
        expected_log_potential(chain, posterior) + posterior.entropy(), logz, atol=2e-6
    )
    for index in np.ndindex(batch):
        initial, transition, local = (np.asarray(x[index]) for x in chain[:3])
        paths = np.array(list(itertools.product(range(3), repeat=steps)))
        scores = np.log(initial[paths[:, 0]]) + local[np.arange(steps), paths].sum(
            axis=-1
        )
        if steps > 1:
            scores += np.log(
                transition[np.arange(steps - 1), paths[:, :-1], paths[:, 1:]]
            ).sum(axis=-1)
        weights = np.exp(scores - scores.max())
        z = weights.sum()
        weights /= z
        np.testing.assert_allclose(logz[index], scores.max() + np.log(z), atol=2e-6)
        for t in range(steps):
            for k in range(3):
                np.testing.assert_allclose(
                    posterior.state_probs[index + (t, k)],
                    weights[paths[:, t] == k].sum(),
                    atol=2e-6,
                )
                if t < steps - 1:
                    for j in range(3):
                        expected = weights[
                            (paths[:, t] == k) & (paths[:, t + 1] == j)
                        ].sum()
                        np.testing.assert_allclose(
                            posterior.pair_probs[index + (t, k, j)], expected, atol=2e-6
                        )


def test_discrete_owned_lengths_exclude_empty_and_padded_factors_under_jit():
    lengths = Lengths(jnp.array([0, 1, 3]))
    initial = jnp.broadcast_to(jnp.array([0.25, 0.75]), (3, 2))
    initial = initial.at[0].set(jnp.nan)
    chain = DiscreteChain(
        initial_probs=initial,
        transition_probs=lengths.transitions().clear(
            jnp.full((3, 3, 2, 2), 0.5), fill=jnp.nan
        ),
        state_log_potentials=lengths.clear(jnp.zeros((3, 4, 2)), fill=jnp.nan),
        lengths=lengths,
    )
    chain.validate()
    posterior, logz = jax.jit(DiscreteChain.forward_backward)(chain)
    np.testing.assert_allclose(logz, 0, atol=1e-6)
    initial_entropy = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
    np.testing.assert_allclose(
        posterior.entropy(),
        [0, initial_entropy, initial_entropy + 2 * np.log(2)],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        expected_log_potential(chain, posterior) + posterior.entropy(),
        logz,
        atol=1e-6,
    )
    np.testing.assert_array_equal(posterior.lengths.values, lengths.values)
    np.testing.assert_allclose(posterior.state_probs[2, 0], [0.25, 0.75], atol=1e-6)
    np.testing.assert_allclose(posterior.state_probs[2, 1:3], 0.5, atol=1e-6)
