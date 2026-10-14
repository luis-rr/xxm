"""Native batch prefixes preserve independent chain inference and reductions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.core.inference.chains.batch_helpers import (
    discrete_chain,
    discrete_chain_structure,
    discrete_marginals,
    expected_log_potential,
    gaussian_chain,
    gaussian_chain_structure,
    gaussian_marginals,
)
from xxm.core.inference.chains.discrete import DiscreteChain


@pytest.mark.parametrize(
    ('make_chain', 'make_marginals'),
    [
        (gaussian_chain_structure, gaussian_marginals),
        (discrete_chain_structure, discrete_marginals),
    ],
)
def test_batch_transformations_and_scalar_selection(make_chain, make_marginals):
    chain = make_chain((2, 3), 2)
    # Batch transformations are structural. Construct marginals directly rather
    # than paying for another forward-backward pass; inference/JIT are covered
    # by the dense/enumerated tests above and dedicated chain tests.
    marginals = make_marginals((2, 3), 2)
    for obj in (chain, marginals):
        transformed = (
            obj.broadcast(1, axis=1)
            .squeeze(1)
            .move_axis(0, 1)
            .permute(jnp.array([2, 0, 1]), axis=0)
            .select((1, 1))
        )
        expected = obj.select((1, 0))
        assert type(transformed) is type(obj)
        assert transformed.batch_shape == ()
        for actual, wanted in zip(
            jax.tree.leaves(transformed), jax.tree.leaves(expected)
        ):
            np.testing.assert_array_equal(actual, wanted)
        with pytest.raises(IndexError):
            obj.select((0, 0, 0))
        with pytest.raises(IndexError):
            obj.select(None)
        with pytest.raises(ValueError):
            obj.move_axis(0, 2)
    with pytest.raises(ValueError, match='shape'):
        expected_log_potential(chain.select(0), marginals)


def test_batch_gradients_are_chain_local():
    gaussian = gaussian_chain((2, 3), 4)
    posterior, _ = gaussian.forward_backward()
    gradient = jax.jit(
        jax.grad(
            lambda h: (
                gaussian._replace(information_vectors=h).forward_backward()[1].sum()
            )
        )
    )(gaussian.information_vectors)
    np.testing.assert_allclose(gradient, posterior.means, atol=2e-6)
    discrete = discrete_chain((2, 3), 4)
    posterior, _ = discrete.forward_backward()
    gradient = jax.jit(
        jax.grad(
            lambda local: (
                discrete._replace(state_log_potentials=local)
                .forward_backward()[1]
                .sum()
            )
        )
    )(discrete.state_log_potentials)
    np.testing.assert_allclose(gradient, posterior.state_probs, atol=2e-6)


def test_state_permutation_is_distinct_from_batch_permutation():
    chain = discrete_chain((2,), 2)
    posterior, _ = chain.forward_backward()
    p = jnp.array([2, 0, 1])
    relabeled = posterior.permute_states(p)
    expected, _ = DiscreteChain(
        chain.initial_probs[..., p],
        chain.transition_probs[..., p, :][..., p],
        chain.state_log_potentials[..., p],
        chain.lengths,
    ).forward_backward()
    np.testing.assert_allclose(relabeled.state_probs, expected.state_probs, atol=2e-6)
    np.testing.assert_allclose(relabeled.pair_probs, expected.pair_probs, atol=2e-6)
    assert relabeled.batch_shape == (2,)
