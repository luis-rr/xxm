"""Native batch prefixes preserve independent chain inference and reductions."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.affine import Affine
from xxm.core.chains.discrete import (
    DiscreteChain,
    DiscreteChainMarginals,
    DiscretePotential,
)
from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.core.dists.gaussian import Gaussian, LinearGaussian


def gaussian_chain(batch, steps):
    d = 2
    rng = np.random.default_rng(32)
    initial = Gaussian(
        jnp.asarray(rng.normal(size=batch + (d,))),
        jnp.broadcast_to(jnp.eye(d), batch + (d, d)),
    )
    pairs = LinearGaussian(
        Affine(
            jnp.asarray(rng.normal(scale=0.1, size=batch + (steps - 1, d, d))),
            jnp.asarray(rng.normal(size=batch + (steps - 1, d))),
        ),
        jnp.broadcast_to(jnp.eye(d), batch + (steps - 1, d, d)),
    )
    chain = GaussianChain.from_pair_potentials(
        GaussianPotential.from_moments(initial),
        GaussianPairPotential.from_linear_conditional(pairs),
    )
    local = GaussianPotential.from_moments(
        Gaussian(
            jnp.asarray(rng.normal(size=batch + (steps, d))),
            jnp.broadcast_to(jnp.eye(d), batch + (steps, d, d)),
        )
    )
    return chain.add_local_potential(local)


def discrete_chain(batch, steps):
    rng = np.random.default_rng(13)
    k = 3
    initial = jax.nn.softmax(jnp.asarray(rng.normal(size=batch + (k,))), axis=-1)
    transitions = jax.nn.softmax(
        jnp.asarray(rng.normal(size=batch + (steps - 1, k, k))), axis=-1
    )
    local = DiscretePotential(jnp.asarray(rng.normal(size=batch + (steps, k))))
    return DiscreteChain.from_markov_prior(initial, transitions).add_local_potential(
        local
    )


def gaussian_chain_structure(batch, steps):
    d = 2
    return GaussianChain(
        diagonal_precision_blocks=jnp.broadcast_to(jnp.eye(d), batch + (steps, d, d)),
        lower_precision_blocks=jnp.zeros(batch + (steps - 1, d, d)),
        information_vectors=jnp.zeros(batch + (steps, d)),
        log_constant=jnp.zeros(batch),
    )


def discrete_chain_structure(batch, steps):
    k = 3
    return DiscreteChain(
        initial_probs=jnp.full(batch + (k,), 1.0 / k),
        transition_probs=jnp.full(batch + (steps - 1, k, k), 1.0 / k),
        state_log_potentials=jnp.zeros(batch + (steps, k)),
    )


def gaussian_marginals(batch, steps):
    d = 2
    return GaussianChainMarginals(
        means=jnp.zeros(batch + (steps, d)),
        covariances=jnp.broadcast_to(jnp.eye(d), batch + (steps, d, d)),
        cross_covariances=jnp.zeros(batch + (steps - 1, d, d)),
    )


def discrete_marginals(batch, steps):
    k = 3
    state_probs = jnp.full(batch + (steps, k), 1.0 / k)
    pair_probs = state_probs[..., :-1, :, None] * state_probs[..., 1:, None, :]
    return DiscreteChainMarginals(state_probs, pair_probs)


# Pair the meaningful edge cases rather than compiling the Cartesian product of
# every batch shape with every sequence length. The scalar one-step path, a
# multi-axis batch, and an empty batch all remain covered.
@pytest.mark.parametrize(
    ('batch', 'steps'),
    [
        ((), 1),
        ((2, 3), 3),
        ((0, 2), 3),
    ],
)
def test_gaussian_batch_matches_dense_precision(batch, steps):
    chain = gaussian_chain(batch, steps)
    posterior, logz = jax.jit(GaussianChain.forward_backward)(chain)
    precision = chain.dense_precision()
    covariance = jnp.linalg.inv(precision)
    h = chain.information_vectors.reshape(batch + (steps * 2,))
    mean = jnp.linalg.solve(precision, h[..., None])[..., 0]
    expected_logz = chain.log_constant + 0.5 * (
        jnp.sum(h * mean, axis=-1)
        - jnp.linalg.slogdet(precision)[1]
        + steps * 2 * jnp.log(2 * jnp.pi)
    )
    assert chain.batch_shape == posterior.batch_shape == batch
    assert logz.shape == posterior.entropy().shape == batch
    assert posterior.paired_marginals().batch_shape == batch + (steps - 1,)
    np.testing.assert_allclose(posterior.means.reshape(mean.shape), mean, atol=2e-6)
    np.testing.assert_allclose(logz, expected_logz, atol=3e-6)
    np.testing.assert_allclose(
        posterior.entropy() + posterior.expected_log_potential(chain), logz, atol=3e-6
    )
    for t in range(steps):
        np.testing.assert_allclose(
            posterior.covariances[..., t, :, :],
            covariance[..., 2 * t : 2 * t + 2, 2 * t : 2 * t + 2],
            atol=2e-6,
        )
        if t < steps - 1:
            np.testing.assert_allclose(
                posterior.cross_covariances[..., t, :, :],
                covariance[..., 2 * t : 2 * t + 2, 2 * t + 2 : 2 * t + 4],
                atol=2e-6,
            )
    values = jnp.ones(batch + (steps, 2))
    flat = values.reshape(batch + (steps * 2,))
    expected = (
        -0.5 * jnp.einsum('...i,...ij,...j->...', flat, precision, flat)
        + jnp.sum(h * flat, axis=-1)
        + chain.log_constant
    )
    np.testing.assert_allclose(chain.log_potential(values), expected, atol=2e-6)


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
        posterior.expected_log_potential(chain) + posterior.entropy(), logz, atol=2e-6
    )
    for index in np.ndindex(batch):
        initial, transition, local = (np.asarray(x[index]) for x in chain)
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
        for actual, wanted in zip(transformed, expected):
            np.testing.assert_array_equal(actual, wanted)
        with pytest.raises(IndexError):
            obj.select((0, 0, 0))
        with pytest.raises(IndexError):
            obj.select(None)
        with pytest.raises(ValueError):
            obj.move_axis(0, 2)
    with pytest.raises(ValueError, match='shapes must match'):
        marginals.expected_log_potential(chain.select(0))


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
    ).forward_backward()
    np.testing.assert_allclose(relabeled.state_probs, expected.state_probs, atol=2e-6)
    np.testing.assert_allclose(relabeled.pair_probs, expected.pair_probs, atol=2e-6)
    assert relabeled.batch_shape == (2,)
