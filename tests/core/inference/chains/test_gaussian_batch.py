"""Native batch prefixes preserve independent chain inference and reductions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.core.inference.chains.batch_helpers import (
    expected_log_potential,
    gaussian_chain,
)
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import GaussianChainMarginals
from xxm.core.inference.chains.gaussian import GaussianChain
from xxm.core.lengths import Lengths


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
        posterior.entropy() + expected_log_potential(chain, posterior), logz, atol=3e-6
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


def test_gaussian_owned_lengths_reduce_only_genuine_unary_and_pair_terms():
    lengths = Lengths(jnp.array([0, 1, 3]))
    chain = GaussianChain(
        diagonal_precision_blocks=jnp.ones((3, 4, 1, 1)),
        lower_precision_blocks=jnp.zeros((3, 3, 1, 1)),
        information_vectors=jnp.zeros((3, 4, 1)),
        log_constant=-0.5 * lengths.values * jnp.log(2 * jnp.pi),
        lengths=lengths,
    )
    chain.validate()
    posterior, logz = jax.jit(GaussianChain.forward_backward)(chain)
    np.testing.assert_allclose(logz, 0, atol=1e-6)
    expected_entropy = 0.5 * lengths.values * (1 + np.log(2 * np.pi))
    np.testing.assert_allclose(posterior.entropy(), expected_entropy, atol=1e-6)
    np.testing.assert_allclose(
        expected_log_potential(chain, posterior), -expected_entropy, atol=1e-6
    )
    dirty = chain._replace(
        diagonal_precision_blocks=lengths.clear(
            chain.diagonal_precision_blocks, fill=jnp.nan
        ),
        lower_precision_blocks=lengths.transitions().clear(
            chain.lower_precision_blocks, fill=jnp.nan
        ),
    )
    np.testing.assert_allclose(
        expected_log_potential(dirty, posterior), -expected_entropy, atol=1e-6
    )
    shifted = posterior.affine(Affine(jnp.ones((3, 1, 1)), jnp.ones((3, 1))))
    assert shifted.lengths is posterior.lengths


def test_empty_gaussian_marginals_keep_zero_lengths_and_entropy():
    posterior = GaussianChainMarginals(
        means=jnp.zeros((2, 0, 1)),
        covariances=jnp.zeros((2, 0, 1, 1)),
        cross_covariances=jnp.zeros((2, 0, 1, 1)),
        lengths=Lengths(jnp.zeros(2, dtype=int)),
    )
    assert posterior.max_steps == 0
    np.testing.assert_array_equal(jax.jit(lambda p: p.entropy())(posterior), [0, 0])
