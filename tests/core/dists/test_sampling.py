"""Batch-major sampling and temporal operations, with unequal axis lengths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.dists.discrete import Categorical
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson


def gaussian(batch=(2, 3), dim=2):
    return Gaussian(
        jnp.arange(np.prod(batch) * dim, dtype=float).reshape(batch + (dim,)),
        jnp.broadcast_to(jnp.eye(dim), batch + (dim, dim)),
    )


@pytest.mark.parametrize('sample_shape', [(), (2, 2), (0,)])
def test_distribution_samples_follow_batch(sample_shape):
    key = jax.random.key(7)
    g = gaussian()
    shape = g.batch_shape + sample_shape
    expanded = g.broadcast(sample_shape, axis=2)
    np.testing.assert_array_equal(g.sample(key, sample_shape), expanded.sample(key))

    p = Poisson(jnp.log(jnp.arange(1.0, 13.0).reshape(2, 3, 2)))
    np.testing.assert_array_equal(
        p.sample(key, sample_shape), p.broadcast(sample_shape, axis=2).sample(key)
    )

    c = Categorical(jax.nn.one_hot(jnp.arange(6).reshape(2, 3) % 4, 4))
    result = c.sample(key, sample_shape)
    assert result.shape == shape
    np.testing.assert_array_equal(
        result,
        jnp.broadcast_to(
            jnp.argmax(c.probs, axis=-1).reshape((2, 3) + (1,) * len(sample_shape)),
            shape,
        ),
    )


def test_categorical_sampling_is_jittable():
    categorical = Categorical(jax.nn.one_hot(jnp.arange(6).reshape(2, 3) % 4, 4))
    result = jax.jit(lambda key: categorical.sample(key, (2,)))(jax.random.key(7))
    assert result.shape == (2, 3, 2)
