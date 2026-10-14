import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core import batch


@pytest.mark.parametrize(
    'index',
    [(0, 2), (-2, -3), (Ellipsis, -1), ([1, 0], slice(None)), slice(0, 10)],
)
def test_validate_selection_accepts_valid_indices_and_clipped_slices(index):
    batch.validate_selection(index, (2, 3))


@pytest.mark.parametrize('shape', [(), (2, 3), (2, 0)])
def test_flatten_round_trip_preserves_nested_leaves(shape):
    tree = {'matrix': jnp.zeros(shape + (2, 2)), 'scalar': jnp.ones(shape)}
    restored = batch.unflatten_batch(batch.flatten_batch(tree, shape), shape)
    for actual, expected in zip(
        jax.tree.leaves(restored), jax.tree.leaves(tree), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_shape_helpers_preserve_prefix_and_expand_trailing_axes():
    assert batch.split_prefix((2, 3, 4), (2, 3)) == (4,)
    assert batch.split_prefix((2,), ()) == (2,)
    assert batch.expand_trailing(jnp.array(True), 3).shape == (1, 1, 1)
    values = jnp.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(
        batch.expand_trailing(values, 4), values[..., None, None]
    )


def test_pool_samples_preserves_model_batch():
    values = jnp.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

    pooled = batch.pool_samples(
        values,
        batch_shape=(2,),
        sample_shape=(3, 4),
    )

    assert pooled.shape == (2, 12, 5)

    np.testing.assert_array_equal(pooled, values.reshape(pooled.shape))


def test_pool_samples_empty_sample_shape():
    values = jnp.arange(10).reshape(2, 5)

    pooled = batch.pool_samples(
        values,
        batch_shape=(2,),
        sample_shape=(),
    )

    assert pooled.shape == (2, 1, 5)

    np.testing.assert_array_equal(pooled, values.reshape(pooled.shape))
