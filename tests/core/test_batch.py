import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core import batch


@pytest.mark.parametrize(
    'index',
    [(2, 0), (-3, 0), (0, 3), (0, -4), ([0, 2], 0), (0, jnp.array([0, -4]))],
)
def test_validate_selection_rejects_out_of_bounds_indices(index):
    with pytest.raises(IndexError, match='out of bounds'):
        batch.validate_selection(index, (2, 3))


@pytest.mark.parametrize(
    'index',
    [(0, 2), (-2, -3), (Ellipsis, -1), ([1, 0], slice(None)), slice(0, 10)],
)
def test_validate_selection_accepts_valid_indices_and_clipped_slices(index):
    batch.validate_selection(index, (2, 3))


@pytest.mark.parametrize('index', [0, -1])
def test_validate_selection_rejects_integer_indices_on_empty_axes(index):
    with pytest.raises(IndexError, match='out of bounds'):
        batch.validate_selection(index, (0,))


@pytest.mark.parametrize('shape', [(), (2, 3), (2, 0)])
def test_flatten_round_trip_preserves_nested_leaves(shape):
    tree = {'matrix': jnp.zeros(shape + (2, 2)), 'scalar': jnp.ones(shape)}
    restored = batch.unflatten_batch(batch.flatten_batch(tree, shape), shape)
    for actual, expected in zip(
        jax.tree.leaves(restored), jax.tree.leaves(tree), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_shape_helpers_validate_prefix_and_rank():
    assert batch.split_prefix((2, 3, 4), (2, 3)) == (4,)
    assert batch.split_prefix((2,), ()) == (2,)
    for shape in [(3, 2), (2,)]:
        with pytest.raises(ValueError, match='must begin with'):
            batch.split_prefix(shape, (2, 3))
    assert batch.expand_trailing(jnp.array(True), 3).shape == (1, 1, 1)
    values = jnp.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(
        batch.expand_trailing(values, 4), values[..., None, None]
    )
    with pytest.raises(ValueError, match='cannot expand rank'):
        batch.expand_trailing(values, 1)
    with pytest.raises(ValueError, match='shapes must match'):
        batch.require_same_shape((2,), (3,))


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


def test_batch_helpers_reject_misaligned_leaves():
    with pytest.raises(ValueError, match='PyTree leaf shape must begin'):
        batch.flatten_batch(jnp.ones((3, 2)), (2,))
    with pytest.raises(ValueError, match='flattened batch size'):
        batch.unflatten_batch(jnp.ones((5, 2)), (2, 3))
    with pytest.raises(ValueError, match='flattened batch size'):
        batch.unflatten_batch(jnp.array(1), ())
    with pytest.raises(ValueError, match='PyTree leaf shape must begin'):
        batch.pool_samples(jnp.ones((2, 4, 1)), (2,), (3,))
    with pytest.raises(ValueError, match='target shape must begin'):
        batch.align_array(jnp.ones((2, 1)), (2,), (3, 2))
    with pytest.raises(ValueError, match='values shape must begin'):
        batch.weighted_sum(jnp.ones((3, 2)), jnp.ones(2), (2,), 0)


@pytest.mark.parametrize(
    'shape, indices, message',
    [
        ((), jnp.array(0), 'at least one batch axis'),
        ((2, 3), jnp.zeros(2), 'integer dtype'),
        ((2, 3), jnp.zeros(3, dtype=int), 'indices shape must begin'),
    ],
)
def test_take_along_last_batch_rejects_invalid_indices(shape, indices, message):
    with pytest.raises(ValueError, match=message):
        batch.take_along_last_batch(jnp.zeros(shape + (4,)), shape, indices)
