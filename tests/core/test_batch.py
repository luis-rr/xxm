import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core import batch
from xxm.core.data import Dataset, Sequences
from xxm.core.dists.gaussian import Gaussian, PairedGaussian
from xxm.core.lengths import Lengths


def _gaussian():
    return Gaussian(
        mean=jnp.arange(12.0).reshape(2, 3, 2),
        covariance=jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2)),
    )


@pytest.mark.parametrize(
    'index',
    [
        None,
        (slice(None), None),
        True,
        np.bool_(False),
        (True, False),
        np.array([[True, False, True], [False, True, False]]),
        (0, 0, 0),
        (Ellipsis, 0, 0, 0),
        (Ellipsis, Ellipsis),
        0.5,
    ],
)
def test_select_rejects_new_axes_boolean_indices_and_intrinsic_indexing(index):
    with pytest.raises(IndexError):
        _gaussian().select(index)


def test_select_expands_only_the_batch_prefix():
    gaussian = _gaussian()
    selected = gaussian.select((1, Ellipsis))
    assert selected.batch_shape == (3,)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[1])
    np.testing.assert_array_equal(selected.covariance, gaussian.covariance[1])

    selected = gaussian.select((Ellipsis, 2))
    assert selected.batch_shape == (2,)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[:, 2])

    selected = gaussian.select((1, 2))
    assert selected.batch_shape == ()
    assert isinstance(selected, Gaussian)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[1, 2])
    with pytest.raises(IndexError):
        selected.select(0)


def test_select_integer_arrays_is_jittable():
    gaussian = _gaussian()
    indices = jnp.array([2, 0])
    selected = jax.jit(lambda g, i: g.select((1, i)))(gaussian, indices)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[1, indices])
    np.testing.assert_array_equal(selected.covariance, gaussian.covariance[1, indices])
    np.testing.assert_array_equal(
        gaussian.select([1, 0]).mean,
        gaussian.mean[jnp.array([1, 0])],
    )


def test_paired_batch_operations_preserve_all_components():
    left = _gaussian()
    right = left._replace(mean=left.mean + 20.0)
    cross = jnp.arange(24.0).reshape(2, 3, 2, 2)
    paired = PairedGaussian(left, right, cross)

    def transform(p):
        return (
            p.broadcast(1, axis=1)
            .squeeze(1)
            .permute(jnp.array([2, 0, 1]), axis=1)
            .move_axis(1, 0)
            .select((slice(None), 1))
        )

    selected = jax.jit(transform)(paired)
    assert selected.batch_shape == (3,)
    order = jnp.array([2, 0, 1])
    np.testing.assert_array_equal(selected.left.mean, left.mean[1, order])
    np.testing.assert_array_equal(selected.right.mean, right.mean[1, order])
    np.testing.assert_array_equal(selected.left.covariance, left.covariance[1, order])
    np.testing.assert_array_equal(selected.right.covariance, right.covariance[1, order])
    np.testing.assert_array_equal(selected.cross_covariance, cross[1, order])


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


@pytest.mark.parametrize('kind', ['sequences', 'dataset', 'arbitrary', 'lengths'])
def test_custom_pytrees_transform_under_jit(kind):
    lengths = Lengths(jnp.array([[3, 2, 1], [1, 2, 3]]))
    values = jnp.arange(36.0).reshape(2, 3, 3, 2)
    sequences = Sequences(values, lengths)
    mask = lengths.to_mask(3)
    obj = {
        'sequences': sequences,
        'dataset': Dataset(sequences, sequences, mask),
        'arbitrary': mask,
        'lengths': lengths,
    }[kind]

    @jax.jit
    def transform(obj):
        return (
            obj.broadcast(1, axis=1)
            .squeeze(1)
            .permute(jnp.array([2, 0, 1]), axis=1)
            .move_axis(1, 0)
            .select((slice(None), 1))
        )

    result = transform(obj)
    assert type(result) is type(obj)
    assert result.batch_shape == (3,)
    for actual, original in zip(
        jax.tree.leaves(result), jax.tree.leaves(obj), strict=True
    ):
        np.testing.assert_array_equal(actual, original[1, jnp.array([2, 0, 1])])


@pytest.mark.parametrize('kind', ['arbitrary', 'lengths'])
def test_mask_flatten_preserves_type_and_values(kind):
    lengths = Lengths(jnp.array([[3, 2, 1], [1, 2, 3]]))
    mask = {
        'arbitrary': lengths.to_mask(3),
        'lengths': lengths,
    }[kind]
    flat = jax.jit(lambda m: m.flatten())(mask)
    assert type(flat) is type(mask)
    assert flat.batch_shape == (6,)
    np.testing.assert_array_equal(
        flat.values, mask.values.reshape((6,) + mask.values.shape[2:])
    )


def test_concatenate_and_split_preserve_lengths_and_singleton_axes_under_jit():
    values = jnp.arange(36.0).reshape(2, 3, 3, 2)

    sequences = Sequences.from_padded(
        values, Lengths(jnp.array([[3, 2, 1], [1, 2, 3]]))
    )

    parts = jax.jit(lambda item: batch.split(item, (1, 0, 2), axis=-1))(sequences)

    assert tuple(part.batch_shape for part in parts) == ((2, 1), (2, 0), (2, 2))

    restored = jax.jit(lambda *items: batch.concatenate(items, axis=1))(*parts)

    assert type(restored) is Sequences

    np.testing.assert_array_equal(restored.values, sequences.values)

    np.testing.assert_array_equal(restored.lengths.values, sequences.lengths.values)


def test_concatenate_rejects_batch_and_intrinsic_mismatches_without_padding():
    sequences = Sequences.from_padded(jnp.zeros((2, 3, 1)), Lengths.full((2,), 3))
    with pytest.raises(ValueError):
        batch.concatenate([])
    with pytest.raises(ValueError):
        batch.concatenate([sequences.select(0)])
    with pytest.raises(ValueError):
        batch.concatenate([sequences, sequences.broadcast(1)])
    with pytest.raises(ValueError):
        batch.concatenate([sequences, Lengths(jnp.ones(2, dtype=int))])
    with pytest.raises(ValueError, match='intrinsic'):
        batch.concatenate(
            [
                sequences,
                Sequences.from_padded(jnp.zeros((1, 4, 1)), Lengths.full((1,), 4)),
            ]
        )
    with pytest.raises(ValueError):
        batch.concatenate([_gaussian(), _gaussian().select((slice(None), slice(1)))])


@pytest.mark.parametrize('sizes', [(1,), (1, -1, 2), (0.5, 1.5)])
def test_split_requires_static_nonnegative_sizes_covering_the_axis(sizes):
    with pytest.raises((ValueError, TypeError)):
        batch.split(_gaussian(), sizes)


def test_split_empty_axis_accepts_empty_sizes():
    assert batch.split(Lengths(jnp.zeros(0, dtype=int)), ()) == ()
