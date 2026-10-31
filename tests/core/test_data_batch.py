import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core import batch
from xxm.core.data import Dataset, Sequences
from xxm.core.dists.gaussian import FullGaussian, FullGaussianNoise
from xxm.core.lengths import Lengths


def _gaussian():
    return FullGaussian(
        jnp.arange(12.0).reshape(2, 3, 2), FullGaussianNoise.identity(2, (2, 3))
    )


def test_sequences_get_preserves_negative_indices_and_crops_valid_prefix():
    sequences = Sequences.from_padded(
        jnp.arange(24).reshape(2, 3, 4), Lengths(jnp.array([[4, 3, 2], [1, 2, 3]]))
    )
    np.testing.assert_array_equal(sequences.get((-1, -1)), sequences.values[1, 2, :3])
    np.testing.assert_array_equal(
        sequences.get((jnp.array(0), jnp.array(2))), sequences.values[0, 2, :2]
    )


@pytest.mark.parametrize('shape', [(), (3,), (2, 3), (2, 1, 3), (2, 0, 3)])
def test_sequences_items_follow_batch_order_and_crop_and_squeeze_arrays(shape):
    lengths = (jnp.arange(np.prod(shape), dtype=int) % 4).reshape(shape)
    values = jnp.arange(np.prod(shape) * 4).reshape(shape + (4, 1))
    sequences = Sequences.from_padded(values, Lengths(lengths))

    items = sequences.items()
    assert iter(items) is items
    pairs = list(items)
    assert [index for index, _ in pairs] == list(np.ndindex(shape))
    for index, array in pairs:
        expected = jnp.squeeze(values[index][: int(lengths[index])])
        np.testing.assert_array_equal(array, expected)
        np.testing.assert_array_equal(array, sequences.get(index))


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


def test_concatenate_preserves_lengths_and_singleton_axes_under_jit():
    values = jnp.arange(36.0).reshape(2, 3, 3, 2)

    sequences = Sequences.from_padded(
        values, Lengths(jnp.array([[3, 2, 1], [1, 2, 3]]))
    )

    parts = tuple(
        sequences.select((slice(None), part))
        for part in (slice(0, 1), slice(1, 1), slice(1, 3))
    )

    assert tuple(part.batch_shape for part in parts) == ((2, 1), (2, 0), (2, 2))

    restored = jax.jit(lambda *items: batch.concatenate(items, axis=1))(*parts)

    assert type(restored) is Sequences

    np.testing.assert_array_equal(restored.values, sequences.values)

    np.testing.assert_array_equal(restored.lengths.values, sequences.lengths.values)
