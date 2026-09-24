import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.data import Dataset
from xxm.core.mask import ArbitraryMask


def test_visibility_and_validity_survive_padding_stacking_and_jit():
    observations = [jnp.ones((3, 1)), jnp.ones((1, 1))]
    visibility = [jnp.array([True, False, True]), jnp.array([False])]
    inputs = [jnp.arange(3.0)[:, None], jnp.ones((1, 1))]
    data = Dataset.from_sequences(observations, inputs=inputs, visible=visibility)
    np.testing.assert_array_equal(data.lengths, [3, 1])
    np.testing.assert_array_equal(
        data.valid().materialize(3), [[True, True, True], [True, False, False]]
    )
    expected_visible = [[True, False, True], [False, False, False]]
    np.testing.assert_array_equal(data.visible.values, expected_visible)
    np.testing.assert_array_equal(data.observation_weights(), expected_visible)
    assert data.observations.valid() is data.observations.mask
    for actual, expected in zip(data.observations.unpack(), observations, strict=True):
        np.testing.assert_array_equal(actual, expected)

    stacked = Dataset.stack(
        [
            Dataset.from_sequence(y, inputs=u, visible=v)
            for y, u, v in zip(observations, inputs, visibility, strict=True)
        ]
    )
    for actual, expected in zip(
        jax.tree.leaves(stacked), jax.tree.leaves(data), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)

    transformed = jax.jit(lambda d: d.astype(jnp.float32).broadcast(1).squeeze(0))(data)
    np.testing.assert_array_equal(transformed.visible.values, expected_visible)
    np.testing.assert_array_equal(transformed.lengths, data.lengths)
    np.testing.assert_array_equal(transformed.inputs.values, data.inputs.values)


def test_dataset_optional_arguments_use_none_and_default_to_valid_visibility():
    data = Dataset.from_sequences(
        [jnp.ones((2, 1)), jnp.ones((1, 1))], inputs=None, visible=None
    )
    assert data.inputs.values.shape == (2, 2, 0)
    np.testing.assert_array_equal(data.visible.values, [[True, True], [True, False]])
    visibility = ArbitraryMask(jnp.array([[True, False], [False, True]]))
    batch = Dataset.from_batch(jnp.ones((2, 2, 1)), visible=visibility)
    assert batch.visible is visibility
    with pytest.raises(ValueError, match='false outside valid'):
        Dataset.from_padded(jnp.ones((2, 2, 1)), jnp.array([2, 1]), visible=visibility)
