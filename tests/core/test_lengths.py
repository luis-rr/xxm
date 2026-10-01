"""Structural temporal lengths remain separate from observation visibility."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm import Lengths
from xxm.core.data import Dataset, Sequences, WeightedObservations
from xxm.core.mask import ArbitraryMask


def test_lengths_clear_event_axes_and_transition_prefixes_under_jit():
    lengths = Lengths.from_values([[0, 1], [2, 3]])
    values = jnp.full((2, 2, 4, 2), jnp.nan)
    values = values.at[1, 0, :2].set(7)
    values = values.at[1, 1, :3].set(9)
    values = values.at[0, 1, :1].set(5)
    cleared = jax.jit(lambda extent, x: extent.clear(x))(lengths, values)
    np.testing.assert_array_equal(cleared[0, 0], 0)
    np.testing.assert_array_equal(cleared[0, 1, 0], 5)
    np.testing.assert_array_equal(cleared[0, 1, 1:], 0)
    np.testing.assert_array_equal(cleared[1, 0, :2], 7)
    np.testing.assert_array_equal(cleared[1, 0, 2:], 0)
    np.testing.assert_array_equal(cleared[1, 1, :3], 9)
    np.testing.assert_array_equal(cleared[1, 1, 3:], 0)
    np.testing.assert_array_equal(lengths.transitions().values, [[0, 0], [1, 2]])
    assert lengths.to_mask(0).values.shape == (2, 2, 0)


def test_lengths_bounds_and_batch_alignment_are_explicit():
    lengths = Lengths.from_values([0, 3])
    lengths.validate_values(max_steps=3)

    with pytest.raises(ValueError):
        lengths.validate_values(max_steps=2)

    with pytest.raises(ValueError):
        lengths.validate_values(min_length=1, max_steps=3)


def test_sequences_store_lengths_and_crop_without_losing_dense_horizon():
    lengths = Lengths.from_values([3, 1])
    sequences = Sequences.from_padded(jnp.arange(6).reshape(2, 3, 1), lengths)
    assert sequences.lengths is lengths
    assert sequences.max_steps == 3
    np.testing.assert_array_equal(sequences.get(1), [[3]])


def test_sequences_allow_scalar_events_and_empty_time():
    scalar = Sequences.from_sequence(jnp.arange(2))
    assert (scalar.batch_shape, scalar.event_shape, scalar.max_steps) == ((), (), 2)
    stacked = Sequences.stack(
        [scalar, Sequences.from_sequence(jnp.empty((0,), dtype=int))]
    )
    assert stacked.values.shape == (2, 2)
    np.testing.assert_array_equal(stacked.lengths.values, [2, 0])
    np.testing.assert_array_equal(stacked.values, [[0, 1], [0, 0]])

    vectors = Sequences.from_sequences([jnp.empty((0, 1)), jnp.ones((2, 1))])
    assert vectors.values.shape == (2, 2, 1)
    assert vectors.event_shape == (1,)
    np.testing.assert_array_equal(vectors.lengths.values, [0, 2])
    empty = Sequences.from_padded(jnp.empty((2, 0, 1)), Lengths.full((2,), 0))
    assert empty.max_steps == 0
    assert empty.event_shape == (1,)


def test_select_time_preserves_heterogeneous_prefixes_and_scalar_events():
    sequences = Sequences.from_padded(
        jnp.arange(12).reshape(3, 4), Lengths.from_values([4, 2, 1])
    )
    transitions = sequences.select_time(start=1)
    assert transitions.event_shape == ()
    np.testing.assert_array_equal(transitions.lengths.values, [3, 1, 0])
    np.testing.assert_array_equal(
        transitions.time_major(), [[1, 5, 0], [2, 0, 0], [3, 0, 0]]
    )

    bounded = sequences.select_time(start=-3, stop=3)
    np.testing.assert_array_equal(bounded.lengths.values, [2, 1, 0])
    np.testing.assert_array_equal(bounded.values, transitions.values[:, :2])
    empty = sequences.select_time(start=4)
    assert empty.time_major().shape == (0, 3)
    np.testing.assert_array_equal(empty.lengths.values, [0, 0, 0])


def test_weighted_observations_validate_lengths_and_outside_weights_on_host():
    lengths = Lengths.from_values([0, 2])
    values = jnp.ones((2, 3, 1))
    observations = WeightedObservations(
        values, jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), lengths
    )
    observations.validate()
    assert observations.max_steps == 3
    transformed = jax.jit(lambda data: data.broadcast(1).squeeze(0))(observations)
    np.testing.assert_array_equal(transformed.lengths.values, lengths.values)
    empty = WeightedObservations.from_sequence(jnp.zeros((0, 1)))
    assert empty.max_steps == 0
    assert int(empty.lengths.values) == 0


def test_visibility_remains_arbitrary_with_argument_free_reductions():
    visible = ArbitraryMask(jnp.array([[True, False, True], [False, True, False]]))
    assert visible.max_steps == 3
    np.testing.assert_array_equal(visible.num_valid(), [2, 1])
    assert visible.materialize() is visible.values
    np.testing.assert_array_equal(visible.adjacent_pairs().values, False)
    data = Dataset.from_padded(
        jnp.ones((2, 3, 1)), Lengths.from_values([3, 2]), visible=visible
    )
    weighted = data.weighted_observations()
    assert weighted.lengths is data.lengths
    np.testing.assert_array_equal(weighted.weights, visible.values)
