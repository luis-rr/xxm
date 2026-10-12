import jax.numpy as jnp
import numpy as np
import pytest

from xxm.arhmm.data import ARDataset
from xxm.core.data import Dataset
from xxm.core.lengths import Lengths


@pytest.mark.parametrize('num_lags', [1, 2, 3])
def test_prepared_rows_preserve_lag_and_target_order(num_lags):
    observations = jnp.array(
        [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]
    )
    data = ARDataset.from_dataset(Dataset.from_array(observations), num_lags)
    assert data.max_steps == 5 - num_lags
    assert data.num_lags == num_lags
    assert data.observation_dim == 2
    assert data.predictors.shape == (5 - num_lags, num_lags, 2)
    np.testing.assert_array_equal(data.targets, observations[num_lags:])
    for row in range(data.max_steps):
        np.testing.assert_array_equal(
            data.predictors[row], observations[row : row + num_lags][::-1]
        )


def test_prepared_rows_preserve_structural_batch():
    observations = jnp.arange(72.0).reshape(2, 3, 6, 2)
    data = ARDataset.from_dataset(
        Dataset.from_padded(observations, Lengths.full((2, 3), 6)), num_lags=2
    )
    assert data.predictors.shape == (2, 3, 4, 2, 2)
    assert (data.max_steps, data.num_lags, data.observation_dim) == (4, 2, 2)
    np.testing.assert_array_equal(data.targets, observations[..., 2:, :])
    for i, j in np.ndindex(2, 3):
        standalone = ARDataset.from_dataset(
            Dataset.from_array(observations[i, j]), num_lags=2
        )
        np.testing.assert_array_equal(data.predictors[i, j], standalone.predictors)
