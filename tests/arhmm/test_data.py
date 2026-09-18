import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.arhmm.data import ARObservations


@pytest.mark.parametrize('num_lags', [1, 2, 3])
def test_prepared_rows_preserve_lag_and_target_order(num_lags):
    observations = jnp.array(
        [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0], [4.0, 14.0]]
    )
    data = ARObservations.from_observations(observations, num_lags)
    assert data.num_steps == 5 - num_lags
    assert data.num_lags == num_lags
    assert data.output_dim == 2
    assert data.predictors.shape == (5 - num_lags, num_lags, 2)
    np.testing.assert_array_equal(data.targets, observations[num_lags:])
    for row in range(data.num_steps):
        np.testing.assert_array_equal(
            data.predictors[row], observations[row : row + num_lags][::-1]
        )

    prepared_jit = jax.jit(
        ARObservations.from_observations, static_argnames=('num_lags',)
    )(observations, num_lags)
    for actual, expected in zip(prepared_jit, data, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    'shape,num_lags',
    [((5,), 1), ((5, 1), 0), ((5, 1), -1), ((2, 1), 2), ((1, 1), 2)],
)
def test_invalid_observation_structure(shape, num_lags):
    with pytest.raises(ValueError):
        ARObservations.from_observations(jnp.zeros(shape), num_lags)


def test_prepared_rows_preserve_structural_batch():
    observations = jnp.arange(72.0).reshape(2, 3, 6, 2)
    data = ARObservations.from_observations(observations, num_lags=2)
    assert data.predictors.shape == (2, 3, 4, 2, 2)
    assert (data.num_steps, data.num_lags, data.output_dim) == (4, 2, 2)
    np.testing.assert_array_equal(data.targets, observations[..., 2:, :])
    for i, j in np.ndindex(2, 3):
        standalone = ARObservations.from_observations(observations[i, j], 2)
        np.testing.assert_array_equal(data.predictors[i, j], standalone.predictors)
