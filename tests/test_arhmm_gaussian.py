import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.discrete.chain import DiscreteChainMarginals
from xxm.core.discrete.emissions_ar import ARGaussianEmissions, lagged_observations

ATOL = 1e-6
FIT_ATOL = 2e-5


def test_ar_gaussian_properties():
    emissions = ARGaussianEmissions(
        coefficients=jnp.zeros((3, 2, 4, 4)),
        biases=jnp.zeros((3, 4)),
        covariances=jnp.tile(jnp.eye(4), (3, 1, 1)),
    )

    assert emissions.num_states == 3
    assert emissions.lag == 2
    assert emissions.num_dims == 4


def test_ar_gaussian_lagged_observations():
    emissions = ARGaussianEmissions(
        coefficients=jnp.zeros((1, 2, 1, 1)),
        biases=jnp.zeros((1, 1)),
        covariances=jnp.ones((1, 1, 1)),
    )

    observations = jnp.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    history = lagged_observations(observations, lag=emissions.lag, num_dims=emissions.num_dims)

    expected = jnp.array(
        [
            [[1.0], [0.0]],
            [[2.0], [1.0]],
        ]
    )

    np.testing.assert_allclose(history, expected, atol=ATOL)


def test_ar_gaussian_conditional_means_known_solution():
    emissions = ARGaussianEmissions(
        coefficients=jnp.array(
            [
                [
                    [[1.0, 0.0], [0.0, 2.0]],
                    [[0.5, 0.0], [0.0, -1.0]],
                ]
            ]
        ),
        biases=jnp.array([[1.0, -1.0]]),
        covariances=jnp.array([jnp.eye(2)]),
    )

    observations = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    means = emissions.conditional_means(observations)

    expected = jnp.array(
        [
            [[4.5, 5.0]],
            [[7.5, 7.0]],
        ]
    )

    np.testing.assert_allclose(means, expected, atol=ATOL)


def test_ar_gaussian_log_likelihoods_known_solution():
    emissions = ARGaussianEmissions(
        coefficients=jnp.array([[[[1.0]]]]),
        biases=jnp.array([[0.0]]),
        covariances=jnp.array([[[4.0]]]),
    )

    observations = jnp.array(
        [
            [1.0],
            [2.0],
            [4.0],
        ]
    )

    log_likelihoods = emissions.log_likelihoods(observations)

    # y1 | y0 ~ N(1, 4), residual = 1
    # y2 | y1 ~ N(2, 4), residual = 2
    constant = jnp.log(2 * jnp.pi) + jnp.log(4.0)

    expected = jnp.array(
        [
            [0.0],
            [-0.5 * (constant + 1.0**2 / 4.0)],
            [-0.5 * (constant + 2.0**2 / 4.0)],
        ]
    )

    np.testing.assert_allclose(log_likelihoods, expected, atol=ATOL)


def test_ar_gaussian_log_likelihoods_shortest_valid_sequence():
    emissions = ARGaussianEmissions(
        coefficients=jnp.array(
            [
                [
                    [[0.5]],
                    [[0.25]],
                ]
            ]
        ),
        biases=jnp.array([[0.0]]),
        covariances=jnp.array([[[1.0]]]),
    )

    observations = jnp.array(
        [
            [1.0],
            [2.0],
            [3.0],
        ]
    )

    log_likelihoods = emissions.log_likelihoods(observations)

    assert log_likelihoods.shape == (3, 1)

    np.testing.assert_allclose(
        log_likelihoods[:2],
        0.0,
        atol=ATOL,
    )

    assert jnp.isfinite(log_likelihoods[2, 0])


def test_ar_gaussian_fit_recovers_known_ar2_parameters():
    coefficient_1 = 0.5
    coefficient_2 = 0.3
    bias = -0.4

    values = [-1.0, 0.5]

    for _ in range(30):
        values.append(coefficient_1 * values[-1] + coefficient_2 * values[-2] + bias)

    observations = jnp.asarray(values)[:, None]

    posterior = DiscreteChainMarginals(
        # Important: posterior has length T, including the padded
        # first L time points.
        state_marginals=jnp.ones((observations.shape[0], 1)),
        # Mock pair marginals and log normalizer, since they are not used in the fit.
        pair_marginals=jnp.ones((observations.shape[0] - 1, 1, 1)),
        log_normalizer=jnp.zeros((1,)),
    )

    emissions = ARGaussianEmissions(
        coefficients=jnp.zeros((1, 2, 1, 1)),
        biases=jnp.zeros((1, 1)),
        covariances=jnp.ones((1, 1, 1)),
    )

    fitted = emissions.fit_params(
        observations,
        posterior,
    )

    np.testing.assert_allclose(
        fitted.coefficients,
        jnp.array(
            [
                [
                    [[coefficient_1]],
                    [[coefficient_2]],
                ]
            ]
        ),
        atol=FIT_ATOL,
    )

    np.testing.assert_allclose(
        fitted.biases,
        [[bias]],
        atol=FIT_ATOL,
    )

    # The data are deterministic, so residual variance should be
    # approximately zero.
    np.testing.assert_allclose(
        fitted.covariances,
        0.0,
        atol=FIT_ATOL,
    )


def test_ar_gaussian_permute():
    emissions = ARGaussianEmissions(
        coefficients=jnp.arange(3.0).reshape(3, 1, 1, 1),
        biases=jnp.array([[10.0], [20.0], [30.0]]),
        covariances=jnp.array([[[1.0]], [[2.0]], [[3.0]]]),
    )

    permutation = jnp.array([2, 0, 1])

    permuted = emissions.permute(permutation)

    np.testing.assert_array_equal(
        permuted.coefficients,
        emissions.coefficients[permutation],
    )
    np.testing.assert_array_equal(
        permuted.biases,
        emissions.biases[permutation],
    )
    np.testing.assert_array_equal(
        permuted.covariances,
        emissions.covariances[permutation],
    )


def test_ar_gaussian_sample_follows_ar_recurrence():
    emissions = ARGaussianEmissions(
        coefficients=jnp.array([[[[0.5]]]]),
        biases=jnp.array([[1.0]]),
        covariances=jnp.array([[[0.25]]]),
    )

    states = jnp.zeros(4, dtype=int)
    key = jax.random.key(0)

    observations = emissions.sample(key, states)

    # Reconstruct the expected samples explicitly from the same
    # innovations and the AR recurrence, starting from zero history.
    expected = []

    history = jnp.array([0.0])
    scan_key = key

    for _ in range(states.shape[0]):
        scan_key, observation_key = jax.random.split(scan_key)

        mean = 0.5 * history + 1.0

        observation = jax.random.multivariate_normal(
            observation_key,
            mean=mean,
            cov=jnp.array([[0.25]]),
        )

        expected.append(observation)
        history = observation

    expected = jnp.stack(expected)

    np.testing.assert_allclose(
        observations,
        expected,
        atol=ATOL,
    )


def test_ar_gaussian_methods_are_jittable():
    emissions = ARGaussianEmissions(
        coefficients=jnp.array(
            [
                [
                    [[0.5]],
                    [[0.2]],
                ]
            ]
        ),
        biases=jnp.array([[0.1]]),
        covariances=jnp.array([[[0.5]]]),
    )

    observations = jnp.array(
        [
            [0.0],
            [1.0],
            [0.5],
            [-0.5],
            [0.25],
        ]
    )

    posterior = DiscreteChainMarginals(
        state_marginals=jnp.ones((observations.shape[0], 1)),
        # Mock pair marginals and log normalizer, since they are not used in the fit.
        pair_marginals=jnp.ones((observations.shape[0] - 1, 1, 1)),
        log_normalizer=jnp.zeros((1,)),
    )

    states = jnp.zeros(5, dtype=int)
    key = jax.random.key(0)

    conditional_means_jit = jax.jit(
        lambda emissions, observations: emissions.conditional_means(observations)
    )

    log_likelihoods_jit = jax.jit(
        lambda emissions, observations: emissions.log_likelihoods(observations)
    )

    fit_params_jit = jax.jit(
        lambda emissions, observations, posterior: emissions.fit_params(observations, posterior)
    )

    sample_jit = jax.jit(lambda emissions, key, states: emissions.sample(key, states))

    np.testing.assert_allclose(
        conditional_means_jit(emissions, observations),
        emissions.conditional_means(observations),
        atol=ATOL,
    )

    np.testing.assert_allclose(
        log_likelihoods_jit(emissions, observations),
        emissions.log_likelihoods(observations),
        atol=ATOL,
    )

    fitted = fit_params_jit(
        emissions,
        observations,
        posterior,
    )
    fitted_eager = emissions.fit_params(
        observations,
        posterior,
    )

    np.testing.assert_allclose(
        fitted.coefficients,
        fitted_eager.coefficients,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fitted.biases,
        fitted_eager.biases,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fitted.covariances,
        fitted_eager.covariances,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        sample_jit(emissions, key, states),
        emissions.sample(key, states),
        atol=ATOL,
    )
