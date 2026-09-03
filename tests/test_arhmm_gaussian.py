import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.emissions.discrete_ar import AREmissions, lagged_observations

ATOL = 1e-6
FIT_ATOL = 2e-5


def _flatten_lagged_coefficients(coefficients: jax.Array) -> jax.Array:
    """Reshape per-lag matrices (K, L, N_out, N_in) into (K, N_out, L * N_in).

    The predictor concatenates lag blocks from lag 1 to lag L, so the
    coefficient columns must be reordered to match.
    """
    num_states, num_lags, num_dims_out, num_dims_in = coefficients.shape

    return jnp.moveaxis(coefficients, 1, 2).reshape(
        num_states,
        num_dims_out,
        num_lags * num_dims_in,
    )


def _make_emissions(
    coefficients: jax.Array,  # (K, N_out, L, N_in)
    biases: jax.Array,  # (K, N_out)
    covariances: jax.Array,  # (K, N_out, N_out)
) -> AREmissions[LinearGaussian]:
    return AREmissions(
        model=LinearGaussian(
            affine=Affine(
                coefficients=coefficients,
                bias=biases,
            ),
            covariance=covariances,
        ),
    )


def test_ar_gaussian_properties():
    emissions = _make_emissions(
        coefficients=jnp.zeros((3, 4, 2, 4)),
        biases=jnp.zeros((3, 4)),
        covariances=jnp.tile(
            jnp.eye(4),
            (3, 1, 1),
        ),
    )

    assert emissions.num_states == 3
    assert emissions.num_lags == 2
    assert emissions.output_dim == 4
    assert emissions.model.input_shape == (2, 4)


def test_ar_gaussian_lagged_observations():
    emissions = _make_emissions(
        coefficients=jnp.zeros((1, 1, 2, 1)),
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

    history = lagged_observations(
        observations,
        num_lags=emissions.num_lags,
    )

    expected = jnp.array(
        [
            [[1.0], [0.0]],
            [[2.0], [1.0]],
        ]
    )

    np.testing.assert_allclose(
        history,
        expected,
        atol=ATOL,
    )


def test_ar_gaussian_conditional_means_known_solution():
    emissions = _make_emissions(
        coefficients=jnp.array(
            [
                [
                    [
                        [1.0, 0.0],
                        [0.5, 0.0],
                    ],
                    [
                        [0.0, 2.0],
                        [0.0, -1.0],
                    ],
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

    means = emissions.conditional(observations).mean

    expected = jnp.array(
        [
            [[4.5, 5.0]],
            [[7.5, 7.0]],
        ]
    )

    np.testing.assert_allclose(
        means,
        expected,
        atol=ATOL,
    )


def test_ar_gaussian_log_likelihoods_known_solution():
    emissions = _make_emissions(
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
            [-0.5 * (constant + 1.0**2 / 4.0)],
            [-0.5 * (constant + 2.0**2 / 4.0)],
        ]
    )

    np.testing.assert_allclose(
        log_likelihoods,
        expected,
        atol=ATOL,
    )


def test_ar_gaussian_log_likelihoods_shortest_valid_sequence():
    emissions = _make_emissions(
        coefficients=jnp.array(
            [
                [
                    [
                        [0.5],
                        [0.25],
                    ]
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

    # history = (y1, y0) = (2, 1)
    # mean = 0.5 * 2 + 0.25 * 1 = 1.25
    # residual = 3 - 1.25 = 1.75
    expected = -0.5 * (jnp.log(2 * jnp.pi) + 1.75**2)

    assert log_likelihoods.shape == (1, 1)

    np.testing.assert_allclose(
        log_likelihoods[0, 0],
        expected,
        atol=ATOL,
    )


def test_ar_gaussian_fit_recovers_known_ar2_parameters():
    coefficient_1 = 0.5
    coefficient_2 = 0.3
    bias = -0.4

    values = [-1.0, 0.5]

    for _ in range(30):
        values.append(coefficient_1 * values[-1] + coefficient_2 * values[-2] + bias)

    observations = jnp.asarray(values)[:, None]

    emissions = _make_emissions(
        coefficients=jnp.zeros((1, 1, 2, 1)),
        biases=jnp.zeros((1, 1)),
        covariances=jnp.ones((1, 1, 1)),
    )

    num_steps = observations.shape[0] - emissions.num_lags

    posterior = DiscreteChainMarginals(
        state_probs=jnp.ones((num_steps, 1)),
        pair_probs=jnp.ones(
            (
                num_steps - 1,
                1,
                1,
            )
        ),
    )

    fitted = emissions.fit_params(
        observations,
        posterior,
    )

    expected_coefficients = jnp.array(
        [
            [
                [
                    [coefficient_1],
                    [coefficient_2],
                ]
            ]
        ]
    )

    np.testing.assert_allclose(
        fitted.model.affine.coefficients,
        expected_coefficients,
        atol=FIT_ATOL,
    )

    np.testing.assert_allclose(
        fitted.model.affine.bias,
        [[bias]],
        atol=FIT_ATOL,
    )

    np.testing.assert_allclose(
        fitted.model.covariance,
        0.0,
        atol=FIT_ATOL,
    )


def test_ar_gaussian_permute():
    emissions = _make_emissions(
        coefficients=jnp.arange(3.0).reshape(3, 1, 1, 1),
        biases=jnp.array([[10.0], [20.0], [30.0]]),
        covariances=jnp.array([[[1.0]], [[2.0]], [[3.0]]]),
    )

    permutation = jnp.array([2, 0, 1])

    permuted = emissions.permute(permutation)

    np.testing.assert_array_equal(
        permuted.model.affine.coefficients,
        emissions.model.affine.coefficients[permutation],
    )
    np.testing.assert_array_equal(
        permuted.model.affine.bias,
        emissions.model.affine.bias[permutation],
    )
    np.testing.assert_array_equal(
        permuted.model.covariance,
        emissions.model.covariance[permutation],
    )


def test_ar_gaussian_sample_follows_ar_recurrence():
    emissions = _make_emissions(
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
    emissions = _make_emissions(
        coefficients=jnp.array(
            [
                [
                    [
                        [0.5],
                        [0.2],
                    ]
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

    num_steps = observations.shape[0] - emissions.num_lags

    posterior = DiscreteChainMarginals(
        state_probs=jnp.ones((num_steps, 1)),
        pair_probs=jnp.ones(
            (
                num_steps - 1,
                1,
                1,
            )
        ),
    )

    states = jnp.zeros(5, dtype=int)
    key = jax.random.key(0)

    conditional_means_jit = jax.jit(
        lambda emissions, observations: emissions.conditional(observations).mean
    )

    log_likelihoods_jit = jax.jit(
        lambda emissions, observations: emissions.log_likelihoods(observations)
    )

    fit_params_jit = jax.jit(
        lambda emissions, observations, posterior: emissions.fit_params(
            observations,
            posterior,
        )
    )

    sample_jit = jax.jit(
        lambda emissions, key, states: emissions.sample(
            key,
            states,
        )
    )

    np.testing.assert_allclose(
        conditional_means_jit(
            emissions,
            observations,
        ),
        emissions.conditional(observations).mean,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        log_likelihoods_jit(
            emissions,
            observations,
        ),
        emissions.log_likelihoods(observations),
        atol=ATOL,
    )

    fitted_jit = fit_params_jit(
        emissions,
        observations,
        posterior,
    )

    fitted = emissions.fit_params(
        observations,
        posterior,
    )

    for actual, expected in zip(
        jax.tree_util.tree_leaves(fitted_jit),
        jax.tree_util.tree_leaves(fitted),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL,
        )

    np.testing.assert_allclose(
        sample_jit(
            emissions,
            key,
            states,
        ),
        emissions.sample(
            key,
            states,
        ),
        atol=ATOL,
    )


def test_ar_gaussian_sample_uses_zero_history():
    emissions = _make_emissions(
        coefficients=jnp.array(
            [
                [
                    [
                        [0.5],
                        [0.2],
                    ]
                ]
            ]
        ),
        biases=jnp.array([[0.1]]),
        covariances=jnp.array([[[0.5]]]),
    )

    states = jnp.zeros(5, dtype=int)
    key = jax.random.key(0)

    initial_history = jnp.zeros(
        (
            emissions.num_lags,
            emissions.output_dim,
        )
    )

    autonomous = emissions.sample(
        key,
        states,
    )

    continuation = emissions.sample_continuation(
        key,
        states,
        initial_history,
    )

    np.testing.assert_allclose(
        autonomous,
        continuation,
        atol=ATOL,
    )
