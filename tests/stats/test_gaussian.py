import jax
import jax.numpy as jnp
import numpy as np

from xxm.stats import gaussian

ATOL = 1e-5


def _normal_log_prob(x: float, mean: float, variance: float) -> float:
    return -0.5 * (np.log(2.0 * np.pi * variance) + (x - mean) ** 2 / variance)


def test_log_likelihoods_matches_univariate_gaussians():
    observations = jnp.array([[0.0], [2.0]])
    means = jnp.array([[[0.0], [1.0]]])
    covariances = jnp.array([[[1.0]], [[4.0]]])

    actual = gaussian.log_likelihoods(observations, means, covariances)

    expected = np.array(
        [
            [
                _normal_log_prob(0.0, 0.0, 1.0),
                _normal_log_prob(0.0, 1.0, 4.0),
            ],
            [
                _normal_log_prob(2.0, 0.0, 1.0),
                _normal_log_prob(2.0, 1.0, 4.0),
            ],
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_fit_weighted_matches_hard_assignments():
    observations = jnp.array([[0.0], [2.0], [10.0], [14.0]])
    weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    means, covariances = gaussian.fit_weighted(observations, weights)

    np.testing.assert_allclose(means, [[1.0], [12.0]], atol=ATOL)
    np.testing.assert_allclose(covariances, [[[1.0]], [[4.0]]], atol=ATOL)


def test_fit_linear_recovers_exact_affine_map():
    inputs = jnp.array([[-1.0], [0.0], [1.0], [2.0]])
    outputs = 2.0 * inputs + 1.0

    fit = gaussian.fit_linear(inputs, outputs)

    np.testing.assert_allclose(fit.coefficients, [[2.0]], atol=ATOL)
    np.testing.assert_allclose(fit.bias, [1.0], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[0.0]], atol=ATOL)


def test_fit_linear_from_moments_uses_unregularized_covariance_for_noise():
    # x has variance 1 and y = 2x exactly. With ridge=1, the fitted
    # coefficient is 1, but the residual covariance is still computed from
    # the original input covariance: 4 - 2 - 2 + 1 = 1.
    fit = gaussian.fit_linear_from_moments(
        input_mean=jnp.array([0.0]),
        output_mean=jnp.array([0.0]),
        input_second_moment=jnp.array([[1.0]]),
        output_second_moment=jnp.array([[4.0]]),
        output_input_moment=jnp.array([[2.0]]),
        ridge=1.0,
    )

    np.testing.assert_allclose(fit.coefficients, [[1.0]], atol=ATOL)
    np.testing.assert_allclose(fit.bias, [0.0], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[1.0]], atol=ATOL)


def test_fit_weighted_linear_recovers_state_specific_affine_maps():
    inputs = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    outputs = jnp.array([[1.0], [3.0], [8.0], [7.0]])
    weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    fit = gaussian.fit_weighted_linear(inputs, outputs, weights)

    np.testing.assert_allclose(
        fit.coefficients,
        [[[2.0]], [[-1.0]]],
        atol=ATOL,
    )
    np.testing.assert_allclose(fit.bias, [[1.0], [10.0]], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[[0.0]], [[0.0]]], atol=ATOL)


def test_public_routines_are_jittable():
    observations = jnp.array([[0.0], [2.0], [10.0], [14.0]])
    means = jnp.array([[[1.0], [12.0]]])
    covariances = jnp.array([[[1.0]], [[4.0]]])
    weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    inputs = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    outputs = jnp.array([[1.0], [3.0], [8.0], [7.0]])

    @jax.jit
    def run(observations, means, covariances, weights, inputs, outputs):
        log_likelihoods = gaussian.log_likelihoods(
            observations,
            means,
            covariances,
        )
        weighted_means, weighted_covariances = gaussian.fit_weighted(
            observations,
            weights,
        )
        linear_fit = gaussian.fit_linear(inputs, outputs)
        weighted_linear_fit = gaussian.fit_weighted_linear(
            inputs,
            outputs,
            weights,
        )

        return (
            log_likelihoods,
            weighted_means,
            weighted_covariances,
            linear_fit,
            weighted_linear_fit,
        )

    result = run(observations, means, covariances, weights, inputs, outputs)
    jax.block_until_ready(result)

    assert result[0].shape == (4, 2)
    assert result[3].coefficients.shape == (1, 1)
    assert result[4].coefficients.shape == (2, 1, 1)
