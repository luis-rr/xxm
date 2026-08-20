import jax
import jax.numpy as jnp
import numpy as np

from xxm.stats import gaussian_fit
from xxm.core.affine import Affine
from xxm.stats.gaussian import Gaussian, LinearGaussian

ATOL = 1e-5


def _normal_log_prob(x: float, mean: float, variance: float) -> float:
    return -0.5 * (np.log(2.0 * np.pi * variance) + (x - mean) ** 2 / variance)


def test_log_likelihoods_matches_univariate_gaussians():
    observations = jnp.array([[0.0], [2.0]])
    means = jnp.array([[0.0], [1.0]])
    covariances = jnp.array([[[1.0]], [[4.0]]])

    actual = Gaussian(mean=means, covariance=covariances).log_prob_broadcast(observations)

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

    fit = gaussian_fit.gaussian_from_samples_weighted(observations, weights)

    np.testing.assert_allclose(fit.mean, [[1.0], [12.0]], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[[1.0]], [[4.0]]], atol=ATOL)


def test_fit_linear_recovers_exact_affine_map():
    inputs = jnp.array([[-1.0], [0.0], [1.0], [2.0]])
    outputs = 2.0 * inputs + 1.0

    fit = gaussian_fit.linear_from_pairs(inputs, outputs)

    np.testing.assert_allclose(fit.affine.coefficients, [[2.0]], atol=ATOL)
    np.testing.assert_allclose(fit.affine.bias, [1.0], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[0.0]], atol=ATOL)


def test_fit_linear_from_moments_uses_unregularized_covariance_for_noise():
    # x has variance 1 and y = 2x exactly. With ridge=1, the fitted
    # coefficient is 1, but the residual covariance is still computed from
    # the original input covariance: 4 - 2 - 2 + 1 = 1.
    fit = gaussian_fit.linear_from_moments(
        input_mean=jnp.array([0.0]),
        output_mean=jnp.array([0.0]),
        input_second_moment=jnp.array([[1.0]]),
        output_second_moment=jnp.array([[4.0]]),
        output_input_moment=jnp.array([[2.0]]),
        ridge=1.0,
    )

    np.testing.assert_allclose(fit.affine.coefficients, [[1.0]], atol=ATOL)
    np.testing.assert_allclose(fit.affine.bias, [0.0], atol=ATOL)
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

    fit = gaussian_fit.linear_from_pairs_weighted(inputs, outputs, weights)

    np.testing.assert_allclose(
        fit.affine.coefficients,
        [[[2.0]], [[-1.0]]],
        atol=ATOL,
    )
    np.testing.assert_allclose(fit.affine.bias, [[1.0], [10.0]], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[[0.0]], [[0.0]]], atol=ATOL)


def test_public_routines_are_jittable():
    observations = jnp.array([[0.0], [2.0], [10.0], [14.0]])
    means = jnp.array([[1.0], [12.0]])
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
        log_likelihoods = Gaussian(mean=means, covariance=covariances).log_prob_broadcast(
            observations
        )
        weighted_fit = gaussian_fit.gaussian_from_samples_weighted(
            observations,
            weights,
        )
        linear_fit = gaussian_fit.linear_from_pairs(inputs, outputs)
        weighted_linear_fit = gaussian_fit.linear_from_pairs_weighted(
            inputs,
            outputs,
            weights,
        )

        return (
            log_likelihoods,
            weighted_fit,
            linear_fit,
            weighted_linear_fit,
        )

    result = run(observations, means, covariances, weights, inputs, outputs)
    jax.block_until_ready(result)

    assert result[0].shape == (4, 2)
    assert result[2].affine.coefficients.shape == (1, 1)
    assert result[3].affine.coefficients.shape == (2, 1, 1)


def test_linear_gaussian_conditional_broadcasts_covariance():
    num_time_steps = 5
    num_models = 3
    input_dim = 2
    output_dim = 4

    model = LinearGaussian(
        affine=Affine(
            coefficients=jnp.zeros((num_models, output_dim, input_dim)),
            bias=jnp.zeros((num_models, output_dim)),
        ),
        covariance=jnp.broadcast_to(
            jnp.eye(output_dim),
            (num_models, output_dim, output_dim),
        ),
    )

    inputs = jnp.zeros(
        (num_time_steps, num_models, input_dim),
    )

    conditional = model.conditional(inputs)

    assert model.covariance.shape == (
        num_models,
        output_dim,
        output_dim,
    )

    assert conditional.mean.shape == (
        num_time_steps,
        num_models,
        output_dim,
    )

    assert conditional.covariance.shape == (
        num_time_steps,
        num_models,
        output_dim,
        output_dim,
    )
