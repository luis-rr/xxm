import math

import jax
import jax.numpy as jnp
import numpy as np

from xxm.stats import poisson

ATOL = 1e-5
FIT_ATOL = 1e-4


def _poisson_log_prob(count: int, rate: float) -> float:
    return count * np.log(rate) - rate - math.lgamma(count + 1)


def test_log_likelihoods_matches_known_poisson_probabilities():
    observations = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    rates = jnp.array([[[1.0, 2.0], [0.5, 4.0]]])

    actual = poisson.log_likelihoods(observations, jnp.log(rates))

    expected = np.array(
        [
            [
                _poisson_log_prob(0, 1.0) + _poisson_log_prob(1, 2.0),
                _poisson_log_prob(0, 0.5) + _poisson_log_prob(1, 4.0),
            ],
            [
                _poisson_log_prob(2, 1.0) + _poisson_log_prob(3, 2.0),
                _poisson_log_prob(2, 0.5) + _poisson_log_prob(3, 4.0),
            ],
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_expected_log_likelihood_matches_gaussian_moment_formula():
    # z ~ N(1, 0.5), eta = 2z, y = 2.
    # E[eta] = 2, Var[eta] = 2, so E[exp(eta)] = exp(3).
    actual = poisson.expected_log_likelihood(
        observations=jnp.array([[2.0]]),
        means=jnp.array([[1.0]]),
        covariances=jnp.array([[[0.5]]]),
        coefficients=jnp.array([[2.0]]),
        bias=jnp.array([0.0]),
    )

    expected = 2.0 * 2.0 - np.exp(3.0) - math.lgamma(3.0)

    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_deterministic_inputs_match_zero_covariance_marginals():
    observations = jnp.array([[1.0], [3.0]])
    means = jnp.array([[0.0], [1.0]])
    coefficients = jnp.array([[0.5]])
    bias = jnp.array([-0.2])

    deterministic = poisson.expected_log_likelihood_per_output(
        observations=observations,
        means=means,
        covariances=None,
        coefficients=coefficients,
        bias=bias,
    )
    zero_covariance = poisson.expected_log_likelihood_per_output(
        observations=observations,
        means=means,
        covariances=jnp.zeros((2, 1, 1)),
        coefficients=coefficients,
        bias=bias,
    )

    np.testing.assert_allclose(deterministic, zero_covariance, atol=ATOL)


def test_sample_weights_ignore_zero_weight_samples():
    observations = jnp.array([[2.0], [100.0]])
    means = jnp.zeros((2, 1))
    coefficients = jnp.zeros((1, 1))
    bias = jnp.zeros(1)

    actual = poisson.expected_log_likelihood(
        observations=observations,
        means=means,
        covariances=None,
        coefficients=coefficients,
        bias=bias,
        sample_weights=jnp.array([1.0, 0.0]),
    )

    expected = _poisson_log_prob(2, 1.0)
    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_fit_weighted_matches_weighted_sample_means():
    observations = jnp.array([[1.0], [3.0], [4.0], [8.0]])
    weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    log_rates = poisson.fit_weighted(observations, weights)

    np.testing.assert_allclose(
        log_rates,
        np.log([[2.0], [6.0]]),
        atol=ATOL,
    )


def test_fit_weighted_keeps_zero_rates_finite():
    log_rates = poisson.fit_weighted(
        observations=jnp.zeros((2, 1)),
        weights=jnp.ones((2, 1)),
    )

    assert np.isfinite(np.asarray(log_rates)).all()
    np.testing.assert_allclose(jnp.exp(log_rates), [[1e-8]], atol=1e-10)


def test_fit_linear_recovers_two_point_poisson_mle():
    # With two observations and two parameters, the optimum can match both
    # positive counts exactly: lambda(0)=1 and lambda(1)=2.
    fit = poisson.fit_linear(
        inputs=jnp.array([[0.0], [1.0]]),
        outputs=jnp.array([[1.0], [2.0]]),
        coefficients=jnp.zeros((1, 1)),
        bias=jnp.zeros(1),
    )

    np.testing.assert_allclose(fit.coefficients, [[np.log(2.0)]], atol=FIT_ATOL)
    np.testing.assert_allclose(fit.bias, [0.0], atol=FIT_ATOL)


def test_fit_linear_from_marginals_matches_known_ridge_solution():
    # Optimize the average Poisson log likelihood minus
    # 0.5 * ridge * ||coefficients||^2. For x = {-1, 1}, y = {1, 3},
    # choosing this ridge gives the exact optimum below.
    ridge = 3.0 / (13.0 * np.log(1.5))

    fit = poisson.fit_linear_from_marginals(
        observations=jnp.array([[1.0], [3.0]]),
        means=jnp.array([[-1.0], [1.0]]),
        covariances=None,
        coefficients=jnp.zeros((1, 1)),
        bias=jnp.zeros(1),
        ridge=ridge,
    )

    np.testing.assert_allclose(
        fit.coefficients,
        [[np.log(1.5)]],
        atol=FIT_ATOL,
    )
    np.testing.assert_allclose(
        fit.bias,
        [np.log(24.0 / 13.0)],
        atol=FIT_ATOL,
    )


def test_fit_weighted_linear_recovers_state_specific_two_point_mles():
    inputs = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    outputs = jnp.array([[1.0], [2.0], [4.0], [2.0]])
    weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    fit = poisson.fit_weighted_linear(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        coefficients=jnp.zeros((2, 1, 1)),
        bias=jnp.zeros((2, 1)),
    )

    expected_coefficients = np.array([[[np.log(2.0)]], [[-np.log(2.0)]]])
    expected_bias = np.array([[0.0], [np.log(16.0)]])

    np.testing.assert_allclose(
        fit.coefficients,
        expected_coefficients,
        atol=FIT_ATOL,
    )
    np.testing.assert_allclose(fit.bias, expected_bias, atol=FIT_ATOL)


def test_public_routines_are_jittable():
    observations = jnp.array([[1.0], [2.0]])
    means = jnp.array([[0.0], [1.0]])
    covariances = jnp.array([[[0.1]], [[0.1]]])
    coefficients = jnp.zeros((1, 1))
    bias = jnp.zeros(1)
    weights = jnp.ones((2, 1))
    log_rates = jnp.zeros((1, 1, 1))

    @jax.jit
    def run_marginal_routines(
        observations,
        means,
        covariances,
        coefficients,
        bias,
        weights,
        log_rates,
    ):
        log_likelihoods = poisson.log_likelihoods(observations, log_rates)
        expected_per_output = poisson.expected_log_likelihood_per_output(
            observations,
            means,
            covariances,
            coefficients,
            bias,
        )
        expected_total = poisson.expected_log_likelihood(
            observations,
            means,
            covariances,
            coefficients,
            bias,
        )
        fitted_rates = poisson.fit_weighted(observations, weights)
        marginal_fit = poisson.fit_linear_from_marginals(
            observations=observations,
            means=means,
            covariances=covariances,
            coefficients=coefficients,
            bias=bias,
            max_iter=2,
            ridge=0.1,
        )

        return (
            log_likelihoods,
            expected_per_output,
            expected_total,
            fitted_rates,
            marginal_fit,
        )

    marginal_result = run_marginal_routines(
        observations,
        means,
        covariances,
        coefficients,
        bias,
        weights,
        log_rates,
    )
    jax.block_until_ready(marginal_result)

    weighted_inputs = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    weighted_outputs = jnp.array([[1.0], [2.0], [4.0], [2.0]])
    state_weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )

    @jax.jit
    def run_deterministic_fits(inputs, outputs, weights):
        linear_fit = poisson.fit_linear(
            inputs=inputs[:2],
            outputs=outputs[:2],
            coefficients=jnp.zeros((1, 1)),
            bias=jnp.zeros(1),
            max_iter=2,
            ridge=0.1,
        )
        weighted_fit = poisson.fit_weighted_linear(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            coefficients=jnp.zeros((2, 1, 1)),
            bias=jnp.zeros((2, 1)),
            max_iter=2,
            ridge=0.1,
        )
        return linear_fit, weighted_fit

    deterministic_result = run_deterministic_fits(
        weighted_inputs,
        weighted_outputs,
        state_weights,
    )
    jax.block_until_ready(deterministic_result)

    assert marginal_result[0].shape == (2, 1)
    assert marginal_result[-1].coefficients.shape == (1, 1)
    assert deterministic_result[0].coefficients.shape == (1, 1)
    assert deterministic_result[1].coefficients.shape == (2, 1, 1)
