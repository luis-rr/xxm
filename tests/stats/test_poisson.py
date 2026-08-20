import math

import jax
import jax.numpy as jnp
import numpy as np

from xxm.stats import poisson_fit
from xxm.stats.gaussian import Affine, Gaussian
from xxm.stats.poisson import LinearPoisson, Poisson

ATOL = 1e-5
FIT_ATOL = 1e-4


def _poisson_log_prob(count: int, rate: float) -> float:
    return count * np.log(rate) - rate - math.lgamma(count + 1)


def test_log_likelihoods_matches_known_poisson_probabilities():
    observations = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    rates = jnp.array([[1.0, 2.0], [0.5, 4.0]])

    actual = Poisson(log_rates=jnp.log(rates)).log_prob_broadcast(observations)

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
    linear_model = LinearPoisson(
        affine=Affine(coefficients=jnp.array([[2.0]]), bias=jnp.array([0.0]))
    )

    actual = linear_model.expected_log_prob(
        observations=jnp.array([[2.0]]),
        inputs=Gaussian(mean=jnp.array([[1.0]]), covariance=jnp.array([[[0.5]]])),
    )

    expected = 2.0 * 2.0 - np.exp(3.0) - math.lgamma(3.0)

    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_deterministic_inputs_match_zero_covariance_marginals():
    observations = jnp.array([[1.0], [3.0]])
    means = jnp.array([[0.0], [1.0]])
    linear_model = LinearPoisson(
        affine=Affine(coefficients=jnp.array([[0.5]]), bias=jnp.array([-0.2]))
    )

    deterministic = linear_model.conditional(means).log_prob_each(observations)
    zero_covariance = linear_model.expected_log_prob_each(
        observations=observations,
        inputs=Gaussian(mean=means, covariance=jnp.zeros((2, 1, 1))),
    )

    np.testing.assert_allclose(deterministic, zero_covariance, atol=ATOL)


def test_sample_weights_ignore_zero_weight_samples():
    # Deterministic inputs (no covariance): log rates equal the bias since
    # coefficients are zero, matching a Poisson(rate=1.0) for both samples.
    observations = jnp.array([[2.0], [100.0]])
    means = jnp.zeros((2, 1))
    linear_model = LinearPoisson(affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)))
    sample_weights = jnp.array([1.0, 0.0])

    log_probs = linear_model.conditional(means).log_prob_each(observations)
    actual = jnp.sum(sample_weights[:, None] * log_probs)

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

    fit = poisson_fit.poisson_from_pairs_weighted(observations, weights)

    np.testing.assert_allclose(
        fit.log_rates,
        np.log([[2.0], [6.0]]),
        atol=ATOL,
    )


def test_fit_weighted_keeps_zero_rates_finite():
    fit = poisson_fit.poisson_from_pairs_weighted(
        observations=jnp.zeros((2, 1)),
        weights=jnp.ones((2, 1)),
    )

    assert np.isfinite(np.asarray(fit.log_rates)).all()


def test_fit_linear_recovers_two_point_poisson_mle():
    # With two observations and two parameters, the optimum can match both
    # positive counts exactly: lambda(0)=1 and lambda(1)=2.
    fit = poisson_fit.linear_from_pairs(
        inputs=jnp.array([[0.0], [1.0]]),
        outputs=jnp.array([[1.0], [2.0]]),
        initial_affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)),
    )

    np.testing.assert_allclose(fit.affine.coefficients, [[np.log(2.0)]], atol=FIT_ATOL)
    np.testing.assert_allclose(fit.affine.bias, [0.0], atol=FIT_ATOL)


def test_fit_linear_from_marginals_matches_known_ridge_solution():
    # Optimize the average Poisson log likelihood minus
    # 0.5 * ridge * ||coefficients||^2. For x = {-1, 1}, y = {1, 3},
    # choosing this ridge gives the exact optimum below.
    ridge = 3.0 / (13.0 * np.log(1.5))

    fit = poisson_fit.linear_from_marginals(
        observations=jnp.array([[1.0], [3.0]]),
        input_means=jnp.array([[-1.0], [1.0]]),
        input_covariances=None,
        initial_affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)),
        ridge=ridge,
    )

    np.testing.assert_allclose(
        fit.affine.coefficients,
        [[np.log(1.5)]],
        atol=FIT_ATOL,
    )
    np.testing.assert_allclose(
        fit.affine.bias,
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

    fit = poisson_fit.linear_from_pairs_weighted(
        inputs=inputs,
        outputs=outputs,
        weights=weights,
        initial_affine=Affine(coefficients=jnp.zeros((2, 1, 1)), bias=jnp.zeros((2, 1))),
    )

    expected_coefficients = np.array([[[np.log(2.0)]], [[-np.log(2.0)]]])
    expected_bias = np.array([[0.0], [np.log(16.0)]])

    np.testing.assert_allclose(
        fit.affine.coefficients,
        expected_coefficients,
        atol=FIT_ATOL,
    )
    np.testing.assert_allclose(fit.affine.bias, expected_bias, atol=FIT_ATOL)


def test_public_routines_are_jittable():
    observations = jnp.array([[1.0], [2.0]])
    means = jnp.array([[0.0], [1.0]])
    covariances = jnp.array([[[0.1]], [[0.1]]])
    coefficients = jnp.zeros((1, 1))
    bias = jnp.zeros(1)
    weights = jnp.ones((2, 1))
    log_rates = jnp.zeros((1, 1))

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
        linear_model = LinearPoisson(affine=Affine(coefficients=coefficients, bias=bias))
        inputs = Gaussian(mean=means, covariance=covariances)

        log_likelihoods = Poisson(log_rates=log_rates).log_prob_broadcast(observations)
        expected_per_output = linear_model.expected_log_prob_each(
            observations=observations,
            inputs=inputs,
        )
        expected_total = linear_model.expected_log_prob(
            observations=observations,
            inputs=inputs,
        )
        fitted = poisson_fit.poisson_from_pairs_weighted(observations, weights)
        marginal_fit = poisson_fit.linear_from_marginals(
            observations=observations,
            input_means=means,
            input_covariances=covariances,
            initial_affine=Affine(coefficients=coefficients, bias=bias),
            max_iter=2,
            ridge=0.1,
        )

        return (
            log_likelihoods,
            expected_per_output,
            expected_total,
            fitted.log_rates,
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
        linear_fit = poisson_fit.linear_from_pairs(
            inputs=inputs[:2],
            outputs=outputs[:2],
            initial_affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)),
            max_iter=2,
            ridge=0.1,
        )
        weighted_fit = poisson_fit.linear_from_pairs_weighted(
            inputs=inputs,
            outputs=outputs,
            weights=weights,
            initial_affine=Affine(coefficients=jnp.zeros((2, 1, 1)), bias=jnp.zeros((2, 1))),
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
    assert marginal_result[-1].affine.coefficients.shape == (1, 1)
    assert deterministic_result[0].affine.coefficients.shape == (1, 1)
    assert deterministic_result[1].affine.coefficients.shape == (2, 1, 1)
