import math

import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson

ATOL = 1e-5


FIT_ATOL = 1e-4


def _poisson_log_prob(count: int, rate: float) -> float:
    return count * np.log(rate) - rate - math.lgamma(count + 1)


def test_log_likelihoods_matches_known_poisson_probabilities():
    observations = jnp.array([[0.0, 1.0], [2.0, 3.0], [1.0, 0.0]])
    rates = jnp.array([[1.0, 2.0], [0.5, 4.0]])

    actual = Poisson(log_rates=jnp.log(rates)).log_prob_broadcast(observations)

    expected = np.array(
        [
            [
                _poisson_log_prob(0, 1.0) + _poisson_log_prob(1, 2.0),
                _poisson_log_prob(2, 1.0) + _poisson_log_prob(3, 2.0),
                _poisson_log_prob(1, 1.0) + _poisson_log_prob(0, 2.0),
            ],
            [
                _poisson_log_prob(0, 0.5) + _poisson_log_prob(1, 4.0),
                _poisson_log_prob(2, 0.5) + _poisson_log_prob(3, 4.0),
                _poisson_log_prob(1, 0.5) + _poisson_log_prob(0, 4.0),
            ],
        ]
    )

    assert actual.shape == (2, 3)  # Receiver batch, then observations.
    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_expected_log_likelihood_matches_gaussian_moment_formula():
    # z ~ N(1, 0.5), eta = 2z, y = 2.
    # E[eta] = 2, Var[eta] = 2, so E[exp(eta)] = exp(3).
    linear_model = LinearPoisson(
        affine=Affine(coefficients=jnp.array([[2.0]]), bias=jnp.array([0.0]))
    ).broadcast((1,))

    actual = linear_model.expected_log_prob(
        values=jnp.array([[2.0]]),
        inputs=Gaussian(mean=jnp.array([[1.0]]), covariance=jnp.array([[[0.5]]])),
    )

    expected = 2.0 * 2.0 - np.exp(3.0) - math.lgamma(3.0)

    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_deterministic_inputs_match_zero_covariance_marginals():
    observations = jnp.array([[1.0], [3.0]])
    means = jnp.array([[0.0], [1.0]])
    linear_model = LinearPoisson(
        affine=Affine(coefficients=jnp.array([[0.5]]), bias=jnp.array([-0.2]))
    ).broadcast((2,))

    deterministic = linear_model.conditional(means).log_prob_each(observations)
    zero_covariance = linear_model.expected_log_prob_each(
        values=observations,
        inputs=Gaussian(mean=means, covariance=jnp.zeros((2, 1, 1))),
    )

    np.testing.assert_allclose(deterministic, zero_covariance, atol=ATOL)
