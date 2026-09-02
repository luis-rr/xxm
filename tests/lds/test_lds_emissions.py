import typing

import jax
import numpy as np
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions


class MockPosterior(typing.NamedTuple):
    means: jax.Array
    covariances: jax.Array

    def raw_second_moments(self) -> jax.Array:
        return self.covariances + jnp.einsum(
            'ti,tj->tij',
            self.means,
            self.means,
        )


# ---------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------


def test_gaussian_potential_matches_known_value():
    emissions = GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(
                coefficients=jnp.array(
                    [
                        [1.0, 2.0],
                        [0.0, 1.0],
                    ]
                ),
                bias=jnp.array([1.0, -1.0]),
            ),
            covariance=jnp.diag(jnp.array([1.0, 4.0])),
        ),
    )

    potential = emissions.compute_potential(jnp.array([[2.0, 3.0]]))

    expected_precision = jnp.array(
        [
            [[1.0, 2.0], [2.0, 4.25]],
        ]
    )
    expected_information = jnp.array([[1.0, 3.0]])
    expected_constant = -0.5 * (5.0 + jnp.log(4.0) + 2.0 * jnp.log(2.0 * jnp.pi))

    np.testing.assert_allclose(
        potential.precision_blocks,
        expected_precision,
    )
    np.testing.assert_allclose(
        potential.information_vectors,
        expected_information,
    )
    np.testing.assert_allclose(
        potential.log_constant,
        expected_constant,
    )


def test_gaussian_log_likelihood_matches_known_value():
    emissions = GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(
                coefficients=jnp.array(
                    [
                        [1.0, 2.0],
                        [0.0, 1.0],
                    ]
                ),
                bias=jnp.array([1.0, -1.0]),
            ),
            covariance=jnp.diag(jnp.array([1.0, 4.0])),
        ),
    )

    value = emissions.log_likelihood(
        observations=jnp.array([[1.0, 0.0]]),
        latents=jnp.array([[1.0, -1.0]]),
    )

    expected = -0.5 * (2.0 + jnp.log(4.0) + 2.0 * jnp.log(2.0 * jnp.pi))

    np.testing.assert_allclose(value, expected)


def test_gaussian_fit_recovers_known_parameters():
    latents = jnp.array(
        [
            [-1.0],
            [-1.0],
            [0.0],
            [0.0],
            [1.0],
            [1.0],
        ]
    )

    residuals = jnp.array(
        [
            [0.5],
            [-0.5],
            [0.5],
            [-0.5],
            [0.5],
            [-0.5],
        ]
    )

    observations = 2.0 * latents + 1.0 + residuals

    posterior = MockPosterior(
        means=latents,
        covariances=jnp.zeros((6, 1, 1)),
    )

    emissions = GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)),
            covariance=jnp.eye(1),
        ),
    )

    fitted = emissions.fit_params(observations, posterior)  # type: ignore

    np.testing.assert_allclose(fitted.model.affine.coefficients, [[2.0]], atol=1e-6)
    np.testing.assert_allclose(fitted.model.affine.bias, [1.0], atol=1e-6)
    np.testing.assert_allclose(
        fitted.model.covariance,
        [[0.25]],
        atol=1e-6,
    )


def test_gaussian_sample_has_expected_shape():
    emissions = GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.ones((3, 2)), bias=jnp.zeros(3)),
            covariance=jnp.eye(3),
        ),
    )

    sample = emissions.sample(
        jax.random.key(0),
        jnp.zeros(2),
    )

    assert sample.shape == (3,)
    assert jnp.all(jnp.isfinite(sample))


def test_gaussian_emissions_potential_has_one_factor_per_observation():
    # y_t | x_t ~ N(2 x_t + 1, 4)
    emissions = GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.array([[2.0]]), bias=jnp.array([1.0])),
            covariance=jnp.array([[4.0]]),
        ),
    )

    observations = jnp.array(
        [
            [1.0],
            [3.0],
            [5.0],
        ]
    )

    potential = emissions.compute_potential(observations)

    # For R = 4 and C = 2:
    #
    # J = C^T R^-1 C = 1
    #
    # h_t = C^T R^-1 (y_t - 1)
    #     = [0, 1, 2]
    #
    # c_t = -1/2 (y_t - 1)^2 / 4
    #       -1/2 log(4)
    #       -1/2 log(2 pi)
    residuals = np.array([0.0, 2.0, 4.0])

    expected_precision = np.ones((3, 1, 1))
    expected_information = np.array([[0.0], [1.0], [2.0]])
    expected_log_constant = (
        -0.5 * residuals**2 / 4.0 - 0.5 * np.log(4.0) - 0.5 * np.log(2.0 * np.pi)
    )

    assert potential.batch_shape == (3,)
    assert potential.variable_dim == 1

    np.testing.assert_allclose(
        potential.precision_blocks,
        expected_precision,
    )
    np.testing.assert_allclose(
        potential.information_vectors,
        expected_information,
    )
    np.testing.assert_allclose(
        potential.log_constant,
        expected_log_constant,
    )


# ---------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------


def test_poisson_rates_match_known_values():
    emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(coefficients=jnp.array([[jnp.log(2.0)]]), bias=jnp.zeros(1)),
        ),
    )

    rates = emissions.rates(jnp.array([[0.0], [1.0], [2.0]]))

    np.testing.assert_allclose(
        rates,
        [[1.0], [2.0], [4.0]],
        rtol=1e-6,
    )


def test_poisson_log_likelihood_matches_known_scalar_value():
    emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)),
        ),
    )

    value = emissions.log_likelihood(
        observations=jnp.array([[2.0]]),
        latents=jnp.array([[0.0]]),
    )

    np.testing.assert_allclose(
        value,
        -1.0 - jnp.log(2.0),
        rtol=1e-6,
    )


def test_poisson_log_likelihood_handles_zero_count():
    emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(
                coefficients=jnp.zeros((1, 1)), bias=jnp.array([jnp.log(2.0)])
            ),
        ),
    )

    value = emissions.log_likelihood(
        observations=jnp.array([[0.0]]),
        latents=jnp.array([[0.0]]),
    )

    np.testing.assert_allclose(value, -2.0, rtol=1e-6)


def test_poisson_local_potential_matches_value_gradient_and_hessian():
    emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(coefficients=jnp.array([[2.0]]), bias=jnp.zeros(1)),
        ),
    )

    observations = jnp.array([[3.0]])
    reference = jnp.array([[0.5]])

    potential = emissions.compute_local_potential(
        observations,
        reference,
    )

    def log_likelihood(x):
        return emissions.log_likelihood(
            observations,
            x.reshape(1, 1),
        )

    x0 = reference[0, 0]

    value = log_likelihood(x0)
    gradient = jax.grad(log_likelihood)(x0)
    hessian = jax.grad(jax.grad(log_likelihood))(x0)

    precision = potential.precision_blocks[0, 0, 0]
    information = potential.information_vectors[0, 0]

    quadratic_value = (
        -0.5 * precision * x0**2 + information * x0 + potential.log_constant
    )
    quadratic_gradient = -precision * x0 + information
    quadratic_hessian = -precision

    np.testing.assert_allclose(quadratic_value, value, rtol=1e-6)
    np.testing.assert_allclose(quadratic_gradient, gradient, rtol=1e-6)
    np.testing.assert_allclose(quadratic_hessian, hessian, rtol=1e-6)


def test_poisson_fit_recovers_known_parameters():
    latents = jnp.tile(
        jnp.array([[-1.0], [0.0], [1.0]]),
        (8, 1),
    )

    true_emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(
                coefficients=jnp.array([[jnp.log(2.0)]]), bias=jnp.array([jnp.log(2.0)])
            ),
        ),
    )

    observations = true_emissions.rates(latents)

    posterior = MockPosterior(
        means=latents,
        covariances=jnp.zeros((latents.shape[0], 1, 1)),
    )

    initial_emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(coefficients=jnp.zeros((1, 1)), bias=jnp.zeros(1)),
        ),
    )

    fitted = initial_emissions.fit_params(observations, posterior)  # type: ignore

    np.testing.assert_allclose(
        fitted.model.affine.coefficients,
        true_emissions.model.affine.coefficients,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        fitted.model.affine.bias,
        true_emissions.model.affine.bias,
        atol=1e-3,
    )


def test_poisson_sample_has_expected_shape_and_values():
    emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(coefficients=jnp.ones((3, 2)), bias=jnp.zeros(3)),
        ),
    )

    sample = emissions.sample(
        jax.random.key(0),
        jnp.zeros(2),
    )

    assert sample.shape == (3,)
    assert jnp.all(sample >= 0)
    assert jnp.issubdtype(sample.dtype, jnp.integer)


# ---------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------


def test_gaussian_methods_are_jittable():
    emissions = GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.eye(2), bias=jnp.zeros(2)),
            covariance=jnp.eye(2),
        ),
    )

    observations = jnp.ones((3, 2))
    latents = jnp.zeros((3, 2))
    posterior = MockPosterior(
        means=latents,
        covariances=jnp.broadcast_to(jnp.eye(2), (3, 2, 2)),
    )

    jax.jit(emissions.compute_potential)(observations)
    jax.jit(emissions.log_likelihood)(observations, latents)
    jax.jit(emissions.sample)(jax.random.key(0), latents[0])
    jax.jit(emissions.fit_params)(observations, posterior)


def test_poisson_methods_are_jittable():
    emissions = PoissonEmissions(
        model=LinearPoisson(
            affine=Affine(coefficients=jnp.eye(2), bias=jnp.zeros(2)),
        ),
    )

    observations = jnp.ones((3, 2))
    latents = jnp.zeros((3, 2))
    posterior = MockPosterior(
        means=latents,
        covariances=jnp.broadcast_to(jnp.eye(2), (3, 2, 2)),
    )

    jax.jit(emissions.rates)(latents)
    jax.jit(emissions.log_likelihood)(observations, latents)
    jax.jit(emissions.compute_local_potential)(observations, latents)
    jax.jit(emissions.sample)(jax.random.key(0), latents[0])
    jax.jit(emissions.fit_params)(observations, posterior)
