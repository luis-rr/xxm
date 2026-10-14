import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.potentials.gaussian import GaussianPairPotential, GaussianPotential

jax.config.update('jax_enable_x64', True)


RTOL = 1e-7
ATOL = 1e-6


def test_gaussian_potential_from_moments_fields():
    mean = jnp.array([1.0, -0.5])
    covariance = jnp.array([[2.0, 0.3], [0.3, 1.5]])

    potential = GaussianPotential.from_moments(Gaussian(mean, covariance))

    expected_precision = np.linalg.inv(np.array(covariance))
    expected_information = expected_precision @ np.array(mean)

    np.testing.assert_allclose(
        potential.precision_blocks, expected_precision, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        potential.information_vectors, expected_information, atol=ATOL, rtol=RTOL
    )


def test_gaussian_potential_from_moments_log_density():
    # Evaluate the canonical potential at a point and compare to log N(x; mean, cov).
    mean = jnp.array([1.0, -0.5])
    covariance = jnp.array([[2.0, 0.3], [0.3, 1.5]])
    x = jnp.array([0.5, 0.2])

    potential = GaussianPotential.from_moments(Gaussian(mean, covariance))

    log_f = (
        -0.5 * x @ potential.precision_blocks @ x
        + potential.information_vectors @ x
        + potential.log_constant
    )

    cov_inv = np.linalg.inv(np.array(covariance))
    residual = np.array(x) - np.array(mean)
    expected = (
        -0.5 * residual @ cov_inv @ residual
        - 0.5 * np.linalg.slogdet(np.array(covariance))[1]
        - 0.5 * mean.shape[0] * np.log(2.0 * np.pi)
    )

    np.testing.assert_allclose(log_f, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_potential_from_moments_is_normalized():
    # A potential built from moments is a normalized Gaussian, so its log normalizer is 0.
    mean = jnp.array([1.0, -0.5])
    covariance = jnp.array([[2.0, 0.3], [0.3, 1.5]])

    potential = GaussianPotential.from_moments(Gaussian(mean, covariance))

    J = np.array(potential.precision_blocks)
    h = np.array(potential.information_vectors)
    c = float(potential.log_constant)
    d = mean.shape[0]

    log_normalizer = (
        c
        + 0.5 * h @ np.linalg.solve(J, h)
        - 0.5 * np.linalg.slogdet(J)[1]
        + 0.5 * d * np.log(2.0 * np.pi)
    )

    np.testing.assert_allclose(log_normalizer, 0.0, atol=ATOL)


def test_gaussian_pair_potential_from_linear_conditional_fields():
    A = jnp.array([[0.8, 0.1], [-0.2, 0.9]])
    b = jnp.array([0.5, -0.3])
    Q = jnp.array([[1.5, 0.2], [0.2, 1.0]])

    potential = GaussianPairPotential.from_linear_conditional(
        LinearGaussian(affine=Affine(coefficients=A, bias=b), covariance=Q)
    )

    Q_inv = np.linalg.inv(np.array(Q))

    np.testing.assert_allclose(
        potential.left_precision,
        np.array(A).T @ Q_inv @ np.array(A),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(potential.right_precision, Q_inv, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        potential.lower_precision, -Q_inv @ np.array(A), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        potential.left_information,
        -np.array(A).T @ Q_inv @ np.array(b),
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        potential.right_information, Q_inv @ np.array(b), atol=ATOL, rtol=RTOL
    )


def test_gaussian_pair_potential_log_constant():
    A = jnp.array([[0.8, 0.1], [-0.2, 0.9]])
    b = jnp.array([0.5, -0.3])
    Q = jnp.array([[1.5, 0.2], [0.2, 1.0]])

    potential = GaussianPairPotential.from_linear_conditional(
        LinearGaussian(affine=Affine(coefficients=A, bias=b), covariance=Q)
    )

    Q_inv = np.linalg.inv(np.array(Q))
    d = b.shape[0]
    expected = (
        -0.5 * np.array(b) @ Q_inv @ np.array(b)
        - 0.5 * np.linalg.slogdet(np.array(Q))[1]
        - 0.5 * d * np.log(2.0 * np.pi)
    )

    np.testing.assert_allclose(potential.log_constant, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_pair_potential_log_density():
    # Evaluate log f(x0, x1) and compare to log N(x1; A x0 + b, Q).
    A = jnp.array([[0.8, 0.1], [-0.2, 0.9]])
    b = jnp.array([0.5, -0.3])
    Q = jnp.array([[1.5, 0.2], [0.2, 1.0]])
    x0 = jnp.array([1.0, 0.5])
    x1 = jnp.array([0.2, 0.8])

    potential = GaussianPairPotential.from_linear_conditional(
        LinearGaussian(affine=Affine(coefficients=A, bias=b), covariance=Q)
    )

    log_f = (
        -0.5 * x0 @ potential.left_precision @ x0
        - x1 @ potential.lower_precision @ x0
        - 0.5 * x1 @ potential.right_precision @ x1
        + potential.left_information @ x0
        + potential.right_information @ x1
        + potential.log_constant
    )

    conditional_mean = np.array(A) @ np.array(x0) + np.array(b)
    Q_inv = np.linalg.inv(np.array(Q))
    residual = np.array(x1) - conditional_mean
    d = b.shape[0]
    expected = (
        -0.5 * residual @ Q_inv @ residual
        - 0.5 * np.linalg.slogdet(np.array(Q))[1]
        - 0.5 * d * np.log(2.0 * np.pi)
    )

    np.testing.assert_allclose(log_f, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_potential_from_moments_batched():
    means = jnp.array(
        [
            [[1.0, -0.5], [0.2, 0.7]],
            [[-0.3, 0.4], [0.8, -0.1]],
        ]
    )  # (2, 2, D)

    covariances = jnp.array(
        [
            [
                [[2.0, 0.3], [0.3, 1.5]],
                [[1.2, 0.1], [0.1, 0.9]],
            ],
            [
                [[1.5, -0.2], [-0.2, 1.1]],
                [[0.8, 0.1], [0.1, 1.4]],
            ],
        ]
    )  # (2, 2, D, D)

    result = GaussianPotential.from_moments(Gaussian(means, covariances))

    expected = jax.vmap(
        jax.vmap(
            lambda mean, covariance: GaussianPotential.from_moments(
                Gaussian(mean, covariance)
            )
        )
    )(means, covariances)

    np.testing.assert_allclose(
        result.precision_blocks,
        expected.precision_blocks,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        result.information_vectors,
        expected.information_vectors,
        atol=ATOL,
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        result.log_constant,
        expected.log_constant,
        atol=ATOL,
        rtol=RTOL,
    )

    assert result.precision_blocks.shape == (2, 2, 2, 2)
    assert result.information_vectors.shape == (2, 2, 2)
    assert result.log_constant.shape == (2, 2)

    jitted = jax.jit(GaussianPotential.from_moments)(Gaussian(means, covariances))
    for eager_field, jitted_field in zip(result, jitted, strict=True):
        np.testing.assert_allclose(jitted_field, eager_field)


def test_gaussian_pair_potential_from_linear_conditional_batched():
    matrices = jnp.array(
        [
            [[0.8, 0.1], [-0.2, 0.9]],
            [[1.0, -0.1], [0.3, 0.7]],
            [[0.6, 0.2], [0.1, 0.8]],
        ]
    )
    biases = jnp.array(
        [
            [0.5, -0.3],
            [-0.2, 0.4],
            [0.1, 0.2],
        ]
    )
    covariances = jnp.array(
        [
            [[1.5, 0.2], [0.2, 1.0]],
            [[1.2, -0.1], [-0.1, 0.9]],
            [[0.8, 0.1], [0.1, 1.3]],
        ]
    )

    result = GaussianPairPotential.from_linear_conditional(
        LinearGaussian(
            affine=Affine(coefficients=matrices, bias=biases), covariance=covariances
        )
    )

    expected = jax.vmap(
        lambda matrix, bias, covariance: GaussianPairPotential.from_linear_conditional(
            LinearGaussian(
                affine=Affine(coefficients=matrix, bias=bias), covariance=covariance
            )
        )
    )(matrices, biases, covariances)

    np.testing.assert_allclose(
        result.left_precision, expected.left_precision, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        result.right_precision, expected.right_precision, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        result.lower_precision, expected.lower_precision, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        result.left_information, expected.left_information, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        result.right_information, expected.right_information, atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        result.log_constant, expected.log_constant, atol=ATOL, rtol=RTOL
    )

    assert result.left_precision.shape == (3, 2, 2)
    assert result.left_information.shape == (3, 2)
    assert result.log_constant.shape == (3,)

    jitted = jax.jit(GaussianPairPotential.from_linear_conditional)(
        LinearGaussian(Affine(matrices, biases), covariances)
    )
    for eager_field, jitted_field in zip(result, jitted, strict=True):
        np.testing.assert_allclose(jitted_field, eager_field, atol=ATOL, rtol=RTOL)
