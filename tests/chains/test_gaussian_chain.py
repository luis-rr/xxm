import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.gaussian.chain import (
    GaussianChain,
    GaussianPairPotential,
    GaussianPotential,
)
from xxm.stats.gaussian import Affine, Gaussian, LinearGaussian

jax.config.update('jax_enable_x64', True)

RTOL = 1e-7
ATOL = 1e-6


def make_chain() -> GaussianChain:
    diagonal_precision_blocks = jnp.array(
        [
            [[4.0, 0.5], [0.5, 3.0]],
            [[5.0, 0.2], [0.2, 4.0]],
            [[6.0, 0.4], [0.4, 5.0]],
        ]
    )
    lower_precision_blocks = jnp.array(
        [
            [[0.2, -0.1], [-0.05, 0.1]],
            [[0.15, 0.05], [0.02, 0.12]],
        ]
    )
    information_vectors = jnp.array(
        [
            [0.3, -0.2],
            [0.1, 0.4],
            [-0.2, 0.5],
        ]
    )
    return GaussianChain(
        diagonal_precision_blocks=diagonal_precision_blocks,
        lower_precision_blocks=lower_precision_blocks,
        information_vectors=information_vectors,
        log_constant=jnp.array(0.0),  # Placeholder for log constant
    )


def test_dense_precision_matches_block_representation():
    chain = make_chain()

    result = chain.dense_precision()

    expected = np.array(
        [
            [4.0, 0.5, 0.2, -0.05, 0.0, 0.0],
            [0.5, 3.0, -0.1, 0.1, 0.0, 0.0],
            [0.2, -0.1, 5.0, 0.2, 0.15, 0.02],
            [-0.05, 0.1, 0.2, 4.0, 0.05, 0.12],
            [0.0, 0.0, 0.15, 0.05, 6.0, 0.4],
            [0.0, 0.0, 0.02, 0.12, 0.4, 5.0],
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_gaussian_chain_mean_matches_dense_reference():
    chain = make_chain()

    marginals = chain.forward_backward()

    result = marginals.means

    dense_precision_matrix = np.asarray(chain.dense_precision())
    dense_information_vectors = np.asarray(chain.information_vectors).reshape(-1)
    expected = np.linalg.solve(dense_precision_matrix, dense_information_vectors)
    expected = expected.reshape((3, 2))

    np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_chain_covariances_match_dense_inverse():
    chain = make_chain()

    marginals = chain.forward_backward()
    result = marginals.covariances

    dense_precision_matrix = np.asarray(chain.dense_precision())
    dense_inverse = np.linalg.inv(dense_precision_matrix)
    expected = []
    for t in range(len(chain.diagonal_precision_blocks)):
        start = t * 2
        stop = start + 2
        expected.append(dense_inverse[start:stop, start:stop])

    np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_chain_cross_covariances_match_dense_inverse():
    chain = make_chain()

    marginals = chain.forward_backward()
    result = marginals.cross_covariances

    dense_precision_matrix = np.asarray(chain.dense_precision())
    dense_inverse = np.linalg.inv(dense_precision_matrix)
    expected = []
    for t in range(len(chain.lower_precision_blocks)):
        start = t * 2
        stop = start + 2
        next_start = (t + 1) * 2
        next_stop = next_start + 2
        expected.append(dense_inverse[start:stop, next_start:next_stop])

    np.testing.assert_allclose(result, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_chain_single_time_step():
    chain = GaussianChain(
        diagonal_precision_blocks=jnp.array([[[3.0, 0.2], [0.2, 4.0]]]),
        lower_precision_blocks=jnp.zeros((0, 2, 2)),
        information_vectors=jnp.array([[0.5, -0.3]]),
        log_constant=jnp.array(0.0),
    )

    marginals = chain.forward_backward()
    posterior = marginals

    dense_precision_matrix = np.asarray(chain.dense_precision())
    dense_information_vectors = np.asarray(chain.information_vectors).reshape(-1)
    expected_mean = np.linalg.solve(dense_precision_matrix, dense_information_vectors)

    np.testing.assert_allclose(posterior.means, expected_mean.reshape((1, 2)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        posterior.covariances[0], np.linalg.inv(dense_precision_matrix), atol=ATOL, rtol=RTOL
    )
    assert posterior.cross_covariances.shape == (0, 2, 2)


def test_gaussian_chain_scalar_latent():
    chain = GaussianChain(
        diagonal_precision_blocks=jnp.array([[[4.0]], [[5.0]], [[6.0]]]),
        lower_precision_blocks=jnp.array([[[0.3]], [[0.2]]]),
        information_vectors=jnp.array([[1.0], [0.5], [-0.25]]),
        log_constant=jnp.array(0.0),
    )

    marginals = chain.forward_backward()
    posterior = marginals
    dense_precision_matrix = np.asarray(chain.dense_precision())
    dense_information_vectors = np.asarray(chain.information_vectors).reshape(-1)

    expected_mean = np.linalg.solve(dense_precision_matrix, dense_information_vectors)

    np.testing.assert_allclose(posterior.means, expected_mean.reshape((3, 1)), atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        posterior.covariances[:, 0, 0],
        np.diag(np.linalg.inv(dense_precision_matrix)),
        atol=ATOL,
        rtol=RTOL,
    )


def test_gaussian_chain_covariances_are_symmetric_and_positive_definite():
    chain = make_chain()

    marginals = chain.forward_backward()
    posterior = marginals

    for covariance in posterior.covariances:
        np.testing.assert_allclose(covariance, covariance.T)
        eigenvalues = np.linalg.eigvalsh(np.asarray(covariance))
        assert np.all(eigenvalues > 0.0)


def test_gaussian_chain_log_normalizer_matches_dense_reference():
    chain = make_chain()

    marginals = chain.forward_backward()
    posterior = marginals
    dense_precision_matrix = np.asarray(chain.dense_precision())
    h = np.asarray(chain.information_vectors).reshape(-1)

    dense_log_det = np.linalg.slogdet(dense_precision_matrix)[1]
    dense_quadratic = (
        0.5
        * h
        @ np.linalg.solve(
            dense_precision_matrix,
            h,
        )
    )

    expected = dense_quadratic - 0.5 * dense_log_det + 0.5 * h.size * np.log(2.0 * np.pi)

    np.testing.assert_allclose(posterior.log_normalizer, expected, atol=ATOL, rtol=RTOL)


def test_gaussian_chain_jit():
    chain = make_chain()

    eager = chain.forward_backward()
    jitted = jax.jit(lambda c: c.forward_backward())(chain)

    np.testing.assert_allclose(
        np.asarray(jitted.means), np.asarray(eager.means), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(np.asarray(jitted.covariances), np.asarray(eager.covariances))
    np.testing.assert_allclose(
        np.asarray(jitted.cross_covariances), np.asarray(eager.cross_covariances)
    )
    np.testing.assert_allclose(jitted.log_normalizer, eager.log_normalizer)


# --- GaussianPotential ---


def test_gaussian_potential_from_moments_fields():
    mean = jnp.array([1.0, -0.5])
    covariance = jnp.array([[2.0, 0.3], [0.3, 1.5]])

    potential = GaussianPotential.from_moments(Gaussian(mean, covariance))

    expected_precision = np.linalg.inv(np.array(covariance))
    expected_information = expected_precision @ np.array(mean)

    np.testing.assert_allclose(potential.precision_blocks, expected_precision, atol=ATOL, rtol=RTOL)
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


# --- GaussianPairPotential ---


def test_gaussian_pair_potential_from_linear_conditional_fields():
    A = jnp.array([[0.8, 0.1], [-0.2, 0.9]])
    b = jnp.array([0.5, -0.3])
    Q = jnp.array([[1.5, 0.2], [0.2, 1.0]])

    potential = GaussianPairPotential.from_linear_conditional(
        LinearGaussian(affine=Affine(coefficients=A, bias=b), covariance=Q)
    )

    Q_inv = np.linalg.inv(np.array(Q))

    np.testing.assert_allclose(
        potential.left_precision, np.array(A).T @ Q_inv @ np.array(A), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(potential.right_precision, Q_inv, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(
        potential.lower_precision, -Q_inv @ np.array(A), atol=ATOL, rtol=RTOL
    )
    np.testing.assert_allclose(
        potential.left_information, -np.array(A).T @ Q_inv @ np.array(b), atol=ATOL, rtol=RTOL
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


# --- Batched GaussianPotential ---


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
            lambda mean, covariance: GaussianPotential.from_moments(Gaussian(mean, covariance))
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


def test_gaussian_potential_from_moments_batched_jit():
    means = jnp.array(
        [
            [1.0, -0.5],
            [0.2, 0.7],
        ]
    )
    covariances = jnp.array(
        [
            [[2.0, 0.3], [0.3, 1.5]],
            [[1.2, 0.1], [0.1, 0.9]],
        ]
    )

    eager = GaussianPotential.from_moments(Gaussian(means, covariances))
    jitted = jax.jit(
        lambda mean, covariance: GaussianPotential.from_moments(Gaussian(mean, covariance))
    )(means, covariances)

    np.testing.assert_allclose(jitted.precision_blocks, eager.precision_blocks)
    np.testing.assert_allclose(jitted.information_vectors, eager.information_vectors)
    np.testing.assert_allclose(jitted.log_constant, eager.log_constant)


# --- Batched GaussianPairPotential ---


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
        LinearGaussian(affine=Affine(coefficients=matrices, bias=biases), covariance=covariances)
    )

    expected = jax.vmap(
        lambda matrix, bias, covariance: GaussianPairPotential.from_linear_conditional(
            LinearGaussian(affine=Affine(coefficients=matrix, bias=bias), covariance=covariance)
        )
    )(matrices, biases, covariances)

    np.testing.assert_allclose(result.left_precision, expected.left_precision, atol=ATOL, rtol=RTOL)
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
    np.testing.assert_allclose(result.log_constant, expected.log_constant, atol=ATOL, rtol=RTOL)

    assert result.left_precision.shape == (3, 2, 2)
    assert result.left_information.shape == (3, 2)
    assert result.log_constant.shape == (3,)


def test_gaussian_pair_potential_from_linear_conditional_batched_jit():
    matrices = jnp.array(
        [
            [[0.8, 0.1], [-0.2, 0.9]],
            [[1.0, -0.1], [0.3, 0.7]],
        ]
    )
    biases = jnp.array(
        [
            [0.5, -0.3],
            [-0.2, 0.4],
        ]
    )
    covariances = jnp.array(
        [
            [[1.5, 0.2], [0.2, 1.0]],
            [[1.2, -0.1], [-0.1, 0.9]],
        ]
    )

    eager = GaussianPairPotential.from_linear_conditional(
        LinearGaussian(affine=Affine(coefficients=matrices, bias=biases), covariance=covariances)
    )
    jitted = jax.jit(
        lambda matrix, bias, covariance: GaussianPairPotential.from_linear_conditional(
            LinearGaussian(affine=Affine(coefficients=matrix, bias=bias), covariance=covariance)
        )
    )(matrices, biases, covariances)

    for eager_field, jitted_field in zip(eager, jitted):
        np.testing.assert_allclose(jitted_field, eager_field, atol=ATOL, rtol=RTOL)


# --- GaussianChain + local potentials ---


def test_gaussian_chain_add_time_indexed_local_potential():
    chain = make_chain()

    means = jnp.array(
        [
            [0.1, -0.2],
            [0.3, 0.4],
            [-0.1, 0.2],
        ]
    )
    covariances = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))

    potential = GaussianPotential.from_moments(Gaussian(means, covariances))

    result = chain.add_local_potential(potential)

    np.testing.assert_allclose(
        result.diagonal_precision_blocks,
        chain.diagonal_precision_blocks + potential.precision_blocks,
    )
    np.testing.assert_allclose(
        result.information_vectors,
        chain.information_vectors + potential.information_vectors,
    )
    np.testing.assert_allclose(
        result.log_constant,
        chain.log_constant + jnp.sum(potential.log_constant),
    )
    np.testing.assert_allclose(
        result.lower_precision_blocks,
        chain.lower_precision_blocks,
    )


def test_gaussian_chain_add_local_potential_does_not_broadcast_over_time():
    chain = make_chain()

    potential = GaussianPotential.from_moments(
        Gaussian(mean=jnp.zeros(2), covariance=jnp.eye(2)),
    )

    with pytest.raises(ValueError):
        chain.add_local_potential(potential)


def test_gaussian_chain_add_local_potential_requires_matching_variable_dim():
    chain = make_chain()

    potential = GaussianPotential.from_moments(
        Gaussian(mean=jnp.zeros((3, 1)), covariance=jnp.ones((3, 1, 1))),
    )

    with pytest.raises(ValueError):
        chain.add_local_potential(potential)


def test_gaussian_chain_add_local_potential_jit():
    chain = make_chain()

    potential = GaussianPotential.from_moments(
        Gaussian(mean=jnp.zeros((3, 2)), covariance=jnp.broadcast_to(jnp.eye(2), (3, 2, 2))),
    )

    eager = chain.add_local_potential(potential)
    jitted = jax.jit(lambda chain, potential: chain.add_local_potential(potential))(
        chain,
        potential,
    )

    for eager_field, jitted_field in zip(eager, jitted):
        np.testing.assert_allclose(jitted_field, eager_field)


def test_gaussian_chain_rejects_batched_chain():
    chain = make_chain()

    batched_chain = GaussianChain(
        diagonal_precision_blocks=jnp.stack(
            [chain.diagonal_precision_blocks, chain.diagonal_precision_blocks]
        ),
        lower_precision_blocks=jnp.stack(
            [chain.lower_precision_blocks, chain.lower_precision_blocks]
        ),
        information_vectors=jnp.stack([chain.information_vectors, chain.information_vectors]),
        log_constant=jnp.zeros(2),
    )

    with pytest.raises(ValueError):
        batched_chain.forward_backward()
