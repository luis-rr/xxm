import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.dists.gaussian import (
    Gaussian,
    LinearGaussian,
    PairedGaussian,
)
from xxm.core.optim import gaussian as gaussian_fit

ATOL = 1e-5


def _normal_log_prob(x: float, mean: float, variance: float) -> float:
    return -0.5 * (np.log(2.0 * np.pi * variance) + (x - mean) ** 2 / variance)


def test_log_likelihoods_matches_univariate_gaussians():
    observations = jnp.array([[0.0], [2.0], [-1.0]])
    means = jnp.array([[0.0], [1.0]])
    covariances = jnp.array([[[1.0]], [[4.0]]])

    actual = Gaussian.from_moments(means, covariances).log_prob_broadcast(observations)

    expected = np.array(
        [
            [
                _normal_log_prob(0.0, 0.0, 1.0),
                _normal_log_prob(2.0, 0.0, 1.0),
                _normal_log_prob(-1.0, 0.0, 1.0),
            ],
            [
                _normal_log_prob(0.0, 1.0, 4.0),
                _normal_log_prob(2.0, 1.0, 4.0),
                _normal_log_prob(-1.0, 1.0, 4.0),
            ],
        ]
    )

    assert actual.shape == (2, 3)  # Receiver batch, then observations.
    np.testing.assert_allclose(actual, expected, atol=ATOL)


def test_linear_gaussian_conditional_broadcasts_covariance():
    num_steps = 5
    num_models = 3
    input_dim = 2
    output_dim = 4

    model = LinearGaussian.from_components(
        jnp.zeros((num_models, output_dim, input_dim)),
        jnp.zeros((num_models, output_dim)),
        jnp.broadcast_to(
            jnp.eye(output_dim),
            (num_models, output_dim, output_dim),
        ),
    )

    inputs = jnp.zeros(
        (num_models, num_steps, input_dim),
    )

    conditional = model.conditional(inputs)

    assert model.covariance.shape == (
        num_models,
        output_dim,
        output_dim,
    )

    assert conditional.mean.shape == (
        num_models,
        num_steps,
        output_dim,
    )

    assert conditional.covariance.shape == (
        num_models,
        num_steps,
        output_dim,
        output_dim,
    )

    np.testing.assert_allclose(
        conditional.covariance,
        jnp.broadcast_to(model.covariance[:, None], conditional.covariance.shape),
        atol=ATOL,
    )


def test_paired_gaussian_assembles_joint_moments():
    paired = PairedGaussian(
        left=Gaussian.from_moments(jnp.array([1.0]), jnp.array([[2.0]])),
        right=Gaussian.from_moments(jnp.array([3.0]), jnp.array([[4.0]])),
        cross_covariance=jnp.array([[5.0]]),
    )

    np.testing.assert_allclose(
        paired.mean,
        [1.0, 3.0],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        paired.covariance,
        [
            [2.0, 5.0],
            [5.0, 4.0],
        ],
        atol=ATOL,
    )


def test_moment_match_combines_within_and_between_covariance():
    distributions = Gaussian.from_moments(
        jnp.array([[0.0], [2.0]]), jnp.array([[[1.0]], [[3.0]]])
    )

    fit = gaussian_fit.from_moment_match(distributions)

    np.testing.assert_allclose(fit.mean, [1.0], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[3.0]], atol=ATOL)


def test_paired_moment_match_combines_cross_covariance():
    distributions = PairedGaussian(
        left=Gaussian.from_moments(
            jnp.array([[0.0], [2.0]]), jnp.array([[[1.0]], [[1.0]]])
        ),
        right=Gaussian.from_moments(
            jnp.array([[1.0], [5.0]]), jnp.array([[[2.0]], [[2.0]]])
        ),
        cross_covariance=jnp.array([[[0.5]], [[0.5]]]),
    )

    fit = gaussian_fit.paired_from_moment_match(distributions)

    np.testing.assert_allclose(fit.left.mean, [1.0], atol=ATOL)
    np.testing.assert_allclose(fit.left.covariance, [[2.0]], atol=ATOL)

    np.testing.assert_allclose(fit.right.mean, [3.0], atol=ATOL)
    np.testing.assert_allclose(fit.right.covariance, [[6.0]], atol=ATOL)

    np.testing.assert_allclose(
        fit.cross_covariance,
        [[2.5]],
        atol=ATOL,
    )


def test_linear_from_marginals_accounts_for_input_uncertainty():
    inputs = Gaussian.from_moments(
        jnp.array([[0.0], [2.0]]), jnp.array([[[1.0]], [[1.0]]])
    )
    outputs = jnp.array([[1.0], [5.0]])

    fit = gaussian_fit.linear_from_marginals(
        inputs,
        outputs,
        ridge=0.0,
        covariance_floor=0.0,
    )

    np.testing.assert_allclose(
        fit.affine.coefficients,
        [[1.0]],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fit.affine.bias,
        [2.0],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fit.covariance,
        [[2.0]],
        atol=ATOL,
    )


def test_paired_moment_match_preserves_positive_definiteness():
    means = jnp.array(
        [
            [272.0, -1113.0, 819.0, -11.0],
            [-145.0, 594.0, -437.0, 6.0],
        ],
        dtype=jnp.float32,
    )

    joint_covariance = jnp.array(
        [
            [34.643, -25.674, -48.165, -5.416],
            [-25.674, 108.154, -78.849, -13.374],
            [-48.165, -78.849, 217.448, 1.916],
            [-5.416, -13.374, 1.916, 244.210],
        ],
        dtype=jnp.float32,
    )

    joint_covariances = jnp.broadcast_to(
        joint_covariance,
        (2, 4, 4),
    )

    distributions = PairedGaussian(
        left=Gaussian.from_moments(means[:, :2], joint_covariances[:, :2, :2]),
        right=Gaussian.from_moments(means[:, 2:], joint_covariances[:, 2:, 2:]),
        cross_covariance=joint_covariances[:, 2:, :2],
    )

    # The input distributions are valid.
    assert jnp.isfinite(jnp.linalg.cholesky(distributions.covariance)).all()

    fit = gaussian_fit.paired_from_moment_match(distributions)

    # Moment matching should preserve positive definiteness.
    assert jnp.isfinite(jnp.linalg.cholesky(fit.covariance)).all()


@pytest.mark.parametrize(
    'batch, query, weighted, paired',
    [
        ((2, 3), (), False, False),
        ((2, 3), (2, 1), True, False),
        ((2, 3), (), True, True),
        ((), (3,), True, True),
    ],
)
def test_moment_matching_preserves_structural_and_query_batches(
    batch, query, weighted, paired
):
    shape = batch + (4,)
    means = jnp.arange(np.prod(shape) * 2, dtype=float).reshape(shape + (2,)) / 20
    covariances = jnp.broadcast_to(jnp.array([[2.0, 0.3], [0.3, 1.0]]), shape + (2, 2))
    distributions = Gaussian.from_moments(means, covariances)
    weights = None
    if weighted:
        weights = (
            jnp.arange(np.prod(batch + query + (4,)), dtype=float).reshape(
                batch + query + (4,)
            )
            % 7
            + 1
        )
    if paired:
        distributions = PairedGaussian(
            Gaussian.from_moments(means[..., :1], covariances[..., :1, :1]),
            Gaussian.from_moments(means[..., 1:], covariances[..., 1:, 1:]),
            covariances[..., 1:, :1],
        )
        fit = gaussian_fit.paired_from_moment_match
    else:
        fit = gaussian_fit.from_moment_match
    actual = jax.jit(fit)(distributions, weights)
    normalized = (
        np.ones(shape) / 4
        if weights is None
        else np.asarray(weights) / np.asarray(weights).sum(-1, keepdims=True)
    )
    means = np.asarray(means).reshape(batch + (1,) * len(query) + (4, 2))
    covariance = np.asarray(covariances).reshape(batch + (1,) * len(query) + (4, 2, 2))
    expected_mean = (normalized[..., None] * means).sum(-2)
    expected_second = (
        normalized[..., None, None]
        * (covariance + means[..., :, None] * means[..., None, :])
    ).sum(-3)
    expected_covariance = (
        expected_second - expected_mean[..., :, None] * expected_mean[..., None, :]
    )
    assert actual.batch_shape == batch + query
    np.testing.assert_allclose(actual.mean, expected_mean, atol=2e-6)
    np.testing.assert_allclose(actual.covariance, expected_covariance, atol=5e-6)
