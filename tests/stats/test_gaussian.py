import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian, PairedGaussian
from xxm.core.optim import gaussian as gaussian_fit

ATOL = 1e-5


def _normal_log_prob(x: float, mean: float, variance: float) -> float:
    return -0.5 * (np.log(2.0 * np.pi * variance) + (x - mean) ** 2 / variance)


def test_log_likelihoods_matches_univariate_gaussians():
    observations = jnp.array([[0.0], [2.0], [-1.0]])
    means = jnp.array([[0.0], [1.0]])
    covariances = jnp.array([[[1.0]], [[4.0]]])

    actual = Gaussian(mean=means, covariance=covariances).log_prob_broadcast(
        observations
    )

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


def test_fit_weighted_matches_hard_assignments():
    observations = jnp.array([[0.0], [2.0], [10.0], [14.0]])
    weights = jnp.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    ).T

    fit = gaussian_fit.from_samples_weighted(
        observations,
        weights,
        covariance_floor=0.0,
    )

    np.testing.assert_allclose(fit.mean, [[1.0], [12.0]], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[[1.0]], [[4.0]]], atol=ATOL)


def test_fit_linear_recovers_exact_affine_map():
    inputs = jnp.array(
        [
            [1023.875],
            [1024.0],
            [1024.125],
        ]
    )
    outputs = 2.0 * inputs + 1.0

    eager = gaussian_fit.linear_from_samples(
        inputs,
        outputs,
        ridge=0.0,
        covariance_floor=0.0,
    )
    jitted = jax.jit(gaussian_fit.linear_from_samples)(
        inputs,
        outputs,
        ridge=0.0,
        covariance_floor=0.0,
    )

    for fit in (eager, jitted):
        np.testing.assert_allclose(
            fit.affine.coefficients,
            [[2.0]],
            atol=ATOL,
        )
        np.testing.assert_allclose(
            fit.affine.bias,
            [1.0],
            atol=ATOL,
        )
        np.testing.assert_allclose(
            fit.covariance,
            [[0.0]],
            atol=ATOL,
        )


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
    ).T

    fit = gaussian_fit.linear_from_samples_weighted(
        inputs,
        outputs,
        weights,
        ridge=0.0,
        covariance_floor=0.0,
    )

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
    ).T
    inputs = jnp.array([[0.0], [1.0], [2.0], [3.0]])
    outputs = jnp.array([[1.0], [3.0], [8.0], [7.0]])

    @jax.jit
    def run(observations, means, covariances, weights, inputs, outputs):
        log_likelihoods = Gaussian(
            mean=means, covariance=covariances
        ).log_prob_broadcast(observations)
        weighted_fit = gaussian_fit.from_samples_weighted(
            observations,
            weights,
            covariance_floor=0.0,
        )
        linear_fit = gaussian_fit.linear_from_samples(
            inputs,
            outputs,
            ridge=0.0,
            covariance_floor=0.0,
        )
        weighted_linear_fit = gaussian_fit.linear_from_samples_weighted(
            inputs,
            outputs,
            weights,
            ridge=0.0,
            covariance_floor=0.0,
        )

        return (
            log_likelihoods,
            weighted_fit,
            linear_fit,
            weighted_linear_fit,
        )

    result = run(observations, means, covariances, weights, inputs, outputs)
    jax.block_until_ready(result)

    assert result[0].shape == (2, 4)
    assert result[2].affine.coefficients.shape == (1, 1)
    assert result[3].affine.coefficients.shape == (2, 1, 1)


def test_weighted_fits_support_multi_axis_batches():
    values = jnp.arange(15.0).reshape(5, 3)
    weights = jnp.arange(30.0).reshape(2, 3, 5) + 1.0

    fit = gaussian_fit.from_samples_weighted(
        values,
        weights,
        covariance_floor=0.0,
    )

    assert fit.batch_shape == (2, 3)
    for index in np.ndindex((2, 3)):
        expected = gaussian_fit.from_samples_weighted(
            values,
            weights[index],
            covariance_floor=0.0,
        )
        np.testing.assert_allclose(fit.mean[index], expected.mean, atol=ATOL)
        np.testing.assert_allclose(
            fit.covariance[index],
            expected.covariance,
            atol=ATOL,
        )


def test_linear_gaussian_conditional_broadcasts_covariance():
    num_steps = 5
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
    with pytest.raises(ValueError, match='target shape must begin with'):
        model.conditional(jnp.swapaxes(inputs, 0, 1))


def test_fit_linear_preserves_structured_input_shape():
    inputs = jnp.array(
        [
            [[0.0], [0.0]],
            [[1.0], [0.0]],
            [[0.0], [1.0]],
            [[1.0], [1.0]],
        ]
    )  # (T, L, I)

    outputs = 2.0 * inputs[:, 0] - 3.0 * inputs[:, 1] + 1.0

    fit = gaussian_fit.linear_from_samples(
        inputs,
        outputs,
        ridge=0.0,
        covariance_floor=0.0,
    )

    np.testing.assert_allclose(
        fit.affine.coefficients,
        [[[2.0], [-3.0]]],
        atol=ATOL,
    )

    np.testing.assert_allclose(
        fit.affine.bias,
        [1.0],
        atol=ATOL,
    )

    assert fit.input_shape == (2, 1)


def test_paired_gaussian_assembles_joint_moments():
    paired = PairedGaussian(
        left=Gaussian(
            mean=jnp.array([1.0]),
            covariance=jnp.array([[2.0]]),
        ),
        right=Gaussian(
            mean=jnp.array([3.0]),
            covariance=jnp.array([[4.0]]),
        ),
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
    distributions = Gaussian(
        mean=jnp.array([[0.0], [2.0]]),
        covariance=jnp.array([[[1.0]], [[3.0]]]),
    )

    fit = gaussian_fit.from_moment_match(distributions)

    np.testing.assert_allclose(fit.mean, [1.0], atol=ATOL)
    np.testing.assert_allclose(fit.covariance, [[3.0]], atol=ATOL)


def test_paired_moment_match_combines_cross_covariance():
    distributions = PairedGaussian(
        left=Gaussian(
            mean=jnp.array([[0.0], [2.0]]),
            covariance=jnp.array([[[1.0]], [[1.0]]]),
        ),
        right=Gaussian(
            mean=jnp.array([[1.0], [5.0]]),
            covariance=jnp.array([[[2.0]], [[2.0]]]),
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
    inputs = Gaussian(
        mean=jnp.array([[0.0], [2.0]]),
        covariance=jnp.array([[[1.0]], [[1.0]]]),
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
        left=Gaussian(
            mean=means[:, :2],
            covariance=joint_covariances[:, :2, :2],
        ),
        right=Gaussian(
            mean=means[:, 2:],
            covariance=joint_covariances[:, 2:, 2:],
        ),
        cross_covariance=joint_covariances[:, 2:, :2],
    )

    # The input distributions are valid.
    assert jnp.isfinite(jnp.linalg.cholesky(distributions.covariance)).all()

    fit = gaussian_fit.paired_from_moment_match(distributions)

    # Moment matching should preserve positive definiteness.
    assert jnp.isfinite(jnp.linalg.cholesky(fit.covariance)).all()


@pytest.mark.parametrize(
    'batch, query, weighted',
    [
        ((2, 3), (), False),
        ((2, 3), (), True),
        ((), (3,), True),
        ((2, 3), (2, 1), True),
    ],
)
@pytest.mark.parametrize('paired', [False, True])
def test_moment_matching_preserves_structural_and_query_batches(
    batch, query, weighted, paired
):
    shape = batch + (4,)
    means = jnp.arange(np.prod(shape) * 2, dtype=float).reshape(shape + (2,)) / 20
    covariances = jnp.broadcast_to(jnp.array([[2.0, 0.3], [0.3, 1.0]]), shape + (2, 2))
    distributions = Gaussian(means, covariances)
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
            Gaussian(means[..., :1], covariances[..., :1, :1]),
            Gaussian(means[..., 1:], covariances[..., 1:, 1:]),
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


@pytest.mark.parametrize('shape', [(), (4,), (1, 3, 4), (2, 3, 5)])
def test_moment_matching_rejects_misaligned_weights(shape):
    distributions = Gaussian(jnp.zeros((2, 3, 4, 1)), jnp.ones((2, 3, 4, 1, 1)))
    with pytest.raises(ValueError, match='weights'):
        gaussian_fit.from_moment_match(distributions, jnp.ones(shape))


@pytest.mark.parametrize('linear', [False, True])
@pytest.mark.parametrize('shape', [(4, 1), (3,)])
def test_grouped_fit_rejects_malformed_assignments(linear, shape):
    values = jnp.ones((4, 1))
    assignments = jnp.zeros(shape, dtype=int)
    options = {'covariance_floor': 0.0}
    with pytest.raises(ValueError, match='assignments'):
        if linear:
            gaussian_fit.linear_from_samples_grouped(
                values, values, assignments, 2, ridge=1e-5, **options
            )
        else:
            gaussian_fit.from_samples_grouped(values, assignments, 2, **options)
