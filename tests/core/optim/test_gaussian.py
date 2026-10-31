import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.dists.gaussian import FullGaussian
from xxm.core.optim import gaussian_full as gaussian_full_fit

ATOL = 1e-5


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

    fit = gaussian_full_fit.from_samples_weighted(
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

    eager = gaussian_full_fit.linear_from_samples(
        inputs,
        outputs,
        ridge=0.0,
        covariance_floor=0.0,
    )
    jitted = jax.jit(gaussian_full_fit.linear_from_samples)(
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

    fit = gaussian_full_fit.linear_from_samples_weighted(
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
        log_likelihoods = FullGaussian.from_moments(
            means, covariances
        ).log_prob_broadcast(observations)
        weighted_fit = gaussian_full_fit.from_samples_weighted(
            observations,
            weights,
            covariance_floor=0.0,
        )
        linear_fit = gaussian_full_fit.linear_from_samples(
            inputs,
            outputs,
            ridge=0.0,
            covariance_floor=0.0,
        )
        weighted_linear_fit = gaussian_full_fit.linear_from_samples_weighted(
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

    fit = gaussian_full_fit.from_samples_weighted(
        values,
        weights,
        covariance_floor=0.0,
    )

    assert fit.batch_shape == (2, 3)
    for index in np.ndindex((2, 3)):
        expected = gaussian_full_fit.from_samples_weighted(
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

    fit = gaussian_full_fit.linear_from_samples(
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
