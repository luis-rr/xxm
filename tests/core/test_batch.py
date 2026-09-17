import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.dists.gaussian import Gaussian, PairedGaussian


def _gaussian():
    return Gaussian(
        mean=jnp.arange(12.0).reshape(2, 3, 2),
        covariance=jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2)),
    )


@pytest.mark.parametrize(
    'index',
    [
        None,
        (slice(None), None),
        True,
        np.bool_(False),
        [True, False],
        np.array([[True, False, True], [False, True, False]]),
        (0, 0, 0),
        (Ellipsis, 0, 0, 0),
        (Ellipsis, Ellipsis),
        0.5,
    ],
)
def test_select_rejects_new_axes_boolean_indices_and_intrinsic_indexing(index):
    with pytest.raises(IndexError):
        _gaussian().select(index)


def test_select_expands_only_the_batch_prefix():
    gaussian = _gaussian()
    selected = gaussian.select((1, Ellipsis))
    assert selected.batch_shape == (3,)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[1])
    np.testing.assert_array_equal(selected.covariance, gaussian.covariance[1])

    selected = gaussian.select((Ellipsis, 2))
    assert selected.batch_shape == (2,)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[:, 2])

    selected = gaussian.select((1, 2))
    assert selected.batch_shape == ()
    assert isinstance(selected, Gaussian)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[1, 2])
    with pytest.raises(IndexError):
        selected.select(0)


def test_select_integer_arrays_is_jittable():
    gaussian = _gaussian()
    indices = jnp.array([2, 0])
    selected = jax.jit(lambda g, i: g.select((1, i)))(gaussian, indices)
    np.testing.assert_array_equal(selected.mean, gaussian.mean[1, indices])
    np.testing.assert_array_equal(selected.covariance, gaussian.covariance[1, indices])
    np.testing.assert_array_equal(
        gaussian.select([1, 0]).mean, gaussian.mean[jnp.array([1, 0])]
    )


def test_paired_batch_operations_preserve_all_components():
    left = _gaussian()
    right = left._replace(mean=left.mean + 20.0)
    cross = jnp.arange(24.0).reshape(2, 3, 2, 2)
    paired = PairedGaussian(left, right, cross)

    def transform(p):
        return (
            p.broadcast(1, axis=1)
            .squeeze(1)
            .permute(jnp.array([2, 0, 1]), axis=1)
            .move_axis(1, 0)
            .select((slice(None), 1))
        )

    selected = jax.jit(transform)(paired)
    assert selected.batch_shape == (3,)
    order = jnp.array([2, 0, 1])
    np.testing.assert_array_equal(selected.left.mean, left.mean[1, order])
    np.testing.assert_array_equal(selected.right.mean, right.mean[1, order])
    np.testing.assert_array_equal(selected.left.covariance, left.covariance[1, order])
    np.testing.assert_array_equal(selected.right.covariance, right.covariance[1, order])
    np.testing.assert_array_equal(selected.cross_covariance, cross[1, order])
