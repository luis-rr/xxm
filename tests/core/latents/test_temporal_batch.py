"""Batch-major sampling and temporal operations, with unequal axis lengths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.data import Sequences
from xxm.core.dists.gaussian import Gaussian, GaussianNoise
from xxm.core.latents.hermite import HermiteGaussian
from xxm.core.latents.random_walk import GatedGaussianRandomWalk, GaussianRandomWalk
from xxm.core.lengths import Lengths, LengthsSpec


def gaussian(batch=(2, 3), dim=2):
    return Gaussian(
        jnp.arange(np.prod(batch) * dim, dtype=float).reshape(batch + (dim,)),
        GaussianNoise.identity(dim, batch),
    )


def hermite():
    # The common-batch test is about structural operations, so construct a
    # valid Hermite Gaussian directly from its flattened coefficient Gaussian.
    return HermiteGaussian.from_coefficient_gaussian(
        Gaussian(jnp.zeros((2, 3, 4)), GaussianNoise.identity(4, (2, 3))),
        variable_dim=2,
    )


@pytest.mark.parametrize('steps', [0, 1, 5])
def test_random_walk_time_follows_batch(steps):
    rw = GaussianRandomWalk(
        gaussian(),
        gaussian()._replace(
            loc=jnp.ones((2, 3, 2)),
        ),
    )
    key = jax.random.key(19)

    ls = LengthsSpec.from_params(
        Lengths.full(rw.batch_shape, steps),
    )

    actual = jax.jit(lambda k, ls: rw.sample(k, ls))(
        key,
        ls,
    )

    assert actual.values.shape == (2, 3, steps, 2)
    assert actual.lengths.batch_shape == rw.batch_shape
    np.testing.assert_array_equal(
        actual.lengths.values,
        jnp.full(rw.batch_shape, steps),
    )

    if steps:
        ki, ke = jax.random.split(key)

        initial = rw.initial.sample(ki)

        np.testing.assert_allclose(
            actual.values[..., 0, :],
            initial,
        )

        increments = rw.innovation.sample(
            ke,
            (steps - 1,),
        )

        np.testing.assert_allclose(
            actual.values[..., 1:, :],
            initial[..., None, :] + jnp.cumsum(increments, axis=-2),
        )


@pytest.mark.parametrize('steps', [0, 1, 5])
def test_gated_random_walk_aligns_gates_and_replicates(steps):
    # Structural batch B=(3,), intrinsic gates K=2.
    initial = gaussian(batch=(3, 2), dim=1)
    active = initial._replace(loc=initial.mean + 10.0)
    inactive = initial._replace(loc=initial.mean - 10.0)

    rw = GatedGaussianRandomWalk(
        initial,
        active,
        inactive,
    )

    ls = LengthsSpec.from_params(
        Lengths.full(rw.batch_shape, steps),
    )

    gates = (
        jnp.arange(3 * max(steps - 1, 0)).reshape(
            3,
            max(steps - 1, 0),
        )
        % 2
    )

    key = jax.random.key(20)
    gate_sequences = Sequences.from_padded(gates, ls.lengths.transitions())

    actual = jax.jit(
        lambda k, ls, gates: rw.sample(
            key=k,
            ls=ls,
            gates=gates,
        )
    )(
        key,
        ls,
        gate_sequences,
    )

    assert rw.batch_shape == (3,)
    assert rw.num_gates == 2

    # Once exposed as a generic LinearGaussian, K is simply part of
    # that object's batch structure.
    assert rw.active_transition_dist().batch_shape == rw.batch_shape + (rw.num_gates,)

    assert actual.values.shape == (3, steps, 2, 1)

    np.testing.assert_array_equal(
        actual.lengths.values,
        jnp.full(rw.batch_shape, steps),
    )

    if steps:
        ki, ke = jax.random.split(key)

        first = initial.sample(ki)[..., None, :]

        mask = gates[..., None, :] == jnp.arange(2).reshape(1, 2, 1)

        increments = Gaussian.from_moments(
            jnp.where(
                mask[..., None],
                active.mean[..., None, :],
                inactive.mean[..., None, :],
            ),
            jnp.broadcast_to(
                initial.covariance[..., None, :, :],
                (
                    3,
                    2,
                    steps - 1,
                    1,
                    1,
                ),
            ),
        ).sample(ke)

        expected = jnp.concatenate(
            (
                first,
                first
                + jnp.cumsum(
                    increments,
                    axis=-2,
                ),
            ),
            axis=-2,
        )

        np.testing.assert_allclose(
            actual.values,
            jnp.moveaxis(expected, -3, -2),
        )

    np.testing.assert_array_equal(
        rw.permute_gates(jnp.array([1, 0])).initial.mean,
        initial.mean[:, ::-1],
    )


def test_hermite_uses_each_sequences_normalized_transition_grid():
    spline = HermiteGaussian.from_coefficient_gaussian(
        gaussian(batch=(3,), dim=1),
        variable_dim=1,
    )

    ls = LengthsSpec(
        Lengths.from_values([4, 2, 1]),
        4,
    ).transitions()

    weights = spline.sample_from_coefficients(
        jnp.ones((3, 1)),
        ls,
    )

    np.testing.assert_array_equal(
        weights.lengths.values,
        [3, 1, 0],
    )
    assert weights.values.shape == (3, 3, 1)

    # With two knots, zero integral means opposite endpoints and a zero midpoint.
    assert abs(float(weights.values[0, 0, 0])) > 0.1
    np.testing.assert_allclose(
        weights.values[0, 1],
        0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        weights.values[0, 2],
        -weights.values[0, 0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        weights.values[1, 0],
        weights.values[0, 0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        weights.values[1, 1:],
        0,
    )
    np.testing.assert_array_equal(
        weights.values[2],
        0,
    )


def test_hermite_function_prior_constructor():
    # Keep explicit coverage of the factory, but use its minimal valid problem.
    spline = HermiteGaussian.from_function_prior(
        variable_dim=1,
        num_knots=2,
        alpha_amp=1.0,
        alpha_smooth=1.0,
    )

    assert spline.batch_shape == ()
    assert spline.variable_dim == 1
    assert spline.num_knots == 2
    assert spline.coefficients.mean.shape == (1,)


@pytest.mark.parametrize(
    'process',
    [
        'random_walk',
        'gated',
        'hermite',
    ],
)
def test_temporal_common_batch_api(process):
    g = gaussian()

    gated_g = gaussian(
        batch=(2, 3, 4),
    )

    obj = {
        'random_walk': lambda: GaussianRandomWalk(
            g,
            g,
        ),
        'gated': lambda: GatedGaussianRandomWalk(
            gated_g,
            gated_g,
            gated_g,
        ),
        'hermite': hermite,
    }[process]()

    assert obj.select((1, 2)).batch_shape == ()

    assert obj.broadcast(
        (1, 4),
        axis=1,
    ).batch_shape == (
        2,
        1,
        4,
        3,
    )

    assert obj.broadcast(
        1,
        axis=1,
    ).squeeze(1).batch_shape == (2, 3)

    assert obj.move_axis(
        0,
        1,
    ).batch_shape == (
        3,
        2,
    )

    assert obj.permute(
        jnp.array([2, 0, 1]),
        axis=1,
    ).batch_shape == (
        2,
        3,
    )

    if process == 'gated':
        assert obj.num_gates == 4

    transformed = jax.jit(
        lambda model: (
            model.broadcast(1, axis=1)
            .squeeze(1)
            .permute(jnp.array([2, 0, 1]), axis=1)
            .move_axis(1, 0)
            .select((slice(None), 1))
        )
    )(obj)
    assert transformed.batch_shape == (3,)
    for actual, original in zip(
        jax.tree.leaves(transformed), jax.tree.leaves(obj), strict=True
    ):
        np.testing.assert_array_equal(actual, original[1, jnp.array([2, 0, 1])])
    if process == 'gated':
        assert transformed.num_gates == 4
        assert transformed.initial.batch_shape == (3, 4)
    elif process == 'hermite':
        assert transformed.variable_dim == obj.variable_dim
        assert transformed.basis == obj.basis
