"""Batch-major sampling and temporal operations, with unequal axis lengths."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian
from xxm.core.dists.poisson import Poisson
from xxm.core.latents.hermite import HermiteSpline
from xxm.core.latents.random_walk import GatedGaussianRandomWalk, GaussianRandomWalk


def gaussian(batch=(2, 3), dim=2):
    return Gaussian(
        jnp.arange(np.prod(batch) * dim, dtype=float).reshape(batch + (dim,)),
        jnp.broadcast_to(jnp.eye(dim), batch + (dim, dim)),
    )


@pytest.mark.parametrize('sample_shape', [(), (5,), (5, 4), (0,)])
def test_distribution_samples_follow_batch(sample_shape):
    key = jax.random.key(7)
    g = gaussian()
    shape = g.batch_shape + sample_shape
    expanded = g.broadcast(sample_shape, axis=2)
    np.testing.assert_array_equal(g.sample(key, sample_shape), expanded.sample(key))
    p = Poisson(jnp.log(jnp.arange(1.0, 13.0).reshape(2, 3, 2)))
    np.testing.assert_array_equal(
        p.sample(key, sample_shape), p.broadcast(sample_shape, axis=2).sample(key)
    )
    c = Categorical(jax.nn.one_hot(jnp.arange(6).reshape(2, 3) % 4, 4))
    result = jax.jit(lambda k: c.sample(k, sample_shape))(key)
    assert result.shape == shape
    np.testing.assert_array_equal(
        result,
        jnp.broadcast_to(
            jnp.argmax(c.probs, axis=-1).reshape((2, 3) + (1,) * len(sample_shape)),
            shape,
        ),
    )


@pytest.mark.parametrize('steps', [0, 1, 5])
def test_random_walk_time_follows_batch(steps):
    rw = GaussianRandomWalk(gaussian(), gaussian()._replace(mean=jnp.ones((2, 3, 2))))
    key = jax.random.key(19)
    actual = jax.jit(lambda k: rw.sample(k, steps))(key)
    assert actual.shape == (2, 3, steps, 2)
    if steps:
        ki, ke = jax.random.split(key)
        initial = rw.initial.sample(ki)
        np.testing.assert_allclose(actual[..., 0, :], initial)
        increments = rw.innovation.sample(ke, (steps - 1,))
        np.testing.assert_allclose(
            actual[..., 1:, :], initial[..., None, :] + jnp.cumsum(increments, axis=-2)
        )


@pytest.mark.parametrize('steps', [0, 1, 5])
def test_gated_random_walk_aligns_gates_and_replicates(steps):
    # Structural batch B=(3,), intrinsic gates K=2.
    initial = gaussian(batch=(3, 2), dim=1)
    active = initial._replace(mean=initial.mean + 10.0)
    inactive = initial._replace(mean=initial.mean - 10.0)

    rw = GatedGaussianRandomWalk(
        initial,
        active,
        inactive,
    )

    gates = (
        jnp.arange(3 * max(steps - 1, 0)).reshape(
            3,
            max(steps - 1, 0),
        )
        % 2
    )

    key = jax.random.key(20)

    actual = jax.jit(
        lambda k, gates: rw.sample(
            k,
            steps,
            gates,
        )
    )(
        key,
        gates,
    )

    assert rw.batch_shape == (3,)
    assert rw.num_gates == 2

    # Once exposed as a generic LinearGaussian, K is simply part of
    # that object's batch structure.
    assert rw.active_transition_dist().batch_shape == rw.batch_shape + (rw.num_gates,)

    assert actual.shape == (
        3,
        2,
        steps,
        1,
    )

    if steps:
        ki, ke = jax.random.split(key)

        first = initial.sample(ki)[..., None, :]

        mask = gates[..., None, :] == jnp.arange(2).reshape(1, 2, 1)

        increments = Gaussian(
            mean=jnp.where(
                mask[..., None],
                active.mean[..., None, :],
                inactive.mean[..., None, :],
            ),
            covariance=jnp.broadcast_to(
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

        np.testing.assert_allclose(
            actual,
            jnp.concatenate(
                (
                    first,
                    first
                    + jnp.cumsum(
                        increments,
                        axis=-2,
                    ),
                ),
                axis=-2,
            ),
        )

    np.testing.assert_array_equal(
        rw.permute_gates(jnp.array([1, 0])).initial.mean,
        initial.mean[:, ::-1],
    )

    with pytest.raises(
        ValueError,
        match='gates must have shape',
    ):
        rw.sample(
            key,
            steps,
            gates.T[..., None],
        )


def hermite():
    return HermiteSpline.from_function_prior(
        num_motifs=2, num_knots=3, alpha_amp=1.0, alpha_smooth=1.0, batch_shape=(2, 3)
    )


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

    with pytest.raises(IndexError):
        obj.select((0, 0, 0))

    with pytest.raises(IndexError):
        obj.select(None)

    if process == 'gated':
        assert obj.num_gates == 4
