import math

import jax
import jax.numpy as jnp
import numpy as np

from xxm.arhmm.emissions import AREmissions
from xxm.core.affine import Affine
from xxm.core.dists.discrete import Categorical, DiscreteChainMarginals
from xxm.core.dists.gaussian import (
    Gaussian,
    GaussianChainMarginals,
    GaussianNoise,
    LinearGaussian,
)
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.emissions import continuous, discrete
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
    StateConditionedGaussian,
)
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics
from xxm.core.lengths import Lengths

B = (2, 3)
K, T, D, O, L = 2, 3, 2, 2, 1
CHECK_INDICES = ((0, 0), (1, 2))
FIT_SELECTION = (slice(None), 0)
FIT_B = (2,)
FIT_INDICES = (0, 1)


def _values(shape):
    return jnp.sin(jnp.arange(math.prod(shape), dtype=jnp.float32)).reshape(shape)


def _gaussian(batch, dim=D):
    mean = _values(batch + (dim,))
    return Gaussian(mean, GaussianNoise.identity(dim, batch, dtype=mean.dtype))


def _linear(batch, output_dim=D, input_shape=(D,)):
    affine = Affine(
        0.1 * _values(batch + (output_dim,) + input_shape),
        0.2 * _values(batch + (output_dim,)),
    )
    return LinearGaussian(
        affine, GaussianNoise.identity(output_dim, batch, dtype=affine.dtype)
    )


def _categorical(batch):
    return Categorical(jax.nn.softmax(_values(batch + (K,)), axis=-1))


def _discrete_posterior(batch=B, num_steps=T):
    probs = jax.nn.softmax(_values(batch + (num_steps, K)), axis=-1)
    return DiscreteChainMarginals(
        probs,
        probs[..., :-1, :, None] * probs[..., 1:, None, :],
        Lengths(jnp.full(batch, num_steps)),
    )


def _continuous_posterior(batch=B, num_steps=T):
    means = _values(batch + (num_steps, D))
    return GaussianChainMarginals(
        means,
        jnp.broadcast_to(
            0.2 * jnp.eye(D, dtype=means.dtype),
            batch + (num_steps, D, D),
        ),
        jnp.zeros(batch + (num_steps - 1, D, D), dtype=means.dtype),
        Lengths(jnp.full(batch, num_steps)),
    )


def _assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_allclose(left, right, rtol=2e-4, atol=2e-5)


def _component(kind):
    if kind == 'categorical_initial':
        return CategoricalInitial(_categorical(B))
    if kind == 'categorical_transitions':
        return CategoricalTransitions(_categorical(B + (K,)))
    if kind == 'gaussian_initial':
        return GaussianInitial(_gaussian(B))
    if kind == 'gaussian_dynamics':
        return GaussianLinearDynamics(_linear(B))
    if kind == 'state_gaussian':
        return StateConditionedGaussian(_gaussian(B + (K,)))
    if kind == 'switching':
        return GaussianLinearSwitchingDynamics(_linear(B + (K,)))
    if kind == 'discrete_gaussian':
        return discrete.GaussianEmissions(_gaussian(B + (K,), O))
    if kind == 'discrete_poisson':
        return discrete.PoissonEmissions(Poisson(_values(B + (K, O))))
    if kind == 'continuous_gaussian':
        return continuous.GaussianEmissions(_linear(B, O))
    if kind == 'continuous_poisson':
        return continuous.PoissonEmissions(LinearPoisson(_linear(B, O).affine))
    if kind == 'ar_gaussian':
        return AREmissions(_linear(B + (K,), O, (L, O)))
    if kind == 'ar_poisson':
        return AREmissions(LinearPoisson(_linear(B + (K,), O, (L, O)).affine))
    raise ValueError(kind)
