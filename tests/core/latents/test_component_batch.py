"""Independent structural batches on reusable model components."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.core.component_batch_helpers import (
    FIT_B,
    FIT_INDICES,
    FIT_SELECTION,
    B,
    D,
    K,
    T,
    _assert_tree_close,
    _categorical,
    _component,
    _continuous_posterior,
    _discrete_posterior,
    _gaussian,
)
from xxm.core.data import Sequences
from xxm.core.dists.gaussian import Gaussian, GaussianChainMarginals
from xxm.core.inference.chains.discrete import DiscreteChain
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
    compute_chain_potentials,
)
from xxm.core.latents.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
    StateConditionedGaussian,
)
from xxm.core.lengths import Lengths, LengthsSpec


@pytest.fixture
def component(kind):
    return _component(kind)


@pytest.mark.parametrize('query_shape', [(), (T,), (2, T)])
def test_aligned_state_selection_matches_manual_rows(query_shape):
    states = jnp.arange(math.prod(B + query_shape)).reshape(B + query_shape) % K
    transitions = CategoricalTransitions(_categorical(B + (K,)))
    gaussian = StateConditionedGaussian(_gaussian(B + (K,)))
    selected_probs = transitions.conditional(states)
    selected_gaussian = gaussian.conditional(states)
    assert selected_probs.batch_shape == B + query_shape
    assert selected_gaussian.batch_shape == B + query_shape
    for i, j in np.ndindex(B):
        np.testing.assert_array_equal(
            selected_probs.probs[i, j], transitions.dist.probs[i, j, states[i, j]]
        )
        np.testing.assert_array_equal(
            selected_gaussian.mean[i, j], gaussian.dist.mean[i, j, states[i, j]]
        )
        np.testing.assert_array_equal(
            selected_gaussian.covariance[i, j],
            gaussian.dist.covariance[i, j, states[i, j]],
        )
    assert gaussian.compute_potential().batch_shape == B + (K,)
    assert gaussian.sample(jax.random.key(0), states).shape == B + query_shape + (D,)
    _assert_tree_close(
        jax.jit(lambda model, z: model.conditional(z))(gaussian, states),
        selected_gaussian,
    )


def test_discrete_latent_fit_and_sampling():
    # Multi-axis structural semantics are covered above. Fitting only needs a
    # non-scalar batch to verify independent parameter updates.
    initial = CategoricalInitial(_categorical(B)).select(FIT_SELECTION)
    transitions = CategoricalTransitions(_categorical(B + (K,))).select(FIT_SELECTION)
    posterior = _discrete_posterior().select(FIT_SELECTION)
    for component in (initial, transitions):
        fitted = component.fit_params(posterior)
        assert fitted.batch_shape == FIT_B
        for index in FIT_INDICES:
            _assert_tree_close(
                fitted.select(index),
                component.select(index).fit_params(posterior.select(index)),
            )
    states = initial.sample(jax.random.key(0))
    assert states.shape == FIT_B
    ls = LengthsSpec(Lengths.full(FIT_B, T), T)
    trajectory = transitions.sample(jax.random.key(1), states, ls)
    assert trajectory.values.shape == FIT_B + (T,)
    assert trajectory.event_shape == ()
    np.testing.assert_array_equal(trajectory.lengths.values, ls.lengths.values)
    np.testing.assert_array_equal(trajectory.values[..., 0], states)
    assert bool(jnp.all((trajectory.values >= 0) & (trajectory.values < K)))
    one_step = LengthsSpec(Lengths.full(FIT_B, 1), 1)
    short_trajectory = transitions.sample(jax.random.key(1), states, one_step)
    assert short_trajectory.values.shape == FIT_B + (1,)
    np.testing.assert_array_equal(
        short_trajectory.lengths.values, one_step.lengths.values
    )
    chain = DiscreteChain.from_potentials(
        compute_chain_potentials(
            initial, transitions, LengthsSpec(Lengths.full((), T), T)
        )
    )
    assert chain.transition_probs.shape == FIT_B + (T - 1, K, K)
    assert chain.state_log_potentials.shape == FIT_B + (T, K)


@pytest.mark.parametrize('kind', ['gaussian_initial', 'gaussian_dynamics', 'switching'])
def test_gaussian_latent_sampling_and_potentials(kind, component):
    # Parameter fitting is exercised by the owning model/statistics tests and by
    # the representative batched-fit test below. Here we only need to verify
    # that the wrapper preserves structural batch axes during model operations.
    component = component.select(FIT_SELECTION)
    posterior = _continuous_posterior().select(FIT_SELECTION)

    key = jax.random.key(2)
    if kind == 'gaussian_initial':
        assert component.sample(key).shape == FIT_B + (D,)
    else:
        initial = posterior.means[..., 0, :]
        inputs = Sequences.from_padded(
            jnp.empty(FIT_B + (T, 0), dtype=initial.dtype),
            posterior.lengths,
        )
        if kind == 'switching':
            states = Sequences.from_padded(
                jnp.zeros(FIT_B + (T,), dtype=jnp.int32), posterior.lengths
            )
            sampled = component.sample(
                key,
                initial,
                states,
                inputs=inputs,
            )
        else:
            sampled = component.sample(key, initial, inputs=inputs)
        assert sampled.values.shape == FIT_B + (T, D)
        np.testing.assert_array_equal(sampled.lengths.values, posterior.lengths.values)
        np.testing.assert_array_equal(sampled.values[..., 0, :], initial)
    if kind == 'switching':
        assert component.compute_pair_potentials(
            Sequences.from_padded(
                jnp.empty(FIT_B + (T, 0), dtype=posterior.means.dtype),
                posterior.lengths,
            )
        ).batch_shape == FIT_B + (T - 1, K)


def test_singleton_state_axis_is_not_squeezed():
    component = StateConditionedGaussian(_gaussian((1, 1)))
    assert component.batch_shape == (1,)
    squeezed = component.squeeze()
    assert squeezed.batch_shape == ()
    assert squeezed.num_states == 1
    assert squeezed.dist.batch_shape == (1,)


def test_gaussian_initial_excludes_zero_length_sequences_and_padded_moments():
    posterior = GaussianChainMarginals(
        means=jnp.array([[[jnp.nan], [jnp.nan]], [[2.0], [jnp.nan]]]),
        covariances=jnp.array([[[[jnp.nan]], [[jnp.nan]]], [[[3.0]], [[jnp.nan]]]]),
        cross_covariances=jnp.zeros((2, 1, 1, 1)),
        lengths=Lengths(jnp.array([0, 1])),
    )
    initial = GaussianInitial(Gaussian.from_moments(jnp.zeros(1), jnp.eye(1)))
    fitted = jax.jit(lambda p: initial.fit_params(p, covariance_floor=0))(posterior)
    np.testing.assert_allclose(fitted.dist.mean, [2.0], atol=1e-6)
    np.testing.assert_allclose(fitted.dist.covariance, [[3.0]], atol=1e-6)


def test_gaussian_dynamics_ignore_padded_transition_moments_and_inputs():
    posterior = GaussianChainMarginals(
        means=jnp.array([[0.0], [1.0], [2.0], [jnp.nan]]),
        covariances=jnp.array([[[1.0]], [[1.0]], [[1.0]], [[jnp.nan]]]),
        cross_covariances=jnp.array([[[0.0]], [[0.0]], [[jnp.nan]]]),
        lengths=Lengths(jnp.array(3)),
    )
    dynamics = GaussianLinearDynamics.from_components(
        latent_coefficients=jnp.eye(1),
        input_coefficients=jnp.ones((1, 1)),
        bias=jnp.zeros(1),
        covariance=jnp.eye(1),
    )
    inputs = Sequences.from_padded(
        jnp.array([[0.0], [0.0], [1.0], [jnp.nan]]), posterior.lengths
    )
    actual = jax.jit(lambda p, u: dynamics.fit_params(p, inputs=u))(posterior, inputs)
    cropped = GaussianChainMarginals(
        posterior.means[:3],
        posterior.covariances[:3],
        posterior.cross_covariances[:2],
        posterior.lengths,
    )
    expected = dynamics.fit_params(cropped, inputs=inputs.select_time(start=0, stop=3))
    _assert_tree_close(actual, expected)
