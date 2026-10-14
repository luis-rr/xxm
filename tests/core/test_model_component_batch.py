"""Independent structural batches on reusable model components."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.core.component_batch_helpers import (
    CHECK_INDICES,
    B,
    D,
    K,
    L,
    O,
    T,
    _assert_tree_close,
    _component,
    _continuous_posterior,
    _discrete_posterior,
    _gaussian,
    _linear,
    _values,
)
from xxm.arhmm.data import ARDataset
from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.data import Dataset, Sequences, WeightedObservations
from xxm.core.dists.discrete import DiscreteChainMarginals
from xxm.core.emissions import continuous, discrete
from xxm.core.latents.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
)
from xxm.core.lengths import Lengths


@pytest.fixture
def component(kind):
    return _component(kind)


@pytest.mark.parametrize(
    'kind',
    [
        'categorical_initial',
        'categorical_transitions',
        'gaussian_initial',
        'gaussian_dynamics',
        'state_gaussian',
        'switching',
        'discrete_gaussian',
        'discrete_poisson',
        'continuous_gaussian',
        'continuous_poisson',
        'ar_gaussian',
        'ar_poisson',
    ],
)
def test_structural_operations_preserve_intrinsic_axes(component):
    assert component.batch_shape == B
    assert component.select(1).batch_shape == (3,)
    _assert_tree_close(
        component.select((1, 2)), jax.tree.map(lambda x: x[1, 2], component)
    )

    moved = component.move_axis(-1, 0)
    assert moved.batch_shape == (3, 2)
    _assert_tree_close(moved, jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), component))

    order = jnp.array([2, 0, 1])
    _assert_tree_close(
        component.permute(order, axis=-1),
        jax.tree.map(lambda x: x[:, order], component),
    )
    expanded = component.broadcast((1,), axis=-1)
    assert expanded.batch_shape == B + (1,)
    _assert_tree_close(expanded.squeeze(-1), component)
    _assert_tree_close(expanded.squeeze(), component)

    if component.dist.batch_shape == B + (K,):
        assert moved.dist.batch_shape == (3, 2, K)
        assert expanded.dist.batch_shape == B + (1, K)
        with pytest.raises(IndexError):
            component.select((0, 0, 0))
        with pytest.raises(ValueError):
            component.move_axis(2, 0)
        standalone = component.select((0, 0))
        assert standalone.batch_shape == ()
        assert standalone.squeeze().dist.batch_shape == (K,)


@pytest.mark.parametrize('batch_shape', [(), B])
def test_flatten_round_trip(batch_shape):
    dist = _gaussian(batch_shape + (K,))
    flat = batch.flatten_batch(dist, batch_shape)
    assert flat.batch_shape == (math.prod(batch_shape), K)
    _assert_tree_close(batch.unflatten_batch(flat, batch_shape), dist)


@pytest.mark.parametrize(
    'component_type',
    [
        GaussianInitial,
        GaussianLinearDynamics,
        continuous.GaussianEmissions,
        continuous.PoissonEmissions,
    ],
)
def test_from_latents_matches_independent_sequences(component_type):
    latents = _values(B + (T, D))
    observations = 2.0 + _values(B + (T, O))
    has_observations = component_type in (
        continuous.GaussianEmissions,
        continuous.PoissonEmissions,
    )
    args = (latents, observations) if has_observations else (latents,)
    kwargs = (
        {'inputs': jnp.empty(B + (T, 0), dtype=latents.dtype)}
        if component_type is GaussianLinearDynamics
        else {}
    )
    fitted = component_type.from_latents(*args, **kwargs)
    assert fitted.batch_shape == B
    for i, j in CHECK_INDICES:
        args_i = (
            (latents[i, j], observations[i, j])
            if has_observations
            else (latents[i, j],)
        )
        kwargs_i = {name: values[i, j] for name, values in kwargs.items()}
        _assert_tree_close(
            fitted.select((i, j)), component_type.from_latents(*args_i, **kwargs_i)
        )


@pytest.mark.parametrize(
    'kind',
    [
        'gaussian_initial',
        'gaussian_dynamics',
        'state_gaussian',
        'switching',
        'continuous_gaussian',
        'continuous_poisson',
    ],
)
@pytest.mark.parametrize('global_alignment', [False, True])
def test_alignment_matches_standalone_components(kind, component, global_alignment):
    alignment = Affine(jnp.array([[1.2, 0.1], [0.0, 0.9]]), jnp.array([0.2, -0.1]))
    if not global_alignment:
        alignment = alignment.broadcast(B)
        alignment = alignment._replace(bias=alignment.bias + 0.1 * _values(B + (D,)))
    else:
        alignment = alignment.broadcast(B)
    is_emission = kind.startswith('continuous_')
    aligned = (
        component.compose_input(alignment)
        if is_emission
        else component.align(alignment)
    )
    assert aligned.batch_shape == B
    for i, j in np.ndindex(B):
        standalone = component.select((i, j))
        alignment_i = alignment.select((i, j))
        expected = (
            standalone.compose_input(alignment_i)
            if is_emission
            else standalone.align(alignment_i)
        )
        _assert_tree_close(aligned.select((i, j)), expected)

    @pytest.mark.parametrize(
        'kind',
        [
            'gaussian_initial',
            'gaussian_dynamics',
            'state_gaussian',
            'switching',
            'continuous_gaussian',
            'continuous_poisson',
        ],
    )
    def test_batched_alignment_requires_explicit_alignment_batch(kind, component):
        alignment = Affine(jnp.eye(D), jnp.zeros(D))
        with pytest.raises(ValueError, match='shapes must match'):
            if kind.startswith('continuous_'):
                component.compose_input(alignment)
            else:
                component.align(alignment)


@pytest.mark.parametrize(
    'kind',
    [
        'categorical_initial',
        'categorical_transitions',
        'state_gaussian',
        'switching',
        'discrete_gaussian',
        'discrete_poisson',
        'ar_gaussian',
        'ar_poisson',
    ],
)
def test_state_permutation_preserves_structural_batch(kind, component):
    order = jnp.array([1, 0])
    permuted = component.permute_states(order)
    assert permuted.batch_shape == B
    for i, j in np.ndindex(B):
        _assert_tree_close(
            permuted.select((i, j)), component.select((i, j)).permute_states(order)
        )
    if kind == 'categorical_transitions':
        np.testing.assert_array_equal(
            permuted.dist.probs, component.dist.probs[..., order, :][..., order]
        )


@pytest.mark.parametrize(
    'kind',
    ['discrete_gaussian', 'discrete_poisson', 'ar_gaussian', 'ar_poisson', 'switching'],
)
def test_unoccupied_states_retain_each_batchs_parameters(kind, component):
    num_steps = T - L if kind.startswith('ar_') else T
    active = jnp.arange(math.prod(B)).reshape(B) % K
    state_probs = jnp.broadcast_to(
        jax.nn.one_hot(active, K)[..., None, :], B + (num_steps, K)
    )
    posterior = DiscreteChainMarginals(
        state_probs,
        state_probs[..., :-1, :, None] * state_probs[..., 1:, None, :],
        Lengths(jnp.full(B, num_steps)),
    )
    if kind == 'switching':
        fitted = component.fit_params(
            posterior,
            _continuous_posterior(),
            inputs=Sequences.from_padded(jnp.empty((T, 0)), Lengths.full((), T)),
        )
    else:
        observations = 2.0 + _values(B + (T, O))
        data = (
            ARDataset.from_dataset(
                Dataset.from_padded(observations, Lengths.full(B, T)), L
            )
            if kind.startswith('ar_')
            else WeightedObservations(
                observations, jnp.ones(B + (T,), dtype=bool), Lengths(jnp.full(B, T))
            )
        )
        fitted = component.fit_params(
            posterior,
            data.weighted_observations() if isinstance(data, ARDataset) else data,
        )
    unused = np.asarray(jnp.arange(K) != active[..., None])
    for actual, current in zip(
        jax.tree.leaves(fitted), jax.tree.leaves(component), strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(actual)[unused], np.asarray(current)[unused]
        )


def test_representative_batched_fits_are_jittable():
    posterior = _continuous_posterior()
    dynamics = GaussianLinearDynamics(_linear(B))
    inputs = Sequences.from_padded(
        jnp.empty((T, 0), dtype=posterior.means.dtype), Lengths.full((), T)
    )
    _assert_tree_close(
        jax.jit(lambda model, q, u: model.fit_params(q, inputs=u))(
            dynamics, posterior, inputs
        ),
        dynamics.fit_params(posterior, inputs=inputs),
    )
    emissions = discrete.GaussianEmissions(_gaussian(B + (K,), O))
    observations = _values(B + (T, O))
    data = WeightedObservations(
        observations, jnp.ones(B + (T,), dtype=bool), Lengths(jnp.full(B, T))
    )
    states = _discrete_posterior()
    _assert_tree_close(
        jax.jit(lambda model, y, q: model.fit_params(q, y))(emissions, data, states),
        emissions.fit_params(states, data),
    )
