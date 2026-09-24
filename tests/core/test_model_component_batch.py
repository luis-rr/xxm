"""Independent structural batches on reusable model components."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.arhmm.data import ARObservations
from xxm.arhmm.emissions import AREmissions
from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.chains.gaussian import GaussianChainMarginals
from xxm.core.data import WeightedObservations
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson, Poisson
from xxm.core.emissions import continuous, discrete
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
    homogeneous_chain,
)
from xxm.core.latents.gaussian import (
    GaussianInitial,
    GaussianLinearDynamics,
    StateConditionedGaussian,
)
from xxm.core.latents.switching import GaussianLinearSwitchingDynamics

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
    return Gaussian(
        mean,
        jnp.broadcast_to(jnp.eye(dim, dtype=mean.dtype), batch + (dim, dim)),
    )


def _linear(batch, output_dim=D, input_shape=(D,)):
    affine = Affine(
        0.1 * _values(batch + (output_dim,) + input_shape),
        0.2 * _values(batch + (output_dim,)),
    )
    return LinearGaussian(
        affine,
        jnp.broadcast_to(
            jnp.eye(output_dim, dtype=affine.dtype), batch + (output_dim, output_dim)
        ),
    )


def _categorical(batch):
    return Categorical(jax.nn.softmax(_values(batch + (K,)), axis=-1))


def _discrete_posterior(batch=B, num_steps=T):
    probs = jax.nn.softmax(_values(batch + (num_steps, K)), axis=-1)
    return DiscreteChainMarginals(
        probs,
        probs[..., :-1, :, None] * probs[..., 1:, None, :],
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
    assert gaussian.compute_potentials().batch_shape == B + (K,)
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
    trajectory = transitions.sample(jax.random.key(1), states, T)
    assert trajectory.shape == FIT_B + (T,)
    np.testing.assert_array_equal(trajectory[..., 0], states)
    assert bool(jnp.all((trajectory >= 0) & (trajectory < K)))
    assert transitions.sample(jax.random.key(1), states, 1).shape == FIT_B + (1,)
    chain = homogeneous_chain(initial, transitions, T)
    assert chain.transition_probs.shape == FIT_B + (T - 1, K, K)
    assert chain.state_log_potentials.shape == FIT_B + (T, K)


def test_discrete_latent_fit_requires_matching_posterior_batch():
    posterior = _discrete_posterior(batch=(4,), num_steps=2)

    with pytest.raises(ValueError, match='shapes must match'):
        CategoricalInitial(_categorical(B)).fit_params(posterior)

    with pytest.raises(ValueError, match='shapes must match'):
        CategoricalTransitions(_categorical(B + (K,))).fit_params(posterior)

    matched = _discrete_posterior(batch=B, num_steps=2)
    assert CategoricalInitial(_categorical(B)).fit_params(matched).batch_shape == B
    assert (
        CategoricalTransitions(_categorical(B + (K,))).fit_params(matched).batch_shape
        == B
    )


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
        if kind == 'switching':
            states = jnp.zeros(FIT_B + (T - 1,), dtype=jnp.int32)
            sampled = component.sample(key, initial, states)
        else:
            sampled = component.sample(
                key, initial, T, inputs=jnp.empty(FIT_B + (T, 0), dtype=initial.dtype)
            )
        assert sampled.shape == FIT_B + (T, D)
        np.testing.assert_array_equal(sampled[..., 0, :], initial)
    if kind == 'switching':
        assert component.compute_pair_potentials().batch_shape == FIT_B + (K,)


def test_gaussian_latent_fit_requires_receiver_batch_prefix():
    posterior = _continuous_posterior(batch=(4,), num_steps=2)

    with pytest.raises(ValueError, match='posterior batch shape must begin with'):
        GaussianInitial(_gaussian(B)).fit_params(posterior, weights=jnp.ones((4, 2)))

    with pytest.raises(ValueError, match='posterior batch shape must begin with'):
        GaussianLinearDynamics(_linear(B)).fit_params(
            posterior,
            inputs=jnp.empty((4, 1, 0), dtype=posterior.means.dtype),
            weights=jnp.ones((4, 1)),
        )

    matched = _continuous_posterior(batch=B, num_steps=2)
    fitted_initial = GaussianInitial(_gaussian(B)).fit_params(
        matched, weights=jnp.ones(B + (2,))
    )
    fitted_dynamics = GaussianLinearDynamics(_linear(B)).fit_params(
        matched,
        inputs=jnp.empty(B + (1, 0), dtype=matched.means.dtype),
        weights=jnp.ones(B + (1,)),
    )
    assert fitted_initial.batch_shape == B
    assert fitted_dynamics.batch_shape == B


@pytest.mark.parametrize(
    'kind', ['discrete_gaussian', 'discrete_poisson', 'ar_gaussian', 'ar_poisson']
)
def test_state_emission_likelihood_and_sampling(kind, component):
    # Fitting is deliberately not repeated here: HMM/AR-HMM tests verify the
    # M-steps, stats tests verify batched Gaussian/Poisson fitting, and the
    # unoccupied-state regression below still drives fit_params through every
    # state-emission wrapper. This test owns batch alignment/evaluation/sampling.
    component = component.select(FIT_SELECTION)
    observations = (
        jnp.arange(math.prod(FIT_B + (T, O))).reshape(FIT_B + (T, O)) % 4
    ).astype(jnp.float32)
    is_ar = kind.startswith('ar_')
    data = ARObservations.from_observations(observations, L) if is_ar else observations
    likelihood = component.log_likelihoods(data)
    assert likelihood.shape == FIT_B + (T - L if is_ar else T, K)
    for index in FIT_INDICES:
        standalone = component.select(index)
        data_i = (
            ARObservations.from_observations(observations[index], L)
            if is_ar
            else observations[index]
        )
        np.testing.assert_allclose(
            likelihood[index], standalone.log_likelihoods(data_i), atol=2e-5
        )
    states = jnp.zeros(FIT_B + (T,), dtype=jnp.int32)
    assert component.sample(jax.random.key(3), states).shape == FIT_B + (T, O)
    if is_ar:
        assert isinstance(data, ARObservations)
        assert component.conditional(data.predictors).batch_shape == FIT_B + (T - L, K)
        assert component.sample_continuation(
            jax.random.key(3), states, observations[..., :L, :]
        ).shape == FIT_B + (T, O)
        with pytest.raises(ValueError, match='initial_history'):
            component.sample_continuation(
                jax.random.key(3), states, observations[0, :L]
            )


@pytest.mark.parametrize('kind', ['continuous_gaussian', 'continuous_poisson'])
def test_continuous_emission_evaluation(kind, component):
    # Fit correctness/JIT is covered in lds/test_lds_emissions.py; the low-level
    # Gaussian and Poisson fitting tests cover multi-axis batches. Keep this
    # generic test focused on wrapper batch alignment and derived operations.
    component = component.select(FIT_SELECTION)
    posterior = _continuous_posterior().select(FIT_SELECTION)
    latents = posterior.means
    observations = (
        jnp.arange(math.prod(FIT_B + (T, O))).reshape(FIT_B + (T, O)) % 3
    ).astype(jnp.float32)
    data = WeightedObservations(observations, jnp.ones(FIT_B + (T,), dtype=bool))
    likelihood = component.log_likelihood(data, latents)
    assert likelihood.shape == FIT_B
    assert component.conditional(latents).batch_shape == FIT_B + (T,)
    assert component.sample(jax.random.key(4), latents).shape == FIT_B + (T, O)
    assert component.invert(observations).shape == FIT_B + (T, D)
    potential = (
        component.compute_potential(data)
        if kind == 'continuous_gaussian'
        else component.compute_local_potential(data, latents)
    )
    assert potential.batch_shape == FIT_B + (T,)
    for index in FIT_INDICES:
        standalone = component.select(index)
        np.testing.assert_allclose(
            likelihood[index],
            standalone.log_likelihood(data.select(index), latents[index]),
            atol=2e-5,
        )
        expected_potential = (
            standalone.compute_potential(data.select(index))
            if kind == 'continuous_gaussian'
            else standalone.compute_local_potential(data.select(index), latents[index])
        )
        _assert_tree_close(potential.select(index), expected_potential)
        _assert_tree_close(
            component.observation_mean(posterior)[index],
            standalone.observation_mean(posterior.select(index)),
        )
        if kind == 'continuous_poisson':
            _assert_tree_close(
                component.expected_log_likelihood(data, posterior)[index],
                standalone.expected_log_likelihood(
                    data.select(index), posterior.select(index)
                ),
            )


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
    )
    if kind == 'switching':
        fitted = component.fit_params(posterior, _continuous_posterior())
    else:
        observations = 2.0 + _values(B + (T, O))
        data = (
            ARObservations.from_observations(observations, L)
            if kind.startswith('ar_')
            else observations
        )
        fitted = component.fit_params(data, posterior)
    unused = np.asarray(jnp.arange(K) != active[..., None])
    for actual, current in zip(
        jax.tree.leaves(fitted), jax.tree.leaves(component), strict=True
    ):
        np.testing.assert_array_equal(
            np.asarray(actual)[unused], np.asarray(current)[unused]
        )


def test_singleton_state_axis_is_not_squeezed():
    component = StateConditionedGaussian(_gaussian((1, 1)))
    assert component.batch_shape == (1,)
    squeezed = component.squeeze()
    assert squeezed.batch_shape == ()
    assert squeezed.num_states == 1
    assert squeezed.dist.batch_shape == (1,)


def test_representative_batched_fits_are_jittable():
    posterior = _continuous_posterior()
    dynamics = GaussianLinearDynamics(_linear(B))
    inputs = jnp.empty(B + (T - 1, 0), dtype=posterior.means.dtype)
    weights = jnp.ones(B + (T - 1,))
    _assert_tree_close(
        jax.jit(lambda model, q, u, w: model.fit_params(q, inputs=u, weights=w))(
            dynamics, posterior, inputs, weights
        ),
        dynamics.fit_params(posterior, inputs=inputs, weights=weights),
    )
    emissions = discrete.GaussianEmissions(_gaussian(B + (K,), O))
    observations = _values(B + (T, O))
    states = _discrete_posterior()
    _assert_tree_close(
        jax.jit(lambda model, y, q: model.fit_params(y, q))(
            emissions, observations, states
        ),
        emissions.fit_params(observations, states),
    )
