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
    D,
    K,
    L,
    O,
    T,
    _assert_tree_close,
    _component,
    _continuous_posterior,
)
from xxm.arhmm.data import ARDataset
from xxm.core.data import Dataset, Sequences, WeightedObservations
from xxm.core.lengths import Lengths


@pytest.fixture
def component(kind):
    return _component(kind)


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
    data = (
        ARDataset.from_dataset(
            Dataset.from_padded(observations, Lengths.full(FIT_B, T)), L
        )
        if is_ar
        else WeightedObservations(
            observations,
            jnp.ones(FIT_B + (T,), dtype=bool),
            Lengths(jnp.full(FIT_B, T)),
        )
    )
    likelihood = component.log_likelihoods(
        data.weighted_observations() if isinstance(data, ARDataset) else data
    )
    assert likelihood.shape == FIT_B + (T - L if is_ar else T, K)
    for index in FIT_INDICES:
        standalone = component.select(index)
        data_i = (
            ARDataset.from_array(observations[index], L)
            if is_ar
            else WeightedObservations.from_array(observations[index])
        )
        np.testing.assert_allclose(
            likelihood[index],
            standalone.log_likelihoods(
                data_i.weighted_observations()
                if isinstance(data_i, ARDataset)
                else data_i
            ),
            atol=2e-5,
        )
    states = Sequences.from_padded(
        jnp.zeros(FIT_B + (T,), dtype=jnp.int32), Lengths.full(FIT_B, T)
    )
    sampled = component.sample(jax.random.key(3), states)
    assert sampled.values.shape == FIT_B + (T, O)
    np.testing.assert_array_equal(sampled.lengths.values, states.lengths.values)
    if is_ar:
        assert isinstance(data, ARDataset)
        assert component.conditional(data.predictors).batch_shape == FIT_B + (T - L, K)
        assert component.sample_continuation(
            jax.random.key(3), states, observations[..., :L, :]
        ).values.shape == FIT_B + (T, O)
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
    data = WeightedObservations(
        observations,
        jnp.ones(FIT_B + (T,), dtype=bool),
        Lengths(jnp.full(FIT_B, T)),
    )
    likelihood = component.log_likelihood(data, latents)
    assert likelihood.shape == FIT_B
    assert component.conditional(latents).batch_shape == FIT_B + (T,)
    latent_sequences = Sequences.from_padded(latents, posterior.lengths)
    sampled = component.sample(jax.random.key(4), latent_sequences)
    assert sampled.values.shape == FIT_B + (T, O)
    np.testing.assert_array_equal(sampled.lengths.values, posterior.lengths.values)
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
