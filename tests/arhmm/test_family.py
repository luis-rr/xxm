import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xxm
from xxm import arhmm, hmm
from xxm.arhmm.data import ARObservations
from xxm.arhmm.learning import em_step


def _model(family):
    params = {
        'initial_probs': jnp.array([0.6, 0.4]),
        'transition_probs': jnp.array([[0.8, 0.2], [0.3, 0.7]]),
        'emission_coefficients': jnp.array([[[[0.2], [-0.1]]], [[[0.1], [0.15]]]]),
        'emission_bias': jnp.array([[0.1], [-0.2]]),
    }
    if family == 'gaussian':
        return arhmm.GaussianARHMM.from_params(
            **params, emission_covariances=jnp.array([[[0.7]], [[1.2]]])
        )
    return arhmm.PoissonARHMM.from_params(**params)


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
@pytest.mark.parametrize('num_targets', [1, 3])
def test_exact_posterior_matches_enumerated_paths(family, num_targets):
    model = _model(family)
    observations = jnp.array([[0.0], [1.0], [2.0], [0.0], [1.0]])[: 2 + num_targets]
    values = np.asarray(observations[:, 0])
    coefficients = np.asarray(model.states.affine.coefficients[:, 0, :, 0])
    bias = np.asarray(model.states.affine.bias[:, 0])
    means = np.array(
        [
            bias
            + coefficients[:, 0] * values[t - 1]
            + coefficients[:, 1] * values[t - 2]
            for t in range(2, len(values))
        ]
    )
    targets = values[2:, None]
    if family == 'gaussian':
        variance = np.array([0.7, 1.2])
        log_likelihoods = -0.5 * (
            np.log(2 * np.pi * variance) + (targets - means) ** 2 / variance
        )
        conditional_mean = means
    else:
        log_likelihoods = (
            targets * means
            - np.exp(means)
            - np.array([math.lgamma(float(y) + 1) for y in values[2:]])[:, None]
        )
        conditional_mean = np.exp(means)

    data = ARObservations.from_observations(observations, 2)
    np.testing.assert_allclose(
        model._model.emissions.log_likelihoods(data), log_likelihoods, atol=1e-6
    )
    np.testing.assert_allclose(
        model._model.emissions.compute_potential(data).log_values,
        log_likelihoods,
        atol=1e-6,
    )
    assert model.states_conditional(observations).batch_shape == (num_targets, 2)

    paths = list(itertools.product(range(2), repeat=num_targets))
    initial = np.array([0.6, 0.4])
    transitions = np.array([[0.8, 0.2], [0.3, 0.7]])
    log_weights = np.array(
        [
            np.log(initial[path[0]])
            + sum(np.log(transitions[a, b]) for a, b in itertools.pairwise(path))
            + sum(log_likelihoods[t, state] for t, state in enumerate(path))
            for path in paths
        ]
    )
    normalizer = np.logaddexp.reduce(log_weights)
    weights = np.exp(log_weights - normalizer)
    expected_states = np.zeros((num_targets, 2))
    expected_pairs = np.zeros((num_targets - 1, 2, 2))
    for path, weight in zip(paths, weights, strict=True):
        for t, state in enumerate(path):
            expected_states[t, state] += weight
        for t, (a, b) in enumerate(itertools.pairwise(path)):
            expected_pairs[t, a, b] += weight

    inferred = model.infer(observations)
    np.testing.assert_allclose(inferred.objective, normalizer, atol=2e-6)
    np.testing.assert_allclose(
        inferred.posterior.state_probs, expected_states, atol=2e-6
    )
    np.testing.assert_allclose(inferred.posterior.pair_probs, expected_pairs, atol=2e-6)
    np.testing.assert_array_equal(
        model.most_likely_states(inferred.posterior), expected_states.argmax(-1)
    )
    np.testing.assert_allclose(
        model.observation_mean(observations, inferred.posterior),
        (expected_states * conditional_mean).sum(-1, keepdims=True),
        atol=2e-6,
    )

    permutation = jnp.array([1, 0])
    permuted = model.permute_states(permutation)
    permuted_inferred = permuted.infer(observations)
    np.testing.assert_allclose(permuted_inferred.objective, normalizer, atol=2e-6)
    np.testing.assert_allclose(
        permuted_inferred.posterior.state_probs,
        expected_states[:, ::-1],
        atol=2e-6,
    )
    for actual, expected in zip(
        jax.tree.leaves(permuted.states), jax.tree.leaves(model.states), strict=True
    ):
        np.testing.assert_array_equal(actual, expected[permutation])


@pytest.mark.parametrize('facade', [arhmm.GaussianARHMM, arhmm.PoissonARHMM])
def test_fitting_and_multiple_initializations(facade):
    observations = jnp.array(
        [[0.0], [1.0], [2.0], [0.0], [1.0], [3.0], [1.0], [2.0], [0.0], [2.0]]
    )
    initialized = facade.via_kmeans(
        jax.random.key(2), observations, num_states=2, num_lags=2
    )
    assert initialized.states.affine.coefficients.shape == (2, 1, 2, 1)
    inferred = arhmm.infer_exact(initialized._model, observations)
    updated = em_step(inferred, observations)
    fit = initialized.fit(observations, num_iters=2, progress=False)
    assert fit.posterior.state_probs.shape == (8, 2)
    assert fit.model.states.affine.coefficients.shape == (2, 1, 2, 1)
    assert fit.objective_trace.shape == (3,)
    np.testing.assert_allclose(fit.objective_trace[1], updated.objective, atol=2e-5)
    for leaf in jax.tree.leaves(fit):
        assert np.isfinite(leaf).all()

    multiple = facade.fit_many(
        (initialized, initialized), observations, num_iters=2, progress=False
    )
    np.testing.assert_allclose(
        multiple.objective_traces,
        np.broadcast_to(fit.objective_trace, (2, 3)),
        atol=2e-5,
    )


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_continuation_uses_chronological_history(family):
    model = _model(family)
    key = jax.random.key(7)
    history = jnp.array([[2.0], [1.0]])
    states, actual = model.sample(key, 3, initial_history=history)
    assert states.shape == (3,)
    assert actual.shape == (3, 1)
    _, observation_key = jax.random.split(key)
    expected = []
    recent, older = history[1], history[0]
    for state in states:
        observation_key, sample_key = jax.random.split(observation_key)
        coefficients = model.states.affine.coefficients[state, 0, :, 0]
        mean = (
            model.states.affine.bias[state]
            + coefficients[0] * recent
            + coefficients[1] * older
        )
        if isinstance(model, arhmm.GaussianARHMM):
            covariance = model.states.covariance[state]
            value = jax.random.multivariate_normal(sample_key, mean, covariance)
        else:
            value = jax.random.poisson(sample_key, jnp.exp(mean))
        expected.append(value)
        older, recent = recent, value
    np.testing.assert_allclose(actual, jnp.stack(expected), atol=1e-6)

    autonomous = model.sample(key, 3)
    zero_history = model.sample(key, 3, initial_history=jnp.zeros((2, 1)))
    for a, b in zip(autonomous, zero_history, strict=True):
        np.testing.assert_array_equal(a, b)


def test_public_family_boundary():
    assert xxm.GaussianARHMM is arhmm.GaussianARHMM
    assert xxm.PoissonARHMM is arhmm.PoissonARHMM
    assert not hasattr(hmm, 'GaussianARHMM')
    assert not hasattr(hmm, 'PoissonARHMM')


@pytest.mark.parametrize('facade', [xxm.GaussianSLDS, xxm.PoissonSLDS])
def test_slds_arhmm_warm_start(facade):
    observations = jnp.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [3.0, 2.0],
            [2.0, 4.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ]
    )
    model = facade.via_arhmm(
        jax.random.key(3),
        observations,
        num_states=2,
        latent_dim=1,
        num_arhmm_iters=1,
        progress=False,
    )
    assert model.num_states == 2
    assert model.latent_dim == 1
    for leaf in jax.tree.leaves(model):
        assert np.isfinite(leaf).all()
