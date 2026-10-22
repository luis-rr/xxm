import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xxm
from xxm import arhmm, hmm
from xxm.arhmm.data import ARDataset
from xxm.core.data import Dataset, DatasetCollection


def core(family):
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
    model = core(family)
    observations = jnp.array([[0.0], [1.0], [2.0], [0.0], [1.0]])[: 2 + num_targets]
    values = np.asarray(observations[:, 0])
    coefficients = np.asarray(model.states().affine.coefficients[:, 0, :, 0])
    bias = np.asarray(model.states().affine.bias[:, 0])
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

    data = ARDataset.from_dataset(Dataset.from_array(observations), num_lags=2)
    np.testing.assert_allclose(
        model.core.emissions.get().log_likelihoods(data.weighted_observations()),
        log_likelihoods,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model.core.emissions.get()
        .compute_potential(data.weighted_observations())
        .log_values,
        log_likelihoods,
        atol=1e-6,
    )
    dataset = DatasetCollection.from_datasets((Dataset.from_array(observations),))
    assert model.states_conditional(dataset).dist.batch_shape == (1, num_targets, 2)

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

    inferred = model.infer(dataset)
    np.testing.assert_allclose(
        inferred.posterior.state_probs, expected_states[None], atol=2e-6
    )
    np.testing.assert_allclose(
        inferred.posterior.pair_probs, expected_pairs[None], atol=2e-6
    )
    np.testing.assert_array_equal(
        model.most_likely_states(inferred.posterior), expected_states.argmax(-1)[None]
    )
    np.testing.assert_allclose(
        model.observation_mean(dataset, inferred.posterior),
        (expected_states * conditional_mean).sum(-1, keepdims=True)[None],
        atol=2e-6,
    )

    permutation = jnp.array([1, 0])
    permuted = model.permute_states(permutation)
    permuted_inferred = permuted.infer(dataset)
    np.testing.assert_allclose(
        permuted_inferred.posterior.state_probs,
        expected_states[None, :, ::-1],
        atol=2e-6,
    )
    for actual, expected in zip(
        jax.tree.leaves(permuted.states()), jax.tree.leaves(model.states()), strict=True
    ):
        np.testing.assert_array_equal(actual, expected[permutation])


@pytest.mark.parametrize('facade', [arhmm.GaussianARHMM, arhmm.PoissonARHMM])
def test_fitting_and_multiple_initializations(facade):
    # This is a facade/wiring test. One short sequence and one EM step exercise
    # k-means initialization plus fit without duplicating the explicit em_step
    # calculation already covered by lower-level/integration tests.
    observations = jnp.array([[0.0], [1.0], [2.0], [1.0]])
    data = DatasetCollection.from_datasets((Dataset.from_array(observations),))
    initialized = facade.via_kmeans(
        jax.random.key(2), data, num_states=2, num_lags=1
    ).get()
    assert initialized.states().affine.coefficients.shape == (2, 1, 1, 1)

    fit = initialized.fit(data, num_iters=1, progress=False)

    assert fit.inferred.posterior.state_probs.shape == (1, 3, 2)
    assert fit.inferred.model.states().affine.coefficients.shape == (2, 1, 1, 1)
    assert fit.objective_trace.shape == (2,)


@pytest.mark.parametrize('family', ['gaussian', 'poisson'])
def test_continuation_uses_chronological_history(family):
    model = core(family)
    key = jax.random.key(7)
    history = jnp.array([[2.0], [1.0]])
    states, actual = model.sample(key, 3, initial_history=history)
    states = states.get()
    actual = actual.get()
    assert states.shape == (3,)
    assert actual.shape == (3,)
    _, observation_key = jax.random.split(key)
    expected = []
    recent, older = history[1], history[0]
    for state in states:
        observation_key, sample_key = jax.random.split(observation_key)
        coefficients = model.states().affine.coefficients[state, 0, :, 0]
        mean = (
            model.states().affine.bias[state]
            + coefficients[0] * recent
            + coefficients[1] * older
        )
        if isinstance(model, arhmm.GaussianARHMM):
            covariance = model.states().covariance[state]
            value = jax.random.multivariate_normal(sample_key, mean, covariance)
        else:
            value = jax.random.poisson(sample_key, jnp.exp(mean))
        expected.append(value)
        older, recent = recent, value
    np.testing.assert_allclose(actual, jnp.stack(expected).squeeze(), atol=1e-6)

    autonomous = model.sample(key, 3)
    zero_history = model.sample(key, 3, initial_history=jnp.zeros((2, 1)))
    for a, b in zip(autonomous, zero_history, strict=True):
        np.testing.assert_array_equal(a.values.squeeze(), b.values.squeeze())


def test_public_family_boundary():
    assert xxm.GaussianARHMM is arhmm.GaussianARHMM
    assert xxm.PoissonARHMM is arhmm.PoissonARHMM
    assert not hasattr(hmm, 'GaussianARHMM')
    assert not hasattr(hmm, 'PoissonARHMM')
