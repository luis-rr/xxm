import jax
import jax.numpy as jnp
import numpy as np

from xxm.discrete_chain import DiscreteChainMarginals
from xxm.hmm.core import LatentInitialModel, LatentTransitionModel, Model
from xxm.hmm.emissions import GaussianEmissions, PoissonEmissions
from xxm.hmm.learning import em_step

jax.config.update('jax_enable_x64', True)

RTOL = 1e-7
ATOL = 1e-6


def make_posterior(
    state_marginals: jnp.ndarray,
    pair_marginals: jnp.ndarray,
) -> DiscreteChainMarginals:
    T, K = state_marginals.shape

    return DiscreteChainMarginals(
        # forward_probs=jnp.zeros((T, K)),
        # backward_probs=jnp.zeros((T, K)),
        # log_scaling_factors=jnp.zeros(T),
        state_marginals=state_marginals,
        pair_marginals=pair_marginals,
        log_normalizer=jnp.log(jnp.sum(state_marginals[0])),
    )


def test_m_step_initial_probs():
    posterior = make_posterior(
        state_marginals=jnp.array(
            [
                [0.8, 0.2],
                [0.5, 0.5],
                [0.1, 0.9],
            ]
        ),
        pair_marginals=jnp.zeros((2, 2, 2)),
    )

    initial = LatentInitialModel(initial_probs=jnp.array([0.5, 0.5]))

    result = initial.fit_params(posterior=posterior)

    np.testing.assert_allclose(result.initial_probs, [0.8, 0.2])


def test_m_step_transition_probs():
    state_marginals = jnp.array(
        [
            [0.8, 0.2],
            [0.5, 0.5],
            [0.3, 0.7],
        ]
    )

    pair_marginals = jnp.array(
        [
            [
                [0.6, 0.2],
                [0.1, 0.1],
            ],
            [
                [0.3, 0.2],
                [0.1, 0.4],
            ],
        ]
    )

    posterior = make_posterior(
        state_marginals,
        pair_marginals,
    )

    transitions = LatentTransitionModel(transition_probs=jnp.array([[0.5, 0.5], [0.5, 0.5]]))

    result = transitions.fit_params(posterior)

    expected = np.array(
        [
            [(0.6 + 0.3) / (0.8 + 0.5), (0.2 + 0.2) / (0.8 + 0.5)],
            [(0.1 + 0.1) / (0.2 + 0.5), (0.1 + 0.4) / (0.2 + 0.5)],
        ]
    )

    np.testing.assert_allclose(result.transition_probs, expected)
    np.testing.assert_allclose(result.transition_probs.sum(axis=1), 1.0)


def test_poisson_log_likelihoods():
    emissions = PoissonEmissions(
        rates=jnp.array(
            [
                [1.0],
                [2.0],
            ]
        )
    )

    observations = jnp.array(
        [
            [0.0],
            [1.0],
            [2.0],
        ]
    )

    result = emissions.log_likelihoods(observations)

    # log P(x | lambda) = x log(lambda) - lambda - log(x!)
    expected = np.array(
        [
            [-1.0, -2.0],
            [-1.0, np.log(2.0) - 2.0],
            [-1.0 - np.log(2.0), 2 * np.log(2.0) - 2.0 - np.log(2.0)],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_poisson_m_step():
    observations = jnp.array(
        [
            [1.0],
            [3.0],
            [5.0],
        ]
    )

    posterior = make_posterior(
        state_marginals=jnp.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
        pair_marginals=jnp.zeros((2, 2, 2)),
    )

    emissions = PoissonEmissions(
        rates=jnp.ones((2, 1)),
    )

    result = emissions.fit_params(observations, posterior)

    expected = np.array(
        [
            [(1.0 + 0.5 * 3.0) / 1.5],
            [(0.5 * 3.0 + 5.0) / 1.5],
        ]
    )

    np.testing.assert_allclose(result.rates, expected)


def test_gaussian_log_likelihoods():
    emissions = GaussianEmissions(
        means=jnp.array(
            [
                [0.0],
                [2.0],
            ]
        ),
        covariances=jnp.array(
            [
                [[1.0]],
                [[1.0]],
            ]
        ),
    )

    observations = jnp.array(
        [
            [0.0],
            [1.0],
        ]
    )

    result = emissions.log_likelihoods(observations)

    c = -0.5 * np.log(2 * np.pi)

    expected = np.array(
        [
            [c, c - 2.0],
            [c - 0.5, c - 0.5],
        ]
    )

    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_gaussian_m_step():
    observations = jnp.array(
        [
            [0.0],
            [2.0],
            [4.0],
        ]
    )

    posterior = make_posterior(
        state_marginals=jnp.array(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
        pair_marginals=jnp.zeros((2, 2, 2)),
    )

    emissions = GaussianEmissions(
        means=jnp.zeros((2, 1)),
        covariances=jnp.ones((2, 1, 1)),
    )

    result = emissions.fit_params(observations, posterior)

    expected_means = np.array(
        [
            [(0.0 + 0.5 * 2.0) / 1.5],
            [(0.5 * 2.0 + 4.0) / 1.5],
        ]
    )

    np.testing.assert_allclose(
        result.means,
        expected_means,
        atol=1e-6,
    )

    # Compute expected weighted variances independently.
    gamma = np.asarray(posterior.state_marginals)
    x = np.asarray(observations[:, 0])

    expected_variances = []
    for k in range(2):
        mean = expected_means[k, 0]
        variance = np.sum(gamma[:, k] * (x - mean) ** 2) / gamma[:, k].sum()
        expected_variances.append([[variance]])

    np.testing.assert_allclose(
        result.covariances,
        expected_variances,
        atol=1e-6,
    )


def test_em_step_is_jit_compatible():
    model = Model(
        initial=LatentInitialModel(initial_probs=jnp.array([0.5, 0.5])),
        transitions=LatentTransitionModel(
            transition_probs=jnp.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                ]
            )
        ),
        emissions=PoissonEmissions(
            rates=jnp.array(
                [
                    [1.0],
                    [5.0],
                ]
            ),
        ),
    )

    observations = jnp.array(
        [
            [0.0],
            [1.0],
            [4.0],
            [5.0],
        ]
    )

    eager_model, eager_log_likelihood = em_step(model, observations)
    jitted_model, jitted_log_likelihood = jax.jit(em_step)(model, observations)

    np.testing.assert_allclose(
        jitted_model.initial.initial_probs,
        eager_model.initial.initial_probs,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jitted_model.transitions.transition_probs,
        eager_model.transitions.transition_probs,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jitted_model.emissions.rates,
        eager_model.emissions.rates,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        jitted_log_likelihood,
        eager_log_likelihood,
        atol=1e-6,
    )
