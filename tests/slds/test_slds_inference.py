import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.data import Dataset, DatasetCollection, WeightedObservations
from xxm.core.dists.discrete import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.inference.chains.discrete import DiscreteChain
from xxm.core.inference.chains.gaussian import GaussianChain
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.core.lengths import Lengths
from xxm.slds.core import GaussianLinearSwitchingDynamics, Model
from xxm.slds.inference import (
    LatentPotentials,
    infer_variational,
)

ATOL = 1e-5


def _mock_gaussian_emissions(
    noise_covariance: jax.Array,
) -> GaussianEmissions:
    return GaussianEmissions(
        dist=LinearGaussian.from_components(jnp.eye(1), jnp.zeros(1), noise_covariance),
    )


def _single_state_model() -> Model:
    return Model(
        state_initial=CategoricalInitial(
            dist=Categorical(
                probs=jnp.array([1.0]),
            )
        ),
        transitions=CategoricalTransitions(
            dist=Categorical(
                probs=jnp.array([[1.0]]),
            )
        ),
        latent_initial=StateConditionedGaussian(
            dist=Gaussian.from_moments(jnp.array([[0.0]]), jnp.array([[[1.0]]])),
        ),
        dynamics=GaussianLinearSwitchingDynamics(
            dist=LinearGaussian.from_components(
                jnp.array([[[0.8]]]), jnp.array([[0.2]]), jnp.array([[[0.5]]])
            ),
        ),
        emissions=ContinuousEmissionsCollection(
            (_mock_gaussian_emissions(noise_covariance=jnp.array([[0.25]])),),
        ),
    )


def _two_state_model() -> Model:
    return Model(
        state_initial=CategoricalInitial(
            dist=Categorical(
                probs=jnp.array([0.7, 0.3]),
            )
        ),
        transitions=CategoricalTransitions(
            dist=Categorical(
                probs=jnp.array(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                    ]
                )
            )
        ),
        latent_initial=StateConditionedGaussian(
            dist=Gaussian.from_moments(
                jnp.array(
                    [
                        [0.0],
                        [0.0],
                    ]
                ),
                jnp.array(
                    [
                        [[1.0]],
                        [[1.0]],
                    ]
                ),
            ),
        ),
        dynamics=GaussianLinearSwitchingDynamics(
            dist=LinearGaussian.from_components(
                jnp.array(
                    [
                        [[0.8]],
                        [[-0.5]],
                    ]
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                    ]
                ),
                jnp.array(
                    [
                        [[0.5]],
                        [[0.5]],
                    ]
                ),
            ),
        ),
        emissions=ContinuousEmissionsCollection(
            (_mock_gaussian_emissions(noise_covariance=jnp.array([[0.25]])),),
        ),
    )


def test_single_state_slds_matches_gaussian_chain():
    model = _single_state_model()
    observations = jnp.array([[0.2], [1.0]])
    data = Dataset.from_array(observations)

    posterior = infer_variational(
        model,
        DatasetCollection.from_datasets((data,)),
        num_iters=0,
        initial_latents=jnp.zeros((1, observations.shape[0], 1)),
    ).posterior

    state_probs = jnp.ones((observations.shape[0], 1))

    initial_potential = model.latent_initial.compute_potential().weighted_sum(
        state_probs[0], axis=0
    )
    pair_potentials = model.dynamics.compute_pair_potentials(data.inputs).weighted_sum(
        state_probs[1:], axis=1
    )
    observation_potential = model.emissions.get().compute_potential(
        WeightedObservations.from_array(observations)
    )

    expected, _ = (
        GaussianChain.from_pair_potentials(
            initial_potential,
            pair_potentials,
            Lengths(jnp.asarray(observations.shape[0])),
        )
        .add_local_potential(observation_potential)
        .forward_backward()
    )

    np.testing.assert_allclose(
        posterior.discrete.state_probs,
        1.0,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        posterior.continuous.means,
        expected.means[None],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        posterior.continuous.covariances,
        expected.covariances[None],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        posterior.continuous.cross_covariances,
        expected.cross_covariances[None],
        atol=ATOL,
    )


def test_zero_iterations_uses_initial_latents():
    model = _two_state_model()
    observations = jnp.zeros((1, 1))
    data = Dataset.from_array(observations)
    initial_latents = jnp.zeros((1, observations.shape[0], 1))

    posterior = infer_variational(
        model,
        DatasetCollection.from_datasets((data,)),
        initial_latents=initial_latents,
        num_iters=0,
    ).posterior

    potentials = LatentPotentials.from_data(
        model, DatasetCollection.from_datasets((data,))
    )

    expected_potential = potentials.cond.compute_discrete_potential_at(initial_latents)
    expected_posterior = (
        DiscreteChain.from_local_potentials(potentials.prior, expected_potential)
        .forward_backward()
        .posterior
    )

    np.testing.assert_allclose(
        posterior.discrete.state_probs,
        expected_posterior.state_probs,
        atol=ATOL,
    )

    np.testing.assert_allclose(
        posterior.discrete.pair_probs,
        expected_posterior.pair_probs,
        atol=ATOL,
    )

    assert posterior.continuous.means.shape == (1, 1, 1)


def test_infer_variational_is_jittable():

    model = _two_state_model()
    observations = jnp.array([[0.0], [0.5]])
    data = Dataset.from_array(observations)
    initial_latents = jnp.zeros((1, observations.shape[0], 1))

    eager = infer_variational(
        model,
        DatasetCollection.from_datasets((data,)),
        num_iters=1,
        initial_latents=initial_latents,
    )

    inference_jit = jax.jit(
        infer_variational,
        static_argnames=('num_iters',),
    )

    jitted = inference_jit(
        model,
        DatasetCollection.from_datasets((data,)),
        num_iters=1,
        initial_latents=initial_latents,
    )

    jax.block_until_ready(jitted)

    np.testing.assert_allclose(
        jitted.posterior.discrete.state_probs,
        eager.posterior.discrete.state_probs,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.posterior.discrete.pair_probs,
        eager.posterior.discrete.pair_probs,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.posterior.continuous.means,
        eager.posterior.continuous.means,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.posterior.continuous.covariances,
        eager.posterior.continuous.covariances,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.posterior.continuous.cross_covariances,
        eager.posterior.continuous.cross_covariances,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.objective,
        eager.objective,
        atol=ATOL,
    )
