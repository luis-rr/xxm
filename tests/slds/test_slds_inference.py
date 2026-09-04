import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.chains.gaussian import GaussianChain
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.models.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.models.gaussian import StateConditionedGaussian
from xxm.slds.core import GaussianLinearSwitchingDynamics, Model
from xxm.slds.inference import infer_variational

ATOL = 1e-5


def _mock_gaussian_emissions(
    noise_covariance: jax.Array,
) -> GaussianEmissions:
    return GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(
                coefficients=jnp.eye(1),
                bias=jnp.zeros(1),
            ),
            covariance=noise_covariance,
        ),
    )


def _single_state_model() -> Model:
    return Model(
        state_initial=CategoricalInitial(
            model=Categorical(
                probs=jnp.array([1.0]),
            )
        ),
        transitions=CategoricalTransitions(
            model=Categorical(
                probs=jnp.array([[1.0]]),
            )
        ),
        latent_initial=StateConditionedGaussian(
            model=Gaussian(
                mean=jnp.array([[0.0]]),
                covariance=jnp.array([[[1.0]]]),
            ),
        ),
        dynamics=GaussianLinearSwitchingDynamics(
            model=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array([[[0.8]]]),
                    bias=jnp.array([[0.2]]),
                ),
                covariance=jnp.array([[[0.5]]]),
            ),
        ),
        emissions=_mock_gaussian_emissions(
            noise_covariance=jnp.array([[0.25]]),
        ),
    )


def _two_state_model() -> Model:
    return Model(
        state_initial=CategoricalInitial(
            model=Categorical(
                probs=jnp.array([0.7, 0.3]),
            )
        ),
        transitions=CategoricalTransitions(
            model=Categorical(
                probs=jnp.array(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                    ]
                )
            )
        ),
        latent_initial=StateConditionedGaussian(
            model=Gaussian(
                mean=jnp.array(
                    [
                        [0.0],
                        [0.0],
                    ]
                ),
                covariance=jnp.array(
                    [
                        [[1.0]],
                        [[1.0]],
                    ]
                ),
            ),
        ),
        dynamics=GaussianLinearSwitchingDynamics(
            model=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array(
                        [
                            [[0.8]],
                            [[-0.5]],
                        ]
                    ),
                    bias=jnp.array(
                        [
                            [0.0],
                            [1.0],
                        ]
                    ),
                ),
                covariance=jnp.array(
                    [
                        [[0.5]],
                        [[0.5]],
                    ]
                ),
            ),
        ),
        emissions=_mock_gaussian_emissions(
            noise_covariance=jnp.array([[0.25]]),
        ),
    )


def test_single_state_slds_matches_gaussian_chain():
    model = _single_state_model()
    observations = jnp.array([[0.2], [1.0], [-0.3]])

    posterior, _ = infer_variational(
        model,
        observations,
        num_iters=3,
    )

    state_probs = jnp.ones((observations.shape[0], 1))

    initial_potential = model.latent_initial.compute_potentials().weighted_sum(
        state_probs[0]
    )
    pair_potentials = model.dynamics.compute_pair_potentials().weighted_sum(
        state_probs[1:]
    )
    observation_potential = model.emissions.compute_potential(observations)

    expected, _ = (
        GaussianChain.from_pair_potentials(
            initial_potential,
            pair_potentials,
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
        expected.means,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        posterior.continuous.covariances,
        expected.covariances,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        posterior.continuous.cross_covariances,
        expected.cross_covariances,
        atol=ATOL,
    )


def test_zero_iterations_uses_discrete_prior():
    model = _two_state_model()
    observations = jnp.zeros((4, 1))

    posterior, _ = infer_variational(
        model,
        observations,
        num_iters=0,
    )

    # There are T = 4 discrete states and T - 1 = 3 transitions.
    #
    # Starting from [0.7, 0.3]:
    #   p(z1) = [0.69, 0.31]
    #   p(z2) = [0.683, 0.317]
    expected = np.array(
        [
            [0.7000, 0.3000],
            [0.6900, 0.3100],
            [0.6830, 0.3170],
            [0.6781, 0.3219],
        ]
    )

    np.testing.assert_allclose(
        posterior.discrete.state_probs,
        expected,
        atol=ATOL,
    )

    assert posterior.continuous.means.shape == (4, 1)


def test_infer_variational_is_jittable():
    model = _two_state_model()
    observations = jnp.array(
        [
            [0.0],
            [0.5],
            [1.0],
            [0.2],
        ]
    )

    eager, eager_elbo = infer_variational(
        model,
        observations,
        num_iters=2,
    )

    inference_jit = jax.jit(
        infer_variational,
        static_argnames=('num_iters',),
    )

    jitted, jitted_elbo = inference_jit(
        model,
        observations,
        num_iters=2,
    )

    jax.block_until_ready(jitted)

    np.testing.assert_allclose(
        jitted.discrete.state_probs,
        eager.discrete.state_probs,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.discrete.pair_probs,
        eager.discrete.pair_probs,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.continuous.means,
        eager.continuous.means,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.continuous.covariances,
        eager.continuous.covariances,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted.continuous.cross_covariances,
        eager.continuous.cross_covariances,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        jitted_elbo,
        eager_elbo,
        atol=ATOL,
    )
