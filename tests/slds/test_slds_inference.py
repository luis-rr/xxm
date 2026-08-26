import typing

import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.chains.gaussian import (
    GaussianChain,
    GaussianChainMarginals,
    GaussianPotential,
)
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.models.discrete import CategoricalInitial
from xxm.hmm.model import CategoricalTransition

from xxm.core.models.gaussian import GaussianInitial
from xxm.slds.model import (
    Model,
    Posterior,
    SwitchingLinearGaussianDynamics,
)
from xxm.slds.inference import infer_variational
from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian

ATOL = 1e-5


def _mock_gaussian_emissions(noise_covariance: jax.Array) -> GaussianEmissions:
    return GaussianEmissions(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.eye(1), bias=jnp.zeros(1)),
            covariance=noise_covariance,
        ),
    )


def test_switching_dynamics_fit_recovers_known_linear_gaussian_model():
    # x[t+1] = 2 x[t] + 1 + eps, eps ~ N(0, 0.5)
    #
    # Starting from x0 ~ N(0, 1):
    #   E[x]   = [0, 1, 3]
    #   Var[x] = [1, 4.5, 18.5]
    #   Cov(x0, x1) = 2
    #   Cov(x1, x2) = 9
    continuous = GaussianChainMarginals(
        means=jnp.array([[0.0], [1.0], [3.0]]),
        covariances=jnp.array([[[1.0]], [[4.5]], [[18.5]]]),
        cross_covariances=jnp.array([[[2.0]], [[9.0]]]),
    )

    discrete = DiscreteChainMarginals(
        state_probs=jnp.ones((2, 1)),
        pair_probs=jnp.ones((1, 1, 1)),
    )

    posterior = Posterior(
        discrete=discrete,
        continuous=continuous,
    )

    dynamics = SwitchingLinearGaussianDynamics(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.zeros((1, 1, 1)), bias=jnp.zeros((1, 1))),
            covariance=jnp.ones((1, 1, 1)),
        ),
    )

    fitted = dynamics.fit_params(posterior)

    np.testing.assert_allclose(fitted.model.affine.coefficients, [[[2.0]]], atol=ATOL)
    np.testing.assert_allclose(fitted.model.affine.bias, [[1.0]], atol=ATOL)
    np.testing.assert_allclose(fitted.model.covariance, [[[0.5]]], atol=ATOL)


def test_switching_dynamics_fit_is_jittable():
    continuous = GaussianChainMarginals(
        means=jnp.array([[0.0], [1.0], [3.0]]),
        covariances=jnp.array([[[1.0]], [[4.5]], [[18.5]]]),
        cross_covariances=jnp.array([[[2.0]], [[9.0]]]),
    )

    discrete = DiscreteChainMarginals(
        state_probs=jnp.ones((2, 1)),
        pair_probs=jnp.ones((1, 1, 1)),
    )

    posterior = Posterior(discrete, continuous)

    dynamics = SwitchingLinearGaussianDynamics(
        model=LinearGaussian(
            affine=Affine(coefficients=jnp.zeros((1, 1, 1)), bias=jnp.zeros((1, 1))),
            covariance=jnp.ones((1, 1, 1)),
        ),
    )

    fitted = jax.jit(lambda dynamics, posterior: dynamics.fit_params(posterior))(
        dynamics, posterior
    )

    jax.block_until_ready(fitted)

    np.testing.assert_allclose(fitted.model.affine.coefficients, [[[2.0]]], atol=ATOL)
    np.testing.assert_allclose(fitted.model.affine.bias, [[1.0]], atol=ATOL)
    np.testing.assert_allclose(fitted.model.covariance, [[[0.5]]], atol=ATOL)


class _IdentityGaussianEmissions(typing.NamedTuple):
    covariance: jax.Array

    def get_potential(
        self,
        observations: jax.Array,
    ) -> GaussianPotential:
        covariances = jnp.broadcast_to(
            self.covariance,
            (observations.shape[0],) + self.covariance.shape,
        )
        return GaussianPotential.from_moments(
            Gaussian(mean=observations, covariance=covariances),
        )


def _single_state_model() -> Model:
    return Model(
        state_initial=CategoricalInitial(
            Categorical(probs=jnp.array([1.0])),
        ),
        transitions=CategoricalTransition(
            Categorical(probs=jnp.array([[1.0]])),
        ),
        latent_initial=GaussianInitial(
            model=Gaussian(mean=jnp.array([0.0]), covariance=jnp.array([[1.0]])),
        ),
        dynamics=SwitchingLinearGaussianDynamics(
            model=LinearGaussian(
                affine=Affine(coefficients=jnp.array([[[0.8]]]), bias=jnp.array([[0.2]])),
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
            Categorical(probs=jnp.array([0.7, 0.3])),
        ),
        transitions=CategoricalTransition(
            Categorical(
                jnp.array(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                    ]
                )
            ),
        ),
        latent_initial=GaussianInitial(
            model=Gaussian(mean=jnp.array([0.0]), covariance=jnp.array([[1.0]])),
        ),
        dynamics=SwitchingLinearGaussianDynamics(
            model=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array([[[0.8]], [[-0.5]]]),
                    bias=jnp.array([[0.0], [1.0]]),
                ),
                covariance=jnp.array([[[0.5]], [[0.5]]]),
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

    # With K=1, q(z_t=0)=1 and the SLDS reduces to a Gaussian chain.
    state_probs = jnp.ones((observations.shape[0] - 1, 1))

    pair_potentials = model.dynamics.get_pair_potentials().weighted_sum(state_probs)
    initial_potential = GaussianPotential.from_moments(
        Gaussian(
            mean=model.latent_initial.model.mean, covariance=model.latent_initial.model.covariance
        ),
    )
    observation_potential = model.emissions.get_potential(observations)

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

    # There are T - 1 = 3 discrete time steps.
    #
    # Starting from [0.7, 0.3]:
    #   p(z1) = [0.69, 0.31]
    #   p(z2) = [0.683, 0.317]
    expected = np.array(
        [
            [0.700, 0.300],
            [0.690, 0.310],
            [0.683, 0.317],
        ]
    )

    np.testing.assert_allclose(
        posterior.discrete.state_probs,
        expected,
        atol=ATOL,
    )

    assert posterior.continuous.means.shape == (4, 1)


def test_inference_exact_is_jittable():
    model = _two_state_model()
    observations = jnp.array([[0.0], [0.5], [1.0], [0.2]])

    eager, _ = infer_variational(
        model,
        observations,
        num_iters=2,
    )

    inference_jit = jax.jit(
        infer_variational,
        static_argnames=('num_iters',),
    )

    jitted, jitted_log_normalizer = inference_jit(
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
