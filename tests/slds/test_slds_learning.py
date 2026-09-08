import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscreteChainMarginals
from xxm.core.chains.gaussian import GaussianChainMarginals
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.inference import Inferred
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.slds.core import (
    GaussianLinearSwitchingDynamics,
    Model,
    Posterior,
)
from xxm.slds.learning import variational_em_step

ATOL = 1e-5


def _posterior() -> Posterior:
    # x[t] = 2 x[t-1] + 1 + eps, eps ~ N(0, 0.5)
    #
    # Starting from x0 ~ N(0, 1):
    #   E[x]   = [0, 1, 3]
    #   Var[x] = [1, 4.5, 18.5]
    #   Cov(x0, x1) = 2
    #   Cov(x1, x2) = 9
    continuous = GaussianChainMarginals(
        means=jnp.array(
            [
                [0.0],
                [1.0],
                [3.0],
            ]
        ),
        covariances=jnp.array(
            [
                [[1.0]],
                [[4.5]],
                [[18.5]],
            ]
        ),
        cross_covariances=jnp.array(
            [
                [[2.0]],
                [[9.0]],
            ]
        ),
    )

    discrete = DiscreteChainMarginals(
        state_probs=jnp.ones((3, 1)),
        pair_probs=jnp.ones((2, 1, 1)),
    )

    return Posterior(
        discrete=discrete,
        continuous=continuous,
    )


def _dynamics() -> GaussianLinearSwitchingDynamics:
    return GaussianLinearSwitchingDynamics(
        dist=LinearGaussian(
            affine=Affine(
                coefficients=jnp.zeros((1, 1, 1)),
                bias=jnp.zeros((1, 1)),
            ),
            covariance=jnp.ones((1, 1, 1)),
        ),
    )


def _model() -> Model:
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
            dist=Gaussian(
                mean=jnp.array([[7.0]]),
                covariance=jnp.array([[[2.0]]]),
            )
        ),
        dynamics=_dynamics(),
        emissions=GaussianEmissions(
            dist=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array([[1.0]]),
                    bias=jnp.array([0.0]),
                ),
                covariance=jnp.array([[1.0]]),
            )
        ),
    )


def test_switching_dynamics_fit_recovers_known_linear_gaussian_model():
    post = _posterior()
    fitted = _dynamics().fit_params(
        discrete=post.discrete,
        continuous=post.continuous,
    )

    np.testing.assert_allclose(
        fitted.dist.affine.coefficients,
        [[[2.0]]],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fitted.dist.affine.bias,
        [[1.0]],
        atol=ATOL,
    )
    np.testing.assert_allclose(
        fitted.dist.covariance,
        [[[0.5]]],
        atol=ATOL,
    )


def test_model_fit_params_keeps_latent_initial_fixed():
    model = _model()

    fitted = model.fit_params(
        observations=jnp.array(
            [
                [0.0],
                [1.0],
                [3.0],
            ]
        ),
        posterior=_posterior(),
    )

    np.testing.assert_allclose(
        fitted.latent_initial.dist.mean,
        model.latent_initial.dist.mean,
    )
    np.testing.assert_allclose(
        fitted.latent_initial.dist.covariance,
        model.latent_initial.dist.covariance,
    )


def test_variational_em_step_returns_finite_objective():
    model = _model()
    observations = jnp.array([[0.0], [1.0], [3.0]])

    inferred = variational_em_step(
        Inferred(model=model, posterior=_posterior(), objective=jnp.array(0.0)),
        observations,
        num_inference_iters=2,
    )

    assert np.isfinite(inferred.objective)

    np.testing.assert_allclose(
        inferred.model.latent_initial.dist.mean,
        model.latent_initial.dist.mean,
    )
    np.testing.assert_allclose(
        inferred.model.latent_initial.dist.covariance,
        model.latent_initial.dist.covariance,
    )
