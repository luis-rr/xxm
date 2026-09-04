import jax
import numpy as np
from jax import numpy as jnp

from tests.lds.lds_helpers import make_model, make_observations
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.continuous import PoissonEmissions
from xxm.core.models.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.optim.laplace import (
    _NewtonSearchModel,
    _NewtonSearchParams,
)
from xxm.core.optim.newton import NewtonSearch
from xxm.lds.core import Model
from xxm.lds.inference import (
    infer_exact,
    infer_laplace,
    to_chain,
)


def make_scalar_poisson_model() -> Model[PoissonEmissions]:
    return Model(
        initial=GaussianInitial(
            model=Gaussian(mean=jnp.zeros(1), covariance=jnp.eye(1))
        ),
        dynamics=GaussianLinearDynamics(
            model=LinearGaussian(
                affine=Affine(coefficients=jnp.eye(1), bias=jnp.zeros(1)),
                covariance=jnp.eye(1),
            ),
        ),
        emissions=PoissonEmissions(
            model=LinearPoisson(
                affine=Affine(coefficients=jnp.ones((1, 1)), bias=jnp.zeros(1))
            ),
        ),
    )


def test_exact_inference_returns_one_posterior_per_observation():
    posterior, log_normalizer = infer_exact(make_model(), make_observations())

    assert posterior.means.shape == (3, 2)
    assert posterior.covariances.shape == (3, 2, 2)
    assert posterior.cross_covariances.shape == (2, 2, 2)
    assert np.isfinite(log_normalizer)


def test_exact_inference_is_jittable():
    model = make_model()
    observations = make_observations()

    eager, log_normalizer = infer_exact(model, observations)
    jitted, jitted_log_normalizer = jax.jit(infer_exact)(model, observations)

    np.testing.assert_allclose(jitted.means, eager.means)
    np.testing.assert_allclose(jitted.covariances, eager.covariances)
    np.testing.assert_allclose(jitted_log_normalizer, log_normalizer)


def test_laplace_recovers_known_scalar_map():
    model = make_scalar_poisson_model()

    posterior, _log_normalizer = infer_laplace(model, jnp.array([[1.0]]))

    np.testing.assert_allclose(posterior.means, [[0.0]], atol=1e-6)


def test_laplace_newton_steps_do_not_decrease_objective():
    model = make_scalar_poisson_model()
    observations = jnp.array([[3.0]])

    num_steps = observations.shape[0]

    laplace_model = _NewtonSearchModel(
        latent_chain=to_chain(model=model, num_steps=num_steps),
        emissions=model.emissions,
        observations=observations,
    )

    initial_params = _NewtonSearchParams(
        latents=model.compute_prior_means(num_steps),
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=laplace_model,
        max_line_search_iters=10,
        tol=1e-6,
    )

    state = search.initial_state(params=initial_params)

    objectives = [state.objective]

    for _ in range(5):
        state = search._newton_step(state)
        objectives.append(state.objective)

    assert np.all(np.diff(np.asarray(objectives)) >= 0.0)
