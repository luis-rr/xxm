import jax
import numpy as np
from jax import numpy as jnp

from tests.lds.lds_helpers import make_model, make_observations
from xxm.core.gaussian.emissions import PoissonEmissions
from xxm.lds.core import LinearGaussianDynamicsModel, GaussianInitialModel, Model
from xxm.lds.inference import (
    _NewtonSearchModel,
    _NewtonSearchParams,
    inference_exact,
    inference_laplace,
)
from xxm.newton import NewtonSearch


def make_scalar_poisson_model() -> Model[PoissonEmissions]:
    return Model(
        initial=GaussianInitialModel(mean=jnp.zeros(1), covariance=jnp.eye(1)),
        dynamics=LinearGaussianDynamicsModel(
            matrix=jnp.eye(1),
            bias=jnp.zeros(1),
            noise_covariance=jnp.eye(1),
        ),
        emissions=PoissonEmissions(readout=jnp.ones((1, 1)), bias=jnp.zeros(1)),
    )


def test_exact_inference_returns_one_posterior_per_observation():
    posterior = inference_exact(make_model(), make_observations())

    assert posterior.means.shape == (3, 2)
    assert posterior.covariances.shape == (3, 2, 2)
    assert posterior.cross_covariances.shape == (2, 2, 2)
    assert np.isfinite(posterior.log_normalizer)


def test_exact_inference_is_jittable():
    model = make_model()
    observations = make_observations()

    eager = inference_exact(model, observations)
    jitted = jax.jit(inference_exact)(model, observations)

    np.testing.assert_allclose(jitted.means, eager.means)
    np.testing.assert_allclose(jitted.covariances, eager.covariances)
    np.testing.assert_allclose(jitted.log_normalizer, eager.log_normalizer)


def test_laplace_recovers_known_scalar_map():
    model = make_scalar_poisson_model()

    posterior = inference_laplace(model, jnp.array([[1.0]]))

    np.testing.assert_allclose(posterior.means, [[0.0]], atol=1e-6)


def test_laplace_newton_steps_do_not_decrease_objective():
    model = make_scalar_poisson_model()
    observations = jnp.array([[3.0]])

    num_time_steps = observations.shape[0]

    laplace_model = _NewtonSearchModel(
        latent_chain=model.to_gaussian_chain(num_time_steps),
        emissions=model.emissions,
        observations=observations,
    )

    initial_params = _NewtonSearchParams(
        latents=model.get_prior_mean_latents(num_time_steps),
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
