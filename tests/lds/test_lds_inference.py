import jax
import numpy as np
import pytest
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.data import Dataset, DatasetCollection
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.dists.poisson import LinearPoisson
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import GaussianEmissions, PoissonEmissions
from xxm.core.inference.chains.gaussian import GaussianChain
from xxm.core.inference.laplace import (
    _NewtonSearchModel,
    _NewtonSearchParams,
)
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.core.lengths import Lengths
from xxm.core.optim.newton import NewtonSearch, OptimParams
from xxm.lds.core import Model
from xxm.lds.inference import (
    compute_latent_potentials,
    infer_exact,
    infer_laplace,
)

from .lds_helpers import make_model, make_observations


def make_scalar_poisson_model() -> Model[PoissonEmissions]:
    return Model(
        initial=GaussianInitial(
            dist=Gaussian(mean=jnp.zeros(1), covariance=jnp.eye(1))
        ),
        dynamics=GaussianLinearDynamics(
            dist=LinearGaussian(
                affine=Affine(coefficients=jnp.eye(1), bias=jnp.zeros(1)),
                covariance=jnp.eye(1),
            ),
        ),
        emissions=ContinuousEmissionsCollection(
            (
                PoissonEmissions(
                    dist=LinearPoisson(
                        affine=Affine(coefficients=jnp.ones((1, 1)), bias=jnp.zeros(1))
                    ),
                ),
            ),
        ),
    )


def test_exact_inference_returns_one_posterior_per_observation():
    data = DatasetCollection.from_datasets((Dataset.from_array(make_observations()),))
    inferred = infer_exact(make_model(), data)
    posterior = inferred.posterior
    log_normalizer = inferred.objective

    assert posterior.batch_shape == (1,)
    assert posterior.means.shape == (1, 3, 2)
    assert posterior.covariances.shape == (1, 3, 2, 2)
    assert posterior.cross_covariances.shape == (1, 2, 2, 2)
    assert log_normalizer.shape == ()
    assert np.isfinite(log_normalizer)


def test_exact_inference_is_jittable():
    model = make_model()
    observations = DatasetCollection.from_datasets(
        (Dataset.from_array(make_observations()),)
    )

    eager = infer_exact(model, observations)
    jitted = jax.jit(infer_exact)(model, observations)

    np.testing.assert_allclose(jitted.posterior.means, eager.posterior.means)
    np.testing.assert_allclose(
        jitted.posterior.covariances, eager.posterior.covariances
    )
    np.testing.assert_allclose(jitted.objective, eager.objective)


@pytest.mark.parametrize(
    ('make', 'infer'),
    [(make_model, infer_exact), (make_scalar_poisson_model, infer_laplace)],
)
def test_withheld_observations_do_not_change_inference(make, infer):
    model = make()
    dim = model.emissions.get().observation_dim
    values = jnp.ones((3, dim))
    visible = jnp.array([True, False, True])
    original = Dataset.from_array(values, visible=visible)
    changed = Dataset.from_array(values.at[1].set(100.0), visible=visible)
    expected = infer(model, DatasetCollection.from_datasets((original,)))
    actual = infer(model, DatasetCollection.from_datasets((changed,)))
    np.testing.assert_allclose(
        actual.posterior.means, expected.posterior.means, atol=1e-5
    )
    np.testing.assert_allclose(actual.objective, expected.objective, atol=1e-5)
    np.testing.assert_allclose(
        model.log_joint(
            DatasetCollection.from_datasets((original,)), expected.posterior.means
        ),
        model.log_joint(
            DatasetCollection.from_datasets((changed,)), expected.posterior.means
        ),
        atol=1e-5,
    )


def test_hidden_timesteps_keep_known_inputs_and_padding_excludes_transitions():
    model = Model(
        initial=GaussianInitial(Gaussian(jnp.zeros(1), jnp.eye(1))),
        dynamics=GaussianLinearDynamics.from_params(
            latent_coefficients=jnp.eye(1),
            input_coefficients=jnp.array([[2.0]]),
            bias=jnp.zeros(1),
            covariance=jnp.eye(1),
        ),
        emissions=ContinuousEmissionsCollection(
            (
                GaussianEmissions(
                    LinearGaussian(Affine(jnp.eye(1), jnp.zeros(1)), jnp.eye(1))
                ),
            ),
        ),
    )
    data = Dataset.from_padded(
        jnp.full((4, 1), 100.0),
        Lengths(jnp.array(3)),
        inputs=jnp.array([[99.0], [1.0], [2.0], [1000.0]]),
        visible=jnp.zeros(4, dtype=bool),
    )
    result = jax.jit(infer_exact)(model, DatasetCollection.from_datasets((data,)))
    np.testing.assert_allclose(
        result.posterior.means[0, :3, 0], [0.0, 2.0, 6.0], atol=1e-5
    )
    np.testing.assert_allclose(result.objective, 0.0, atol=1e-5)
    potential = model.initial.compute_potential()
    np.testing.assert_allclose(potential.precision_blocks, [[1.0]])
    np.testing.assert_allclose(potential.information_vectors, [0.0])
    np.testing.assert_allclose(
        potential.log_constant, -0.5 * np.log(2 * np.pi), atol=1e-6
    )


def test_laplace_recovers_known_scalar_map():
    model = make_scalar_poisson_model()

    data = DatasetCollection.from_datasets((Dataset.from_array(jnp.array([[1.0]])),))
    posterior = infer_laplace(model, data).posterior

    assert posterior.batch_shape == (1,)
    np.testing.assert_allclose(posterior.means, [[[0.0]]], atol=1e-6)


def test_laplace_newton_steps_do_not_decrease_objective():
    model = make_scalar_poisson_model()
    observations = jnp.array([[3.0]])

    data = Dataset.from_array(observations)

    laplace_model = _NewtonSearchModel(
        latent_chain=GaussianChain.from_potentials(
            compute_latent_potentials(model, data.inputs)
        ),
        emissions=model.emissions.get(),
        observations=data.weighted_observations(),
    )

    initial_params = _NewtonSearchParams(
        latents=model.compute_prior_means(
            inputs=data.inputs,
        ).get(squeeze=False),
    )

    search = NewtonSearch[_NewtonSearchParams](
        model=laplace_model,
        optim_params=OptimParams(
            max_line_search_iters=10,
            tol=1e-6,
            max_iter=100,
        ),
    )

    state = search.initial_state(params=initial_params)

    objectives = [state.objective]

    for _ in range(5):
        state = search._newton_step(state)
        objectives.append(state.objective)

    assert np.all(np.diff(np.asarray(objectives), axis=0) >= 0.0)
