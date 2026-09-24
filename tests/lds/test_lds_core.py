import jax
import numpy as np
import pytest
from jax import numpy as jnp

from xxm.core.latents.gaussian import GaussianLinearDynamics
from xxm.lds.model import GaussianLDS, PoissonLDS

from .lds_helpers import make_model


def test_dynamics_next_mean_applies_matrix_and_bias():
    dynamics = make_model().dynamics

    result = dynamics.dist.conditional(jnp.array([1.0, 2.0])).mean

    assert jnp.allclose(result, jnp.array([1.15, 1.55]))


def test_prior_mean_starts_at_initial_mean():
    model = make_model()

    means = model.compute_prior_means(num_steps=3, inputs=jnp.empty((3, 0)))

    assert jnp.allclose(means[0], model.initial.dist.mean)
    assert jnp.allclose(means[1], model.dynamics.dist.conditional(means[0]).mean)


def test_sample_returns_one_latent_and_observation_per_time_step():
    model = make_model()

    latents, observations = model.sample(
        num_steps=4, key=jax.random.key(0), inputs=jnp.empty((4, 0))
    )

    assert latents.shape == (4, 2)
    assert observations.shape == (4, 2)


@pytest.mark.parametrize('controlled', [False, True])
def test_dynamics_from_params_and_potentials_with_batch_and_query_axes(controlled):
    a = jnp.full((2, 1, 1), 0.5)
    b = jnp.full((2, 1, 1), 2.0) if controlled else None
    dynamics = GaussianLinearDynamics.from_params(
        latent_coefficients=a,
        input_coefficients=b,
        bias=jnp.ones((2, 1)),
        covariance=jnp.full((2, 1, 1), 2.0),
    )
    inputs = jnp.ones((2, 3, int(controlled)))
    conditioned = dynamics.conditional(inputs)
    np.testing.assert_allclose(conditioned.affine.bias, 3.0 if controlled else 1.0)
    assert dynamics.input_coefficients.shape == (2, 1, int(controlled))
    assert dynamics.input_coefficients.dtype == a.dtype
    pair = jax.jit(lambda d, u: d.compute_pair_potentials(u))(dynamics, inputs)
    assert pair.batch_shape == (2, 3)
    np.testing.assert_allclose(pair.left_precision, 0.125)
    np.testing.assert_allclose(pair.right_precision, 0.5)
    np.testing.assert_allclose(pair.lower_precision, -0.25)
    np.testing.assert_allclose(pair.right_information, 1.5 if controlled else 0.5)


@pytest.mark.parametrize('facade', [GaussianLDS, PoissonLDS])
@pytest.mark.parametrize('controlled', [False, True])
def test_facade_input_properties_and_conditioning(facade, controlled):
    kwargs = {'emission_covariance': jnp.eye(1)} if facade is GaussianLDS else {}
    model = facade.from_params(
        initial_mean=jnp.zeros(1),
        initial_covariance=jnp.eye(1),
        dynamics_coefficients=jnp.eye(1),
        dynamics_bias=jnp.zeros(1),
        dynamics_covariance=jnp.eye(1),
        emission_coefficients=jnp.eye(1),
        emission_bias=jnp.zeros(1),
        dynamics_input_coefficients=jnp.full((1, 1), 2.0) if controlled else None,
        **kwargs,
    )
    assert model.input_dim == int(controlled)
    assert model.input_coefficients.shape == (1, int(controlled))
    conditioned = model.dynamics(jnp.array([3.0]) if controlled else None)
    np.testing.assert_allclose(conditioned.affine.bias, [6.0] if controlled else [0.0])
