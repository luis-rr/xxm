import jax
from jax import numpy as jnp

from tests.lds.lds_helpers import make_model


def test_dynamics_next_mean_applies_matrix_and_bias():
    dynamics = make_model().dynamics

    result = dynamics.model.conditional(jnp.array([1.0, 2.0])).mean

    assert jnp.allclose(result, jnp.array([1.15, 1.55]))


def test_prior_mean_starts_at_initial_mean():
    model = make_model()

    means = model.compute_prior_means(num_steps=3)

    assert jnp.allclose(means[0], model.initial.model.mean)
    assert jnp.allclose(means[1], model.dynamics.model.conditional(means[0]).mean)


def test_sample_returns_one_latent_and_observation_per_time_step():
    model = make_model()

    latents, observations = model.sample(num_steps=4, key=jax.random.key(0))

    assert latents.shape == (4, 2)
    assert observations.shape == (4, 2)


def test_log_joint_is_finite_for_sampled_data():
    model = make_model()
    latents, observations = model.sample(num_steps=3, key=jax.random.key(0))

    assert jnp.isfinite(model.log_joint(observations, latents))
