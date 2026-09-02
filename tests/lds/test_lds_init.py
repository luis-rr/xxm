import jax
from jax import numpy as jnp

from xxm.lds.init import init_pca_gaussian, init_pca_gaussian_many


def test_init_pca_gaussian_returns_model_with_requested_latent_dimension():
    observations = jnp.arange(24.0).reshape(8, 3)

    model = init_pca_gaussian(observations, latent_dim=2)

    assert model.initial.model.mean.shape == (2,)
    assert model.dynamics.model.affine.coefficients.shape == (2, 2)
    assert model.emissions.model.affine.coefficients.shape == (3, 2)


def test_init_pca_gaussian_many_returns_one_model_per_floor():
    observations = jnp.arange(24.0).reshape(8, 3)

    models = init_pca_gaussian_many(
        observations, latent_dim=2, covariance_floors=jnp.array([1e-3, 1e-2])
    )

    assert len(models) == 2
    assert all(model.initial.model.mean.shape == (2,) for model in models)


def test_init_pca_gaussian_is_jittable():
    observations = jnp.arange(24.0).reshape(8, 3)

    model = jax.jit(init_pca_gaussian, static_argnames='latent_dim')(
        observations, latent_dim=2
    )

    assert model.emissions.model.covariance.shape == (3, 3)
