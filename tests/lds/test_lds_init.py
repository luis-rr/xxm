import jax
from jax import numpy as jnp

from xxm.lds.init import init_gaussian_via_pca, init_gaussian_via_pca_many


def test_gaussian_via_pca_returns_model_with_requested_latent_dimension():
    observations = jnp.arange(24.0).reshape(8, 3)

    model = init_gaussian_via_pca(observations, latent_dim=2)

    assert model.initial.dist.mean.shape == (2,)
    assert model.dynamics.dist.affine.coefficients.shape == (2, 2)
    assert model.emissions.dist.affine.coefficients.shape == (3, 2)


def test_gaussian_via_pca_many_returns_one_model_per_floor():
    observations = jnp.arange(24.0).reshape(8, 3)

    models = init_gaussian_via_pca_many(
        observations, latent_dim=2, covariance_floors=jnp.array([1e-3, 1e-2])
    )

    assert len(models) == 2
    assert all(model.initial.dist.mean.shape == (2,) for model in models)


def test_gaussian_via_pca_is_jittable():
    observations = jnp.arange(24.0).reshape(8, 3)

    model = jax.jit(init_gaussian_via_pca, static_argnames='latent_dim')(
        observations, latent_dim=2
    )

    assert model.emissions.dist.covariance.shape == (3, 3)
