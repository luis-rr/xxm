import jax
import jax.numpy as jnp

from xxm.lds.init import init_gaussian_via_pca

OBSERVATIONS = jnp.array(
    [
        [0.0, 1.0, 0.5],
        [1.0, 0.0, -0.5],
        [2.0, 1.0, 1.0],
        [3.0, -1.0, 0.0],
    ]
)


def test_gaussian_via_pca_returns_model_with_requested_latent_dimension():
    model = init_gaussian_via_pca(OBSERVATIONS, latent_dim=2)

    assert model.initial.dist.mean.shape == (2,)
    assert model.dynamics.dist.affine.coefficients.shape == (2, 2)
    assert model.emissions.dist.affine.coefficients.shape == (3, 2)


def test_gaussian_via_pca_is_jittable():
    model = jax.jit(init_gaussian_via_pca, static_argnames='latent_dim')(
        OBSERVATIONS, latent_dim=2
    )

    assert model.emissions.dist.covariance.shape == (3, 3)
