import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.data import Sequences
from xxm.lds.init import init_gaussian_via_pca, pca_latents

OBSERVATIONS = jnp.array(
    [
        [0.0, 1.0, 0.5],
        [1.0, 0.0, -0.5],
        [2.0, 1.0, 1.0],
        [3.0, -1.0, 0.0],
    ]
)


def test_gaussian_via_pca_returns_model_with_requested_latent_dimension():
    model = init_gaussian_via_pca(Sequences.from_sequence(OBSERVATIONS), latent_dim=2)

    assert model.initial.dist.mean.shape == (2,)
    assert model.dynamics.dist.affine.coefficients.shape == (2, 2)
    assert model.emissions.dist.affine.coefficients.shape == (3, 2)
    assert model.emissions.dist.covariance.shape == (3, 3)


def test_pca_latents_are_jittable():
    # Dataset validation and masked row selection belong to host initialization.
    # The complete-data PCA numerical kernel remains compilable.
    latents = jax.jit(pca_latents, static_argnames='latent_dim')(
        OBSERVATIONS, latent_dim=2
    )

    assert latents.shape == (4, 2)
    np.testing.assert_allclose(
        latents, pca_latents(OBSERVATIONS, latent_dim=2), atol=1e-6
    )
