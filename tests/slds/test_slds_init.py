import jax
import jax.numpy as jnp
import numpy as np

from xxm.slds.init import init_gaussian_via_pca


def test_gaussian_via_pca_returns_requested_model_structure():
    # Smallest comfortable nontrivial problem: PCA actually reduces 2D
    # observations to one latent dimension, while three latent transitions are
    # available for two switching states.
    observations = jnp.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, -1.0],
        ]
    )

    model = init_gaussian_via_pca(
        key=jax.random.key(0),
        observations=observations,
        num_states=2,
        latent_dim=1,
    )

    assert model.state_initial.dist.probs.shape == (2,)
    assert model.transitions.dist.probs.shape == (2, 2)

    assert model.latent_initial.dist.mean.shape == (2, 1)
    assert model.latent_initial.dist.covariance.shape == (2, 1, 1)

    assert model.dynamics.dist.affine.coefficients.shape == (2, 1, 1)
    assert model.dynamics.dist.affine.bias.shape == (2, 1)
    assert model.dynamics.dist.covariance.shape == (2, 1, 1)

    assert model.emissions.dist.affine.coefficients.shape == (2, 1)
    assert model.emissions.dist.affine.bias.shape == (2,)
    assert model.emissions.dist.covariance.shape == (2, 2)

    np.testing.assert_allclose(model.state_initial.dist.probs.sum(), 1.0)
    np.testing.assert_allclose(model.transitions.dist.probs.sum(axis=-1), 1.0)

    for leaf in jax.tree_util.tree_leaves(model):
        assert jnp.all(jnp.isfinite(leaf))
