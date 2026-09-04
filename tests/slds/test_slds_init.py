import jax
import jax.numpy as jnp
import numpy as np

from xxm.slds.init import init_pca_gaussian


def test_init_pca_gaussian_returns_requested_model_structure():
    time = jnp.linspace(
        0.0,
        2.0 * jnp.pi,
        12,
        endpoint=False,
    )
    observations = jnp.stack(
        [
            jnp.sin(time),
            jnp.cos(time),
            jnp.sin(2.0 * time) + 0.2 * jnp.cos(time),
        ],
        axis=-1,
    )

    model = init_pca_gaussian(
        key=jax.random.key(0),
        observations=observations,
        num_states=2,
        latent_dim=2,
    )

    assert model.state_initial.model.probs.shape == (2,)
    assert model.transitions.model.probs.shape == (2, 2)

    assert model.latent_initial.model.mean.shape == (2, 2)
    assert model.latent_initial.model.covariance.shape == (2, 2, 2)

    assert model.dynamics.model.affine.coefficients.shape == (2, 2, 2)
    assert model.dynamics.model.affine.bias.shape == (2, 2)
    assert model.dynamics.model.covariance.shape == (2, 2, 2)

    assert model.emissions.model.affine.coefficients.shape == (3, 2)
    assert model.emissions.model.affine.bias.shape == (3,)
    assert model.emissions.model.covariance.shape == (3, 3)

    np.testing.assert_allclose(
        model.state_initial.model.probs.sum(),
        1.0,
    )
    np.testing.assert_allclose(
        model.transitions.model.probs.sum(axis=-1),
        1.0,
    )

    for leaf in jax.tree_util.tree_leaves(model):
        assert jnp.all(jnp.isfinite(leaf))
