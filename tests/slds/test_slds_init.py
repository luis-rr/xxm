import jax
import jax.numpy as jnp
import numpy as np
import pytest

import xxm
from xxm.core.data import Dataset, DatasetCollection
from xxm.core.dists.gaussian import DiagonalLinearGaussian
from xxm.slds import GaussianSLDS


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

    model = (
        GaussianSLDS.via_pca(
            key=jax.random.key(0),
            data=DatasetCollection.from_datasets((Dataset.from_array(observations),)),
            num_states=2,
            latent_dim=1,
        )
        .get()
        .core
    )

    assert model.state_initial.dist.probs.shape == (2,)
    assert model.transitions.dist.probs.shape == (2, 2)

    assert model.latent_initial.dist.mean.shape == (2, 1)
    assert model.latent_initial.dist.covariance.shape == (2, 1, 1)

    assert model.dynamics.dist.affine.coefficients.shape == (2, 1, 1)
    assert model.dynamics.dist.affine.bias.shape == (2, 1)
    assert model.dynamics.dist.covariance.shape == (2, 1, 1)

    assert model.emissions.get().dist.affine.coefficients.shape == (2, 1)
    assert model.emissions.get().dist.affine.bias.shape == (2,)
    emissions = model.emissions.get().dist
    assert isinstance(emissions, DiagonalLinearGaussian)
    assert emissions.variances.shape == (2,)

    np.testing.assert_allclose(model.state_initial.dist.probs.sum(), 1.0)
    np.testing.assert_allclose(model.transitions.dist.probs.sum(axis=-1), 1.0)

    for leaf in jax.tree_util.tree_leaves(model):
        assert jnp.all(jnp.isfinite(leaf))


@pytest.mark.parametrize('facade', [xxm.GaussianSLDS, xxm.PoissonSLDS])
def test_slds_arhmm_warm_start(facade):
    observations = jnp.array(
        [
            [0.0],
            [1.0],
            [2.0],
            [1.0],
        ]
    )
    model = facade.via_arhmm(
        jax.random.key(3),
        DatasetCollection.from_datasets((Dataset.from_array(observations),)),
        num_states=2,
        latent_dim=1,
        num_arhmm_iters=1,
        progress=False,
    ).get()
    assert model.num_states == 2
    assert model.latent_dim == 1
    for leaf in jax.tree.leaves(model):
        assert np.isfinite(leaf).all()


@pytest.mark.parametrize('facade', [xxm.GaussianSLDS, xxm.PoissonSLDS])
def test_slds_via_arhmm_matches_explicit_steps(facade):
    observations = jnp.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, -1.0],
            [0.0, 0.0],
        ]
    )
    data = DatasetCollection.from_datasets((Dataset.from_array(observations),))
    key = jax.random.key(3)

    pseudo_latents = facade.pseudo_latents_via_pca(data, latent_dim=1)
    fit = xxm.GaussianARHMM.via_kmeans(
        key, pseudo_latents, num_states=2, num_lags=1, noise='full'
    ).fit(pseudo_latents, num_iters=2)
    assert fit.objective_traces.shape[-1] == 3
    explicit = facade.from_arhmm(fit.inferred, data).get()

    integrated = facade.via_arhmm(
        key,
        data,
        num_states=2,
        latent_dim=1,
        num_arhmm_iters=2,
        progress=False,
    ).get()

    for left, right in zip(
        jax.tree.leaves(explicit), jax.tree.leaves(integrated), strict=True
    ):
        np.testing.assert_allclose(left, right)
