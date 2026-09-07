import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.latents.gaussian import StateConditionedGaussian
from xxm.slds.core import GaussianLinearSwitchingDynamics, Model


def _make_model() -> Model:
    return Model(
        state_initial=CategoricalInitial(
            dist=Categorical(
                probs=jnp.array([0.7, 0.3]),
            )
        ),
        transitions=CategoricalTransitions(
            dist=Categorical(
                probs=jnp.array(
                    [
                        [0.9, 0.1],
                        [0.2, 0.8],
                    ]
                )
            )
        ),
        latent_initial=StateConditionedGaussian(
            dist=Gaussian(
                mean=jnp.array(
                    [
                        [1.0, 2.0],
                        [-1.0, 0.5],
                    ]
                ),
                covariance=jnp.array(
                    [
                        [[1.0, 0.2], [0.2, 0.5]],
                        [[0.7, 0.1], [0.1, 1.2]],
                    ]
                ),
            )
        ),
        dynamics=GaussianLinearSwitchingDynamics(
            dist=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array(
                        [
                            [[0.8, 0.1], [0.0, 0.9]],
                            [[-0.4, 0.2], [0.1, 0.5]],
                        ]
                    ),
                    bias=jnp.array(
                        [
                            [0.2, -0.1],
                            [1.0, 0.5],
                        ]
                    ),
                ),
                covariance=jnp.array(
                    [
                        [[0.4, 0.1], [0.1, 0.3]],
                        [[0.3, 0.0], [0.0, 0.2]],
                    ]
                ),
            )
        ),
        emissions=GaussianEmissions(
            dist=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array(
                        [
                            [1.0, 0.2],
                            [-0.3, 0.8],
                            [0.5, -0.4],
                        ]
                    ),
                    bias=jnp.array([0.1, -0.2, 0.3]),
                ),
                covariance=jnp.diag(jnp.array([0.2, 0.3, 0.4])),
            )
        ),
    )


def test_permute_relabels_all_state_dependent_components():
    model = _make_model()
    permutation = jnp.array([1, 0])

    permuted = model.permute(permutation)

    np.testing.assert_allclose(
        permuted.state_initial.dist.probs,
        model.state_initial.dist.probs[permutation],
    )
    np.testing.assert_allclose(
        permuted.transitions.dist.probs,
        model.transitions.dist.probs[permutation][:, permutation],
    )
    np.testing.assert_allclose(
        permuted.latent_initial.dist.mean,
        model.latent_initial.dist.mean[permutation],
    )
    np.testing.assert_allclose(
        permuted.dynamics.dist.affine.coefficients,
        model.dynamics.dist.affine.coefficients[permutation],
    )


def test_align_preserves_the_model_in_aligned_latent_coordinates():
    model = _make_model()

    alignment = Affine(
        coefficients=jnp.array(
            [
                [0.0, -2.0],
                [1.5, 0.0],
            ]
        ),
        bias=jnp.array([0.3, -0.4]),
    )

    aligned = model.align(alignment)

    state = jnp.array(1)
    latent = jnp.array([0.4, -0.7])
    aligned_latent = alignment.apply(latent)

    expected_initial = model.latent_initial.conditional(state).affine(alignment)
    actual_initial = aligned.latent_initial.conditional(state)

    np.testing.assert_allclose(
        actual_initial.mean,
        expected_initial.mean,
    )
    np.testing.assert_allclose(
        actual_initial.covariance,
        expected_initial.covariance,
    )

    expected_next = (
        model.dynamics.dist.select(state).conditional(latent).affine(alignment)
    )
    actual_next = aligned.dynamics.dist.select(state).conditional(aligned_latent)

    np.testing.assert_allclose(
        actual_next.mean,
        expected_next.mean,
    )
    np.testing.assert_allclose(
        actual_next.covariance,
        expected_next.covariance,
    )

    expected_observation = model.emissions.conditional(latent)
    actual_observation = aligned.emissions.conditional(aligned_latent)

    np.testing.assert_allclose(
        actual_observation.mean,
        expected_observation.mean,
    )
    np.testing.assert_allclose(
        actual_observation.covariance,
        expected_observation.covariance,
    )
