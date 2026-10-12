import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine
from xxm.core.data import Sequences
from xxm.core.dists.categorical import Categorical
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.collection import EmissionsCollection
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.latents.discrete import (
    CategoricalInitial,
    CategoricalTransitions,
)
from xxm.core.lengths import LengthsSpec
from xxm.slds.core import (
    GaussianLinearSwitchingDynamics,
    Model,
    StateConditionedGaussian,
)


def test_categorical_transitions_sample_includes_initial():
    transitions = CategoricalTransitions(
        dist=Categorical(
            probs=jnp.array(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        )
    )

    states = transitions.sample(
        jax.random.key(0),
        initial_state=jnp.array(0),
        ls=LengthsSpec.from_params(4),
    )

    np.testing.assert_array_equal(
        states.values,
        jnp.array([0, 1, 1, 1]),
    )


def _make_model() -> Model:
    state_initial = CategoricalInitial(
        dist=Categorical(
            probs=jnp.array([1.0, 0.0]),
        )
    )

    transitions = CategoricalTransitions(
        dist=Categorical(
            probs=jnp.array(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            )
        )
    )

    latent_initial = StateConditionedGaussian(
        dist=Gaussian(
            mean=jnp.array(
                [
                    [5.0],
                    [-5.0],
                ]
            ),
            covariance=jnp.array(
                [
                    [[1e-8]],
                    [[1e-8]],
                ]
            ),
        )
    )

    dynamics = GaussianLinearSwitchingDynamics(
        dist=LinearGaussian(
            affine=Affine(
                coefficients=jnp.array(
                    [
                        [[0.0]],
                        [[0.0]],
                    ]
                ),
                bias=jnp.array(
                    [
                        [10.0],
                        [-10.0],
                    ]
                ),
            ),
            covariance=jnp.array(
                [
                    [[1e-8]],
                    [[1e-8]],
                ]
            ),
        )
    )

    emissions = GaussianEmissions(
        dist=LinearGaussian(
            affine=Affine(
                coefficients=jnp.array([[1.0]]),
                bias=jnp.array([3.0]),
            ),
            covariance=jnp.array([[1e-8]]),
        )
    )

    return Model(
        state_initial=state_initial,
        transitions=transitions,
        latent_initial=latent_initial,
        dynamics=dynamics,
        emissions=EmissionsCollection((emissions,)),
    )


def test_switching_dynamics_sample_includes_initial():
    dynamics = _make_model().dynamics

    initial_latent = jnp.array([2.0])
    states = Sequences.from_array(jnp.array([0, 0, 1, 1]))

    latents = dynamics.sample(
        jax.random.key(0),
        initial_latent,
        states,
        inputs=Sequences.from_array(jnp.empty((4, 0))),
    )

    assert latents.values.shape == (4, 1)
    np.testing.assert_array_equal(latents.lengths.values, states.lengths.values)

    np.testing.assert_array_equal(
        latents.values[0],
        initial_latent,
    )

    np.testing.assert_allclose(
        latents.values[1:, 0],
        jnp.array([10.0, -10.0, -10.0]),
        atol=1e-3,
    )


def test_sample():
    model = _make_model()

    states, latents, observations = model.sample(
        jax.random.key(0),
        inputs=Sequences.from_array(jnp.empty((4, 0))),
    )
    states = states.get()
    latents = latents.get(squeeze=False)
    observations = observations.get(squeeze=False)

    assert states.shape == (4,)
    assert latents.shape == (4, 1)
    assert observations.shape == (4, 1)

    # z0 selects x0's initial distribution; subsequent states select the
    # dynamics that generate x1, x2, and x3.
    np.testing.assert_array_equal(
        states,
        jnp.array([0, 1, 1, 1]),
    )

    np.testing.assert_allclose(
        latents[:, 0],
        jnp.array([5.0, -10.0, -10.0, -10.0]),
        atol=1e-3,
    )

    np.testing.assert_allclose(
        observations[:, 0],
        latents[:, 0] + 3.0,
        atol=1e-3,
    )


def test_sample_jittable():
    model = _make_model()
    key = jax.random.key(0)

    inputs = Sequences.from_array(jnp.empty((4, 0)))

    eager = model.sample(key, inputs=inputs)

    jitted = jax.jit(model.sample)(key, inputs=inputs)

    for eager_value, jitted_value in zip(
        eager,
        jitted,
        strict=True,
    ):
        np.testing.assert_allclose(
            jitted_value.get(),
            eager_value.get(),
        )
        np.testing.assert_array_equal(
            jitted_value.lengths.values, eager_value.lengths.values
        )
