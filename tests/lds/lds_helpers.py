import jax
from jax import numpy as jnp

from xxm.core.dists.gaussian import FullGaussian, FullLinearGaussian
from xxm.core.emissions.collection import ContinuousEmissionsCollection
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.latents.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.lds.core import Model


def make_model() -> Model[GaussianEmissions]:
    return Model(
        initial=GaussianInitial(
            dist=FullGaussian.from_moments(
                jnp.array([0.5, -0.3]), jnp.array([[2.0, 0.2], [0.2, 1.5]])
            ),
        ),
        dynamics=GaussianLinearDynamics(
            dist=FullLinearGaussian.from_components(
                jnp.array([[0.9, 0.1], [0.0, 0.8]]),
                jnp.array([0.05, -0.05]),
                jnp.array([[0.5, 0.05], [0.05, 0.4]]),
            ),
        ),
        emissions=ContinuousEmissionsCollection(
            (
                GaussianEmissions(
                    dist=FullLinearGaussian.from_components(
                        jnp.eye(2),
                        jnp.array([0.1, -0.2]),
                        jnp.array([[1.5, 0.1], [0.1, 1.0]]),
                    ),
                ),
            ),
        ),
    )


def make_observations() -> jax.Array:
    return jnp.array([[1.0, 0.5], [0.2, -0.3], [-0.5, 0.8]])
