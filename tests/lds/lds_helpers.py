import jax
from jax import numpy as jnp

from xxm.core.gaussian.emissions import GaussianEmissions
from xxm.lds.core import LinearGaussianDynamicsModel, GaussianInitialModel, Model


def make_model() -> Model[GaussianEmissions]:
    return Model(
        initial=GaussianInitialModel(
            mean=jnp.array([0.5, -0.3]),
            covariance=jnp.array([[2.0, 0.2], [0.2, 1.5]]),
        ),
        dynamics=LinearGaussianDynamicsModel(
            matrix=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
            bias=jnp.array([0.05, -0.05]),
            noise_covariance=jnp.array([[0.5, 0.05], [0.05, 0.4]]),
        ),
        emissions=GaussianEmissions(
            readout=jnp.eye(2),
            bias=jnp.array([0.1, -0.2]),
            noise_covariance=jnp.array([[1.5, 0.1], [0.1, 1.0]]),
        ),
    )


def make_observations() -> jax.Array:
    return jnp.array([[1.0, 0.5], [0.2, -0.3], [-0.5, 0.8]])
