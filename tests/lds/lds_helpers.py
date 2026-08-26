import jax
from jax import numpy as jnp

from xxm.core.emissions.continuous import GaussianEmissions
from xxm.lds.model import GaussianInitialModel, LinearGaussianDynamicsModel, Model
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian


def make_model() -> Model[GaussianEmissions]:
    return Model(
        initial=GaussianInitialModel(
            model=Gaussian(
                mean=jnp.array([0.5, -0.3]),
                covariance=jnp.array([[2.0, 0.2], [0.2, 1.5]]),
            ),
        ),
        dynamics=LinearGaussianDynamicsModel(
            model=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.array([[0.9, 0.1], [0.0, 0.8]]),
                    bias=jnp.array([0.05, -0.05]),
                ),
                covariance=jnp.array([[0.5, 0.05], [0.05, 0.4]]),
            ),
        ),
        emissions=GaussianEmissions(
            model=LinearGaussian(
                affine=Affine(
                    coefficients=jnp.eye(2),
                    bias=jnp.array([0.1, -0.2]),
                ),
                covariance=jnp.array([[1.5, 0.1], [0.1, 1.0]]),
            ),
        ),
    )


def make_observations() -> jax.Array:
    return jnp.array([[1.0, 0.5], [0.2, -0.3], [-0.5, 0.8]])
