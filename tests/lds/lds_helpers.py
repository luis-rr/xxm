import jax
from jax import numpy as jnp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian, LinearGaussian
from xxm.core.emissions.continuous import GaussianEmissions
from xxm.core.models.gaussian import GaussianInitial, GaussianLinearDynamics
from xxm.lds.core import Model


def make_model() -> Model[GaussianEmissions]:
    return Model(
        initial=GaussianInitial(
            model=Gaussian(
                mean=jnp.array([0.5, -0.3]),
                covariance=jnp.array([[2.0, 0.2], [0.2, 1.5]]),
            ),
        ),
        dynamics=GaussianLinearDynamics(
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
