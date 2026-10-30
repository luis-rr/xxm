"""Canonical stored parameters for distributions shared by several emission types."""

import jax

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import LinearGaussian
from xxm.core.dists.poisson import LinearPoisson


def linear_gaussian_to_params(dist: LinearGaussian) -> dict[str, jax.Array]:
    """Return the stored parameters of a linear-Gaussian conditional."""
    return {
        'coefficients': dist.affine.coefficients,
        'bias': dist.affine.bias,
        'covariance': dist.covariance,
    }


def linear_gaussian_from_params(
    *,
    coefficients: jax.Array,
    bias: jax.Array,
    covariance: jax.Array,
) -> LinearGaussian:
    """Construct a linear-Gaussian conditional from stored parameters."""
    return LinearGaussian.from_components(coefficients, bias, covariance)


def linear_poisson_to_params(dist: LinearPoisson) -> dict[str, jax.Array]:
    """Return the stored parameters of a linear-Poisson conditional."""
    return {
        'coefficients': dist.affine.coefficients,
        'bias': dist.affine.bias,
    }


def linear_poisson_from_params(
    *,
    coefficients: jax.Array,
    bias: jax.Array,
) -> LinearPoisson:
    """Construct a linear-Poisson conditional from stored parameters."""
    return LinearPoisson(affine=Affine(coefficients, bias))
