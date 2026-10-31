"""Canonical stored parameters for distributions used as emission parameters."""

from typing import TypedDict

import jax

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import (
    DiagonalGaussian,
    DiagonalLinearGaussian,
    FullGaussian,
    FullLinearGaussian,
    Gaussian,
    LinearGaussian,
)
from xxm.core.dists.poisson import LinearPoisson, Poisson


class FullGaussianParams(TypedDict):
    mean: jax.Array
    covariance: jax.Array


class DiagonalGaussianParams(TypedDict):
    mean: jax.Array
    variances: jax.Array


GaussianParams = FullGaussianParams | DiagonalGaussianParams


class PoissonParams(TypedDict):
    log_rates: jax.Array


class FullLinearGaussianParams(TypedDict):
    coefficients: jax.Array
    bias: jax.Array
    covariance: jax.Array


class DiagonalLinearGaussianParams(TypedDict):
    coefficients: jax.Array
    bias: jax.Array
    variances: jax.Array


LinearGaussianParams = FullLinearGaussianParams | DiagonalLinearGaussianParams


class LinearPoissonParams(TypedDict):
    coefficients: jax.Array
    bias: jax.Array


def gaussian_to_params(
    dist: Gaussian,
) -> GaussianParams:
    """Return the stored parameters of a full or diagonal Gaussian."""
    if isinstance(dist, DiagonalGaussian):
        return {'mean': dist.mean, 'variances': dist.variances}
    return {'mean': dist.mean, 'covariance': dist.covariance}


def gaussian_from_params(
    params: GaussianParams,
) -> Gaussian:
    """Construct a full or diagonal Gaussian from stored parameters."""
    if 'variances' in params:
        return DiagonalGaussian.from_moments(params['mean'], params['variances'])
    return FullGaussian.from_moments(params['mean'], params['covariance'])


def poisson_to_params(dist: Poisson) -> PoissonParams:
    """Return the stored parameters of a Poisson."""
    return {'log_rates': dist.log_rates}


def poisson_from_params(params: PoissonParams) -> Poisson:
    """Construct a Poisson from stored parameters."""
    return Poisson(log_rates=params['log_rates'])


def linear_gaussian_to_params(
    dist: LinearGaussian,
) -> LinearGaussianParams:
    """Return the stored parameters of a full or diagonal linear-Gaussian conditional."""
    if isinstance(dist, DiagonalLinearGaussian):
        return {
            'coefficients': dist.affine.coefficients,
            'bias': dist.affine.bias,
            'variances': dist.variances,
        }
    return {
        'coefficients': dist.affine.coefficients,
        'bias': dist.affine.bias,
        'covariance': dist.covariance,
    }


def linear_gaussian_from_params(
    params: LinearGaussianParams,
) -> LinearGaussian:
    """Construct a full or diagonal linear-Gaussian conditional from stored parameters."""
    if 'variances' in params:
        return DiagonalLinearGaussian.from_components(
            params['coefficients'], params['bias'], params['variances']
        )
    return FullLinearGaussian.from_components(
        params['coefficients'], params['bias'], params['covariance']
    )


def linear_poisson_to_params(dist: LinearPoisson) -> LinearPoissonParams:
    """Return the stored parameters of a linear-Poisson conditional."""
    return {
        'coefficients': dist.affine.coefficients,
        'bias': dist.affine.bias,
    }


def linear_poisson_from_params(params: LinearPoissonParams) -> LinearPoisson:
    """Construct a linear-Poisson conditional from stored parameters."""
    return LinearPoisson(affine=Affine(params['coefficients'], params['bias']))
