"""Fitting Gaussian models of any covariance structure.

Package-wide defaults, and the dispatch between `gaussian_full` and
`gaussian_diagonal`, which expose the same observation fits under the same
names.
"""

import typing

import jax

from xxm.core.dists.gaussian import (
    DiagonalGaussian,
    DiagonalLinearGaussian,
    FullGaussian,
    Gaussian,
    LinearGaussian,
)
from xxm.core.optim import gaussian_diagonal as gaussian_diag_fit
from xxm.core.optim import gaussian_full as gaussian_full_fit

DEFAULT_RIDGE = 1e-6  # package-wide default for relative ridge regularization

# package-wide default for relative covariance floor regularization at fitting
DEFAULT_COV_FLOOR = 1e-6

# package-wide default for relative covariance floor regularization at
#  initialization, which is more conservative than the default for fitting
DEFAULT_COV_FLOOR_INIT = 1e-2


CovarianceType = typing.Literal['diagonal', 'full']


class GaussianFit(typing.Protocol):
    """Observation fits shared by the full and diagonal fitting modules."""

    def from_samples_weighted(
        self,
        values: jax.Array,
        weights: jax.Array,
        *,
        covariance_floor: float,
    ) -> Gaussian: ...

    def from_samples_grouped(
        self,
        values: jax.Array,
        assignments: jax.Array,
        num_groups: int,
        *,
        covariance_floor: float,
        sample_weights: jax.Array | None = None,
    ) -> Gaussian: ...

    def linear_from_marginals(
        self,
        inputs: FullGaussian,
        outputs: jax.Array,
        *,
        weights: jax.Array | None = None,
        ridge: float,
        covariance_floor: float,
    ) -> LinearGaussian: ...

    def linear_from_samples(
        self,
        inputs: jax.Array,
        outputs: jax.Array,
        *,
        ridge: float,
        covariance_floor: float,
    ) -> LinearGaussian: ...

    def linear_from_samples_weighted(
        self,
        inputs: jax.Array,
        outputs: jax.Array,
        weights: jax.Array,
        *,
        ridge: float,
        covariance_floor: float,
    ) -> LinearGaussian: ...

    def linear_from_samples_grouped(
        self,
        inputs: jax.Array,
        outputs: jax.Array,
        assignments: jax.Array,
        num_groups: int,
        *,
        ridge: float,
        covariance_floor: float,
    ) -> LinearGaussian: ...


def covariance_type(dist: Gaussian | LinearGaussian) -> CovarianceType:
    """Covariance structure of a Gaussian or linear-Gaussian distribution."""
    if isinstance(dist, DiagonalGaussian | DiagonalLinearGaussian):
        return 'diagonal'
    return 'full'


def gaussian_fit_for(noise: CovarianceType) -> GaussianFit:
    """Fitting module for the given covariance structure."""
    return gaussian_diag_fit if noise == 'diagonal' else gaussian_full_fit


def gaussian_fit_for_dist(dist: Gaussian | LinearGaussian) -> GaussianFit:
    """Fitting module matching the covariance structure of `dist`."""
    return gaussian_fit_for(covariance_type(dist))
