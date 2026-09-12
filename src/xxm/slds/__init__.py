"""Switching linear dynamical system public API."""

from .core import Posterior
from .inference import infer_variational
from .init import (
    init_gaussian_via_arhmm,
    init_gaussian_via_pca,
    init_poisson_via_arhmm,
    init_poisson_via_pca,
)
from .learning import (
    fit_laplace_em,
    fit_variational_em,
)
from .model import GaussianSLDS, PoissonSLDS

__all__ = [
    'GaussianSLDS',
    'PoissonSLDS',
    'Posterior',
    'fit_laplace_em',
    'fit_variational_em',
    'infer_variational',
    'init_gaussian_via_arhmm',
    'init_gaussian_via_pca',
    'init_poisson_via_arhmm',
    'init_poisson_via_pca',
]
