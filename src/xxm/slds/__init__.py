"""Switching linear dynamical system public API."""

from .core import Model, Posterior
from .inference import infer_laplace, infer_variational
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
from .model import Fit, GaussianSLDS, Inferred, PoissonSLDS
from .serialization import load_posterior, save_posterior

__all__ = [
    'Fit',
    'GaussianSLDS',
    'Inferred',
    'Model',
    'PoissonSLDS',
    'Posterior',
    'fit_laplace_em',
    'fit_variational_em',
    'infer_laplace',
    'infer_variational',
    'init_gaussian_via_arhmm',
    'init_gaussian_via_pca',
    'init_poisson_via_arhmm',
    'init_poisson_via_pca',
    'load_posterior',
    'save_posterior',
]
