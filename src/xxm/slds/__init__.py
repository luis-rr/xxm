"""Switching linear dynamical system public API."""

from .core import Posterior
from .inference import infer_variational
from .init import (
    init_arhmm_gaussian,
    init_arhmm_poisson,
    init_pca_gaussian,
    init_pca_poisson,
)
from .learning import (
    fit_laplace_em,
    fit_laplace_em_many,
    fit_variational_em,
    fit_variational_em_many,
)
from .model import GaussianSLDS, PoissonSLDS

__all__ = [
    'GaussianSLDS',
    'PoissonSLDS',
    'Posterior',
    'fit_laplace_em',
    'fit_laplace_em_many',
    'fit_variational_em',
    'fit_variational_em_many',
    'infer_variational',
    'init_arhmm_gaussian',
    'init_arhmm_poisson',
    'init_pca_gaussian',
    'init_pca_poisson',
]
