"""Switching linear dynamical system public API."""

from .core import Posterior
from .inference import infer_variational
from .init import (
    init_arhmm_gaussian,
    init_pca_gaussian,
)
from .learning import (
    fit_variational_em,
    fit_variational_em_many,
)
from .model import GaussianSLDS

__all__ = [
    'GaussianSLDS',
    'Posterior',
    'fit_variational_em',
    'fit_variational_em_many',
    'infer_variational',
    'init_arhmm_gaussian',
    'init_pca_gaussian',
]
