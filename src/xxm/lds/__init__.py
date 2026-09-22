"""Linear dynamical system public API."""

from xxm.core.data import Sequences

from .core import Posterior
from .inference import infer_exact, infer_laplace
from .init import (
    init_gaussian_via_pca,
    init_poisson_via_pca,
)
from .learning import fit_em, fit_laplace_em
from .model import GaussianLDS, PoissonLDS

__all__ = [
    'GaussianLDS',
    'PoissonLDS',
    'Posterior',
    'Sequences',
    'fit_em',
    'fit_laplace_em',
    'infer_exact',
    'infer_laplace',
    'init_gaussian_via_pca',
    'init_poisson_via_pca',
]
