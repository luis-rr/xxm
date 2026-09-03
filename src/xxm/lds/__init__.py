"""Linear dynamical system public API."""

from .align import align_procrustes
from .core import Posterior
from .inference import infer_exact, infer_laplace
from .init import (
    init_pca_gaussian,
    init_pca_gaussian_many,
    init_pca_poisson,
    init_pca_poisson_many,
)
from .learning import (
    fit_em,
    fit_em_many,
    fit_laplace_em,
    fit_laplace_em_many,
)
from .model import (
    GaussianLDS,
    PoissonLDS,
)

__all__ = [
    'GaussianLDS',
    'PoissonLDS',
    'Posterior',
    'align_procrustes',
    'fit_em',
    'fit_em_many',
    'fit_laplace_em',
    'fit_laplace_em_many',
    'infer_exact',
    'infer_laplace',
    'init_pca_gaussian',
    'init_pca_gaussian_many',
    'init_pca_poisson',
    'init_pca_poisson_many',
]
