"""Hidden Markov model public API."""

from .core import Posterior
from .inference import infer_exact
from .init import (
    init_gaussian_ar_via_kmeans,
    init_gaussian_via_kmeans,
    init_poisson_ar_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em, fit_em_many
from .model import (
    GaussianARHMM,
    GaussianHMM,
    PoissonARHMM,
    PoissonHMM,
)

__all__ = [
    'GaussianARHMM',
    'GaussianHMM',
    'PoissonARHMM',
    'PoissonHMM',
    'Posterior',
    'fit_em',
    'fit_em_many',
    'infer_exact',
    'init_gaussian_ar_via_kmeans',
    'init_gaussian_via_kmeans',
    'init_poisson_ar_via_kmeans',
    'init_poisson_via_kmeans',
]
