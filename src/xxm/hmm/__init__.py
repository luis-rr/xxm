"""Hidden Markov model public API."""

from .core import Posterior
from .inference import infer_exact
from .init import (
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em
from .model import (
    GaussianHMM,
    PoissonHMM,
)

__all__ = [
    'GaussianHMM',
    'PoissonHMM',
    'Posterior',
    'fit_em',
    'infer_exact',
    'init_gaussian_via_kmeans',
    'init_poisson_via_kmeans',
]
