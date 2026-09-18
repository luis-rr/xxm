"""Autoregressive Hidden Markov model public API."""

from .core import Posterior
from .inference import infer_exact
from .init import (
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em, fit_em_many
from .model import (
    GaussianARHMM,
    PoissonARHMM,
)

__all__ = [
    'GaussianARHMM',
    'PoissonARHMM',
    'Posterior',
    'fit_em',
    'fit_em_many',
    'infer_exact',
    'init_gaussian_via_kmeans',
    'init_poisson_via_kmeans',
]
