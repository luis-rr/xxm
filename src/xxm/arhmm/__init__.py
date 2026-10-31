"""Autoregressive hidden Markov model public API."""

from xxm.hmm.io import HMMParams

from .core import (
    Model,
    Posterior,
)
from .inference import infer_exact
from .init import (
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em
from .model import (
    GaussianARHMM,
    Inferred,
    PoissonARHMM,
)

__all__ = [
    'GaussianARHMM',
    'HMMParams',
    'Inferred',
    'Model',
    'PoissonARHMM',
    'Posterior',
    'fit_em',
    'infer_exact',
    'init_gaussian_via_kmeans',
    'init_poisson_via_kmeans',
]
