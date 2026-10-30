"""Autoregressive Hidden Markov model public API."""

from xxm.core.io.chains import load_discrete_posterior as load_posterior
from xxm.core.io.chains import save_discrete_posterior as save_posterior

from .core import Model, Posterior
from .inference import infer_exact
from .init import (
    cluster_rows,
    init_from_emissions,
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em
from .model import (
    ARHMM,
    ARHMMCollection,
    ARHMMInferredCollection,
    Fit,
    GaussianARHMM,
    GaussianARHMMCollection,
    Inferred,
    PoissonARHMM,
    PoissonARHMMCollection,
)

__all__ = [
    'ARHMM',
    'ARHMMCollection',
    'ARHMMInferredCollection',
    'Fit',
    'GaussianARHMM',
    'GaussianARHMMCollection',
    'Inferred',
    'Model',
    'PoissonARHMM',
    'PoissonARHMMCollection',
    'Posterior',
    'cluster_rows',
    'fit_em',
    'infer_exact',
    'init_from_emissions',
    'init_gaussian_via_kmeans',
    'init_poisson_via_kmeans',
    'load_posterior',
    'save_posterior',
]
