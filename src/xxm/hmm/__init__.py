"""Hidden Markov model public API."""

from xxm.core.io.chains import load_discrete_posterior as load_posterior
from xxm.core.io.chains import save_discrete_posterior as save_posterior

from .core import Model, Posterior
from .inference import infer_exact
from .init import (
    cluster_observations,
    init_from_emissions,
    init_gaussian_via_kmeans,
    init_poisson_via_kmeans,
)
from .learning import fit_em
from .model import (
    HMM,
    Fit,
    GaussianHMM,
    GaussianHMMCollection,
    HMMCollection,
    HMMInferredCollection,
    Inferred,
    PoissonHMM,
    PoissonHMMCollection,
)

__all__ = [
    'HMM',
    'Fit',
    'GaussianHMM',
    'GaussianHMMCollection',
    'HMMCollection',
    'HMMInferredCollection',
    'Inferred',
    'Model',
    'PoissonHMM',
    'PoissonHMMCollection',
    'Posterior',
    'cluster_observations',
    'fit_em',
    'infer_exact',
    'init_from_emissions',
    'init_gaussian_via_kmeans',
    'init_poisson_via_kmeans',
    'load_posterior',
    'save_posterior',
]
