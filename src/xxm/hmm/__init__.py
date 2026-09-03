"""Hidden Markov model public API."""

from .core import Posterior
from .inference import infer_exact
from .init import init_gaussian, init_gaussian_ar, init_poisson, init_poisson_ar
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
    'init_gaussian',
    'init_gaussian_ar',
    'init_poisson',
    'init_poisson_ar',
]
