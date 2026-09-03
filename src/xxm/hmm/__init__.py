"""Hidden Markov model public API."""

from .align import (
    match_states,
    match_states_by_conditional_mean,
    match_states_by_mean,
    match_true_states,
)
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
    'match_states',
    'match_states_by_conditional_mean',
    'match_states_by_mean',
    'match_true_states',
]
