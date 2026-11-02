"""Switching linear dynamical system public API."""

from .core import (
    Model,
    Posterior,
)
from .inference import (
    infer_laplace,
    infer_variational,
)
from .init import (
    gaussian_emissions_from_pseudo_latents,
    init_from_arhmm,
    poisson_emissions_from_pseudo_latents,
    pseudo_latents_via_log_pca,
    pseudo_latents_via_pca,
)
from .io import SLDSParams
from .learning import (
    fit_laplace_em,
    fit_variational_em,
)
from .model import (
    GaussianSLDS,
    Inferred,
    PoissonSLDS,
)

__all__ = [
    'GaussianSLDS',
    'Inferred',
    'Model',
    'PoissonSLDS',
    'Posterior',
    'SLDSParams',
    'fit_laplace_em',
    'fit_variational_em',
    'gaussian_emissions_from_pseudo_latents',
    'infer_laplace',
    'infer_variational',
    'init_from_arhmm',
    'poisson_emissions_from_pseudo_latents',
    'pseudo_latents_via_log_pca',
    'pseudo_latents_via_pca',
]
