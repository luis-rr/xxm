"""Switching linear dynamical system public API."""

from .core import Model, Posterior
from .inference import infer_laplace, infer_variational
from .init import (
    init_gaussian_via_arhmm,
    init_gaussian_via_pca,
    init_poisson_via_arhmm,
    init_poisson_via_pca,
    init_via_arhmm,
    init_via_pca,
    masked_pca_latents,
)
from .learning import (
    fit_laplace_em,
    fit_variational_em,
)
from .model import (
    SLDS,
    Fit,
    GaussianSLDS,
    GaussianSLDSCollection,
    Inferred,
    LaplaceSLDS,
    LaplaceSLDSCollection,
    PoissonSLDS,
    PoissonSLDSCollection,
    SLDSCollection,
    SLDSInferredCollection,
    VariationalSLDS,
    VariationalSLDSCollection,
)
from .serialization import load_posterior, save_posterior

__all__ = [
    'SLDS',
    'Fit',
    'GaussianSLDS',
    'GaussianSLDSCollection',
    'Inferred',
    'LaplaceSLDS',
    'LaplaceSLDSCollection',
    'Model',
    'PoissonSLDS',
    'PoissonSLDSCollection',
    'Posterior',
    'SLDSCollection',
    'SLDSInferredCollection',
    'VariationalSLDS',
    'VariationalSLDSCollection',
    'fit_laplace_em',
    'fit_variational_em',
    'infer_laplace',
    'infer_variational',
    'init_gaussian_via_arhmm',
    'init_gaussian_via_pca',
    'init_poisson_via_arhmm',
    'init_poisson_via_pca',
    'init_via_arhmm',
    'init_via_pca',
    'load_posterior',
    'masked_pca_latents',
    'save_posterior',
]
