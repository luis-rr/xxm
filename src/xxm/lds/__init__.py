"""Linear dynamical system public API."""

from xxm.core.io.chains import load_gaussian_posterior as load_posterior
from xxm.core.io.chains import save_gaussian_posterior as save_posterior

from .core import Model, Posterior
from .inference import infer_exact, infer_laplace
from .init import (
    init_from_latents,
    init_gaussian_via_pca,
    init_poisson_via_pca,
    masked_pca_latents,
    observed_rows,
)
from .learning import fit_em, fit_laplace_em
from .model import (
    LDS,
    ExactLDS,
    ExactLDSCollection,
    Fit,
    GaussianLDS,
    GaussianLDSCollection,
    Inferred,
    LaplaceLDS,
    LaplaceLDSCollection,
    LDSCollection,
    LDSInferredCollection,
    PoissonLDS,
    PoissonLDSCollection,
)

__all__ = [
    'LDS',
    'ExactLDS',
    'ExactLDSCollection',
    'Fit',
    'GaussianLDS',
    'GaussianLDSCollection',
    'Inferred',
    'LDSCollection',
    'LDSInferredCollection',
    'LaplaceLDS',
    'LaplaceLDSCollection',
    'Model',
    'PoissonLDS',
    'PoissonLDSCollection',
    'Posterior',
    'fit_em',
    'fit_laplace_em',
    'infer_exact',
    'infer_laplace',
    'init_from_latents',
    'init_gaussian_via_pca',
    'init_poisson_via_pca',
    'load_posterior',
    'masked_pca_latents',
    'observed_rows',
    'save_posterior',
]
