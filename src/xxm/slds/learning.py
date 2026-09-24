"""SLDS parameter learning via expectation-maximization."""

from __future__ import annotations

import jax

from xxm.core.data import Dataset
from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.inference import InferenceState
from xxm.core.optim.loop import Fit
from xxm.core.optim.loop import fit_one as _fit_one
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model, Posterior
from .inference import infer_laplace, infer_variational

# ----------------------------------------------------------------------------------------------------
# Variational structured mean-field EM (for Gaussian emissions)


def variational_em_step(
    inferred: InferenceState[Model[QuadraticEmissionsT], Posterior],
    data: Dataset,
    *,
    num_inference_iters: int,
) -> InferenceState[Model[QuadraticEmissionsT], Posterior]:
    """Perform one structured variational EM update."""

    model = inferred.model.fit_params(data, inferred.posterior)

    return infer_variational(
        model,
        data,
        num_iters=num_inference_iters,
        initial_latents=inferred.posterior.continuous.means,
    )


def fit_variational_em(
    model: Model[QuadraticEmissionsT],
    data: Dataset,
    *,
    num_iters: int,
    num_inference_iters: int,
    initial_latents: jax.Array,
    progress: bool | str = 'Variational EM',
) -> Fit[InferenceState[Model[QuadraticEmissionsT], Posterior]]:
    """Fit an SLDS with structured variational EM."""

    inferred = infer_variational(
        model,
        data,
        num_iters=num_inference_iters,
        initial_latents=initial_latents,
    )

    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=lambda inferred, data: variational_em_step(
            inferred,
            data,
            num_inference_iters=num_inference_iters,
        ),
        progress=progress,
    )


# ----------------------------------------------------------------------------------------------------
# Laplace structural EM (for Poisson emissions)


def laplace_em_step(
    inferred: InferenceState[Model[LaplaceEmissionsT], Posterior],
    data: Dataset,
    *,
    num_inference_iters: int,
    params: OptimParams,
) -> InferenceState[Model[LaplaceEmissionsT], Posterior]:
    """Perform one structured Laplace EM update."""

    model = inferred.model.fit_params(
        data,
        inferred.posterior,
    )

    return infer_laplace(
        model,
        data,
        num_iters=num_inference_iters,
        initial_latents=inferred.posterior.continuous.means,
        params=params,
    )


def fit_laplace_em(
    model: Model[LaplaceEmissionsT],
    data: Dataset,
    *,
    num_iters: int,
    num_inference_iters: int,
    initial_latents: jax.Array,
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    progress: bool | str = 'Laplace EM',
) -> Fit[InferenceState[Model[LaplaceEmissionsT], Posterior]]:
    """Fit an SLDS with structured Laplace EM."""

    inferred = infer_laplace(
        model,
        data,
        num_iters=num_inference_iters,
        initial_latents=initial_latents,
        params=laplace_params,
    )

    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=lambda inferred, data: laplace_em_step(
            inferred,
            data,
            num_inference_iters=num_inference_iters,
            params=laplace_params,
        ),
        progress=progress,
    )
