from __future__ import annotations

import jax

from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import infer_laplace, infer_variational

# ----------------------------------------------------------------------------------------------------
# Variational structured mean-field EM (for gaussian emissions)


def variational_em_step(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    *,
    num_inference_iters: int,
) -> tuple[Model[QuadraticEmissionsT], jax.Array]:
    """Perform one structured variational EM update."""

    posterior, elbo = infer_variational(
        model,
        observations,
        num_iters=num_inference_iters,
    )

    new_model = model.fit_params(
        observations,
        posterior,
    )

    return new_model, elbo


def fit_variational_em(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    *,
    num_iters: int,
    num_inference_iters: int,
    progress: bool | str = 'Variational EM',
) -> Fit[Model[QuadraticEmissionsT]]:
    """Fit an SLDS with structured variational EM."""

    return _fit_one(
        model,
        observations,
        num_iters=num_iters,
        step=lambda model, observations: variational_em_step(
            model,
            observations,
            num_inference_iters=num_inference_iters,
        ),
        objective=lambda model, observations: infer_variational(
            model,
            observations,
            num_iters=num_inference_iters,
        )[1],
        progress=progress,
    )


def fit_variational_em_many(
    models: tuple[Model[QuadraticEmissionsT], ...],
    observations: jax.Array,
    *,
    num_iters: int,
    num_inference_iters: int,
    progress: bool | str = 'Multi-Variational EM',
) -> FitCollection[Model[QuadraticEmissionsT]]:
    """Fit multiple SLDS initializations independently."""

    return _fit_many(
        models,
        observations,
        num_iters=num_iters,
        step=lambda model, observations: variational_em_step(
            model,
            observations,
            num_inference_iters=num_inference_iters,
        ),
        objective=lambda model, observations: infer_variational(
            model,
            observations,
            num_iters=num_inference_iters,
        )[1],
        progress=progress,
    )


# ----------------------------------------------------------------------------------------------------
# Laplace structural EM (for poisson emissions)


def laplace_em_step(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    *,
    num_inference_iters: int,
    params: OptimParams,
) -> tuple[Model[LaplaceEmissionsT], jax.Array]:
    """Perform one structured Laplace EM update."""

    inferred = infer_laplace(
        model,
        observations,
        num_iters=num_inference_iters,
        params=params,
    )

    new_model = model.fit_params(
        observations,
        inferred.posterior,
    )

    return new_model, inferred.objective


def fit_laplace_em(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    *,
    num_iters: int,
    num_inference_iters: int,
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    progress: bool | str = 'Laplace EM',
) -> Fit[Model[LaplaceEmissionsT]]:
    """Fit an SLDS with structured Laplace EM."""

    laplace_params = laplace_params or OptimParams()

    return _fit_one(
        model,
        observations,
        num_iters=num_iters,
        step=lambda model, observations: laplace_em_step(
            model,
            observations,
            num_inference_iters=num_inference_iters,
            params=laplace_params,
        ),
        objective=lambda model, observations: (
            infer_laplace(
                model,
                observations,
                params=laplace_params,
                num_iters=num_inference_iters,
            ).objective
        ),
        progress=progress,
    )


def fit_laplace_em_many(
    models: tuple[Model[LaplaceEmissionsT], ...],
    observations: jax.Array,
    *,
    num_iters: int,
    num_inference_iters: int,
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    progress: bool | str = 'Multi-Laplace EM',
) -> FitCollection[Model[LaplaceEmissionsT]]:
    """Fit multiple SLDS initializations independently with Laplace EM."""

    laplace_params = laplace_params or OptimParams()

    return _fit_many(
        models,
        observations,
        num_iters=num_iters,
        step=lambda model, observations: laplace_em_step(
            model,
            observations,
            num_inference_iters=num_inference_iters,
            params=laplace_params,
        ),
        objective=lambda model, observations: (
            infer_laplace(
                model,
                observations,
                num_iters=num_inference_iters,
                params=laplace_params,
            ).objective
        ),
        progress=progress,
    )
