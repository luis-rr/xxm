from __future__ import annotations

import jax

from xxm.core.emissions.continuous import QuadraticEmissionsT
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model
from .inference import infer_variational


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
