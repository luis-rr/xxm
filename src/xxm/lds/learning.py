from __future__ import annotations

import typing

import jax

from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import infer_exact, infer_laplace

ModelT = typing.TypeVar('ModelT')
PosteriorT = typing.TypeVar('PosteriorT')


InferenceFn = typing.Callable[[ModelT, jax.Array], PosteriorT]


def em_step(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
) -> tuple[Model[QuadraticEmissionsT], jax.Array]:

    posterior, log_normalizer = infer_exact(
        model,
        observations,
    )

    new_model = model.fit_params(
        observations,
        posterior,
    )

    return new_model, log_normalizer


def laplace_em_step(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    params: OptimParams,
) -> tuple[Model[LaplaceEmissionsT], jax.Array]:

    posterior, log_normalizer = infer_laplace(
        model,
        observations,
        params=params,
    )

    new_model = model.fit_params(
        observations,
        posterior,
    )

    return new_model, log_normalizer


def fit_em(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[Model[QuadraticEmissionsT]]:

    return _fit_one(
        model,
        observations,
        num_iters=num_iters,
        step=em_step,
        objective=lambda m, o: infer_exact(m, o)[1],
        progress=progress,
    )


def fit_em_many(
    models: tuple[Model[QuadraticEmissionsT], ...],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Multi-EM',
) -> FitCollection[Model[QuadraticEmissionsT]]:

    return _fit_many(
        models,
        observations,
        num_iters=num_iters,
        step=em_step,
        objective=lambda m, o: infer_exact(m, o)[1],
        progress=progress,
    )


def fit_laplace_em(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Laplace EM',
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Fit[Model[LaplaceEmissionsT]]:

    laplace_params = laplace_params or OptimParams()

    return _fit_one(
        model,
        observations,
        num_iters=num_iters,
        step=lambda m, o: laplace_em_step(m, o, params=laplace_params),
        objective=lambda m, o: infer_laplace(
            model=m,
            observations=o,
            params=laplace_params,
        )[1],
        progress=progress,
    )


def fit_laplace_em_many(
    models: tuple[Model[LaplaceEmissionsT], ...],
    observations: jax.Array,
    num_iters: int,
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    progress: bool | str = 'Multi-Laplace EM',
) -> FitCollection[Model[LaplaceEmissionsT]]:

    laplace_params = laplace_params or OptimParams()

    return _fit_many(
        models,
        observations,
        num_iters=num_iters,
        step=lambda m, o: laplace_em_step(m, o, params=laplace_params),
        objective=lambda m, o: infer_laplace(
            model=m,
            observations=o,
            params=laplace_params,
        )[1],
        progress=progress,
    )
