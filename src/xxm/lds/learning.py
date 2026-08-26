from __future__ import annotations

import typing

import jax

from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT

from ..optim.loop import Fit, FitCollection
from ..optim.loop import fit_many as _fit_many
from ..optim.loop import fit_one as _fit_one
from .model import Model
from .inference import inference_exact, inference_laplace

ModelT = typing.TypeVar('ModelT')
PosteriorT = typing.TypeVar('PosteriorT')


InferenceFn = typing.Callable[[ModelT, jax.Array], PosteriorT]


def em_step(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
) -> tuple[Model[QuadraticEmissionsT], jax.Array]:

    posterior, log_normalizer = inference_exact(
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
) -> tuple[Model[LaplaceEmissionsT], jax.Array]:

    posterior, log_normalizer = inference_laplace(
        model,
        observations,
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
) -> Fit[Model[QuadraticEmissionsT]]:

    return _fit_one(
        model,
        observations,
        num_iters,
        step=em_step,
        objective=lambda m, o: inference_exact(m, o)[1],
    )


def fit_em_many(
    models: tuple[Model[QuadraticEmissionsT], ...],
    observations: jax.Array,
    num_iters: int,
) -> FitCollection[Model[QuadraticEmissionsT]]:

    return _fit_many(
        models,
        observations,
        num_iters,
        step=em_step,
        objective=lambda m, o: inference_exact(m, o)[1],
    )


def fit_laplace_em(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    num_iters: int,
) -> Fit[Model[LaplaceEmissionsT]]:

    return _fit_one(
        model,
        observations,
        num_iters,
        step=laplace_em_step,
        objective=lambda m, o: inference_laplace(m, o)[1],
    )


def fit_laplace_em_many(
    models: tuple[Model[LaplaceEmissionsT], ...],
    observations: jax.Array,
    num_iters: int,
) -> FitCollection[Model[LaplaceEmissionsT]]:

    return _fit_many(
        models,
        observations,
        num_iters,
        step=laplace_em_step,
        objective=lambda m, o: inference_laplace(m, o)[1],
    )
