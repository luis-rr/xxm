from __future__ import annotations

import jax

from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model
from .inference import infer_exact


def em_step(
    model: Model,
    observations: jax.Array,
) -> tuple[Model, jax.Array]:

    posterior, log_normalizer = infer_exact(
        model,
        observations,
    )

    new_model = model.fit_params(
        observations,
        posterior,
    )

    return new_model, log_normalizer


def fit_em(
    model: Model,
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[Model]:

    return _fit_one(
        model,
        observations,
        num_iters=num_iters,
        step=em_step,
        objective=lambda m, o: infer_exact(m, o)[1],
        progress=progress,
    )


def fit_em_many(
    models: tuple[Model, ...],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Multi-EM',
) -> FitCollection[Model]:

    return _fit_many(
        models,
        observations,
        num_iters=num_iters,
        step=em_step,
        objective=lambda m, o: infer_exact(m, o)[1],
        progress=progress,
    )
