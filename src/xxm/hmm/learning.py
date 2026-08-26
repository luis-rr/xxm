from __future__ import annotations

import jax

from ..optim.loop import Fit
from ..optim.loop import fit_one as _fit_one
from .model import Model
from .inference import inference_exact


def em_step(
    model: Model,
    observations: jax.Array,
) -> tuple[Model, jax.Array]:

    posterior, log_normalizer = inference_exact(
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
) -> Fit[Model]:

    return _fit_one(
        model,
        observations,
        num_iters,
        step=em_step,
        objective=lambda m, o: inference_exact(m, o)[1],
    )
