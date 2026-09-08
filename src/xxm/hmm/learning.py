"""HMM parameter learning via expectation-maximization."""

from __future__ import annotations

import jax

from xxm.core.inference import Inferred
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model, Posterior
from .inference import infer_exact


def em_step(
    inferred: Inferred[Model, Posterior],
    observations: jax.Array,
) -> Inferred[Model, Posterior]:
    """Perform one EM update from an existing posterior."""

    model = inferred.model.fit_params(
        observations,
        inferred.posterior,
    )

    return infer_exact(
        model,
        observations,
    )


def fit_em(
    model: Model,
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[Inferred[Model, Posterior]]:
    """Fit model via exact expectation-maximization."""

    inferred = infer_exact(
        model,
        observations,
    )

    return _fit_one(
        inferred,
        observations,
        num_iters=num_iters,
        step=em_step,
        progress=progress,
    )


def fit_em_many(
    models: tuple[Model, ...],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Multi-EM',
) -> FitCollection[Inferred[Model, Posterior]]:
    """Fit multiple model initializations in parallel via EM."""

    inferred = tuple(
        infer_exact(
            model,
            observations,
        )
        for model in models
    )

    return _fit_many(
        inferred,
        observations,
        num_iters=num_iters,
        step=em_step,
        progress=progress,
    )
