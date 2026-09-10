"""LDS parameter learning via expectation-maximization."""

from __future__ import annotations

import jax

from xxm.core.chains.gaussian import (
    GaussianChainMarginals as Posterior,
)
from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.inference import Inferred
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import infer_exact, infer_laplace


def em_step(
    inferred: Inferred[
        Model[QuadraticEmissionsT],
        Posterior,
    ],
    observations: jax.Array,
) -> Inferred[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Perform one exact EM update."""

    model = inferred.model.fit_params(
        observations,
        inferred.posterior,
    )

    return infer_exact(
        model,
        observations,
    )


def laplace_em_step(
    inferred: Inferred[
        Model[LaplaceEmissionsT],
        Posterior,
    ],
    observations: jax.Array,
    params: OptimParams,
) -> Inferred[
    Model[LaplaceEmissionsT],
    Posterior,
]:
    """Perform one warm-started Laplace EM update."""

    model = inferred.model.fit_params(
        observations,
        inferred.posterior,
    )

    return infer_laplace(
        model,
        observations,
        initial_latents=inferred.posterior.means,
        params=params,
    )


def fit_em(
    model: Model[QuadraticEmissionsT],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[
    Inferred[
        Model[QuadraticEmissionsT],
        Posterior,
    ]
]:
    """Fit a quadratic-emission LDS by exact expectation-maximization."""

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
    models: tuple[
        Model[QuadraticEmissionsT],
        ...,
    ],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Multi-EM',
) -> FitCollection[
    Inferred[
        Model[QuadraticEmissionsT],
        Posterior,
    ]
]:
    """Fit multiple quadratic-emission LDS initializations by exact EM."""

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


def fit_laplace_em(
    model: Model[LaplaceEmissionsT],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Laplace EM',
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Fit[
    Inferred[
        Model[LaplaceEmissionsT],
        Posterior,
    ]
]:
    """Fit a nonconjugate LDS with Laplace-approximated EM."""

    inferred = infer_laplace(
        model,
        observations,
        params=laplace_params,
    )

    return _fit_one(
        inferred,
        observations,
        num_iters=num_iters,
        step=lambda inferred, observations: laplace_em_step(
            inferred,
            observations,
            params=laplace_params,
        ),
        progress=progress,
    )


def fit_laplace_em_many(
    models: tuple[
        Model[LaplaceEmissionsT],
        ...,
    ],
    observations: jax.Array,
    num_iters: int,
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
    progress: bool | str = 'Multi-Laplace EM',
) -> FitCollection[
    Inferred[
        Model[LaplaceEmissionsT],
        Posterior,
    ]
]:
    """Fit multiple nonconjugate LDS initializations with Laplace EM."""

    inferred = tuple(
        infer_laplace(
            model,
            observations,
            params=laplace_params,
        )
        for model in models
    )

    return _fit_many(
        inferred,
        observations,
        num_iters=num_iters,
        step=lambda inferred, observations: laplace_em_step(
            inferred,
            observations,
            params=laplace_params,
        ),
        progress=progress,
    )
