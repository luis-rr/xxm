"""LDS parameter learning via expectation-maximization."""

from __future__ import annotations

from xxm.core.data import DatasetCollection
from xxm.core.dists.gaussian import GaussianChainMarginals as Posterior
from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.inference import InferenceState
from xxm.core.optim.loop import Fit, Progress
from xxm.core.optim.loop import fit_one as _fit_one
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import infer_exact, infer_laplace


def em_step(
    inferred: InferenceState[
        Model[QuadraticEmissionsT],
        Posterior,
    ],
    data: DatasetCollection,
) -> InferenceState[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Perform one pooled exact EM update across independent sequences."""
    model = inferred.model.fit_params(
        inferred.posterior,
        data,
    )

    return infer_exact(
        model,
        data,
    )


def laplace_em_step(
    inferred: InferenceState[
        Model[LaplaceEmissionsT],
        Posterior,
    ],
    data: DatasetCollection,
    params: OptimParams,
) -> InferenceState[
    Model[LaplaceEmissionsT],
    Posterior,
]:
    """Perform one warm-started pooled Laplace EM update."""
    model = inferred.model.fit_params(
        inferred.posterior,
        data,
    )

    return infer_laplace(
        model,
        data,
        initial_latents=inferred.posterior.means,
        params=params,
    )


def fit_em(
    model: Model[QuadraticEmissionsT],
    data: DatasetCollection,
    num_iters: int,
    progress: Progress = 'EM',
) -> Fit[
    InferenceState[
        Model[QuadraticEmissionsT],
        Posterior,
    ]
]:
    """Fit a quadratic-emission LDS by pooled exact EM."""
    inferred = infer_exact(
        model,
        data,
    )

    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=em_step,
        progress=progress,
    )


def fit_laplace_em(
    model: Model[LaplaceEmissionsT],
    data: DatasetCollection,
    num_iters: int,
    progress: Progress = 'Laplace EM',
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Fit[
    InferenceState[
        Model[LaplaceEmissionsT],
        Posterior,
    ]
]:
    """Fit a nonconjugate LDS with pooled Laplace-approximated EM."""
    inferred = infer_laplace(
        model,
        data,
        params=laplace_params,
    )

    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=lambda inferred, data: laplace_em_step(
            inferred,
            data,
            params=laplace_params,
        ),
        progress=progress,
    )
