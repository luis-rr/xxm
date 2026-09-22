"""LDS parameter learning via expectation-maximization."""

from __future__ import annotations

from xxm.core.chains.gaussian import GaussianChainMarginals as Posterior
from xxm.core.data import Sequences
from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.inference import Inferred
from xxm.core.optim.loop import Fit
from xxm.core.optim.loop import fit_one as _fit_one
from xxm.core.optim.newton import DEFAULT_OPTIM_PARAMS, OptimParams

from .core import Model
from .inference import infer_exact, infer_laplace


def em_step(
    inferred: Inferred[
        Model[QuadraticEmissionsT],
        Posterior,
    ],
    data: Sequences,
) -> Inferred[
    Model[QuadraticEmissionsT],
    Posterior,
]:
    """Perform one pooled exact EM update across independent sequences."""
    model = inferred.model.fit_params(
        data,
        inferred.posterior,
    )

    return infer_exact(
        model,
        data,
    )


def laplace_em_step(
    inferred: Inferred[
        Model[LaplaceEmissionsT],
        Posterior,
    ],
    data: Sequences,
    params: OptimParams,
) -> Inferred[
    Model[LaplaceEmissionsT],
    Posterior,
]:
    """Perform one warm-started pooled Laplace EM update."""
    model = inferred.model.fit_params(
        data,
        inferred.posterior,
    )

    return infer_laplace(
        model,
        data,
        initial_latents=inferred.posterior.means,
        params=params,
    )


def fit_em(
    model: Model[QuadraticEmissionsT],
    data: Sequences,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[
    Inferred[
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
    data: Sequences,
    num_iters: int,
    progress: bool | str = 'Laplace EM',
    laplace_params: OptimParams = DEFAULT_OPTIM_PARAMS,
) -> Fit[
    Inferred[
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
