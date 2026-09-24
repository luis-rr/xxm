"""HMM parameter learning via expectation-maximization."""

from __future__ import annotations

from xxm.core.data import Dataset
from xxm.core.inference import InferenceState
from xxm.core.optim.loop import Fit
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model, Posterior
from .inference import infer_exact


def em_step(
    inferred: InferenceState[Model, Posterior],
    data: Dataset,
) -> InferenceState[Model, Posterior]:
    """Perform one EM update from an existing posterior."""

    model = inferred.model.fit_params(
        data,
        inferred.posterior,
    )

    return infer_exact(
        model,
        data,
    )


def fit_em(
    model: Model,
    data: Dataset,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[InferenceState[Model, Posterior]]:
    """Fit model via exact expectation-maximization."""

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
