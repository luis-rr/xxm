"""AR-HMM expectation-maximization on prepared autoregressive datasets."""

from xxm.core.inference import InferenceState
from xxm.core.optim.loop import Fit
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model, Posterior
from .data import ARDatasetCollection
from .emissions import ConditionalDistT
from .inference import infer_exact


def em_step(
    inferred: InferenceState[Model[ConditionalDistT], Posterior],
    data: ARDatasetCollection,
) -> InferenceState[Model[ConditionalDistT], Posterior]:
    """Perform one EM update from an existing posterior."""

    model = inferred.model.fit_params(
        inferred.posterior,
        data,
    )

    return infer_exact(model, data)


def fit_em(
    model: Model[ConditionalDistT],
    data: ARDatasetCollection,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[InferenceState[Model[ConditionalDistT], Posterior]]:
    """Fit the conditional AR-HMM using expectation-maximization."""

    inferred = infer_exact(model, data)

    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=em_step,
        progress=progress,
    )
