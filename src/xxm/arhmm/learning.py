"""AR-HMM expectation-maximization on prepared autoregressive datasets."""

from xxm.core.inference import InferenceState
from xxm.core.optim.loop import Fit
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model, Posterior
from .data import ARDatasetCollection
from .emissions import ConditionalDistT
from .inference import _infer_prepared


def _em_step(
    inferred: InferenceState[
        Model[ConditionalDistT],
        Posterior,
    ],
    data: ARDatasetCollection,
) -> InferenceState[
    Model[ConditionalDistT],
    Posterior,
]:
    model = inferred.model.fit_params(
        data,
        inferred.posterior,
    )

    return _infer_prepared(
        model,
        data,
    )


def em_step(
    inferred: InferenceState[
        Model[ConditionalDistT],
        Posterior,
    ],
    data: ARDatasetCollection,
) -> InferenceState[
    Model[ConditionalDistT],
    Posterior,
]:
    if data.input_dim != 0:
        raise ValueError('AR-HMM does not currently support known external inputs')

    if data.num_lags != inferred.model.num_lags:
        raise ValueError('Mismatch between data.num_lags and model.num_lags')

    return _em_step(
        inferred,
        data,
    )


def _fit_prepared(
    model: Model[ConditionalDistT],
    data: ARDatasetCollection,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[
    InferenceState[
        Model[ConditionalDistT],
        Posterior,
    ]
]:
    inferred = _infer_prepared(model, data)

    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=_em_step,
        progress=progress,
    )


def fit_em(
    model: Model[ConditionalDistT],
    data: ARDatasetCollection,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[
    InferenceState[
        Model[ConditionalDistT],
        Posterior,
    ]
]:
    if data.input_dim != 0:
        raise ValueError('AR-HMM does not currently support known external inputs')

    if data.num_lags != model.num_lags:
        raise ValueError('Mismatch between data.num_lags and model.num_lags')

    return _fit_prepared(
        model,
        data,
        num_iters=num_iters,
        progress=progress,
    )
