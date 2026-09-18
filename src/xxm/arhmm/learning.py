"""AR-HMM expectation-maximization with fixed prepared regression rows."""

import jax

from xxm.core.inference import Inferred
from xxm.core.optim.loop import Fit, FitCollection
from xxm.core.optim.loop import fit_many as _fit_many
from xxm.core.optim.loop import fit_one as _fit_one

from .core import Model, Posterior
from .data import ARObservations
from .emissions import ConditionalDistT
from .inference import _infer_prepared


def _em_step(
    inferred: Inferred[Model[ConditionalDistT], Posterior],
    data: ARObservations,
) -> Inferred[Model[ConditionalDistT], Posterior]:
    """Update parameters and infer states using the same aligned rows."""
    model = inferred.model.fit_params(data, inferred.posterior)
    return _infer_prepared(model, data)


def em_step(
    inferred: Inferred[Model[ConditionalDistT], Posterior],
    observations: jax.Array,
) -> Inferred[Model[ConditionalDistT], Posterior]:
    """Perform one exact EM update conditional on the initial history."""
    data = ARObservations.from_observations(observations, inferred.model.num_lags)
    return _em_step(inferred, data)


def fit_em(
    model: Model[ConditionalDistT],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'EM',
) -> Fit[Inferred[Model[ConditionalDistT], Posterior]]:
    """Fit via exact EM, preparing autoregressive rows once for the whole fit."""
    data = ARObservations.from_observations(observations, model.num_lags)
    inferred = _infer_prepared(model, data)
    return _fit_one(
        inferred,
        data,
        num_iters=num_iters,
        step=_em_step,
        progress=progress,
    )


def fit_em_many(
    models: tuple[Model[ConditionalDistT], ...],
    observations: jax.Array,
    num_iters: int,
    progress: bool | str = 'Multi-EM',
) -> FitCollection[Inferred[Model[ConditionalDistT], Posterior]]:
    """Fit multiple initializations with the same lag count in parallel."""
    if not models:
        raise ValueError('models must contain at least one initialization')
    num_lags = models[0].num_lags
    if any(model.num_lags != num_lags for model in models):
        raise ValueError('all initializations must have the same num_lags')
    data = ARObservations.from_observations(observations, num_lags)
    inferred = tuple(_infer_prepared(model, data) for model in models)
    return _fit_many(
        inferred,
        data,
        num_iters=num_iters,
        step=_em_step,
        progress=progress,
    )
