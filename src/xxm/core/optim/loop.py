import typing
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax_tqdm.scan_pbar import scan_tqdm

ModelT = typing.TypeVar('ModelT')
DataT = typing.TypeVar('DataT')


class Fit(typing.NamedTuple, typing.Generic[ModelT]):
    """
    Results from fitting a single model.

    Attributes:
        model: The fitted model.
        objective_trace: The value of the objective function (e.g. log-likelihood or ELBO)
            over iterations, including the final value (num_iters + 1,).
    """

    model: ModelT
    objective_trace: jax.Array


class FitCollection(typing.NamedTuple, typing.Generic[ModelT]):
    """Results from fitting multiple models."""

    models: tuple[ModelT, ...]
    objective_traces: jax.Array  # (M, num_iters + 1)

    def best_index(self) -> int:
        """Index of the best fit among those with NaN-free objective traces."""
        valid = self.is_valid()

        if not jnp.any(valid):
            raise ValueError('No fit has a NaN-free objective trace')

        valid_indices = jnp.where(valid)[0]
        valid_final_objectives = self.objective_traces[valid, -1]

        best_valid_index = jnp.argmax(valid_final_objectives)

        return int(valid_indices[best_valid_index])

    def is_valid(self) -> jax.Array:
        """Boolean array indicating which fits have NaN-free objective traces."""
        return ~jnp.isnan(self.objective_traces).any(axis=1)

    def best(self) -> Fit[ModelT]:
        """Return the fit with the highest final objective."""
        index = self.best_index()
        return self.get(index)

    def get(self, index: int) -> Fit[ModelT]:
        """Return the fit at the given index."""
        return Fit(
            model=self.models[index],
            objective_trace=self.objective_traces[index],
        )


def stack_models(values: tuple[ModelT, ...]) -> ModelT:
    """Stack a tuple of pytrees into a single pytree with an additional leading dimension."""

    if not values:
        raise ValueError('Cannot stack an empty tuple of pytrees')

    return jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *values,
    )


def unstack_models(value: ModelT) -> tuple[ModelT, ...]:
    """Unstack a pytree with a leading dimension into a tuple of pytrees."""

    leaves = jax.tree.leaves(value)

    if not leaves:
        raise ValueError('Cannot unstack a pytree with no leaves')

    num_items = leaves[0].shape[0]

    if any(x.shape[0] != num_items for x in leaves):
        raise ValueError('Pytree leaves do not share a common batch dimension')

    def take(index):
        return jax.tree.map(lambda x: x[index], value)

    return tuple(take(i) for i in range(num_items))


FitStep = Callable[
    [ModelT, DataT],
    tuple[ModelT, jax.Array],
]

_ScanStep = Callable[
    [ModelT, jax.Array],
    tuple[ModelT, jax.Array],
]

_ProgressDecorator = Callable[
    [_ScanStep[ModelT]],
    _ScanStep[ModelT],
]

Progress = bool | str | _ProgressDecorator[ModelT]

Objective = Callable[
    [ModelT, DataT],
    jax.Array,
]


def _add_progress_bar(
    step: _ScanStep,
    num_iters: int,
    progress: Progress[ModelT],
) -> _ScanStep:
    """Wrap a scan step with a progress bar if requested."""

    if progress is True:
        decorated = scan_tqdm(
            num_iters,
            tqdm_type='auto',
        )(step)

    elif progress is False:
        decorated = step

    elif isinstance(progress, str):
        decorated = scan_tqdm(
            num_iters,
            tqdm_type='auto',
            desc=progress,
        )(step)

    else:
        decorated = progress(step)

    return decorated


def _scan(
    model: ModelT,
    *,
    num_iters: int,
    step: _ScanStep[ModelT],
    progress: Progress[ModelT],
) -> tuple[ModelT, jax.Array]:
    """Run a fitting scan with optional progress reporting."""

    scan_step = _add_progress_bar(
        step,
        num_iters,
        progress,
    )

    return jax.lax.scan(
        scan_step,
        model,
        xs=jnp.arange(num_iters),
    )


def fit_one(
    model: ModelT,
    data: DataT,
    *,
    num_iters: int,
    step: FitStep[ModelT, DataT],
    objective: Objective[ModelT, DataT],
    progress: Progress[ModelT] = False,
) -> Fit[ModelT]:
    """Fit a single model with the given step function."""

    def _step(
        model: ModelT,
        _: jax.Array,
    ) -> tuple[ModelT, jax.Array]:
        return step(model, data)

    model, objective_trace = _scan(
        model,
        num_iters=num_iters,
        step=_step,
        progress=progress,
    )

    final_value = objective(model, data)

    return Fit(
        model=model,
        objective_trace=jnp.concatenate(
            [
                objective_trace,
                final_value[None],
            ]
        ),
    )


def fit_many(
    models: tuple[ModelT, ...],
    data: DataT,
    *,
    num_iters: int,
    step: FitStep[ModelT, DataT],
    objective: Objective[ModelT, DataT],
    progress: Progress[ModelT] = False,
) -> FitCollection[ModelT]:
    """Fit multiple models independently."""

    stacked_models = stack_models(models)

    def _step(
        models: ModelT,
        _: jax.Array,
    ) -> tuple[ModelT, jax.Array]:
        return jax.vmap(lambda model: step(model, data))(models)

    stacked_models, objective_traces = _scan(
        stacked_models,
        num_iters=num_iters,
        step=_step,
        progress=progress,
    )

    final_values = jax.vmap(lambda model: objective(model, data))(stacked_models)

    objective_traces = jnp.concatenate(
        [
            objective_traces,
            final_values[None, :],
        ],
        axis=0,
    )

    return FitCollection(
        models=unstack_models(stacked_models),
        objective_traces=objective_traces.T,
    )
