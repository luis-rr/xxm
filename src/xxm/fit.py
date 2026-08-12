import typing

import jax
from jax import numpy as jnp

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


def fit_one(
    model: ModelT,
    data: DataT,
    num_iters: int,
    step: typing.Callable[
        [ModelT, DataT],
        tuple[ModelT, jax.Array],
    ],
    objective: typing.Callable[
        [ModelT, DataT],
        jax.Array,
    ],
) -> Fit[ModelT]:
    """Fit a single model with the given step function."""

    def _step(model, _):
        new_model, value = step(model, data)
        return new_model, value

    model, objective_trace = jax.lax.scan(
        _step,
        model,
        xs=None,
        length=num_iters,
    )

    final_value = objective(model, data)

    objective_trace = jnp.concatenate(
        [
            objective_trace,
            final_value[None],
        ]
    )

    return Fit(
        model=model,
        objective_trace=objective_trace,
    )


def fit_many(
    models: tuple[ModelT, ...],
    data: DataT,
    num_iters: int,
    step: typing.Callable[
        [ModelT, DataT],
        tuple[ModelT, jax.Array],
    ],
    objective: typing.Callable[
        [ModelT, DataT],
        jax.Array,
    ],
) -> FitCollection[ModelT]:
    """Fit multiple models independently."""

    batched_models = stack_models(models)

    batched_fit = jax.vmap(
        lambda model: fit_one(
            model,
            data,
            num_iters,
            step,
            objective,
        )
    )(batched_models)

    return FitCollection(
        models=unstack_models(batched_fit.model),
        objective_traces=batched_fit.objective_trace,
    )
