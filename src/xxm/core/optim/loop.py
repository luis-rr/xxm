"""Iterative fitting loops and convergence control."""

import typing
from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax_tqdm.scan_pbar import scan_tqdm


class FitState(typing.Protocol):
    """State carried by an iterative fitting procedure."""

    @property
    def objective(self) -> jax.Array:
        """Current fitting objective."""
        ...


StateT = typing.TypeVar(
    'StateT',
    bound=FitState,
)
DataT = typing.TypeVar('DataT')
PytreeT = typing.TypeVar('PytreeT')


class Fit(typing.NamedTuple, typing.Generic[StateT]):
    """Final fitting state and objective history."""

    state: StateT
    objective_trace: jax.Array


class FitCollection(typing.NamedTuple, typing.Generic[StateT]):
    """Results from fitting multiple states independently."""

    states: tuple[StateT, ...]
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

    def best(self) -> Fit[StateT]:
        """Return the fit with the highest final objective."""

        return self.get(self.best_index())

    def get(self, index: int) -> Fit[StateT]:
        """Return one fit from the collection."""

        return Fit(
            state=self.states[index],
            objective_trace=self.objective_traces[index],
        )


def stack_states(values: tuple[StateT, ...]) -> StateT:
    """Stack pytrees along a new leading dimension."""

    if not values:
        raise ValueError('Cannot stack an empty tuple of pytrees')

    return jax.tree.map(
        lambda *xs: jnp.stack(xs),
        *values,
    )


def unstack_states(value: PytreeT) -> tuple[PytreeT, ...]:
    """Unstack a pytree along its leading dimension."""

    leaves = jax.tree.leaves(value)

    if not leaves:
        raise ValueError('Cannot unstack a pytree with no leaves')

    num_items = leaves[0].shape[0]

    if any(leaf.shape[0] != num_items for leaf in leaves):
        raise ValueError('Pytree leaves do not share a common batch dimension')

    def take(index: int) -> PytreeT:
        return jax.tree.map(lambda x: x[index], value)

    return tuple(take(index) for index in range(num_items))


FitStep = Callable[
    [StateT, DataT],
    StateT,
]

_ScanStep = Callable[
    [StateT, jax.Array],
    tuple[StateT, jax.Array],
]

_ProgressDecorator = Callable[
    [_ScanStep[StateT]],
    _ScanStep[StateT],
]

Progress = bool | str | _ProgressDecorator[StateT]


def _add_progress_bar(
    step: _ScanStep[StateT],
    num_iters: int,
    progress: Progress[StateT],
) -> _ScanStep[StateT]:
    """Wrap a scan step with a progress bar if requested."""

    if progress is False:
        return step

    if progress is True:
        decorated = scan_tqdm(
            num_iters,
            tqdm_type='auto',
        )(step)

    elif isinstance(progress, str):
        decorated = scan_tqdm(
            num_iters,
            tqdm_type='auto',
            desc=progress,
        )(step)

    else:
        return progress(step)

    return typing.cast(
        _ScanStep[StateT],
        decorated,
    )


def _scan(
    state: StateT,
    *,
    num_iters: int,
    step: _ScanStep[StateT],
    progress: Progress[StateT],
) -> tuple[StateT, jax.Array]:
    """Run a fitting scan with optional progress reporting."""

    scan_step = _add_progress_bar(
        step,
        num_iters,
        progress,
    )

    return jax.lax.scan(
        scan_step,
        state,
        xs=jnp.arange(num_iters),
    )


def fit_one(
    state: StateT,
    data: DataT,
    *,
    num_iters: int,
    step: FitStep[StateT, DataT],
    progress: Progress[StateT] = False,
) -> Fit[StateT]:
    """Iteratively update a fitting state."""

    initial_objective = state.objective

    def _step(
        state: StateT,
        _: jax.Array,
    ) -> tuple[StateT, jax.Array]:

        state = step(state, data)

        return (state, state.objective)

    state, objective_trace = _scan(
        state,
        num_iters=num_iters,
        step=_step,
        progress=progress,
    )

    return Fit(
        state=state,
        objective_trace=jnp.concatenate(
            [
                initial_objective[None],
                objective_trace,
            ]
        ),
    )


def fit_many(
    states: tuple[StateT, ...],
    data: DataT,
    *,
    num_iters: int,
    step: FitStep[StateT, DataT],
    progress: Progress[StateT] = False,
) -> FitCollection[StateT]:
    """Fit multiple states independently."""

    stacked_states = stack_states(
        states,
    )

    initial_objectives = jax.vmap(lambda state: state.objective)(stacked_states)

    def _step(
        states: StateT,
        _: jax.Array,
    ) -> tuple[StateT, jax.Array]:

        states = jax.vmap(
            lambda state: step(
                state,
                data,
            )
        )(states)

        objectives = jax.vmap(lambda state: state.objective)(states)

        return (
            states,
            objectives,
        )

    stacked_states, objective_traces = _scan(
        stacked_states,
        num_iters=num_iters,
        step=_step,
        progress=progress,
    )

    objective_traces = jnp.concatenate(
        [
            initial_objectives[None, :],
            objective_traces,
        ],
        axis=0,
    )

    return FitCollection(
        states=unstack_states(stacked_states),
        objective_traces=objective_traces.T,
    )
