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
