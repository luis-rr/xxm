"""
Damped Newton optimization for one or more independent optimization problems.
The objective may be scalar or batched; parameter operations must follow the
same objective batch shape.
"""

import typing

import jax
import jax.numpy as jnp


class FreeParams(typing.Protocol):
    def take_step(
        self, direction: typing.Self, step_size: jax.Array
    ) -> typing.Self: ...
    def relative_change_from(self, other: typing.Self) -> jax.Array: ...
    def where(self, mask: jax.Array, other: typing.Self) -> typing.Self: ...


FreeParamsT = typing.TypeVar('FreeParamsT', bound=FreeParams)


class Model(typing.Protocol[FreeParamsT]):
    def objective(self, params: FreeParamsT) -> jax.Array: ...
    def newton_direction(self, params: FreeParamsT) -> FreeParamsT: ...


ModelT = typing.TypeVar('ModelT', bound=Model)


class NewtonState(typing.NamedTuple, typing.Generic[FreeParamsT]):
    """State for damped Newton optimization of Poisson readout parameters."""

    iteration: jax.Array
    params: FreeParamsT
    objective: jax.Array
    done: jax.Array


@jax.tree_util.register_static
class OptimParams(typing.NamedTuple):
    """Parameters for damped Newton optimization."""

    max_iter: int = 20
    tol: float = 1e-6
    max_line_search_iters: int = 20

    def validate(self) -> None:
        """Validate the parameters for Laplace inference."""

        if self.max_iter < 1:
            raise ValueError('max_iter must be at least 1')

        if self.tol <= 0.0:
            raise ValueError('tol must be positive')

        if self.max_line_search_iters < 0:
            raise ValueError('max_line_search_iters must be non-negative')


DEFAULT_OPTIM_PARAMS = OptimParams()


class NewtonSearch(typing.NamedTuple, typing.Generic[FreeParamsT]):
    model: Model[FreeParamsT]
    optim_params: OptimParams

    def initial_state(
        self,
        params: FreeParamsT,
    ) -> NewtonState[FreeParamsT]:

        current_objective = self.model.objective(params)

        return NewtonState(
            iteration=jnp.asarray(0),
            params=params,
            objective=current_objective,
            done=jnp.zeros_like(current_objective, dtype=bool),
        )

    def _newton_step(
        self, search_state: NewtonState[FreeParamsT]
    ) -> NewtonState[FreeParamsT]:
        """Perform one damped Newton iteration."""
        active = ~search_state.done
        direction = self.model.newton_direction(search_state.params)

        candidate_params, candidate_objective, accepted = (
            self._backtracking_line_search(
                params=search_state.params,
                direction=direction,
                current_objective=search_state.objective,
                active=active,
                max_iter=self.optim_params.max_line_search_iters,
            )
        )

        updated = active & accepted

        next_params = candidate_params.where(updated, search_state.params)
        next_objective = jnp.where(
            updated,
            candidate_objective,
            search_state.objective,
        )

        relative_change = next_params.relative_change_from(search_state.params)

        next_done = search_state.done | (
            active & ((relative_change <= self.optim_params.tol) | ~accepted)
        )

        return NewtonState(
            iteration=search_state.iteration + 1,
            params=next_params,
            objective=next_objective,
            done=next_done,
        )

    def optimize(self, params: FreeParamsT) -> NewtonState[FreeParamsT]:
        """Iteratively refine Poisson readout parameters with damped Newton steps."""
        initial = self.initial_state(params)
        return self._iterate(initial)

    def _iterate(self, initial: NewtonState[FreeParamsT]) -> NewtonState[FreeParamsT]:

        def should_continue(search_state: NewtonState[FreeParamsT]) -> jax.Array:
            return (search_state.iteration < self.optim_params.max_iter) & jnp.any(
                ~search_state.done
            )

        return jax.lax.while_loop(
            should_continue,
            self._newton_step,
            initial,
        )

    def _backtracking_line_search(
        self,
        params: FreeParamsT,
        direction: FreeParamsT,
        current_objective: jax.Array,
        active: jax.Array,
        max_iter: int,
    ) -> tuple[FreeParamsT, jax.Array, jax.Array]:
        """Find an improving step independently for each active neuron."""
        search = LineSearch(
            model=self.model,
            current_objective=current_objective,
            direction=direction,
            initial_params=params,
            active=active,
        )

        final = search.iterate(
            search.initial_state(),
            max_iter=max_iter,
        )

        return (
            final.candidate_params,
            final.candidate_objective,
            final.is_accepted(current_objective),
        )


class LineSearchState(typing.NamedTuple, typing.Generic[FreeParamsT]):
    """State for a batched backtracking line search."""

    iteration: jax.Array
    step_size: jax.Array
    candidate_params: FreeParamsT
    candidate_objective: jax.Array

    def is_accepted(self, current_objective: jax.Array) -> jax.Array:
        return jnp.isfinite(self.candidate_objective) & (
            self.candidate_objective >= current_objective
        )


class LineSearch(typing.NamedTuple, typing.Generic[FreeParamsT]):
    model: Model[FreeParamsT]
    current_objective: jax.Array
    direction: FreeParamsT
    initial_params: FreeParamsT
    active: jax.Array

    def initial_state(self) -> LineSearchState[FreeParamsT]:
        step_size = jnp.ones_like(self.current_objective)
        candidate_params = self.initial_params.take_step(
            self.direction,
            step_size,
        )

        return LineSearchState(
            iteration=jnp.asarray(0),
            step_size=step_size,
            candidate_params=candidate_params,
            candidate_objective=self.model.objective(candidate_params),
        )

    def iterate(
        self, initial: LineSearchState[FreeParamsT], max_iter: int
    ) -> LineSearchState[FreeParamsT]:
        def needs_backtracking(search: LineSearchState[FreeParamsT]) -> jax.Array:
            rejected = self.active & ~search.is_accepted(self.current_objective)

            return (search.iteration < max_iter) & jnp.any(rejected)

        def backtrack(
            search: LineSearchState[FreeParamsT],
        ) -> LineSearchState[FreeParamsT]:
            rejected = self.active & ~search.is_accepted(self.current_objective)

            step_size = jnp.where(
                rejected,
                0.5 * search.step_size,
                search.step_size,
            )

            candidate_params = self.initial_params.take_step(
                self.direction,
                step_size,
            )

            return LineSearchState(
                iteration=search.iteration + 1,
                step_size=step_size,
                candidate_params=candidate_params,
                candidate_objective=self.model.objective(candidate_params),
            )

        return jax.lax.while_loop(
            needs_backtracking,
            backtrack,
            initial,
        )
