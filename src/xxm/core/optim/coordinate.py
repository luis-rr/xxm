"""Coordinate ascent."""

import typing

import jax
import jax.numpy as jnp

StateT = typing.TypeVar('StateT')
ValueT = typing.TypeVar('ValueT')


Evaluator = typing.Callable[[StateT], jax.Array]

Coordinate = typing.Callable[[StateT, jax.Array], ValueT]


Updater = typing.Callable[
    [StateT, jax.Array, ValueT],
    StateT,
]
Proposer = typing.Callable[
    [
        jax.Array,
        typing.Callable[[ValueT], jax.Array],
        ValueT,
    ],
    ValueT,
]


def coordinate_ascent(
    key: jax.Array,
    state: StateT,
    *,
    num_coordinates: int,
    objective: Evaluator[StateT],
    coordinate: Coordinate[StateT, ValueT],
    update: Updater[StateT, ValueT],
    propose: Proposer[ValueT],
    active: jax.Array,
    max_sweeps: int = 50,
) -> StateT:
    """Maximize an objective through sequential coordinate updates."""

    def condition(carry):
        _, _, _, iteration, improved = carry
        return (iteration < max_sweeps) & improved

    def sweep(carry):
        state, score, key, iteration, _ = carry

        def step(index, carry):
            state, score, key, improved = carry

            key, proposal_key = jax.random.split(key)

            def optimize(values):
                state, score = values

                evaluate = lambda value: objective(update(state, index, value))

                value = propose(
                    proposal_key,
                    evaluate,
                    coordinate(state, index),
                )

                candidate = update(state, index, value)
                candidate_score = objective(candidate)

                accepted = jnp.isfinite(candidate_score) & (candidate_score > score)

                state, score = jax.lax.cond(
                    accepted,
                    lambda _: (candidate, candidate_score),
                    lambda _: (state, score),
                    operand=None,
                )

                return state, score, accepted

            def skip(values):
                state, score = values
                return state, score, jnp.asarray(False)

            state, score, accepted = jax.lax.cond(
                active[index],
                optimize,
                skip,
                (state, score),
            )

            return (
                state,
                score,
                key,
                improved | accepted,
            )

        state, score, key, improved = jax.lax.fori_loop(
            0,
            num_coordinates,
            step,
            (
                state,
                score,
                key,
                jnp.asarray(False),
            ),
        )

        return (
            state,
            score,
            key,
            iteration + 1,
            improved,
        )

    state, _, _, _, _ = jax.lax.while_loop(
        condition,
        sweep,
        (
            state,
            objective(state),
            key,
            jnp.asarray(0),
            jnp.asarray(True),
        ),
    )

    return state
