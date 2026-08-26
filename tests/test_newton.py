# tests/test_newton.py

import typing

import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.optim.newton import LineSearch, NewtonSearch


class QuadraticParams(typing.NamedTuple):
    values: jax.Array

    def take_step(
        self,
        direction: typing.Self,
        step_size: jax.Array,
    ) -> typing.Self:
        return self.__class__(
            values=self.values + step_size[..., None] * direction.values,
        )

    def norm(self) -> jax.Array:
        return jnp.linalg.norm(self.values, axis=-1)

    def relative_change_from(self, other: typing.Self) -> jax.Array:
        distance = jnp.linalg.norm(self.values - other.values, axis=-1)
        return distance / (1.0 + other.norm())

    def where(
        self,
        mask: jax.Array,
        other: typing.Self,
    ) -> typing.Self:
        return self.__class__(
            values=jnp.where(mask[..., None], self.values, other.values),
        )


class QuadraticModel(typing.NamedTuple):
    """Concave quadratic with a known maximum at ``target``."""

    target: jax.Array

    def objective(self, params: QuadraticParams) -> jax.Array:
        error = params.values - self.target
        return -0.5 * jnp.sum(error**2, axis=-1)

    def newton_direction(self, params: QuadraticParams) -> QuadraticParams:
        # For -1/2 ||x - target||², Newton reaches the maximum in one step.
        return QuadraticParams(
            values=self.target - params.values,
        )


class WrongDirectionModel(typing.NamedTuple):
    """Quadratic problem whose proposed direction always moves downhill."""

    target: jax.Array

    def objective(self, params: QuadraticParams) -> jax.Array:
        error = params.values - self.target
        return -0.5 * jnp.sum(error**2, axis=-1)

    def newton_direction(self, params: QuadraticParams) -> QuadraticParams:
        return QuadraticParams(
            values=params.values - self.target,
        )


def _run_search(
    model,
    initial_params,
    *,
    max_iter=10,
    max_line_search_iters=10,
    tol=1e-6,
):
    search = NewtonSearch[QuadraticParams](
        model=model,
        max_line_search_iters=max_line_search_iters,
        tol=tol,
    )

    return search.optimize(params=initial_params, max_iter=max_iter)


def test_newton_search_solves_scalar_quadratic():
    """A single optimization problem reaches the analytical optimum."""

    model = QuadraticModel(
        target=jnp.array([2.0, -1.0]),
    )
    initial = QuadraticParams(
        values=jnp.array([0.0, 3.0]),
    )

    final = _run_search(
        model,
        initial,
        max_iter=1,
    )

    np.testing.assert_allclose(
        final.params.values,
        model.target,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        final.objective,
        0.0,
        atol=1e-6,
    )


def test_newton_search_solves_batched_quadratics():
    """Independent batched problems all reach their known optima."""

    model = QuadraticModel(
        target=jnp.array(
            [
                [1.0, 2.0],
                [-3.0, 0.5],
                [0.0, -4.0],
            ]
        ),
    )
    initial = QuadraticParams(
        values=jnp.array(
            [
                [0.0, 0.0],
                [2.0, 2.0],
                [5.0, 1.0],
            ]
        ),
    )

    final = _run_search(
        model,
        initial,
        max_iter=1,
    )

    np.testing.assert_allclose(
        final.params.values,
        model.target,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        final.objective,
        jnp.zeros(3),
        atol=1e-6,
    )


def test_line_search_backtracks_independently():
    """Only batches whose full step fails should be shortened."""

    model = QuadraticModel(
        target=jnp.ones((2, 1)),
    )

    initial_params = QuadraticParams(
        values=jnp.zeros((2, 1)),
    )

    # Batch 0: x=0 -> x=1, accepted with step 1.
    # Batch 1: x=0 -> x=3, rejected;
    #          step 1/2 gives x=1.5, which improves the objective.
    direction = QuadraticParams(
        values=jnp.array(
            [
                [1.0],
                [3.0],
            ]
        ),
    )

    current_objective = model.objective(initial_params)

    search = LineSearch[QuadraticParams](
        model=model,
        current_objective=current_objective,
        direction=direction,
        initial_params=initial_params,
        active=jnp.array([True, True]),
    )

    final = search.iterate(
        search.initial_state(),
        max_iter=10,
    )

    np.testing.assert_allclose(
        final.step_size,
        jnp.array([1.0, 0.5]),
    )
    np.testing.assert_allclose(
        final.candidate_params.values,
        jnp.array(
            [
                [1.0],
                [1.5],
            ]
        ),
    )

    assert np.all(np.asarray(final.is_accepted(current_objective)))


def test_failed_line_search_leaves_parameters_unchanged():
    """A direction with no improving step is rejected cleanly."""

    model = WrongDirectionModel(
        target=jnp.array([[1.0]]),
    )
    initial = QuadraticParams(
        values=jnp.array([[0.0]]),
    )

    final = _run_search(
        model,
        initial,
        max_iter=10,
        max_line_search_iters=3,
    )

    np.testing.assert_allclose(
        final.params.values,
        initial.values,
    )
    np.testing.assert_allclose(
        final.objective,
        model.objective(initial),
    )

    assert bool(final.done[0])


def test_newton_search_is_jittable():
    """The complete Newton + line-search loop should compile under JAX."""

    @jax.jit
    def fit(initial_values, target):
        model = QuadraticModel(target=target)
        initial_params = QuadraticParams(values=initial_values)

        final = _run_search(
            model,
            initial_params,
            max_iter=5,
        )

        return final.params.values, final.objective

    initial = jnp.array(
        [
            [0.0, 2.0],
            [4.0, -3.0],
        ]
    )
    target = jnp.array(
        [
            [1.0, -1.0],
            [2.0, 3.0],
        ]
    )

    params, objective = fit(initial, target)

    np.testing.assert_allclose(
        params,
        target,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        objective,
        jnp.zeros(2),
        atol=1e-6,
    )
