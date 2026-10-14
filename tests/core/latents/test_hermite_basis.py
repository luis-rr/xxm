import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xxm.core.latents.hermite_core import HermiteBasis


@pytest.mark.parametrize('num_knots', [2, 5])
def test_centering_and_penalties(num_knots):
    basis = HermiteBasis.from_num_knots(num_knots)
    quadrature = basis.quadrature(dtype=jnp.float32)
    evaluation = basis.at_positions(quadrature.positions)
    centered = jax.jit(lambda e: e.centered_basis())(evaluation)
    np.testing.assert_allclose(quadrature.weights @ centered, 0, atol=1e-7)
    penalties = jax.jit(lambda b: b.penalty_matrices(dtype=jnp.float32))(basis)
    coefficients = jnp.arange(1, num_knots, dtype=jnp.float32)
    curve = centered @ coefficients
    curvature = (
        evaluation.raw_second_derivative()
        @ basis.coefficient_map(dtype=jnp.float32)
        @ coefficients
    )
    np.testing.assert_allclose(
        coefficients @ penalties.amplitude @ coefficients,
        quadrature.weights @ curve**2,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        coefficients @ penalties.curvature @ coefficients,
        quadrature.weights @ curvature**2,
        rtol=1e-5,
    )


def test_position_shapes_and_derivatives():
    basis = HermiteBasis.from_num_knots(4)
    positions = jnp.array([[0.1, 0.4], [0.7, 0.9]])
    evaluation = basis.at_positions(positions)
    assert evaluation.centered_basis().shape == (2, 2, 3)
    derivative = jax.jacfwd(jax.jacfwd(lambda t: basis.at_positions(t).raw_basis()))
    expected = jax.vmap(jax.vmap(derivative))(positions)
    np.testing.assert_allclose(evaluation.raw_second_derivative(), expected, atol=1e-5)
    assert basis.regularly_spaced(0).centered_basis().shape == (0, 3)
    assert basis.at_positions(jnp.array([0, 1])).positions.dtype == jnp.float32
