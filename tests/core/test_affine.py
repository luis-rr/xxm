import jax
import jax.numpy as jnp
import numpy as np

from xxm.core.affine import Affine

ATOL = 1e-6


def test_affine_vector_input_properties():
    affine = Affine(
        coefficients=jnp.zeros((3, 4)),
        bias=jnp.zeros(3),
    )

    assert affine.batch_shape == ()
    assert affine.input_shape == (4,)
    assert affine.input_ndim == 1
    assert affine.input_size == 4
    assert affine.output_dim == 3


def test_affine_structured_input_properties():
    affine = Affine(
        coefficients=jnp.zeros((2, 3, 4)),
        bias=jnp.zeros(2),
    )

    assert affine.batch_shape == ()
    assert affine.input_shape == (3, 4)
    assert affine.input_ndim == 2
    assert affine.input_size == 12
    assert affine.output_dim == 2


def test_affine_structured_input_known_contraction():
    affine = Affine(
        coefficients=jnp.array(
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                [
                    [-1.0, 0.5],
                    [2.0, -2.0],
                ],
            ]
        ),
        bias=jnp.array([0.5, -1.0]),
    )

    values = jnp.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    actual = affine.apply(values)

    expected = jnp.array(
        [
            30.5,
            -3.0,
        ]
    )

    np.testing.assert_allclose(
        actual,
        expected,
        atol=ATOL,
    )


def test_affine_flatten_and_unflatten_input_round_trip():
    affine = Affine(
        coefficients=jnp.zeros((2, 3, 4)),
        bias=jnp.zeros(2),
    )

    values = jnp.arange(24.0).reshape(
        2,
        3,
        4,
    )

    flattened = affine.input_flatten(values)
    restored = affine.input_unflatten(flattened)

    assert flattened.shape == (2, 12)

    np.testing.assert_array_equal(
        restored,
        values,
    )


def test_affine_reshape_input_preserves_map():
    affine = Affine(
        coefficients=jnp.arange(24.0).reshape(
            2,
            3,
            4,
        ),
        bias=jnp.array([1.0, -1.0]),
    )

    structured_input = jnp.arange(12.0).reshape(
        3,
        4,
    )

    flat_affine = affine.input_reshape((12,))
    flat_input = structured_input.reshape(12)

    np.testing.assert_allclose(
        affine.apply(structured_input),
        flat_affine.apply(flat_input),
        atol=ATOL,
    )


def test_affine_structured_input_is_jittable():
    affine = Affine(
        coefficients=jnp.arange(24.0).reshape(
            2,
            3,
            4,
        ),
        bias=jnp.array([1.0, -1.0]),
    )

    values = jnp.arange(12.0).reshape(
        3,
        4,
    )

    eager = affine.apply(values)
    compiled = jax.jit(lambda model, x: model.apply(x))(
        affine,
        values,
    )

    np.testing.assert_allclose(
        compiled,
        eager,
        atol=ATOL,
    )
