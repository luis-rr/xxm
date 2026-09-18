"""Private array helpers for structural batch axes and aligned queries."""

import math
import operator

import jax
import jax.numpy as jnp


def flatten_batch(tree, batch_shape: tuple[int, ...]):
    """Flatten a shared structural batch prefix to one leading axis."""
    size = math.prod(batch_shape)
    ndim = len(batch_shape)

    def flatten(values):
        assert values.shape[:ndim] == batch_shape
        return values.reshape((size,) + values.shape[ndim:])

    return jax.tree.map(flatten, tree)


def unflatten_batch(tree, batch_shape: tuple[int, ...]):
    """Restore a flattened leading batch axis to `batch_shape`."""
    size = math.prod(batch_shape)

    def unflatten(values):
        assert values.shape[0] == size
        return values.reshape(batch_shape + values.shape[1:])

    return jax.tree.map(unflatten, tree)


def take_along_last_batch(tree, batch_shape, indices):
    """Select the final batch axis independently for aligned `(*B, *Q)` indices."""
    assert batch_shape
    prefix = batch_shape[:-1]
    assert indices.shape[: len(prefix)] == prefix
    assert jnp.issubdtype(indices.dtype, jnp.integer)
    query_shape = indices.shape[len(prefix) :]
    selected_axis = len(prefix) + len(query_shape)

    def take(values):
        assert values.shape[: len(batch_shape)] == batch_shape
        intrinsic_shape = values.shape[len(batch_shape) :]
        expanded = broadcast_array(values, query_shape, axis=len(prefix))
        gather = indices.reshape(indices.shape + (1,) + (1,) * len(intrinsic_shape))
        gather = jnp.broadcast_to(gather, indices.shape + (1,) + intrinsic_shape)
        return jnp.squeeze(
            jnp.take_along_axis(expanded, gather, axis=selected_axis),
            axis=selected_axis,
        )

    return jax.tree.map(take, tree)


def axis_index(axis: int, ndim: int) -> int:
    """Normalize an axis within the batch prefix, rejecting intrinsic axes."""
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f'batch axis {axis} out of range for {ndim} dimensions')
    return axis % ndim


def selection(index, ndim: int) -> tuple:
    """Expand integers, slices and integer arrays over exactly the batch prefix.

    One ellipsis is allowed. New axes belong to `broadcast`; boolean indexing
    is excluded because masks can consume multiple dimensions.
    """
    index = index if isinstance(index, tuple) else (index,)
    ellipses = sum(item is Ellipsis for item in index)
    if ellipses > 1:
        raise IndexError('only one ellipsis is allowed')
    consumed = len(index) - ellipses
    if consumed > ndim:
        raise IndexError('select may only index batch dimensions')
    result = ()
    for item in index:
        if item is Ellipsis:
            result += (slice(None),) * (ndim - consumed)
        else:
            if item is None:
                raise IndexError('select cannot insert axes; use broadcast')
            if not isinstance(item, slice):
                item = jnp.asarray(item)
                if not jnp.issubdtype(item.dtype, jnp.integer):
                    raise IndexError(
                        'select requires integer indices, not boolean or other non-integer indices'
                    )
            result += (item,)
    if not ellipses:
        result += (slice(None),) * (ndim - consumed)
    return result


def insertion(shape, axis: int, ndim: int) -> tuple[tuple[int, ...], int]:
    """Normalize inserted batch sizes and their position."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    shape = tuple(operator.index(size) for size in shape)
    if any(size < 0 for size in shape):
        raise ValueError('batch sizes must be nonnegative')
    return shape, axis_index(axis, ndim + 1)


def broadcast_array(values: jax.Array, shape: tuple[int, ...], axis: int) -> jax.Array:
    """Insert replicated dimensions at a normalized batch position."""
    prefix, suffix = values.shape[:axis], values.shape[axis:]
    return jnp.broadcast_to(
        values.reshape(prefix + (1,) * len(shape) + suffix),
        prefix + shape + suffix,
    )


def squeeze_axes(shape: tuple[int, ...], axis) -> tuple[int, ...]:
    """Select singleton batch axes without touching intrinsic dimensions."""
    if axis is None:
        return tuple(i for i, size in enumerate(shape) if size == 1)
    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(axis_index(i, len(shape)) for i in axes)
    if len(set(axes)) != len(axes) or any(shape[i] != 1 for i in axes):
        raise ValueError('squeeze requires distinct singleton batch axes')
    return axes


def permutation_indices(permutation, size: int) -> jax.Array:
    """Validate the static shape and dtype of a batch permutation."""
    permutation = jnp.asarray(permutation)
    if permutation.shape != (size,) or not jnp.issubdtype(
        permutation.dtype, jnp.integer
    ):
        raise ValueError(f'permutation must be an integer vector of shape {(size,)}')
    return permutation


def align_array(
    values: jax.Array, batch_shape: tuple[int, ...], shape: tuple[int, ...]
) -> jax.Array:
    """Align a parameter to a receiver batch prefix followed by query axes."""
    ndim = len(batch_shape)
    if shape[:ndim] != batch_shape:
        raise ValueError(f'expected batch prefix {batch_shape}, got {shape}')
    return broadcast_array(values, shape[ndim:], ndim)


def require_same(left: tuple[int, ...], right: tuple[int, ...]) -> None:
    """Require explicitly aligned object batches."""
    if left != right:
        raise ValueError(f'batch shapes must match; got {left} and {right}')


def weighted_sum(
    values: jax.Array, weights: jax.Array, batch_shape: tuple[int, ...], axis: int
) -> jax.Array:
    """Reduce one explicit batch axis with matching, unnormalized weights."""
    axis = axis_index(axis, len(batch_shape))
    require_same(batch_shape, weights.shape)
    return jnp.sum(
        values
        * weights.reshape(weights.shape + (1,) * (values.ndim - len(batch_shape))),
        axis=axis,
    )
