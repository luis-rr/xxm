"""Structural batch helpers for batch-aligned arrays and PyTrees."""

import functools
import itertools
import math
import operator
import typing
from types import EllipsisType

import jax
import jax.numpy as jnp
import numpy as np

ShapeT = tuple[int, ...]


class Batchable(typing.Protocol):
    """JAX PyTree whose dynamic leaves share a structural batch prefix.

    Every dynamic leaf is expected to begin with ``batch_shape``. Intrinsic
    dimensions follow that prefix. This is a structural protocol only;
    concrete objects do not need to inherit from it.
    """

    @property
    def batch_shape(self) -> ShapeT: ...


class VariableBatchable(Batchable, typing.Protocol):
    """Batchable object with a transformable semantic variable dimension."""

    @property
    def variable_dim(self) -> int:
        """Number of represented variables."""
        ...

    def reorient_variables(
        self,
        signs: jax.Array,
    ) -> typing.Self:
        """Flip variable coordinate orientations."""
        ...

    def permute_variables(
        self,
        permutation: jax.Array,
    ) -> typing.Self:
        """Relabel variable coordinates."""
        ...

    def broadcast(
        self,
        shape: tuple[int, ...],
        axis: int = 0,
    ) -> typing.Self: ...


_BatchableT = typing.TypeVar('_BatchableT', bound=Batchable)
_LeftT = typing.TypeVar('_LeftT', bound=Batchable)
_RightT = typing.TypeVar('_RightT', bound=Batchable)
_TreeT = typing.TypeVar('_TreeT')


SelItemT = int | jax.Array | list[int] | slice | EllipsisType
SelT = SelItemT | tuple[SelItemT, ...]


def split_prefix(shape: ShapeT, prefix: ShapeT, *, name: str = 'shape') -> ShapeT:
    """Require `shape` to begin with `prefix` and return the remainder."""
    ndim = len(prefix)
    if shape[:ndim] != prefix:
        raise ValueError(f'{name} must begin with {prefix}; got {shape}')
    return shape[ndim:]


def expand_trailing(values: jax.Array, ndim: int) -> jax.Array:
    """Append singleton axes until `values` has rank `ndim`."""
    ndim = operator.index(ndim)
    if ndim < values.ndim:
        raise ValueError(f'cannot expand rank {values.ndim} values to rank {ndim}')
    return values.reshape(values.shape + (1,) * (ndim - values.ndim))


def require_same_shape(*shapes: ShapeT) -> None:
    """Require explicitly aligned shapes."""
    if shapes and any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError(f'shapes must match; got {shapes}')


def _map_batch_prefix(
    tree: _TreeT,
    batch_shape: ShapeT,
    function: typing.Callable[[jax.Array], jax.Array],
) -> _TreeT:
    """Apply an array transform to PyTree leaves sharing `batch_shape`."""

    def apply(values: jax.Array) -> jax.Array:
        split_prefix(values.shape, batch_shape, name='PyTree leaf shape')
        return function(values)

    return typing.cast(_TreeT, jax.tree.map(apply, tree))


def _map_batch_leaves(
    batchable: _BatchableT,
    function: typing.Callable[[jax.Array], jax.Array],
) -> _BatchableT:
    """Apply an array transform to leaves sharing `batchable.batch_shape`."""
    return typing.cast(
        _BatchableT,
        _map_batch_prefix(batchable, batchable.batch_shape, function),
    )


def select(batchable: _BatchableT, index: SelT) -> _BatchableT:
    """Index the structural batch prefix of a batchable PyTree."""
    index = selection(index, len(batchable.batch_shape))
    return _map_batch_leaves(batchable, lambda values: values[index])


def get(batchable: _BatchableT, index: SelT | None = None) -> _BatchableT:
    """Return exactly one batch item, preserving its type.

    An optional selection is applied first. The resulting batch must
    contain exactly one item, regardless of its number of dimensions.

    The returned object always has batch_shape == ().
    """
    if index is not None:
        validate_selection(index, batchable.batch_shape)
        batchable = select(batchable, index)

    if math.prod(batchable.batch_shape) != 1:
        raise ValueError(
            'get requires exactly one batch item; '
            f'got batch_shape {batchable.batch_shape}'
        )

    return squeeze(batchable)


def concatenate(
    batchables: typing.Sequence[_BatchableT],
    axis: int = 0,
) -> _BatchableT:
    """Concatenate an existing batch axis without padding intrinsic axes."""
    if not batchables:
        raise ValueError('batchables must contain at least one item')
    first = batchables[0]
    shape = first.batch_shape
    axis = axis_index(axis, len(shape))
    leaves, structure = jax.tree_util.tree_flatten(first)
    for item in batchables:
        if type(item) is not type(first):
            raise ValueError('batchables must have the same concrete type')
        other_shape = item.batch_shape
        if len(other_shape) != len(shape) or any(
            size != other_shape[i] for i, size in enumerate(shape) if i != axis
        ):
            raise ValueError('non-concatenated batch dimensions must match')
        other_leaves, other_structure = jax.tree_util.tree_flatten(item)
        if other_structure != structure:
            raise ValueError('batchables must have matching PyTree structure')
        for left, right in zip(leaves, other_leaves, strict=True):
            intrinsic = split_prefix(left.shape, shape, name='PyTree leaf shape')
            other_intrinsic = split_prefix(
                right.shape, other_shape, name='PyTree leaf shape'
            )
            if intrinsic != other_intrinsic:
                raise ValueError('batchables must have matching intrinsic shapes')
    return typing.cast(
        _BatchableT,
        jax.tree.map(lambda *values: jnp.concatenate(values, axis=axis), *batchables),
    )


def broadcast(batchable: _BatchableT, shape, axis: int) -> _BatchableT:
    """Insert replicated dimensions into a batchable PyTree."""
    shape, axis = insertion(shape, axis, len(batchable.batch_shape))
    return _map_batch_leaves(
        batchable,
        lambda values: broadcast_array(values, shape, axis),
    )


def squeeze(batchable: _BatchableT, axis=None) -> _BatchableT:
    """Remove singleton dimensions from a batchable PyTree."""
    axes = squeeze_axes(batchable.batch_shape, axis)
    if not axes:
        return batchable
    return _map_batch_leaves(
        batchable,
        lambda values: jnp.squeeze(values, axis=axes),
    )


def permute(batchable: _BatchableT, permutation, axis: int) -> _BatchableT:
    """Reorder entries along one structural batch axis."""
    axis = axis_index(axis, len(batchable.batch_shape))
    permutation = permutation_indices(
        permutation,
        batchable.batch_shape[axis],
    )
    return _map_batch_leaves(
        batchable,
        lambda values: jnp.take(values, permutation, axis=axis),
    )


def move_axis(batchable: _BatchableT, source: int, destination: int) -> _BatchableT:
    """Move one axis within a structural batch prefix."""
    source = axis_index(source, len(batchable.batch_shape))
    destination = axis_index(destination, len(batchable.batch_shape))
    return _map_batch_leaves(
        batchable,
        lambda values: jnp.moveaxis(values, source, destination),
    )


def cartesian_broadcast(left: _LeftT, right: _RightT) -> tuple[_LeftT, _RightT]:
    """Broadcast both objects to `(*L, *R)`, even when their batches match.

    Empty batch shapes need no inserted axes. Each result retains its original
    object type. Structural batch shapes are inferred from the objects rather
    than supplied separately.
    """
    left_shape = left.batch_shape
    right_shape = right.batch_shape

    if right_shape:
        left = broadcast(left, right_shape, axis=len(left_shape))
    if left_shape:
        right = broadcast(right, left_shape, axis=0)

    return left, right


def flatten_batch(tree: _TreeT, batch_shape: ShapeT) -> _TreeT:
    """Flatten a shared structural batch prefix to one leading axis."""
    size = math.prod(batch_shape)
    ndim = len(batch_shape)
    return _map_batch_prefix(
        tree,
        batch_shape,
        lambda values: values.reshape((size,) + values.shape[ndim:]),
    )


def unflatten_batch(tree: _TreeT, batch_shape: ShapeT) -> _TreeT:
    """Restore a flattened leading batch axis to `batch_shape`."""
    size = math.prod(batch_shape)

    def unflatten(values: jax.Array) -> jax.Array:
        if values.ndim < 1 or values.shape[0] != size:
            raise ValueError(
                f'PyTree leaf must begin with flattened batch size {size}; '
                f'got {values.shape}'
            )
        return values.reshape(batch_shape + values.shape[1:])

    return typing.cast(_TreeT, jax.tree.map(unflatten, tree))


def vmap_batch(function, *trees, batch_shape: ShapeT):
    """Map an operation over a shared structural batch prefix."""
    flat = [flatten_batch(tree, batch_shape) for tree in trees]
    return unflatten_batch(jax.vmap(function)(*flat), batch_shape)


def pool_samples(tree: _TreeT, batch_shape: ShapeT, sample_shape: ShapeT) -> _TreeT:
    """Collapse `(*B, *S, *E)` leaves to `(*B, prod(S), *E)`.

    An empty sample shape introduces one sample; zero-length sample axes
    remain empty. Event dimensions are preserved independently for each leaf.
    """
    prefix = batch_shape + sample_shape
    size = math.prod(sample_shape)

    def pool(values: jax.Array) -> jax.Array:
        event_shape = split_prefix(values.shape, prefix, name='PyTree leaf shape')
        return values.reshape(batch_shape + (size,) + event_shape)

    return typing.cast(_TreeT, jax.tree.map(pool, tree))


def take_along_last_batch(
    tree: _TreeT,
    batch_shape: ShapeT,
    indices: jax.Array,
) -> _TreeT:
    """Select the final batch axis independently for aligned `(*B, *Q)` indices."""
    if not batch_shape:
        raise ValueError('take_along_last_batch requires at least one batch axis')

    prefix = batch_shape[:-1]
    query_shape = split_prefix(indices.shape, prefix, name='indices shape')
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError('indices must have an integer dtype')

    selected_axis = len(prefix) + len(query_shape)

    def take(values: jax.Array) -> jax.Array:
        intrinsic_shape = split_prefix(
            values.shape,
            batch_shape,
            name='PyTree leaf shape',
        )
        expanded = broadcast_array(values, query_shape, axis=len(prefix))
        gather = indices.reshape(indices.shape + (1,) + (1,) * len(intrinsic_shape))
        gather = jnp.broadcast_to(
            gather,
            indices.shape + (1,) + intrinsic_shape,
        )
        return jnp.squeeze(
            jnp.take_along_axis(expanded, gather, axis=selected_axis),
            axis=selected_axis,
        )

    return typing.cast(_TreeT, jax.tree.map(take, tree))


def axis_index(axis: int, ndim: int) -> int:
    """Normalize an axis within the batch prefix, rejecting intrinsic axes."""
    axis = operator.index(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f'batch axis {axis} out of range for {ndim} dimensions')
    return axis % ndim


def selection(index: SelT, ndim: int) -> tuple:
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
            continue
        if item is None:
            raise IndexError('select cannot insert axes; use broadcast')
        if not isinstance(item, slice):
            item = jnp.asarray(item)
            if not jnp.issubdtype(item.dtype, jnp.integer):
                raise IndexError(
                    'select requires integer indices, not boolean or other '
                    'non-integer indices'
                )
        result += (item,)

    if not ellipses:
        result += (slice(None),) * (ndim - consumed)
    return result


def validate_selection(index: SelT, shape: ShapeT) -> None:
    """Validate batch selection bounds on the host, outside JAX transforms.

    Integer indices must lie in ``[-size, size)`` for their axis. Slices retain
    ordinary clipping semantics; valid negative indices count from the end.
    """
    index = selection(index, len(shape))
    for axis, (item, size) in enumerate(zip(index, shape, strict=True)):
        if isinstance(item, slice):
            continue
        indices = np.asarray(item)
        if np.any(indices < -size) or np.any(indices >= size):
            raise IndexError(
                f'selection index out of bounds for batch axis {axis} with size {size}'
            )


def iter_keys(batch_shape: ShapeT) -> typing.Iterator[tuple[int, ...]]:
    return itertools.product(*(range(size) for size in batch_shape))


def insertion(shape, axis: int, ndim: int) -> tuple[ShapeT, int]:
    """Normalize inserted batch sizes and their position."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    shape = tuple(operator.index(size) for size in shape)
    if any(size < 0 for size in shape):
        raise ValueError('batch sizes must be nonnegative')
    return shape, axis_index(axis, ndim + 1)


def broadcast_array(
    values: jax.Array,
    shape: ShapeT,
    axis: int,
) -> jax.Array:
    """Insert replicated dimensions at a normalized batch position."""
    prefix, suffix = values.shape[:axis], values.shape[axis:]
    return jnp.broadcast_to(
        values.reshape(prefix + (1,) * len(shape) + suffix),
        prefix + shape + suffix,
    )


def broadcast_trailing(
    values: jax.Array,
    batch_shape: ShapeT,
    trailing_shape: ShapeT,
) -> jax.Array:
    """Broadcast values over a structural batch with fixed trailing shape."""
    values = jnp.asarray(values)

    ndim = len(trailing_shape)

    if ndim and values.shape[-ndim:] != trailing_shape:
        raise ValueError(f'Values must end with {trailing_shape}; got {values.shape}')

    target_shape = batch_shape + trailing_shape

    try:
        return jnp.broadcast_to(values, target_shape)

    except ValueError as error:
        raise ValueError(
            f'Values must broadcast to {target_shape}; got {values.shape}'
        ) from error


def squeeze_axes(shape: ShapeT, axis) -> tuple[int, ...]:
    """Select singleton batch axes without touching intrinsic dimensions."""
    if axis is None:
        return tuple(i for i, size in enumerate(shape) if size == 1)

    axes = axis if isinstance(axis, tuple) else (axis,)
    axes = tuple(axis_index(i, len(shape)) for i in axes)
    if len(set(axes)) != len(axes) or any(shape[i] != 1 for i in axes):
        raise ValueError('squeeze requires distinct singleton batch axes')
    return axes


def pad_axis(values: jax.Array, size: int, *, axis: int) -> jax.Array:
    """Right-pad one axis with zeros to the requested size."""
    current = values.shape[axis]

    if size < current:
        raise ValueError(f'cannot pad axis of size {current} to {size}')

    padding = [(0, 0)] * values.ndim
    padding[axis] = (0, size - current)

    return jnp.pad(values, tuple(padding))


def permutation_indices(permutation, size: int) -> jax.Array:
    """Validate the static shape and dtype of a batch permutation."""
    permutation = jnp.asarray(permutation)
    if permutation.shape != (size,) or not jnp.issubdtype(
        permutation.dtype, jnp.integer
    ):
        raise ValueError(f'permutation must be an integer vector of shape {(size,)}')
    return permutation


def align_array(
    values: jax.Array,
    batch_shape: ShapeT,
    shape: ShapeT,
) -> jax.Array:
    """Align a parameter to a receiver batch prefix followed by query axes."""
    query_shape = split_prefix(shape, batch_shape, name='target shape')
    return broadcast_array(values, query_shape, len(batch_shape))


def flatten_query(
    values: jax.Array,  # (*B, *Q, N)
    batch_shape: ShapeT,
) -> jax.Array:  # (*B, Q, N)
    """Merge the query axes between a batch prefix and one trailing event axis."""
    query_shape = split_prefix(values.shape[:-1], batch_shape, name='values shape')
    return values.reshape(batch_shape + (math.prod(query_shape), values.shape[-1]))


def weighted_sum(
    values: jax.Array,
    weights: jax.Array,
    batch_shape: ShapeT,
    axis: int,
) -> jax.Array:
    """Reduce one explicit batch axis with matching, unnormalized weights."""
    axis = axis_index(axis, len(batch_shape))
    require_same_shape(batch_shape, weights.shape)
    split_prefix(values.shape, batch_shape, name='values shape')
    return jnp.sum(
        values * expand_trailing(weights, values.ndim),
        axis=axis,
    )


def where(
    condition: jax.Array,
    true: _BatchableT,
    false: _BatchableT,
) -> _BatchableT:
    """Select PyTree leaves per batch element, preserving intrinsic dimensions."""
    require_same_shape(
        condition.shape,
        true.batch_shape,
        false.batch_shape,
    )

    def select(a: jax.Array, b: jax.Array) -> jax.Array:
        if a.shape != b.shape:
            raise ValueError(
                f'corresponding leaves must have equal shapes; '
                f'got {a.shape} and {b.shape}'
            )

        return jnp.where(
            expand_trailing(condition, a.ndim),
            a,
            b,
        )

    return typing.cast(
        _BatchableT,
        jax.tree.map(select, true, false),
    )


@jax.tree_util.register_static
class Partition(typing.NamedTuple):
    """Sizes of consecutive segments along a batch axis."""

    sizes: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.sizes)

    def split(
        self,
        tree: _TreeT,
        *,
        axis: int,
    ) -> tuple[_TreeT, ...]:
        boundaries = (0, *itertools.accumulate(self.sizes))

        return tuple(
            jax.tree.map(
                functools.partial(
                    jax.lax.slice_in_dim,
                    start_index=start,
                    limit_index=stop,
                    axis=axis,
                ),
                tree,
            )
            for start, stop in itertools.pairwise(boundaries)
        )

    def get_slice(self, index: int) -> slice:
        index = range(len(self.sizes))[index]
        start = sum(self.sizes[:index])
        return slice(start, start + self.sizes[index])
