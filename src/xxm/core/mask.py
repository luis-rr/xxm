"""Temporal masks with explicit batch semantics."""

import dataclasses
import operator
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch

FillT = jax.Array | bool | int | float


class Mask(typing.Protocol):
    """Operational interface for masks applied to `(*B, T, *E)` arrays."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def align_batch(
        self,
        batch_shape: tuple[int, ...],
    ) -> typing.Self: ...

    def validate(self, values: jax.Array) -> None:
        """Validate compatibility with a temporal array."""
        ...

    def apply(
        self,
        values: jax.Array,
        *,
        fill: FillT = 0,
    ) -> jax.Array:
        """Replace masked entries, broadcasting across trailing event axes."""
        ...

    def num_valid(
        self,
        num_steps: int,
    ) -> jax.Array: ...

    def flatten(self) -> typing.Self: ...

    def materialize(self, num_steps: int) -> jax.Array: ...

    def adjacent_pairs(self) -> typing.Self:
        """Mask pairs `(t, t + 1)`, active only when both entries are active."""
        ...


class NoMask(typing.NamedTuple):
    """Identity mask used when every entry is active."""

    values: jax.Array = jnp.empty((0,), dtype=bool)

    @property
    def batch_shape(self):
        return self.values.shape[:-1]

    def align_batch(
        self,
        batch_shape: tuple[int, ...],
    ) -> typing.Self:
        return type(self)(
            values=jnp.empty(
                (*batch_shape, 0),
                dtype=bool,
            )
        )

    def validate(
        self,
        values: jax.Array,
    ) -> None:
        ndim = len(self.batch_shape)

        if values.shape[:ndim] != self.batch_shape:
            raise ValueError(
                f'values must have batch shape '
                f'{self.batch_shape}; '
                f'got {values.shape[:ndim]}'
            )

        if values.ndim <= ndim:
            raise ValueError('values must contain a time dimension')

    def flatten(self) -> typing.Self:
        return batch.flatten_batch(self, self.batch_shape)

    def adjacent_pairs(self) -> typing.Self:
        """Every adjacent pair remains active."""
        return self

    def apply(
        self,
        values: jax.Array,
        *,
        fill: FillT = 0,
    ) -> jax.Array:
        del fill
        self.validate(values)
        return values

    def num_valid(
        self,
        num_steps: int,
    ) -> jax.Array:
        return jnp.full(
            self.batch_shape,
            num_steps,
            dtype=jnp.int32,
        )

    def materialize(
        self,
        num_steps: int,
    ) -> jax.Array:
        return jnp.ones(
            (*self.batch_shape, num_steps),
            dtype=bool,
        )


NO_MASK = NoMask()


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class ArbitraryMask:
    """Arbitrary boolean temporal mask with shape `(*B, T)`.

    Leading dimensions are structural batch dimensions. The final dimension is
    time. When applied to `(*B, T, *E)` values, the mask broadcasts across all
    trailing event dimensions `*E`.
    """

    values: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, 'values', jnp.asarray(self.values))

        if self.values.ndim < 1:
            raise ValueError('mask values must have shape (*B, T)')
        if self.values.dtype != jnp.bool_:
            raise ValueError('mask values must have boolean dtype')

    @classmethod
    def _unchecked(cls, values: jax.Array) -> typing.Self:
        """Reconstruct an internal mask without host-side validation."""
        instance = object.__new__(cls)
        object.__setattr__(instance, 'values', values)
        return instance

    def tree_flatten(self):
        return (self.values,), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children) -> typing.Self:
        return cls._unchecked(*children)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Structural batch shape."""
        return self.values.shape[:-1]

    @property
    def num_steps(self) -> int:
        """Number of represented timesteps."""
        return self.values.shape[-1]

    def validate(self, values: jax.Array) -> None:
        """Require values to have prefix `(*B, T)` matching this mask."""
        prefix = (*self.batch_shape, self.num_steps)

        if values.ndim < len(prefix):
            raise ValueError(
                f'values must begin with mask shape {prefix}; got {values.shape}'
            )

        if values.shape[: len(prefix)] != prefix:
            raise ValueError(
                f'values must begin with mask shape {prefix}; got {values.shape}'
            )

    def apply(
        self,
        values: jax.Array,
        *,
        fill: FillT = 0,
    ) -> jax.Array:
        """Replace masked entries, broadcasting over trailing event dimensions."""
        self.validate(values)

        mask = batch.expand_trailing(self.values, values.ndim)

        return jnp.where(
            mask,
            values,
            jnp.asarray(fill, dtype=values.dtype),
        )

    def num_valid(
        self,
        num_steps: int,
    ) -> jax.Array:
        if self.num_steps != num_steps:
            raise ValueError(f'mask has {self.num_steps} steps; expected {num_steps}')

        return jnp.sum(
            self.values,
            axis=-1,
        )

    def align_batch(self, batch_shape):
        if self.batch_shape != batch_shape:
            raise ValueError(...)
        return self

    def flatten(self) -> typing.Self:
        return batch.flatten_batch(self, self.batch_shape)

    def adjacent_pairs(self) -> typing.Self:
        """Require both adjacent entries, retaining the structural batch shape."""
        return self._unchecked(self.values[..., :-1] & self.values[..., 1:])

    def materialize(self, num_steps: int) -> jax.Array:
        """Return this explicit mask, optionally checking the time dimension."""
        if num_steps is not None and operator.index(num_steps) != self.num_steps:
            raise ValueError(
                f'mask has {self.num_steps} timesteps; expected {num_steps}'
            )
        return self.values

    def select(self, index: batch.SelT) -> typing.Self:
        """Select only batch axes, retaining the time axis."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another batch position."""
        return batch.move_axis(self, source, destination)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class ContiguousMask:
    """Compact valid-prefix mask represented by lengths `(*B,)`.

    A length may be zero, which is useful for derived objects such as transition
    masks. Structures that require at least one valid timestep, such as
    `Sequences`, enforce that stronger invariant themselves.
    """

    lengths: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, 'lengths', jnp.asarray(self.lengths))

        if not jnp.issubdtype(self.lengths.dtype, jnp.integer):
            raise ValueError('mask lengths must have an integer dtype')
        if bool(jnp.any(self.lengths < 0)):
            raise ValueError('mask lengths must be nonnegative')

    @classmethod
    def _unchecked(cls, lengths: jax.Array) -> typing.Self:
        """Reconstruct an internal contiguous mask without host validation."""
        instance = object.__new__(cls)
        object.__setattr__(instance, 'lengths', lengths)
        return instance

    def tree_flatten(self):
        return (self.lengths,), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children) -> typing.Self:
        return cls._unchecked(*children)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Structural batch shape."""
        return self.lengths.shape

    def validate(self, values: jax.Array) -> None:
        """Require values to begin with this mask's structural batch shape and time."""
        ndim = len(self.batch_shape)

        if values.ndim < ndim + 1:
            raise ValueError(
                'values must have shape (*B, T, *E) for the contiguous mask'
            )

        if values.shape[:ndim] != self.batch_shape:
            raise ValueError(
                f'values must begin with batch shape {self.batch_shape}; '
                f'got {values.shape}'
            )

    def materialize(self, num_steps: int) -> jax.Array:
        """Expand prefix lengths into an explicit boolean mask `(*B, T)`."""
        num_steps = operator.index(num_steps)
        if num_steps < 0:
            raise ValueError('num_steps must be nonnegative')

        return jnp.arange(num_steps) < self.lengths[..., None]

    def apply(
        self,
        values: jax.Array,
        *,
        fill: FillT = 0,
    ) -> jax.Array:
        """Replace values outside each valid prefix."""
        self.validate(values)

        num_steps = values.shape[len(self.batch_shape)]
        return ArbitraryMask(self.materialize(num_steps)).apply(
            values,
            fill=fill,
        )

    def num_valid(
        self,
        num_steps: int,
    ) -> jax.Array:
        del num_steps

        return self.lengths

    def align_batch(self, batch_shape):
        if self.batch_shape != batch_shape:
            raise ValueError(...)
        return self

    def flatten(self) -> typing.Self:
        return batch.flatten_batch(self, self.batch_shape)

    def adjacent_pairs(self) -> typing.Self:
        """A prefix of length L contains max(L - 1, 0) adjacent pairs."""
        one = jnp.ones((), dtype=self.lengths.dtype)
        return self._unchecked(jnp.maximum(self.lengths, one) - one)

    def select(self, index: batch.SelT) -> typing.Self:
        """Select only batch axes."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another batch position."""
        return batch.move_axis(self, source, destination)
