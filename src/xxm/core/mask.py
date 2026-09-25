"""Temporal masks with explicit batch semantics."""

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch

FillT = jax.Array | bool | int | float


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
    def max_steps(self) -> int:
        """Number of represented timesteps."""
        return self.values.shape[-1]

    def validate(self, values: jax.Array) -> None:
        """Require values to have prefix `(*B, T)` matching this mask."""
        prefix = (*self.batch_shape, self.max_steps)

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

    def num_valid(self) -> jax.Array:
        return jnp.sum(
            self.values,
            axis=-1,
        )

    def align_batch(self, batch_shape):
        batch.require_same_shape(self.batch_shape, batch_shape)
        return self

    def flatten(self) -> typing.Self:
        return batch.flatten_batch(self, self.batch_shape)

    def adjacent_pairs(self) -> typing.Self:
        """Require both adjacent entries, retaining the structural batch shape."""
        return self._unchecked(self.values[..., :-1] & self.values[..., 1:])

    def materialize(self) -> jax.Array:
        """Return this explicit mask."""
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
