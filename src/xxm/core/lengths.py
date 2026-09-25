"""Batched temporal extents."""

import operator
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.mask import ArbitraryMask, FillT


class Lengths(typing.NamedTuple):
    """Valid temporal lengths for a structural batch.

    ``values`` has shape ``*B``. Each entry gives the number of genuine
    timesteps in one dense temporal object whose storage horizon is supplied
    separately by that object's ``max_steps``.

    Use ``from_values`` at public construction boundaries when host validation
    is desired. Ordinary PyTree transformations may construct ``Lengths``
    directly without repeating value validation.
    """

    values: jax.Array

    @classmethod
    def from_values(cls, values) -> typing.Self:
        """Normalize and validate nonnegative integer lengths on the host."""
        result = cls(jnp.asarray(values))
        result.validate()
        return result

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Structural batch shape."""
        return self.values.shape

    def validate(
        self,
        *,
        max_steps: int | None = None,
        min_length: int = 0,
    ) -> None:
        """Validate integer lengths and optional temporal bounds on the host."""
        min_length = operator.index(min_length)
        if min_length < 0:
            raise ValueError('min_length must be nonnegative')

        if not jnp.issubdtype(self.values.dtype, jnp.integer):
            raise ValueError('lengths must have an integer dtype')

        if bool(jnp.any(self.values < min_length)):
            raise ValueError(f'lengths must be at least {min_length}')

        if max_steps is None:
            return

        max_steps = operator.index(max_steps)
        if max_steps < min_length:
            raise ValueError(
                f'max_steps must be at least min_length={min_length}; got {max_steps}'
            )

        if bool(jnp.any(self.values > max_steps)):
            raise ValueError(f'lengths must not exceed max_steps={max_steps}')

    def align_batch(
        self,
        batch_shape: tuple[int, ...],
    ) -> typing.Self:
        """Require this object to have the requested structural batch."""
        batch.require_same_shape(self.batch_shape, batch_shape)
        return self

    def to_mask(
        self,
        max_steps: int,
    ) -> ArbitraryMask:
        """Materialize valid prefixes as an explicit boolean temporal mask."""
        max_steps = operator.index(max_steps)
        if max_steps < 0:
            raise ValueError('max_steps must be nonnegative')

        return ArbitraryMask(jnp.arange(max_steps) < self.values[..., None])

    def clear(
        self,
        values: jax.Array,
        *,
        fill: FillT = 0,
    ) -> jax.Array:
        """Replace padded temporal suffixes in ``(*B, T, *E)`` values."""
        ndim = len(self.batch_shape)

        if values.ndim < ndim + 1:
            raise ValueError('values must have shape (*B, T, *E) with a temporal axis')

        if values.shape[:ndim] != self.batch_shape:
            raise ValueError(
                f'values must begin with batch shape {self.batch_shape}; '
                f'got {values.shape}'
            )

        max_steps = values.shape[ndim]

        return self.to_mask(max_steps).apply(
            values,
            fill=fill,
        )

    def transitions(self) -> typing.Self:
        """Return the number of genuine adjacent transitions per batch entry."""
        one = jnp.ones((), dtype=self.values.dtype)

        return type(self)(jnp.maximum(self.values, one) - one)

    def flatten(self) -> typing.Self:
        """Flatten the structural batch to one leading axis."""
        return batch.flatten_batch(self, self.batch_shape)

    def select(
        self,
        index: batch.SelT,
    ) -> typing.Self:
        """Select only structural batch axes."""
        return batch.select(self, index)

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        """Insert replicated structural batch dimensions."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(
        self,
        axis=None,
    ) -> typing.Self:
        """Remove singleton structural batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        """Reorder entries along one structural batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        """Move one structural batch axis to another position."""
        return batch.move_axis(self, source, destination)
