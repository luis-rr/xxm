"""Batched temporal extents."""

import dataclasses
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
        result = cls(jnp.asarray(values))
        return result

    @classmethod
    def full(cls, shape, fill_value) -> typing.Self:
        return cls(
            jnp.full(
                shape,
                fill_value,
                dtype=jnp.int32,
            )
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.values.shape

    def validate_values(
        self,
        *,
        max_steps: int,
        min_length: int = 0,
    ) -> None:

        if min_length < 0:
            raise ValueError('min_length must be nonnegative')

        if not jnp.issubdtype(
            self.values.dtype,
            jnp.integer,
        ):
            raise ValueError('lengths must have an integer dtype')

        if bool(jnp.any(self.values < min_length)):
            raise ValueError(f'lengths must be at least {min_length}')

        if max_steps is not None:
            max_steps = operator.index(max_steps)

            if bool(jnp.any(self.values > max_steps)):
                raise ValueError(f'lengths must not exceed max_steps={max_steps}')

    def to_mask(
        self,
        max_steps: int,
    ) -> ArbitraryMask:
        return ArbitraryMask(jnp.arange(max_steps) < self.values[..., None])

    def clear(
        self,
        values: jax.Array,
        *,
        fill: FillT = 0,
    ) -> jax.Array:
        ndim = len(self.batch_shape)

        if values.ndim < ndim + 1:
            raise ValueError(
                'values must have shape (*B, T, *E) '
                f'with a temporal axis. Got {values.shape}'
            )

        if values.shape[:ndim] != self.batch_shape:
            raise ValueError(
                f'values must begin with batch shape '
                f'{self.batch_shape}; got {values.shape}'
            )

        return self.to_mask(values.shape[ndim]).apply(
            values,
            fill=fill,
        )

    def transitions(self) -> typing.Self:
        return type(self)(
            jnp.maximum(
                self.values - 1,
                0,
            )
        )

    def flatten(self) -> typing.Self:
        return batch.flatten_batch(
            self,
            self.batch_shape,
        )

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        return batch.broadcast(
            self,
            shape,
            axis=axis,
        )

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(
            self,
            axis=axis,
        )

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        return batch.permute(
            self,
            permutation,
            axis=axis,
        )

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        return batch.move_axis(
            self,
            source,
            destination,
        )

    def shift(
        self,
        offset: int,
    ) -> typing.Self:
        return type(self)(self.values + offset)

    def clip_min(self, v: int) -> typing.Self:
        return type(self)(jnp.maximum(self.values, v))

    def clip_max(self, v: int) -> typing.Self:
        return type(self)(jnp.minimum(self.values, v))


LengthsLike = int | typing.Sequence[int] | Lengths


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class LengthsSpec:
    lengths: Lengths
    max_steps: int

    def validate(
        self,
        *,
        min_length: int = 0,
    ) -> None:

        self.lengths.validate_values(
            min_length=min_length,
            max_steps=self.max_steps,
        )

        min_length = operator.index(min_length)

    def tree_flatten(self):
        return (self.lengths,), self.max_steps

    @classmethod
    def tree_unflatten(cls, max_steps, children):
        (lengths,) = children

        return cls(lengths=lengths, max_steps=max_steps)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.lengths.batch_shape

    @classmethod
    def from_params(cls, lengths: LengthsLike, *, min_length: int = 0) -> typing.Self:
        if not isinstance(lengths, Lengths):
            lengths = Lengths.from_values(lengths)

        ls = cls.from_lengths(lengths)
        ls.validate(min_length=min_length)

        return ls

    @classmethod
    def from_lengths(
        cls,
        lengths: Lengths,
        *,
        max_steps: int | None = None,
    ) -> typing.Self:

        if max_steps is None:
            max_steps = int(jnp.max(lengths.values))

        return cls(
            lengths=lengths,
            max_steps=operator.index(max_steps),
        )

    def transitions(self) -> typing.Self:
        return type(self)(
            lengths=self.lengths.transitions(),
            max_steps=max(self.max_steps - 1, 0),
        )
