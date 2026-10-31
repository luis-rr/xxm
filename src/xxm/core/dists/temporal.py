"""Time-indexed distributions with contiguous valid prefixes."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.data import Sequences
from xxm.core.lengths import Lengths


class MomentDist(typing.Protocol):
    """A batched distribution that can carry a time axis."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def variable_dim(self) -> int: ...

    @property
    def mean(self) -> jax.Array: ...

    @property
    def variance(self) -> jax.Array: ...

    def select(self, index: batch.SelT) -> typing.Self: ...


DistT = typing.TypeVar('DistT', bound=MomentDist)


class Temporal(typing.NamedTuple, typing.Generic[DistT]):
    """Time-indexed distributions with contiguous valid prefixes.

    dist batch:       (*B, T, *C)
    lengths.values:   (*B,)

    *B contains structural batch dimensions.
    T is time.
    *C contains additional distribution batch dimensions, e.g. states.

    Generic batch operations affect only *B.
    """

    dist: DistT
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.lengths.batch_shape

    @property
    def time_axis(self) -> int:
        return len(self.batch_shape)

    @property
    def max_steps(self) -> int:
        return self.dist.batch_shape[self.time_axis]

    @property
    def variable_dim(self) -> int:
        return self.dist.variable_dim

    def validate(self) -> None:
        """Validate static shapes without inspecting traced lengths."""
        trailing = batch.split_prefix(
            self.dist.batch_shape,
            self.batch_shape,
            name='distribution batch shape',
        )

        if not trailing:
            raise ValueError(f'{type(self).__name__} requires a time axis')

        if not jnp.issubdtype(self.lengths.values.dtype, jnp.integer):
            raise ValueError('lengths must have an integer dtype')

    def sequences(self, values: jax.Array) -> Sequences:
        """Values aligned with the distribution batch, with padded timesteps cleared."""
        return Sequences.from_padded(values, self.lengths)

    @property
    def mean(self) -> Sequences:
        """Means, with padded timesteps cleared."""
        return self.sequences(self.dist.mean)

    @property
    def variance(self) -> Sequences:
        """Marginal variances, with padded timesteps cleared."""
        return self.sequences(self.dist.variance)

    @property
    def std(self) -> Sequences:
        """Marginal standard deviations, with padded timesteps cleared."""
        return self.sequences(jnp.sqrt(self.dist.variance))

    def select(self, index: batch.SelT) -> typing.Self:
        """Select only structural batch dimensions."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton structural batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one structural batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        """Move an axis within the structural batch prefix."""
        return batch.move_axis(self, source, destination)

    def select_time(
        self,
        *,
        start: int,
        stop: int | None = None,
    ) -> typing.Self:
        """Select a time interval, adjusting valid-prefix lengths."""
        start, stop, _ = slice(start, stop).indices(self.max_steps)

        new_lengths = Lengths(
            jnp.clip(
                self.lengths.values - start,
                0,
                max(stop - start, 0),
            )
        )

        index = (slice(None),) * self.time_axis + (slice(start, stop),)

        return type(self)(
            dist=self.dist.select(index),
            lengths=new_lengths,
        )

    def get(self, index: batch.SelT | None = None) -> typing.Self:
        """Retrieve exactly one unbatched item."""
        return batch.get(self, index)
