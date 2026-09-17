"""Categorical distributions over finite discrete states."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import _batch


class Categorical(typing.NamedTuple):
    r"""Categorical distribution.

    $$\pi_k = p(z=k).$$

    `probs` stores the category probabilities. Leading dimensions are batch dimensions,
    shared across all attributes.
    """

    probs: jax.Array  # (..., K)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.probs.shape[:-1]

    @property
    def num_categories(self) -> int:
        """Number of categories."""
        return self.probs.shape[-1]

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            probs=self.probs[index],
        )

    def permute_categories(
        self,
        permutation: jax.Array,  # (K,)
    ) -> 'Categorical':
        """Relabel categories by permutation."""
        return self._replace(
            probs=self.probs[..., permutation],
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            probs=_batch.broadcast_array(self.probs, shape, axis),
        )

    def astype(
        self,
        dtype: jax.typing.DTypeLike,
    ) -> 'Categorical':
        """Convert to a different data type."""
        return self._replace(
            probs=self.probs.astype(dtype),
        )

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:  # (*sample_shape, ...)
        """Sample category indices."""
        return jax.random.categorical(
            key,
            logits=jnp.log(self.probs),
            shape=sample_shape + self.batch_shape,
        )

    def log_prob(self, values: jax.Array) -> jax.Array:
        """Evaluate category indices with aligned batch and subsequent query axes."""
        probs = _batch.align_array(self.probs, self.batch_shape, values.shape)
        return jnp.take_along_axis(jnp.log(probs), values[..., None], axis=-1)[..., 0]

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            probs=jnp.squeeze(self.probs, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            probs=jnp.take(self.probs, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            probs=jnp.moveaxis(self.probs, source, destination),
        )
