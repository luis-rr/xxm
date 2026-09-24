"""Categorical distributions over finite categories."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch


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

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

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
        return batch.broadcast(self, shape, axis=axis)

    def astype(
        self,
        dtype: jax.typing.DTypeLike,
    ) -> 'Categorical':
        """Convert to a different data type."""
        return self._replace(
            probs=self.probs.astype(dtype),
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Sample indices with shape `(*batch_shape, *sample_shape)`."""
        shape = self.batch_shape + (1,) * len(sample_shape) + (self.num_categories,)
        return jax.random.categorical(
            key,
            logits=jnp.log(self.probs).reshape(shape),
            shape=self.batch_shape + sample_shape,
        )

    def log_prob(self, values: jax.Array) -> jax.Array:
        """Evaluate category indices with aligned batch and subsequent query axes."""
        probs = batch.align_array(self.probs, self.batch_shape, values.shape)
        return jnp.take_along_axis(jnp.log(probs), values[..., None], axis=-1)[..., 0]

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)
