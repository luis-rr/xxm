"""Categorical distributions over finite discrete states."""

import typing

import jax
import jax.numpy as jnp


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

    def select(self, index) -> 'Categorical':
        """Index into batch dimensions."""
        return self.__class__(
            probs=self.probs[index],
        )

    def permute(
        self,
        permutation: jax.Array,  # (K,)
    ) -> 'Categorical':
        """Relabel categories by permutation."""
        return self._replace(
            probs=self.probs[..., permutation],
        )

    def broadcast(
        self,
        batch_shape: tuple[int, ...],
    ) -> 'Categorical':
        """Broadcast over leading batch dimensions."""
        return self.__class__(
            probs=jnp.broadcast_to(
                self.probs,
                batch_shape + self.probs.shape,
            ),
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

    def log_prob(
        self,
        values: jax.Array,  # (...)
    ) -> jax.Array:  # (...)
        """Evaluate log probabilities."""
        return jnp.take_along_axis(
            jnp.log(self.probs),
            values[..., None],
            axis=-1,
        )[..., 0]

    @classmethod
    def from_counts(
        cls,
        counts: jax.Array,  # (..., K)
    ) -> 'Categorical':
        """Construct from category counts, handling zero-count cases with uniform."""
        total = counts.sum(axis=-1, keepdims=True)
        valid = total > 0

        # Handle zero counts with the uniform distribution.
        valid_total = jnp.where(valid, total, counts.shape[-1])
        valid_counts = jnp.where(valid, counts, 1)

        probs = valid_counts / valid_total

        return cls(probs=probs)
