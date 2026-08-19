import typing

import jax
import jax.numpy as jnp


class Categorical(typing.NamedTuple):
    """A categorical distribution.

    Leading dimensions are batch dimensions.
    """

    probs: jax.Array  # (..., K)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.probs.shape[:-1]

    @property
    def num_categories(self) -> int:
        return self.probs.shape[-1]

    def select(self, index) -> 'Categorical':
        """Index into the batch dimensions of the distribution."""
        return self.__class__(
            probs=self.probs[index],
        )

    def permute(
        self,
        permutation: jax.Array,  # (K,)
    ) -> 'Categorical':
        """Return a copy with categories reordered by ``permutation``."""
        return self._replace(
            probs=self.probs[..., permutation],
        )

    def broadcast(
        self,
        batch_shape: tuple[int, ...],
    ) -> 'Categorical':
        """Broadcast the distribution over additional leading batch dimensions."""
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
        """Evaluate log probabilities with aligned batch dimensions."""
        return jnp.take_along_axis(
            jnp.log(self.probs),
            values[..., None],
            axis=-1,
        )[..., 0]

    @classmethod
    def from_counts(
        cls,
        counts: jax.Array,  # (..., K)
        eps: float = 1e-8,
    ) -> 'Categorical':
        """Construct a categorical distribution from category counts."""
        total = jnp.sum(
            counts,
            axis=-1,
            keepdims=True,
        )

        return cls(
            probs=counts / jnp.maximum(total, eps),
        )
