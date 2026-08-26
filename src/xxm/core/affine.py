import typing

import jax
import jax.numpy as jnp


class Affine(typing.NamedTuple):
    """A linear operation:

        y = A x + b

    Leading dimensions are batch dimensions and must be shared between attributes.
    """

    coefficients: jax.Array  # (..., O, I)
    bias: jax.Array  # (..., O)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        coefficients_shape = self.coefficients.shape[:-2]
        bias_shape = self.bias.shape[:-1]
        assert bias_shape == coefficients_shape
        return coefficients_shape

    @property
    def input_dim(self) -> int:
        return self.coefficients.shape[-1]

    @property
    def output_dim(self) -> int:
        return self.coefficients.shape[-2]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return jnp.result_type(self.coefficients, self.bias)

    def norm(self) -> jax.Array:
        """Return the parameter norm for each output."""
        return jnp.sqrt(jnp.sum(self.coefficients**2, axis=-1) + self.bias**2)  # TODO batch?

    def shift(self, center: jax.Array) -> 'Affine':

        return self._replace(
            bias=self.bias + self.coefficients @ center,
        )

    def apply(self, values: jax.Array) -> jax.Array:
        return (
            jnp.einsum(
                '...oi,...i->...o',
                self.coefficients,
                values,
            )
            + self.bias
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> 'Affine':
        return self._replace(
            coefficients=self.coefficients.astype(dtype),
            bias=self.bias.astype(dtype),
        )

    def select(self, index) -> 'Affine':
        """Index into the batch dimensions."""
        return self.__class__(
            coefficients=self.coefficients[index],
            bias=self.bias[index],
        )
