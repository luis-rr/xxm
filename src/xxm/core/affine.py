import math
import typing

import jax
import jax.numpy as jnp


class Affine(typing.NamedTuple):
    """A linear operation:

        y = A x + b

    Inputs are allowed to be tensor-shaped, which resolves as a tensor contraction
    and matrix-vector multiplication.

    Leading dimensions are batch dimensions and must be shared between attributes.

    """

    coefficients: jax.Array  # (..., O, I1, I2, ...)
    bias: jax.Array  # (..., O)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        if self.bias.ndim < 1:
            raise ValueError('bias must have shape (..., O)')

        if self.coefficients.ndim <= self.bias.ndim:
            raise ValueError(
                'coefficients must have shape (..., O, I1, I2, ...) '
                'with at least one input dimension'
            )

        if self.coefficients.shape[: self.bias.ndim] != self.bias.shape:
            raise ValueError(
                'coefficients and bias must have shapes '
                '(..., O, I1, I2, ...) and (..., O) '
                'with matching batch and output dimensions'
            )

        return self.bias.shape[:-1]

    @property
    def input_shape(self) -> tuple[int, ...]:  # (I1, I2, ...)
        _ = self.batch_shape
        return self.coefficients.shape[self.bias.ndim :]

    @property
    def input_ndim(self) -> int:
        return len(self.input_shape)

    @property
    def input_size(self) -> int:
        return math.prod(self.input_shape)

    def input_squeeze(self) -> typing.Self:
        """Remove singleton dimensions from the affine input shape."""
        input_shape = tuple(size for size in self.input_shape if size != 1)

        if not input_shape:
            input_shape = (1,)

        return self.input_reshape(input_shape)

    @property
    def output_dim(self) -> int:
        return self.bias.shape[-1]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return jnp.result_type(self.coefficients, self.bias)

    @property
    def coefficients_flat(self) -> jax.Array:
        """Return coefficients with all input dimensions flattened."""
        return self.coefficients.reshape(
            self.batch_shape + (self.output_dim, self.input_size)
        )

    def input_flatten(self, values: jax.Array) -> jax.Array:
        """Flatten the structured input dimensions of values."""
        if values.shape[-self.input_ndim :] != self.input_shape:
            raise ValueError(
                f'expected trailing input shape {self.input_shape}, got {values.shape}'
            )

        return values.reshape(values.shape[: -self.input_ndim] + (self.input_size,))

    def input_unflatten(self, values: jax.Array) -> jax.Array:
        """Restore a flattened input dimension to this affine input shape."""
        if values.shape[-1] != self.input_size:
            raise ValueError(
                f'expected trailing input size {self.input_size}, '
                f'got {values.shape[-1]}'
            )

        return values.reshape(values.shape[:-1] + self.input_shape)

    def input_reshape(self, input_shape: tuple[int, ...]) -> typing.Self:
        """Return the same affine map with a different input shape."""
        if not input_shape:
            raise ValueError('input_shape must contain at least one dimension')

        if math.prod(input_shape) != self.input_size:
            raise ValueError(
                f'input shape {input_shape} has size {math.prod(input_shape)}, '
                f'expected {self.input_size}'
            )

        return self._replace(
            coefficients=self.coefficients.reshape(
                self.batch_shape
                + (
                    self.output_dim,
                    *input_shape,
                )
            )
        )

    def norm(self) -> jax.Array:
        """Return the parameter norm for each output."""
        return jnp.sqrt(jnp.sum(self.coefficients_flat**2, axis=-1) + self.bias**2)

    def shift(self, center: jax.Array) -> typing.Self:
        """Shift the input origin by ``center``."""
        shift = jnp.einsum(
            '...oi,...i->...o',
            self.coefficients_flat,
            self.input_flatten(center),
        )

        return self._replace(
            bias=self.bias + shift,
        )

    def apply(self, values: jax.Array) -> jax.Array:
        """Apply the affine map to deterministic input values."""
        return (
            jnp.einsum(
                '...oi,...i->...o',
                self.coefficients_flat,
                self.input_flatten(values),
            )
            + self.bias
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        return self._replace(
            coefficients=self.coefficients.astype(dtype),
            bias=self.bias.astype(dtype),
        )

    def select(self, index) -> typing.Self:
        """Index into the batch dimensions."""
        return self.__class__(
            coefficients=self.coefficients[index],
            bias=self.bias[index],
        )
