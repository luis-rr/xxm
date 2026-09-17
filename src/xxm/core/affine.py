"""Affine maps with vector outputs and possibly tensor-shaped inputs."""

import math
import typing

import jax
import jax.numpy as jnp

from xxm.core import _batch


class Affine(typing.NamedTuple):
    r"""Linear-affine map.

    $$f(u) = Wu + b.$$

    `coefficients` stores $W$ and `bias` stores $b$. Inputs may be tensor-shaped,
    resolved as a tensor contraction and matrix-vector multiplication.
    Leading dimensions are batch dimensions, shared across all attributes.
    """

    coefficients: jax.Array  # (..., O, I1, I2, ...)
    bias: jax.Array  # (..., O)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape shared by coefficients and bias."""
        if self.bias.ndim < 1:
            raise ValueError('bias must have shape (..., O)')

        if self.coefficients.ndim <= self.bias.ndim:
            raise ValueError(
                'coefficients must have shape (..., O, I1, I2, ...) '
                'with at least one input dimension.'
                f' Got {self.coefficients.shape} and {self.bias.shape}'
            )

        if self.coefficients.shape[: self.bias.ndim] != self.bias.shape:
            raise ValueError(
                'coefficients and bias must have shapes '
                '(..., O, I1, I2, ...) and (..., O) '
                f'with matching batch and output dimensions. Got {self.coefficients.shape} and {self.bias.shape}'
            )

        return self.bias.shape[:-1]

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            coefficients=jnp.moveaxis(self.coefficients, source, destination),
            bias=jnp.moveaxis(self.bias, source, destination),
        )

    @property
    def input_shape(self) -> tuple[int, ...]:  # (I1, I2, ...)
        """Shape of input to the affine map."""
        _ = self.batch_shape
        return self.coefficients.shape[self.bias.ndim :]

    @property
    def input_ndim(self) -> int:
        """Number of input dimensions."""
        return len(self.input_shape)

    @property
    def input_size(self) -> int:
        """Total size of input, product of input_shape."""
        return math.prod(self.input_shape)

    def squeeze_input(self) -> typing.Self:
        """Remove singleton dimensions from input shape."""
        input_shape = tuple(size for size in self.input_shape if size != 1)

        if not input_shape:
            input_shape = (1,)

        return self.reshape_input(input_shape)

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self.bias.shape[-1]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Data type of the affine map."""
        return jnp.result_type(self.coefficients, self.bias)

    @property
    def coefficients_flat(self) -> jax.Array:
        """Coefficients with all input dimensions flattened to a single axis."""
        return self.coefficients.reshape(
            self.batch_shape + (self.output_dim, self.input_size)
        )

    def flatten_input(self, values: jax.Array) -> jax.Array:
        """Flatten input dimensions of values from structured to single axis."""
        if values.shape[-self.input_ndim :] != self.input_shape:
            raise ValueError(
                f'expected trailing input shape {self.input_shape}, got {values.shape}'
            )

        return values.reshape(values.shape[: -self.input_ndim] + (self.input_size,))

    def unflatten_input(self, values: jax.Array) -> jax.Array:
        """Restore flattened input dimension to this affine's input shape."""
        if values.shape[-1] != self.input_size:
            raise ValueError(
                f'expected trailing input size {self.input_size}, '
                f'got {values.shape[-1]}'
            )

        return values.reshape(values.shape[:-1] + self.input_shape)

    def reshape_input(self, input_shape: tuple[int, ...]) -> typing.Self:
        """Return affine map with input shape reshaped, preserving input size."""
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
        """Frobenius norm of coefficients and bias for each output dimension."""
        return jnp.sqrt(jnp.sum(self.coefficients_flat**2, axis=-1) + self.bias**2)

    def shift(self, center: jax.Array) -> typing.Self:
        r"""
        Translate the input origin so the shifted map $g$ satisfies
        $g(u-\text{center}) = f(u)$.
        """
        shift = jnp.einsum(
            '...oi,...i->...o',
            self.coefficients_flat,
            self.flatten_input(center),
        )

        return self._replace(
            bias=self.bias + shift,
        )

    def apply(self, values: jax.Array) -> jax.Array:
        """Evaluate inputs shaped `(*batch_shape, *query_shape, *input_shape)`."""
        values = self.flatten_input(values)
        coefficients = _batch.align_array(
            self.coefficients_flat, self.batch_shape, values.shape[:-1]
        )
        bias = _batch.align_array(self.bias, self.batch_shape, values.shape[:-1])
        return jnp.einsum('...oi,...i->...o', coefficients, values) + bias

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(
            coefficients=self.coefficients.astype(dtype),
            bias=self.bias.astype(dtype),
        )

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            coefficients=self.coefficients[index],
            bias=self.bias[index],
        )

    def compose(
        self,
        inner: typing.Self,
    ) -> typing.Self:
        r"""Compose affine maps, returning $f_\text{outer} \circ f_\text{inner}$."""
        if self.input_shape != (inner.output_dim,):
            raise ValueError(
                'affine composition requires the outer input shape '
                f'{self.input_shape} to match the inner output dimension '
                f'({inner.output_dim},)'
            )

        _batch.require_same(self.batch_shape, inner.batch_shape)

        coefficients = jnp.einsum(
            '...oi,...ij->...oj',
            self.coefficients_flat,
            inner.coefficients_flat,
        )

        bias = (
            jnp.einsum(
                '...oi,...i->...o',
                self.coefficients_flat,
                inner.bias,
            )
            + self.bias
        )

        batch_shape = self.batch_shape

        return self.__class__(
            coefficients=coefficients.reshape(
                batch_shape
                + (
                    self.output_dim,
                    *inner.input_shape,
                )
            ),
            bias=bias,
        )

    def inverse(self) -> typing.Self:
        """Compute the inverse of this square vector-to-vector affine map."""
        if self.input_shape != (self.output_dim,):
            raise ValueError(
                'affine inversion requires a square vector-to-vector map, '
                f'got {self.input_shape} -> ({self.output_dim},)'
            )

        coefficients = jnp.linalg.inv(
            self.coefficients_flat,
        )

        bias = -jnp.einsum(
            '...oi,...i->...o',
            coefficients,
            self.bias,
        )

        return self.__class__(
            coefficients=coefficients,
            bias=bias,
        )

    def pseudoinverse(self) -> typing.Self:
        """Compute the Moore-Penrose pseudoinverse affine map (vector inputs only)."""
        if self.input_ndim != 1:
            raise ValueError(
                'affine pseudoinverse requires vector-shaped input; '
                f'got {self.input_shape}'
            )

        coefficients = jnp.linalg.pinv(self.coefficients)

        return self.__class__(
            coefficients=coefficients,
            bias=-coefficients @ self.bias,
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            coefficients=_batch.broadcast_array(self.coefficients, shape, axis),
            bias=_batch.broadcast_array(self.bias, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            coefficients=jnp.squeeze(self.coefficients, axis=axes),
            bias=jnp.squeeze(self.bias, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            coefficients=jnp.take(self.coefficients, permutation, axis=axis),
            bias=jnp.take(self.bias, permutation, axis=axis),
        )
