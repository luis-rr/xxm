r"""Additive-noise distributions: a location or affine mean plus zero-mean noise.

`LocDist` is the location family $u = \mu + \epsilon$ and `AffineDist` the
conditional $v = Wu + b + \epsilon$, where $\epsilon$ is a fixed zero-mean noise
distribution that does not depend on the mean. Concrete families such as
`Gaussian` and `LinearGaussian` fix the noise type and add their closed-form
operations. These bases hold only operations that are valid and efficient for
every noise type.

Distributions whose spread depends on the mean, such as `Poisson` and
`LinearPoisson` (a log-link GLM), are not additive and do not belong here.
"""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine


class Noise(typing.Protocol):
    """Zero-mean noise distribution with leading batch dimensions."""

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    @property
    def variable_dim(self) -> int: ...

    @property
    def dtype(self) -> jnp.dtype: ...

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self: ...

    def log_prob(
        self,
        values: jax.Array,  # (*B, *Q, N)
    ) -> jax.Array:  # (*B, *Q)
        ...

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:  # (*B, *S, N)
        ...

    def entropy(self) -> jax.Array:  # (*B,)
        ...

    def permute_variables(self, permutation: jax.Array) -> typing.Self: ...

    def reorient_variables(self, signs: jax.Array) -> typing.Self: ...


NoiseT = typing.TypeVar('NoiseT', bound=Noise)


class LocDist(typing.NamedTuple, typing.Generic[NoiseT]):
    r"""Location distribution $u = \mu + \epsilon$ with zero-mean noise $\epsilon$.

    Leading dimensions are batch dimensions, shared by `loc` and `noise`.
    """

    loc: jax.Array  # (*B, N)
    noise: NoiseT  # (*B)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        loc_shape = self.loc.shape[:-1]
        batch.require_same_shape(loc_shape, self.noise.batch_shape)
        return loc_shape

    @property
    def variable_dim(self) -> int:
        """Dimension of the variable."""
        return self.loc.shape[-1]

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return jnp.result_type(self.loc, self.noise.dtype)

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(loc=self.loc.astype(dtype), noise=self.noise.astype(dtype))

    def residual(
        self,
        values: jax.Array,  # (*B, *Q, N)
    ) -> jax.Array:  # (*B, *Q, N)
        """Values minus the location, with query axes following the batch."""
        if values.shape[-1:] != (self.variable_dim,):
            raise ValueError('values must match the variable dimension')
        return values - batch.align_array(self.loc, self.batch_shape, values.shape[:-1])

    def log_prob(
        self,
        values: jax.Array,  # (*B, *Q, N)
    ) -> jax.Array:  # (*B, *Q)
        """Evaluate aligned values, with query axes following the receiver batch."""
        return self.noise.log_prob(self.residual(values))

    def log_prob_broadcast(self, values: jax.Array) -> jax.Array:
        """Evaluate all values with receiver batch axes first."""
        values = jnp.broadcast_to(values, self.batch_shape + values.shape)
        return self.log_prob(values)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Sample values with shape `(*batch_shape, *sample_shape, N)`."""
        loc = self.loc.reshape(
            self.batch_shape + (1,) * len(sample_shape) + (self.variable_dim,)
        )
        return loc + self.noise.sample(key, sample_shape)

    def mixture_mean(self, weights: jax.Array, *, axis: int) -> jax.Array:
        """Compute a mixture mean over an explicit, aligned batch axis."""
        return batch.weighted_sum(self.loc, weights, self.batch_shape, axis)

    def entropy(self) -> jax.Array:  # (*B,)
        """Differential entropy, which does not depend on the location."""
        return self.noise.entropy()

    def shift_variables(self, offset: jax.Array) -> typing.Self:
        """Shift variable coordinates as x' = x - offset."""
        offset = batch.broadcast_trailing(
            offset,
            self.batch_shape,
            (self.variable_dim,),
        )
        return self._replace(loc=self.loc - offset)

    def permute_variables(self, permutation: jax.Array) -> typing.Self:
        """Reorder variable dimensions globally or independently by batch."""
        permutation = batch.broadcast_trailing(
            permutation,
            self.batch_shape,
            (self.variable_dim,),
        )
        return self._replace(
            loc=jnp.take_along_axis(self.loc, permutation, axis=-1),
            noise=self.noise.permute_variables(permutation),
        )

    def reorient_variables(self, signs: jax.Array) -> typing.Self:
        """Flip variable signs globally or independently by batch.

        Assumes the noise is symmetric, so that flipped noise stays in its family.
        """
        signs = batch.broadcast_trailing(
            signs,
            self.batch_shape,
            (self.variable_dim,),
        )
        return self._replace(
            loc=self.loc * signs,
            noise=self.noise.reorient_variables(signs),
        )


class AffineDist(typing.NamedTuple, typing.Generic[NoiseT]):
    r"""Affine conditional $v = Wu + b + \epsilon$ with zero-mean noise $\epsilon$.

    `affine` encodes $(W, b)$. Tensor-shaped inputs resolve as tensor
    contractions. Leading dimensions are batch dimensions, shared by `affine`
    and `noise`.
    """

    affine: Affine
    noise: NoiseT  # (*B)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        affine_shape = self.affine.batch_shape
        batch.require_same_shape(self.noise.batch_shape, affine_shape)
        return affine_shape

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Input shape of the affine map."""
        return self.affine.input_shape

    @property
    def input_ndim(self) -> int:
        """Number of input dimensions."""
        return self.affine.input_ndim

    @property
    def input_size(self) -> int:
        """Total size of input."""
        return self.affine.input_size

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self.affine.output_dim

    @property
    def dtype(self) -> jnp.dtype:
        """Data type."""
        return jnp.result_type(self.affine.dtype, self.noise.dtype)

    def select(self, index: batch.SelT) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(
            affine=self.affine.astype(dtype),
            noise=self.noise.astype(dtype),
        )

    def reshape_input(self, input_shape: tuple[int, ...]) -> typing.Self:
        """Return same model with input shape reshaped."""
        return self._replace(affine=self.affine.reshape_input(input_shape))

    def conditional_mean(self, values: jax.Array) -> jax.Array:
        """Mean of the conditional distribution at deterministic input."""
        return self.affine.apply(values)

    def log_prob(
        self,
        inputs: jax.Array,  # (*B, *Q, *input_shape)
        values: jax.Array,  # (*B, *Q, O)
    ) -> jax.Array:  # (*B, *Q)
        """Log density of `values` at deterministic `inputs`."""
        return self.noise.log_prob(values - self.conditional_mean(inputs))

    def sample(self, key: jax.Array, values: jax.Array) -> jax.Array:
        """Sample from the conditional distribution at deterministic input."""
        mean = self.conditional_mean(values)  # (*B, *Q, O)
        query_shape = mean.shape[len(self.batch_shape) : -1]
        return mean + self.noise.sample(key, query_shape)

    def compose_input(self, affine: Affine) -> typing.Self:
        """Precompose with $u=W'u'+b'$."""
        return self._replace(affine=self.affine.compose(affine))
