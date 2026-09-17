"""Independent Poisson distributions and log-linear conditional distributions."""

import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core import _batch
from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian


class Poisson(typing.NamedTuple):
    r"""Independent Poisson variables parameterized by log rates.

    $$u_i \sim \operatorname{Poisson}(\exp(\eta_i)).$$

    `log_rates` stores $\eta$. Leading dimensions are batch dimensions,
    shared across all attributes.
    """

    log_rates: jax.Array  # (..., N)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.log_rates.shape[:-1]

    @property
    def variable_dim(self) -> int:
        """Number of independent Poisson variables."""
        return self.log_rates.shape[-1]

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            log_rates=self.log_rates[index],
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> 'Poisson':
        """Convert to a different data type."""
        return self._replace(
            log_rates=self.log_rates.astype(dtype),
        )

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Data type."""
        return self.log_rates.dtype

    @property
    def rates(self) -> jax.Array:
        r"""Poisson rates $\lambda = \exp(\eta)$."""
        return jnp.exp(self.log_rates)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Sample values with shape `(*batch_shape, *sample_shape, D)`."""
        shape = self.batch_shape + (1,) * len(sample_shape) + (self.variable_dim,)
        return jax.random.poisson(
            key,
            lam=self.rates.reshape(shape),
            shape=self.batch_shape + sample_shape + (self.variable_dim,),
        )

    def log_prob_each(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (..., N)
        """Evaluate the log probability separately for each output dimension."""
        if values.shape[-1:] != (self.variable_dim,):
            raise ValueError('values must match the Poisson variable dimension')
        log_rates = _batch.align_array(
            self.log_rates, self.batch_shape, values.shape[:-1]
        )
        return values * log_rates - jnp.exp(log_rates) - jsp.special.gammaln(values + 1)

    def log_prob(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (...)
        """Evaluate log probabilities with aligned batch dimensions."""
        return jnp.sum(
            self.log_prob_each(values),
            axis=-1,
        )

    def log_prob_broadcast(self, values: jax.Array) -> jax.Array:
        """Evaluate all values with receiver batch axes first."""
        values = jnp.broadcast_to(values, self.batch_shape + values.shape)
        return self.log_prob(values)

    def mixture_mean(self, weights: jax.Array, *, axis: int) -> jax.Array:
        """Compute a mixture mean over an explicit, aligned batch axis."""
        return _batch.weighted_sum(self.rates, weights, self.batch_shape, axis)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            log_rates=_batch.broadcast_array(self.log_rates, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            log_rates=jnp.squeeze(self.log_rates, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            log_rates=jnp.take(self.log_rates, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            log_rates=jnp.moveaxis(self.log_rates, source, destination),
        )


class LinearPoisson(typing.NamedTuple):
    r"""
    Linear-Poisson conditional distribution.

    $$v \mid u \sim \operatorname{Poisson}(\exp(Wu + b)).$$

    `affine` encodes $(W, b)$. Tensor-shaped inputs resolve as tensor contractions.
    Leading dimensions are batch dimensions, shared across all attributes.
    """

    affine: Affine

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        return self.affine.batch_shape

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
    def dtype(self) -> jax.typing.DTypeLike:
        """Data type."""
        return jnp.result_type(
            self.affine.dtype,
            self.affine.bias.dtype,
        )

    def reshape_input(self, input_shape: tuple[int, ...]) -> typing.Self:
        """Return same model with input shape reshaped."""
        return self._replace(
            affine=self.affine.reshape_input(input_shape),
        )

    def select(self, index) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(index, len(self.batch_shape))
        return self.__class__(
            affine=self.affine.select(index),
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Convert to a different data type."""
        return self._replace(affine=self.affine.astype(dtype))

    def log_rates(self, values: jax.Array) -> jax.Array:
        r"""
        Log rates $\eta = Wu + b$ at deterministic input.
        """
        return self.affine.apply(values)

    def conditional(self, values: jax.Array) -> Poisson:
        """Conditional distribution at deterministic input."""
        return Poisson(log_rates=self.log_rates(values))

    def log_rate_moments(self, values: Gaussian) -> tuple[jax.Array, jax.Array]:
        """Mean and variance of log rates under Gaussian input."""
        return (values.affine_mean(self.affine), values.affine_variance(self.affine))

    def expected_rates(self, values: Gaussian) -> jax.Array:
        r"""Expected rates $\mathbb{E}[\exp(\eta)]$ under Gaussian input."""
        mean, variance = self.log_rate_moments(values)
        return jnp.exp(mean + 0.5 * variance)

    def expected_log_prob_each(self, values: jax.Array, inputs: Gaussian) -> jax.Array:
        """Expected log probabilities under Gaussian input marginals, per dimension."""
        mean, variance = self.log_rate_moments(inputs)
        if values.shape != mean.shape:
            raise ValueError(
                'observations and input moments must have matching batch/query shapes'
            )

        return (
            values * mean
            - jnp.exp(mean + 0.5 * variance)
            - jsp.special.gammaln(values + 1)
        )

    def expected_log_prob(
        self,
        values: jax.Array,
        inputs: Gaussian,
        weights: jax.Array | None = None,
    ) -> jax.Array:
        r"""Evaluate weighted $\mathbb{E}_{q(u)}[\log p(v\mid u)]$ under Gaussian inputs."""
        log_probs = self.expected_log_prob_each(
            values=values,
            inputs=inputs,
        )

        if weights is not None:
            _batch.require_same(weights.shape, log_probs.shape[:-1])
            log_probs = weights[..., None] * log_probs

        return jnp.sum(log_probs, axis=-1)

    def compose_input(
        self,
        affine: Affine,
    ) -> typing.Self:
        """Precompose with input map."""
        return self._replace(
            affine=self.affine.compose(affine),
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self.__class__(
            affine=self.affine.broadcast(shape, axis=axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self.__class__(
            affine=self.affine.squeeze(axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self.__class__(
            affine=self.affine.permute(permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self.__class__(
            affine=self.affine.move_axis(source, destination),
        )
