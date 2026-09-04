import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from xxm.core.affine import Affine
from xxm.core.dists.gaussian import Gaussian


class Poisson(typing.NamedTuple):
    """Independent Poisson variables parameterized by log rates.

    Leading dimensions are batch dimensions and must be shared between attributes.
    """

    log_rates: jax.Array  # (..., N)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.log_rates.shape[:-1]

    @property
    def variable_dim(self) -> int:
        return self.log_rates.shape[-1]

    def select(self, index) -> 'Poisson':
        """Index into the batch dimensions of the distribution."""
        return Poisson(
            log_rates=self.log_rates[index],
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> 'Poisson':
        return self._replace(
            log_rates=self.log_rates.astype(dtype),
        )

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return self.log_rates.dtype

    @property
    def rates(self) -> jax.Array:
        return jnp.exp(self.log_rates)

    def sample(
        self,
        key: jax.Array,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        return jax.random.poisson(
            key,
            lam=self.rates,
            shape=sample_shape + self.log_rates.shape,
        )

    def log_prob_each(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (..., N)
        """Evaluate the log probability separately for each output dimension."""
        return values * self.log_rates - self.rates - jsp.special.gammaln(values + 1)

    def log_prob(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (...)
        """Evaluate log probabilities with aligned/broadcast-compatible batch dimensions."""
        return jnp.sum(
            self.log_prob_each(values),
            axis=-1,
        )

    def log_prob_broadcast(
        self,
        values: jax.Array,  # (..., N)
    ) -> jax.Array:  # (..., *batch_shape)
        """Evaluate every observation against every batched distribution."""
        values = values.reshape(
            values.shape[:-1] + (1,) * len(self.batch_shape) + (self.variable_dim,)
        )  # (..., 1, ..., 1, N)

        return self.log_prob(values)

    def mixture_mean(self, weights: jax.Array) -> jax.Array:
        """Mean of a mixture over the last batch dimension."""
        return jnp.sum(
            weights[..., :, None] * self.rates,
            axis=-2,
        )


class LinearPoisson(typing.NamedTuple):
    """A linear Poisson model ``y | x ~ Poisson(exp(A x + b))``.

    Leading dimensions are batch dimensions and must be shared between attributes.
    """

    affine: Affine

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.affine.batch_shape

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self.affine.input_shape

    @property
    def input_ndim(self) -> int:
        return self.affine.input_ndim

    @property
    def input_size(self) -> int:
        return self.affine.input_size

    @property
    def output_dim(self) -> int:
        return self.affine.output_dim

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        return jnp.result_type(
            self.affine.dtype,
            self.affine.bias.dtype,
        )

    def reshape_input(self, input_shape: tuple[int, ...]) -> typing.Self:
        """Return the same model with a different affine input shape."""
        return self._replace(
            affine=self.affine.input_reshape(input_shape),
        )

    def select(self, index) -> typing.Self:
        """Index into the batch dimensions."""
        return self.__class__(
            affine=self.affine.select(index),
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        return self._replace(affine=self.affine.astype(dtype))

    def log_rates(self, values: jax.Array) -> jax.Array:
        return self.affine.apply(values)

    def conditional(self, values: jax.Array) -> Poisson:
        """Conditional distribution of outputs for deterministic inputs."""
        return Poisson(log_rates=self.log_rates(values))

    def log_rate_moments(self, values: Gaussian) -> tuple[jax.Array, jax.Array]:
        """Gaussian moments of each linear predictor under Gaussian inputs."""
        return (values.affine_mean(self.affine), values.affine_variance(self.affine))

    def expected_rates(self, values: Gaussian) -> jax.Array:
        mean, variance = self.log_rate_moments(values)
        return jnp.exp(mean + 0.5 * variance)

    def expected_log_prob_each(self, values: jax.Array, inputs: Gaussian) -> jax.Array:
        """Expected Poisson log probability under Gaussian input marginals."""
        mean, variance = self.log_rate_moments(inputs)

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
        log_probs = self.expected_log_prob_each(
            values=values,
            inputs=inputs,
        )

        if weights is not None:
            log_probs = weights[..., None] * log_probs

        return jnp.sum(log_probs)

    def compose_input(
        self,
        affine: Affine,
    ) -> typing.Self:
        """Precompose the conditional model with an affine input map."""
        return self._replace(
            affine=self.affine.compose(affine),
        )
