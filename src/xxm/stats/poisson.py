import typing

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .gaussian import Affine, Gaussian


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


class LinearPoisson(typing.NamedTuple):
    """A linear Poisson model ``y | x ~ Poisson(exp(A x + b))``.

    Leading dimensions are batch dimensions and must be shared between attributes.
    """

    affine: Affine

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.affine.batch_shape

    def select(self, index) -> 'LinearPoisson':
        """Index into the batch dimensions."""
        return self.__class__(
            affine=self.affine.select(index),
        )

    def astype(self, dtype: jax.typing.DTypeLike) -> 'LinearPoisson':
        return self._replace(
            affine=self.affine.astype(dtype),
        )

    def log_rates(self, values: jax.Array) -> jax.Array:
        return self.affine.apply(values)

    def conditional(self, values: jax.Array) -> Poisson:
        """Conditional distribution of outputs for deterministic input values."""
        return Poisson(log_rates=self.log_rates(values))

    def log_rate_moments(self, values: Gaussian) -> tuple[jax.Array, jax.Array]:
        """Gaussian moments of each linear predictor under Gaussian input values."""
        return (
            values.affine_mean(self.affine),
            values.affine_variance(self.affine),
        )

    def expected_rates(self, values: Gaussian) -> jax.Array:
        mean, variance = self.log_rate_moments(values)
        return jnp.exp(mean + 0.5 * variance)

    def expected_log_prob_each(
        self,
        observations: jax.Array,
        inputs: Gaussian,
    ) -> jax.Array:
        """Expected Poisson log probability under Gaussian input marginals."""
        mean, variance = self.log_rate_moments(inputs)
        return (
            observations * mean
            - jnp.exp(mean + 0.5 * variance)
            - jsp.special.gammaln(observations + 1)
        )

    def expected_log_prob(
        self,
        observations: jax.Array,
        inputs: Gaussian,
        sample_weights: jax.Array | None = None,
    ) -> jax.Array:
        log_probs = self.expected_log_prob_each(
            observations=observations,
            inputs=inputs,
        )

        if sample_weights is not None:
            log_probs = sample_weights[..., None] * log_probs

        return jnp.sum(log_probs)
