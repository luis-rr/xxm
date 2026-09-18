"""Aligned autoregressive regression rows and conditioning-history indexing."""

import typing

import jax
import jax.numpy as jnp


def lagged_observations(
    observations: jax.Array, num_lags: int
) -> jax.Array:  # (T-L, L, N)
    """Return histories ordered from lag 1 to lag L."""

    return jnp.stack(
        [
            observations[..., num_lags - i - 1 : observations.shape[-2] - i - 1, :]
            for i in range(num_lags)
        ],
        axis=-2,
    )


class ARObservations(typing.NamedTuple):
    """Prepared rows: target r is y[L+r], predictors run newest to oldest."""

    predictors: jax.Array  # (R, L, D_y)
    targets: jax.Array  # (R, D_y)

    @classmethod
    def from_observations(
        cls,
        observations: jax.Array,
        num_lags: int,
    ) -> typing.Self:
        """Condition on the first L observations and prepare T-L regression rows."""
        if observations.ndim != 2:
            raise ValueError('observations must have shape (T, D_y)')
        if num_lags < 1:
            raise ValueError('num_lags must be at least 1')
        if observations.shape[0] <= num_lags:
            raise ValueError('observations must contain more than num_lags steps')

        predictors = lagged_observations(observations, num_lags)
        targets = observations[num_lags:]
        if predictors.shape != (targets.shape[0], num_lags, targets.shape[1]):
            raise ValueError('predictor and target dimensions must agree')
        return cls(predictors=predictors, targets=targets)

    @property
    def num_steps(self) -> int:
        """Number of modeled targets R = T-L."""
        return self.targets.shape[0]

    @property
    def num_lags(self) -> int:
        """Number of conditioning lags L."""
        return self.predictors.shape[1]

    @property
    def output_dim(self) -> int:
        """Observation dimension D_y."""
        return self.targets.shape[1]
