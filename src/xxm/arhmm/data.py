"""Aligned autoregressive data for independent trajectories."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.data import Dataset, Sequences, WeightedObservations
from xxm.core.lengths import Lengths, LengthsSpec
from xxm.core.mask import ArbitraryMask


def lagged_observations(
    observations: jax.Array,
    num_lags: int,
) -> jax.Array:
    """Return histories ordered from lag 1 to lag L."""
    return jnp.stack(
        [
            observations[
                ...,
                num_lags - i - 1 : observations.shape[-2] - i - 1,
                :,
            ]
            for i in range(num_lags)
        ],
        axis=-2,
    )


class ARDataset(typing.NamedTuple):
    """
    Prepared autoregressive view of a common Dataset.

    Raw time `t = L + r` maps to compact AR time `r`.

    `observations` contains the modeled targets `y[t]`.
    `predictors` contains `(y[t-1], ..., y[t-L])`.
    `inputs` are aligned to the same raw target times.
    `visible[r]` is true only when the target and every observation in its
    predictor window are visible in the source dataset.
    """

    observations: Sequences  # (*B, R, D_y)
    predictors: jax.Array  # (*B, R, L, D_y)
    inputs: Sequences  # (*B, R, D_u)
    visible: ArbitraryMask  # (*B, R)

    @classmethod
    def from_dataset(
        cls,
        data: Dataset,
        num_lags: int,
    ) -> typing.Self:
        """Prepare compact AR rows from a validated source dataset."""
        if num_lags < 1:
            raise ValueError('num_lags must be at least 1')

        if data.max_steps <= num_lags:
            raise ValueError('dataset must contain more than num_lags padded steps')

        if bool(jnp.any(data.lengths.values <= num_lags)):
            raise ValueError(
                'every sequence must contain more than num_lags valid steps'
            )

        lengths = Lengths(data.lengths.values - num_lags)

        observations = Sequences._unchecked(
            values=data.observations.values[..., num_lags:, :],
            lengths=lengths,
        )

        inputs = Sequences._unchecked(
            values=data.inputs.values[..., num_lags:, :],
            lengths=lengths,
        )

        predictors = lagged_observations(
            data.observations.values,
            num_lags,
        )

        raw_visible = data.visible.values

        visibility_windows = jnp.stack(
            [
                raw_visible[
                    ...,
                    num_lags - offset : data.max_steps - offset,
                ]
                for offset in range(num_lags + 1)
            ],
            axis=-1,
        )

        visible = ArbitraryMask._unchecked(
            jnp.all(
                visibility_windows,
                axis=-1,
            )
        )

        prepared = cls(
            observations=observations,
            predictors=predictors,
            inputs=inputs,
            visible=visible,
        )

        prepared.validate()
        return prepared

    @classmethod
    def from_sequence(
        cls,
        observations: jax.Array,
        num_lags: int,
        *,
        inputs: jax.Array | None = None,
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Prepare compact AR rows from a single sequence of observations."""
        return cls.from_dataset(
            Dataset.from_sequence(
                observations,
                inputs=inputs,
                visible=visible,
            ),
            num_lags=num_lags,
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.observations.batch_shape

    @property
    def lengths(self) -> Lengths:
        return self.observations.lengths

    @property
    def max_steps(self) -> int:
        """Number of compact modeled steps R = T - L."""
        return self.observations.max_steps

    @property
    def lengths_spec(self) -> LengthsSpec:
        return LengthsSpec(self.lengths, self.max_steps)

    @property
    def num_lags(self) -> int:
        return self.predictors.shape[-2]

    @property
    def observation_dim(self) -> int:
        return self.observations.values.shape[-1]

    @property
    def output_dim(self) -> int:
        """Compatibility name used by AR emission code."""
        return self.observation_dim

    @property
    def input_dim(self) -> int:
        return self.inputs.values.shape[-1]

    @property
    def dtype(self) -> jnp.dtype:
        return self.observations.values.dtype

    @property
    def targets(self) -> jax.Array:
        """Modeled target observations on the compact timeline."""
        return self.observations.values

    def validate(self) -> None:
        expected_predictors = (
            *self.batch_shape,
            self.max_steps,
            self.num_lags,
            self.observation_dim,
        )

        if self.predictors.shape != expected_predictors:
            raise ValueError(
                f'predictors must have shape {expected_predictors}; '
                f'got {self.predictors.shape}'
            )

        if self.inputs.batch_shape != self.batch_shape:
            raise ValueError('observations and inputs must share their batch shape')

        if self.inputs.max_steps != self.max_steps:
            raise ValueError('observations and inputs must share their time dimension')

        if not bool(jnp.all(self.inputs.lengths.values == self.lengths.values)):
            raise ValueError('observations and inputs must share their lengths')

        if self.visible.batch_shape != self.batch_shape:
            raise ValueError('visibility must share the AR dataset batch shape')

        if self.visible.max_steps != self.max_steps:
            raise ValueError('visibility must share the AR dataset time dimension')

        self.visible.validate(self.observations.values)

        valid = self.lengths.to_mask(self.max_steps).values

        if bool(jnp.any(self.visible.values & ~valid)):
            raise ValueError('visible must be false outside valid AR prefixes')

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        return batch.permute(
            self,
            permutation,
            axis=axis,
        )

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        return batch.move_axis(
            self,
            source,
            destination,
        )

    def transition_inputs(self) -> jax.Array:
        """
        Inputs associated with transitions into compact AR states 1:R.

        Compact input r corresponds to the source input at raw time L+r.
        """
        return self.inputs.values[..., 1:, :]

    def observation_weights(self) -> jax.Array:
        return self.lengths.clear(
            self.visible.values,
        )

    def weighted_observations(self) -> WeightedObservations:
        return WeightedObservations(
            values=self.observations.values,
            weights=self.observation_weights(),
            lengths=self.lengths,
        )

    def safe_predictors(self) -> jax.Array:
        """
        Zero predictor rows whose complete AR likelihood factor is inactive.

        This avoids evaluating nonlinear conditionals on inaccessible history.
        """
        return self.weighted_observations().safe_aligned(self.predictors)
