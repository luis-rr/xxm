"""Shared representation of independent observation sequences."""

import typing

import jax
import jax.numpy as jnp

from xxm.core import _batch


def _pad_time(values: jax.Array, num_steps: int) -> jax.Array:
    """Pad the leading time axis with zeros, preserving event dimensions."""
    padding = ((0, num_steps - values.shape[0]),) + ((0, 0),) * (values.ndim - 1)
    return jnp.pad(values, padding)


def _require_matching_lengths(values, observations, *, name: str) -> None:
    """Require one time-aligned array per observation sequence."""
    if len(values) != len(observations):
        raise ValueError(f'{name} must contain one array per sequence')
    if any(
        value.shape[0] != observation.shape[0]
        for value, observation in zip(values, observations, strict=True)
    ):
        raise ValueError(f'{name} must match their observation sequence lengths')


class WeightedObservations(typing.NamedTuple):
    """
    Observation sequences with one aligned weight per timestep.

    `values` has shape `(*B, T, D_y)` and `weights` has shape `(*B, T)`.
    Leading dimensions are batch dimensions, shared across both attributes.
    """

    values: jax.Array
    weights: jax.Array

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        values_shape = self.values.shape[:-2]
        weights_shape = self.weights.shape[:-1]

        assert values_shape == weights_shape

        return values_shape

    @property
    def num_steps(self) -> int:
        """Number of timesteps."""
        return self.values.shape[-2]

    @property
    def observation_dim(self) -> int:
        """Observation dimension."""
        return self.values.shape[-1]

    @property
    def dtype(self) -> jax.typing.DTypeLike:
        """Observation dtype."""
        return self.values.dtype

    def validate(self) -> None:
        """Validate structural weighted-observation invariants."""
        if self.values.ndim < 2:
            raise ValueError('values must have shape (*B, T, D_y)')

        if self.weights.shape != self.values.shape[:-1]:
            raise ValueError('weights must have shape (*B, T) aligned with values')

    @classmethod
    def from_sequence(
        cls,
        values: jax.Array,
        *,
        weights: jax.Array | None = None,
    ) -> typing.Self:
        """Construct one unbatched observation sequence `(T, D_y)`."""
        values = jnp.asarray(values)

        if values.ndim != 2:
            raise ValueError('values must have shape (T, D_y)')

        if weights is None:
            weights = jnp.ones(
                values.shape[0],
                dtype=bool,
            )
        weights = jnp.asarray(weights)

        data = cls(
            values=values,
            weights=weights,
        )

        data.validate()

        return data

    def unpack_sequence(
        self,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Return one raw sequence `(T, D_y)` and its weights `(T,)`.

        Scalar and singleton-batched weighted observations are accepted.
        """
        if self.batch_shape == ():
            return self.values, self.weights

        if self.batch_shape == (1,):
            return self.values[0], self.weights[0]

        raise ValueError(
            'unpack_sequence requires scalar or singleton-batched observations'
        )

    def select(
        self,
        index,
    ) -> typing.Self:
        """Index only batch dimensions, retaining this object type."""
        index = _batch.selection(
            index,
            len(self.batch_shape),
        )

        return self.__class__(
            values=self.values[index],
            weights=self.weights[index],
        )

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(
            shape,
            axis,
            len(self.batch_shape),
        )

        return self.__class__(
            values=_batch.broadcast_array(
                self.values,
                shape,
                axis,
            ),
            weights=_batch.broadcast_array(
                self.weights,
                shape,
                axis,
            ),
        )

    def squeeze(
        self,
        axis=None,
    ) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(
            self.batch_shape,
            axis,
        )

        return self.__class__(
            values=jnp.squeeze(
                self.values,
                axis=axes,
            ),
            weights=jnp.squeeze(
                self.weights,
                axis=axes,
            ),
        )

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        """Reorder entries along a batch axis."""
        axis = _batch.axis_index(
            axis,
            len(self.batch_shape),
        )

        permutation = _batch.permutation_indices(
            permutation,
            self.batch_shape[axis],
        )

        return self.__class__(
            values=jnp.take(
                self.values,
                permutation,
                axis=axis,
            ),
            weights=jnp.take(
                self.weights,
                permutation,
                axis=axis,
            ),
        )

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        """Move one batch axis to another position."""
        source = _batch.axis_index(
            source,
            len(self.batch_shape),
        )
        destination = _batch.axis_index(
            destination,
            len(self.batch_shape),
        )

        return self.__class__(
            values=jnp.moveaxis(
                self.values,
                source,
                destination,
            ),
            weights=jnp.moveaxis(
                self.weights,
                source,
                destination,
            ),
        )

    def astype(
        self,
        dtype: jax.typing.DTypeLike,
    ) -> typing.Self:
        """Convert observation values to a different dtype."""
        return self._replace(
            values=self.values.astype(dtype),
        )

    def safe_values(self) -> jax.Array:
        """
        Replace zero-weight observations by finite zeros before evaluation.

        This prevents absent observations from producing infinities or NaNs
        before their likelihood contribution is multiplied by zero.
        """
        return self.safe_aligned(self.values)

    def safe_aligned(self, values: jax.Array) -> jax.Array:
        """Zero aligned `(*B, T, *E)` values wherever observation weight is zero."""
        ndim = self.weights.ndim
        if values.shape[:ndim] != self.weights.shape:
            raise ValueError('values must begin with the observation weight shape')

        present = (self.weights != 0).reshape(
            self.weights.shape + (1,) * (values.ndim - ndim)
        )
        return jnp.where(present, values, jnp.zeros((), dtype=values.dtype))

    def weighted_sum(
        self,
        values: jax.Array,
    ) -> jax.Array:
        """Sum aligned weighted values over time, preserving batch dimensions."""
        if values.shape != self.weights.shape:
            raise ValueError(
                'values must have shape (*B, T) aligned with observation weights'
            )

        safe_values = self.safe_aligned(values)

        return jnp.sum(
            self.weights * safe_values,
            axis=-1,
        )


class Sequences(typing.NamedTuple):
    """
    Rectangular representation of independent sequences.

    Shapes are:

    - `observations`: `(S, T, D_y)`
    - `inputs`: `(S, T, D_u)`
    - `lengths`: `(S,)`
    - `mask`: `(S, T)`

    `lengths[s]` gives the valid prefix of sequence `s`. `mask[s, t]`
    indicates whether the complete observation at a valid timestep is
    available.

    Inputs follow the incoming-state convention: `inputs[s, t]` affects
    the transition from `x[s, t - 1]` to `x[s, t]`. Thus `inputs[:, 0]`
    is unused by an autonomous initial distribution.
    """

    observations: jax.Array
    inputs: jax.Array
    lengths: jax.Array
    mask: jax.Array

    @property
    def dtype(self) -> jnp.dtype:
        """Observation dtype."""
        return self.observations.dtype

    def astype(
        self,
        dtype: jax.typing.DTypeLike,
    ) -> typing.Self:
        """
        Convert observations to a different dtype.

        Input, length, and mask dtypes are unchanged.
        """
        return self._replace(
            observations=self.observations.astype(dtype),
        )

    def validate(self) -> None:
        """Validate structural sequence-data invariants."""
        if self.observations.ndim != 3:
            raise ValueError('observations must have shape (S, T, D_y)')

        num_sequences, num_steps = self.observations.shape[:2]

        if num_sequences < 1 or num_steps < 1:
            raise ValueError(
                'sequences must contain at least one sequence and timestep'
            )

        if self.inputs.ndim != 3 or self.inputs.shape[:2] != (num_sequences, num_steps):
            raise ValueError(
                'inputs must have shape (S, T, D_u) aligned with observations'
            )

        if self.lengths.shape != (num_sequences,):
            raise ValueError('lengths must have shape (S,)')

        if self.lengths.dtype != jnp.int32:
            raise ValueError('lengths must have dtype int32')

        if bool(jnp.any((self.lengths < 1) | (self.lengths > num_steps))):
            raise ValueError('lengths must lie between 1 and T')

        if self.mask.shape != (
            num_sequences,
            num_steps,
        ):
            raise ValueError('mask must have shape (S, T)')

        if self.mask.dtype != jnp.bool_:
            raise ValueError('mask must have boolean dtype')

        if bool(jnp.any(self.mask & ~self.valid())):
            raise ValueError('mask must be false outside valid sequence prefixes')

        if not bool(jnp.all(jnp.isfinite(self.observations))):
            raise ValueError('observations must contain only finite values')

        if not bool(jnp.all(jnp.isfinite(self.inputs))):
            raise ValueError('inputs must contain only finite values')

    @classmethod
    def from_sequence(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> typing.Self:
        """Construct from one sequence `(T, D_y)`."""
        observations = jnp.asarray(observations)

        if observations.ndim != 2:
            raise ValueError('observations must have shape (T, D_y)')

        num_steps = observations.shape[0]

        if inputs is not None:
            inputs = jnp.asarray(inputs)
            if inputs.ndim != 2 or inputs.shape[0] != num_steps:
                raise ValueError('inputs must have shape (T, D_u)')

        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.ndim != 1 or mask.shape[0] != num_steps:
                raise ValueError('mask must have shape (T,)')

        return cls.from_batch(
            observations=observations[None],
            inputs=None if inputs is None else inputs[None],
            mask=None if mask is None else mask[None],
        )

    @classmethod
    def from_batch(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> typing.Self:
        """Construct from equal-length sequences `(S, T, D_y)`."""
        observations = jnp.asarray(observations)

        if observations.ndim != 3:
            raise ValueError('observations must have shape (S, T, D_y)')

        num_sequences, num_steps = observations.shape[:2]

        return cls.from_padded(
            observations=observations,
            inputs=inputs,
            lengths=jnp.full(
                (num_sequences,),
                num_steps,
                dtype=jnp.int32,
            ),
            mask=mask,
        )

    @classmethod
    def from_sequences(
        cls,
        observations: typing.Sequence[jax.Array],
        *,
        inputs: typing.Sequence[jax.Array] = (),
        mask: typing.Sequence[jax.Array] = (),
    ) -> typing.Self:
        """Pack unequal-length sequences into rectangular storage."""
        observations = tuple(jnp.asarray(values) for values in observations)

        if len(observations) == 0:
            raise ValueError('observations must contain at least one sequence')

        if any(values.ndim != 2 for values in observations):
            raise ValueError('each observation sequence must have shape (T_s, D_y)')

        if any(values.shape[0] < 1 for values in observations):
            raise ValueError(
                'each observation sequence must contain at least one timestep'
            )

        observation_dim = observations[0].shape[1]
        if any(values.shape[1] != observation_dim for values in observations):
            raise ValueError(
                'all observation sequences must share an observation dimension'
            )

        lengths = jnp.asarray(
            [values.shape[0] for values in observations],
            dtype=jnp.int32,
        )

        num_steps = max(values.shape[0] for values in observations)

        padded_observations = jnp.stack(
            [_pad_time(values, num_steps) for values in observations]
        )

        if len(inputs) == 0:
            padded_inputs = jnp.empty(
                (
                    len(observations),
                    num_steps,
                    0,
                ),
                dtype=padded_observations.dtype,
            )

        else:
            inputs = tuple(jnp.asarray(values) for values in inputs)

            if any(values.ndim != 2 for values in inputs):
                raise ValueError('each input sequence must have shape (T_s, D_u)')

            _require_matching_lengths(inputs, observations, name='inputs')

            input_dim = inputs[0].shape[1]
            if any(values.shape[1] != input_dim for values in inputs):
                raise ValueError('all input sequences must share an input dimension')

            padded_inputs = jnp.stack(
                [_pad_time(values, num_steps) for values in inputs]
            )

        if len(mask) == 0:
            padded_mask = jnp.arange(num_steps)[None, :] < lengths[:, None]

        else:
            mask = tuple(jnp.asarray(values) for values in mask)

            if any(values.ndim != 1 for values in mask):
                raise ValueError('each mask sequence must have shape (T_s,)')

            if any(values.dtype != jnp.bool_ for values in mask):
                raise ValueError('mask must have boolean dtype')

            _require_matching_lengths(mask, observations, name='mask')

            padded_mask = jnp.stack([_pad_time(values, num_steps) for values in mask])

        return cls.from_padded(
            observations=padded_observations,
            inputs=padded_inputs,
            lengths=lengths,
            mask=padded_mask,
        )

    @classmethod
    def from_padded(
        cls,
        observations: jax.Array,
        lengths: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> typing.Self:
        """Construct from explicitly padded rectangular arrays."""
        observations = jnp.asarray(observations)
        lengths = jnp.asarray(lengths)

        if observations.ndim != 3:
            raise ValueError('observations must have shape (S, T, D_y)')

        num_sequences, num_steps = observations.shape[:2]

        if lengths.shape != (num_sequences,):
            raise ValueError('lengths must have shape (S,)')

        if not jnp.issubdtype(lengths.dtype, jnp.integer):
            raise ValueError('lengths must have an integer dtype')

        lengths = lengths.astype(jnp.int32)

        if inputs is None:
            inputs = jnp.empty(
                (
                    num_sequences,
                    num_steps,
                    0,
                ),
                dtype=observations.dtype,
            )
        inputs = jnp.asarray(inputs)

        valid = jnp.arange(num_steps)[None, :] < lengths[:, None]

        if mask is None:
            mask = valid

        mask = jnp.asarray(mask)

        data = cls(
            observations=observations,
            inputs=inputs,
            lengths=lengths,
            mask=mask,
        )

        data.validate()

        return data

    def num_sequences(self) -> int:
        """Number of independent sequences."""
        return self.observations.shape[0]

    def num_steps(self) -> int:
        """Rectangular time dimension."""
        return self.observations.shape[1]

    def observation_dim(self) -> int:
        """Observation dimension."""
        return self.observations.shape[2]

    def input_dim(self) -> int:
        """Known-input dimension."""
        return self.inputs.shape[2]

    def valid(self) -> jax.Array:
        """Timesteps belonging to each sequence."""
        return (
            jnp.arange(
                self.num_steps(),
            )[None, :]
            < self.lengths[:, None]
        )

    def transition_inputs(self) -> jax.Array:
        """Inputs for `x[t-1] -> x[t]`; the boundary input at zero is unused."""
        return self.inputs[:, 1:, :]

    def valid_transitions(self) -> jax.Array:
        """Valid transitions `x[t] -> x[t + 1]`."""
        return self.valid()[:, 1:]

    def observation_weights(self) -> jax.Array:
        """Valid timesteps with an available observation."""
        return self.valid() & self.mask

    def weighted_observations(
        self,
    ) -> WeightedObservations:
        """Observation sequences with validity-aware weights."""
        observations = WeightedObservations(
            values=self.observations,
            weights=self.observation_weights(),
        )

        observations.validate()

        return observations
