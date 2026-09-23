"""Batched temporal values and known data for independent trajectories."""

import dataclasses
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


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Sequences:
    """Temporal values `(*B, T, D)` with valid-prefix lengths `(*B,)`.

    Every batch item is one sequence. Padded tails are storage only; `get`
    retrieves one cropped sequence on the host.

    Public construction normalizes arrays and validates lengths on the host.
    Batch transformations and PyTree reconstruction bypass host validation.
    """

    values: jax.Array
    lengths: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, 'values', jnp.asarray(self.values))
        object.__setattr__(self, 'lengths', jnp.asarray(self.lengths))
        self.validate()

    @classmethod
    def _unchecked(cls, values: jax.Array, lengths: jax.Array) -> typing.Self:
        """Reconstruct internal temporal values without host validation."""
        instance = object.__new__(cls)
        object.__setattr__(instance, 'values', values)
        object.__setattr__(instance, 'lengths', lengths)
        return instance

    def tree_flatten(self):
        return (self.values, self.lengths), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children) -> typing.Self:
        return cls._unchecked(*children)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.values.shape[:-2]

    @property
    def num_steps(self) -> int:
        return self.values.shape[-2]

    def validate(self) -> None:
        """Validate shapes and lengths on the host."""
        if self.values.ndim < 2:
            raise ValueError('values must have shape (*B, T, D)')
        if self.num_steps < 1:
            raise ValueError('sequences must contain at least one timestep')
        if self.lengths.shape != self.batch_shape:
            raise ValueError('lengths must have shape matching batch_shape')
        if not jnp.issubdtype(self.lengths.dtype, jnp.integer):
            raise ValueError('lengths must have an integer dtype')
        if bool(jnp.any((self.lengths < 1) | (self.lengths > self.num_steps))):
            raise ValueError('lengths must lie between 1 and T')

    @classmethod
    def from_sequence(cls, values: jax.Array) -> typing.Self:
        """Construct one unbatched sequence `(T, D)`."""
        values = jnp.asarray(values)
        if values.ndim != 2:
            raise ValueError('values must have shape (T, D)')
        return cls.from_padded(values, jnp.asarray(values.shape[0], dtype=jnp.int32))

    @classmethod
    def from_batch(cls, values: jax.Array) -> typing.Self:
        """Construct equal-length temporal values `(*B, T, D)`."""
        values = jnp.asarray(values)
        if values.ndim < 2:
            raise ValueError('values must have shape (*B, T, D)')
        return cls.from_padded(
            values,
            jnp.full(values.shape[:-2], values.shape[-2], dtype=jnp.int32),
        )

    @classmethod
    def from_padded(
        cls,
        values: jax.Array,
        lengths: jax.Array,
    ) -> typing.Self:
        """Construct explicitly padded temporal values `(*B, T, D)`."""
        return cls(values=values, lengths=lengths)

    @classmethod
    def from_sequences(
        cls,
        values: typing.Sequence[jax.Array],
    ) -> typing.Self:
        """Pad a flat collection of `(T_i, D)` arrays to a common time size."""
        sequences = tuple(cls.from_sequence(value) for value in values)
        return cls.stack(sequences)

    @classmethod
    def stack(
        cls,
        sequences: typing.Sequence['Sequences'],
        *,
        axis: int = 0,
    ) -> typing.Self:
        """Pad time and stack matching batches along a new batch axis."""
        if not sequences:
            raise ValueError('sequences must contain at least one item')
        batch_shape = sequences[0].batch_shape
        value_dim = sequences[0].values.shape[-1]
        for sequence in sequences:
            _batch.require_same(batch_shape, sequence.batch_shape)
            if sequence.values.shape[-1] != value_dim:
                raise ValueError('sequences must share a value dimension')
        _, axis = _batch.insertion((len(sequences),), axis, len(batch_shape))
        num_steps = max(sequence.num_steps for sequence in sequences)
        return cls._unchecked(
            values=jnp.stack(
                [sequence._pad_to_num_steps(num_steps) for sequence in sequences],
                axis=axis,
            ),
            lengths=jnp.stack([sequence.lengths for sequence in sequences], axis=axis),
        )

    def _pad_to_num_steps(self, num_steps: int) -> jax.Array:
        padding = ((0, 0),) * len(self.batch_shape)
        return jnp.pad(
            self.values,
            padding + ((0, num_steps - self.num_steps), (0, 0)),
        )

    def select(self, index) -> typing.Self:
        """Select only batch axes, retaining time and event axes."""
        index = _batch.selection(index, len(self.batch_shape))
        return self._unchecked(self.values[index], self.lengths[index])

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self._unchecked(
            _batch.broadcast_array(self.values, shape, axis),
            _batch.broadcast_array(self.lengths, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self._unchecked(
            jnp.squeeze(self.values, axis=axes),
            jnp.squeeze(self.lengths, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self._unchecked(
            jnp.take(self.values, permutation, axis=axis),
            jnp.take(self.lengths, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another batch position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self._unchecked(
            jnp.moveaxis(self.values, source, destination),
            jnp.moveaxis(self.lengths, source, destination),
        )

    def valid(self) -> jax.Array:
        """Valid prefixes `(*B, T)`."""
        return jnp.arange(self.num_steps) < self.lengths[..., None]

    def get(self, index=None) -> jax.Array:
        """Retrieve exactly one unpadded `(T_i, D)` array on the host."""
        if index is None:
            selected = self
        else:
            selected = self.select(index)

        if selected.batch_shape:
            raise ValueError('get requires selecting exactly one batch item')

        return selected.values[: int(selected.lengths)]

    def unpack(self) -> tuple[jax.Array, ...]:
        """Crop an unbatched sequence or flat collection on the host."""
        if not self.batch_shape:
            return (self.get(()),)
        if len(self.batch_shape) == 1:
            return tuple(self.get(i) for i in range(self.batch_shape[0]))
        raise ValueError('unpack requires at most one batch axis; use get(...)')


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Dataset:
    """Known observations, inputs, and whole-timestep masks for trajectories.

    Both temporal components share batch shape `*B`, padded time size `T`,
    and lengths. The boolean mask has shape `(*B, T)` and controls observation
    likelihoods only. Inputs at `t` affect `x[t-1] -> x[t]`; input zero is unused
    by an autonomous initial distribution.

    Public construction validates compatibility, masks, and finite input values
    on the host. Transformations and PyTree reconstruction skip those checks.
    """

    observations: Sequences
    inputs: Sequences
    mask: jax.Array

    def __post_init__(self) -> None:
        object.__setattr__(self, 'mask', jnp.asarray(self.mask))
        self.validate()

    @classmethod
    def _unchecked(
        cls,
        observations: Sequences,
        inputs: Sequences,
        mask: jax.Array,
    ) -> typing.Self:
        """Reconstruct internal data without host-side value validation."""
        instance = object.__new__(cls)
        object.__setattr__(instance, 'observations', observations)
        object.__setattr__(instance, 'inputs', inputs)
        object.__setattr__(instance, 'mask', mask)
        return instance

    def tree_flatten(self):
        return (self.observations, self.inputs, self.mask), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children) -> typing.Self:
        return cls._unchecked(*children)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.observations.batch_shape

    @property
    def lengths(self) -> jax.Array:
        return self.observations.lengths

    @property
    def num_steps(self) -> int:
        return self.observations.num_steps

    @property
    def observation_dim(self) -> int:
        return self.observations.values.shape[-1]

    @property
    def input_dim(self) -> int:
        return self.inputs.values.shape[-1]

    @property
    def dtype(self) -> jnp.dtype:
        return self.observations.values.dtype

    def astype(self, dtype: jax.typing.DTypeLike) -> typing.Self:
        """Cast observations only, preserving inputs, lengths, and masks.

        Structural invariants are preserved, but narrowing casts may overflow.
        Finiteness is validated at public construction, not after transformations.
        This operation remains JIT-compatible.
        """
        return self._unchecked(
            observations=self.observations._unchecked(
                values=self.observations.values.astype(dtype),
                lengths=self.observations.lengths,
            ),
            inputs=self.inputs,
            mask=self.mask,
        )

    def validate(self) -> None:
        """Validate temporal structure and public data policies on the host."""
        if self.batch_shape != self.inputs.batch_shape:
            raise ValueError('observations and inputs must share their batch shape')
        if self.num_steps != self.inputs.num_steps:
            raise ValueError('observations and inputs must share their time dimension')
        if not bool(jnp.all(self.lengths == self.inputs.lengths)):
            raise ValueError('observations and inputs must share their lengths')
        if self.mask.shape != (*self.batch_shape, self.num_steps):
            raise ValueError('mask must have shape (*B, T)')
        if self.mask.dtype != jnp.bool_:
            raise ValueError('mask must have boolean dtype')
        if bool(jnp.any(self.mask & ~self.valid())):
            raise ValueError('mask must be false outside valid sequence prefixes')
        if not bool(jnp.all(jnp.isfinite(self.observations.values))):
            raise ValueError('observations must contain only finite values')
        if not bool(jnp.all(jnp.isfinite(self.inputs.values))):
            raise ValueError('inputs must contain only finite values')

    @classmethod
    def from_sequence(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> typing.Self:
        """Construct one unbatched trajectory from `(T, D_y)` observations."""
        sequence = Sequences.from_sequence(observations)
        return cls.from_padded(
            sequence.values,
            sequence.lengths,
            inputs=inputs,
            mask=mask,
        )

    @classmethod
    def from_batch(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> typing.Self:
        """Construct equal-length trajectories with arbitrary batch shape."""
        sequences = Sequences.from_batch(observations)
        return cls.from_padded(
            sequences.values,
            sequences.lengths,
            inputs=inputs,
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
        """Pad a flat collection of unequal-length observation trajectories."""
        observations = tuple(jnp.asarray(values) for values in observations)
        sequences = Sequences.from_sequences(observations)
        padded_inputs = None
        if len(inputs):
            inputs = tuple(jnp.asarray(values) for values in inputs)
            if any(values.ndim != 2 for values in inputs):
                raise ValueError('each input sequence must have shape (T_i, D_u)')
            _require_matching_lengths(inputs, observations, name='inputs')
            padded_inputs = Sequences.from_sequences(inputs).values

        padded_mask = None
        if len(mask):
            mask = tuple(jnp.asarray(values) for values in mask)
            if any(values.ndim != 1 for values in mask):
                raise ValueError('each mask sequence must have shape (T_i,)')
            if any(values.dtype != jnp.bool_ for values in mask):
                raise ValueError('mask must have boolean dtype')
            _require_matching_lengths(mask, observations, name='mask')
            padded_mask = jnp.stack(
                [_pad_time(values, sequences.num_steps) for values in mask],
            )

        return cls.from_padded(
            sequences.values,
            sequences.lengths,
            inputs=padded_inputs,
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
        """Construct explicitly padded trajectories `(*B, T, D_y)`."""
        observation_sequences = Sequences.from_padded(observations, lengths)
        if inputs is None:
            inputs = jnp.empty(
                (*observation_sequences.values.shape[:-1], 0),
                dtype=observation_sequences.values.dtype,
            )
        input_sequences = Sequences.from_padded(inputs, observation_sequences.lengths)
        if mask is None:
            mask = observation_sequences.valid()
        return cls(observation_sequences, input_sequences, mask)

    @classmethod
    def stack(
        cls,
        datasets: typing.Sequence['Dataset'],
        *,
        axis: int = 0,
    ) -> typing.Self:
        """Pad time and stack datasets along a new batch axis."""
        if not datasets:
            raise ValueError('datasets must contain at least one item')
        _, axis = _batch.insertion((len(datasets),), axis, len(datasets[0].batch_shape))
        observations = Sequences.stack(
            [data.observations for data in datasets],
            axis=axis,
        )
        inputs = Sequences.stack([data.inputs for data in datasets], axis=axis)
        masks = [
            jnp.pad(
                data.mask,
                ((0, 0),) * len(data.batch_shape)
                + ((0, observations.num_steps - data.num_steps),),
                constant_values=False,
            )
            for data in datasets
        ]
        return cls._unchecked(observations, inputs, jnp.stack(masks, axis=axis))

    def select(self, index) -> typing.Self:
        """Select only trajectory batch dimensions."""
        index = _batch.selection(index, len(self.batch_shape))
        return self._unchecked(
            self.observations.select(index),
            self.inputs.select(index),
            self.mask[index],
        )

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        shape, axis = _batch.insertion(shape, axis, len(self.batch_shape))
        return self._unchecked(
            self.observations.broadcast(shape, axis),
            self.inputs.broadcast(shape, axis),
            _batch.broadcast_array(self.mask, shape, axis),
        )

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        axes = _batch.squeeze_axes(self.batch_shape, axis)
        return self._unchecked(
            self.observations.squeeze(axes),
            self.inputs.squeeze(axes),
            jnp.squeeze(self.mask, axis=axes),
        )

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one batch axis."""
        axis = _batch.axis_index(axis, len(self.batch_shape))
        permutation = _batch.permutation_indices(permutation, self.batch_shape[axis])
        return self._unchecked(
            self.observations.permute(permutation, axis),
            self.inputs.permute(permutation, axis),
            jnp.take(self.mask, permutation, axis=axis),
        )

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another batch position."""
        source = _batch.axis_index(source, len(self.batch_shape))
        destination = _batch.axis_index(destination, len(self.batch_shape))
        return self._unchecked(
            self.observations.move_axis(source, destination),
            self.inputs.move_axis(source, destination),
            jnp.moveaxis(self.mask, source, destination),
        )

    def unpack(self) -> tuple[jax.Array, ...]:
        """Crop observations in an unbatched or flat dataset on the host."""
        return self.observations.unpack()

    def valid(self) -> jax.Array:
        """Timesteps belonging to each trajectory."""
        return self.observations.valid()

    def transition_inputs(self) -> jax.Array:
        """Inputs for `x[t-1] -> x[t]`; input zero is unused."""
        return self.inputs.values[..., 1:, :]

    def valid_transitions(self) -> jax.Array:
        """Valid transitions `x[t] -> x[t+1]`."""
        return self.valid()[..., 1:]

    def observation_weights(self) -> jax.Array:
        """Valid timesteps with an available observation."""
        return self.valid() & self.mask

    def weighted_observations(self) -> WeightedObservations:
        """Observation values with validity-aware weights."""
        return WeightedObservations(
            values=self.observations.values,
            weights=self.observation_weights(),
        )
