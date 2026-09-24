"""Batched temporal values and known data for independent trajectories."""

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.mask import ArbitraryMask, ContiguousMask


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

        batch.require_same_shape(values_shape, weights_shape)

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
        return batch.select(self, index)

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(
        self,
        axis=None,
    ) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        """Reorder entries along a batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        """Move one batch axis to another position."""
        return batch.move_axis(self, source, destination)

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
        batch.split_prefix(values.shape, self.weights.shape, name='values shape')
        present = batch.expand_trailing(self.weights != 0, values.ndim)
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
@dataclasses.dataclass(frozen=True, eq=False, init=False)
class Sequences:
    """Temporal values `(*B, T, D)` with contiguous valid prefixes.

    Every batch item is one sequence. `mask` stores valid-prefix lengths in a
    `ContiguousMask`; padded tails are storage only. The `lengths` property is
    retained as the compact public view of those prefixes, and `get` retrieves
    one cropped sequence on the host.

    Public construction normalizes arrays and validates lengths on the host.
    Batch transformations and PyTree reconstruction bypass host validation.
    """

    values: jax.Array
    mask: ContiguousMask

    def __init__(
        self,
        values: jax.Array,
        mask: ContiguousMask,
    ) -> None:
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'mask', mask)
        self.validate()

    @classmethod
    def _unchecked(
        cls,
        values: jax.Array,
        mask: ContiguousMask,
    ) -> typing.Self:
        """Reconstruct internal temporal values without host validation."""
        instance = object.__new__(cls)
        object.__setattr__(instance, 'values', values)
        object.__setattr__(instance, 'mask', mask)
        return instance

    def tree_flatten(self):
        return (self.values, self.mask), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children) -> typing.Self:
        return cls._unchecked(*children)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.values.shape[:-2]

    @property
    def num_steps(self) -> int:
        return self.values.shape[-2]

    @property
    def lengths(self) -> jax.Array:
        """Valid-prefix lengths `(*B,)`."""
        return self.mask.lengths

    def validate(self) -> None:
        """Validate shapes and sequence-prefix lengths on the host."""
        if self.values.ndim < 2:
            raise ValueError(
                f'values must have shape (*B, T, D). Got {self.values.shape}'
            )

        if self.num_steps < 1:
            raise ValueError(
                f'sequences must contain at least one timestep. Got {self.num_steps}'
            )

        if self.mask.batch_shape != self.batch_shape:
            raise ValueError(
                f'mask lengths must have shape matching batch_shape. Got {self.mask.batch_shape}, expected {self.batch_shape}'
            )

        if bool(jnp.any((self.lengths < 1) | (self.lengths > self.num_steps))):
            raise ValueError(
                f'lengths must lie between 1 and T. Got {self.lengths}, expected 1 to {self.num_steps}'
            )

    @classmethod
    def from_sequence(cls, values: jax.Array) -> typing.Self:
        """Construct one unbatched sequence `(T, D)`."""
        values = jnp.asarray(values)
        if values.ndim != 2:
            raise ValueError(f'values must have shape (T, D). Got {values.shape}')

        return cls.from_padded(
            values,
            jnp.asarray(values.shape[0], dtype=jnp.int32),
        )

    @classmethod
    def from_batch(cls, values: jax.Array) -> typing.Self:
        """Construct equal-length temporal values `(*B, T, D)`."""
        values = jnp.asarray(values)
        if values.ndim < 2:
            raise ValueError(f'values must have shape (*B, T, D). Got {values.shape}')

        return cls.from_padded(
            values,
            jnp.full(
                values.shape[:-2],
                values.shape[-2],
                dtype=jnp.int32,
            ),
        )

    @classmethod
    def from_padded(
        cls,
        values: jax.Array,
        lengths: jax.Array,
    ) -> typing.Self:
        """Construct explicitly padded temporal values `(*B, T, D)`."""
        return cls(
            values=values,
            mask=ContiguousMask(lengths),
        )

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
            batch.require_same_shape(batch_shape, sequence.batch_shape)
            if sequence.values.shape[-1] != value_dim:
                raise ValueError('sequences must share a value dimension')

        _, axis = batch.insertion(
            (len(sequences),),
            axis,
            len(batch_shape),
        )
        num_steps = max(sequence.num_steps for sequence in sequences)

        return cls._unchecked(
            values=jnp.stack(
                [sequence._pad_to_num_steps(num_steps) for sequence in sequences],
                axis=axis,
            ),
            mask=ContiguousMask._unchecked(
                jnp.stack(
                    [sequence.lengths for sequence in sequences],
                    axis=axis,
                )
            ),
        )

    def _pad_to_num_steps(self, num_steps: int) -> jax.Array:
        padding = ((0, 0),) * len(self.batch_shape)
        return jnp.pad(
            self.values,
            padding + ((0, num_steps - self.num_steps), (0, 0)),
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Select only batch axes, retaining time and event axes."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another batch position."""
        return batch.move_axis(self, source, destination)

    def valid(self) -> ContiguousMask:
        """Contiguous validity mask for these sequences."""
        return self.mask

    def get(self, index: batch.SelT | None = None) -> jax.Array:
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
    """Known observations, inputs, and whole-timestep visibility for trajectories.

    Both temporal components share batch shape `*B`, padded time size `T`, and
    contiguous valid prefixes. `visible` marks accessible observations with
    `True` and deliberately withheld observations with `False`; it gates
    observation likelihoods only. Inputs at `t` affect `x[t-1] -> x[t]`; input
    zero is unused by an autonomous initial distribution.

    Public construction validates compatibility, visibility, and finite input values
    on the host. Transformations and PyTree reconstruction skip those checks.
    """

    observations: Sequences
    inputs: Sequences
    visible: ArbitraryMask

    def __post_init__(self) -> None:
        if not isinstance(self.visible, ArbitraryMask):
            object.__setattr__(self, 'visible', ArbitraryMask(self.visible))
        self.validate()

    @classmethod
    def _unchecked(
        cls,
        observations: Sequences,
        inputs: Sequences,
        visible: ArbitraryMask,
    ) -> typing.Self:
        """Reconstruct internal data without host-side value validation."""
        instance = object.__new__(cls)
        object.__setattr__(instance, 'observations', observations)
        object.__setattr__(instance, 'inputs', inputs)
        object.__setattr__(instance, 'visible', visible)
        return instance

    def tree_flatten(self):
        return (self.observations, self.inputs, self.visible), None

    @classmethod
    def tree_unflatten(cls, auxiliary, children) -> typing.Self:
        return cls._unchecked(*children)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.observations.batch_shape

    @property
    def lengths(self) -> jax.Array:
        """Observation/input valid-prefix lengths `(*B,)`."""
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
        """Cast observations only, preserving inputs, prefixes, and visibility.

        Structural invariants are preserved, but narrowing casts may overflow.
        Finiteness is validated at public construction, not after transformations.
        This operation remains JIT-compatible.
        """
        return self._unchecked(
            observations=self.observations._unchecked(
                values=self.observations.values.astype(dtype),
                mask=self.observations.mask,
            ),
            inputs=self.inputs,
            visible=self.visible,
        )

    def validate(self) -> None:
        """Validate temporal structure and public data policies on the host."""
        if self.batch_shape != self.inputs.batch_shape:
            raise ValueError('observations and inputs must share their batch shape')
        if self.num_steps != self.inputs.num_steps:
            raise ValueError('observations and inputs must share their time dimension')
        if not bool(jnp.all(self.lengths == self.inputs.lengths)):
            raise ValueError('observations and inputs must share their lengths')

        if self.visible.batch_shape != self.batch_shape:
            raise ValueError('visibility must share the dataset batch shape')
        if self.visible.num_steps != self.num_steps:
            raise ValueError('visibility must share the dataset time dimension')

        self.visible.validate(self.observations.values)

        valid = self.valid().materialize(self.num_steps)
        if bool(jnp.any(self.visible.values & ~valid)):
            raise ValueError('visible must be false outside valid sequence prefixes')

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
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Construct one unbatched trajectory from `(T, D_y)` observations."""
        sequence = Sequences.from_sequence(observations)
        return cls.from_padded(
            sequence.values,
            sequence.lengths,
            inputs=inputs,
            visible=visible,
        )

    @classmethod
    def from_batch(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Construct equal-length trajectories with arbitrary batch shape."""
        sequences = Sequences.from_batch(observations)
        return cls.from_padded(
            sequences.values,
            sequences.lengths,
            inputs=inputs,
            visible=visible,
        )

    @classmethod
    def from_sequences(
        cls,
        observations: typing.Sequence[jax.Array],
        *,
        inputs: typing.Sequence[jax.Array] | None = None,
        visible: typing.Sequence[jax.Array] | None = None,
    ) -> typing.Self:
        """Pad a flat collection of unequal-length observation trajectories."""
        observations = tuple(jnp.asarray(values) for values in observations)
        sequences = Sequences.from_sequences(observations)

        padded_inputs = None
        if inputs is not None:
            inputs = tuple(jnp.asarray(values) for values in inputs)
            if any(values.ndim != 2 for values in inputs):
                raise ValueError('each input sequence must have shape (T_i, D_u)')
            _require_matching_lengths(
                inputs,
                observations,
                name='inputs',
            )
            padded_inputs = Sequences.from_sequences(inputs).values

        padded_visible = None
        if visible is not None:
            visible = tuple(jnp.asarray(values) for values in visible)
            if any(values.ndim != 1 for values in visible):
                raise ValueError('each visible sequence must have shape (T_i,)')
            if any(values.dtype != jnp.bool_ for values in visible):
                raise ValueError('visible must have boolean dtype')
            _require_matching_lengths(
                visible,
                observations,
                name='visible',
            )
            padded_visible = jnp.stack(
                [_pad_time(values, sequences.num_steps) for values in visible],
            )

        return cls.from_padded(
            sequences.values,
            sequences.lengths,
            inputs=padded_inputs,
            visible=padded_visible,
        )

    @classmethod
    def from_padded(
        cls,
        observations: jax.Array,
        lengths: jax.Array,
        *,
        inputs: jax.Array | None = None,
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Construct explicitly padded trajectories `(*B, T, D_y)`."""
        observation_sequences = Sequences.from_padded(
            observations,
            lengths,
        )

        if inputs is None:
            inputs = jnp.empty(
                (*observation_sequences.values.shape[:-1], 0),
                dtype=observation_sequences.values.dtype,
            )

        input_sequences = Sequences.from_padded(
            inputs,
            observation_sequences.lengths,
        )

        if visible is None:
            visibility = observation_sequences.valid().materialize(
                observation_sequences.num_steps
            )

            visibility = ArbitraryMask(visibility)

        elif isinstance(visible, ArbitraryMask):
            visibility = visible
        else:
            visibility = ArbitraryMask(visible)

        return cls(
            observation_sequences,
            input_sequences,
            visibility,
        )

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

        _, axis = batch.insertion(
            (len(datasets),),
            axis,
            len(datasets[0].batch_shape),
        )

        observations = Sequences.stack(
            [data.observations for data in datasets],
            axis=axis,
        )
        inputs = Sequences.stack(
            [data.inputs for data in datasets],
            axis=axis,
        )

        visibility = [
            jnp.pad(
                data.visible.values,
                ((0, 0),) * len(data.batch_shape)
                + ((0, observations.num_steps - data.num_steps),),
                constant_values=False,
            )
            for data in datasets
        ]

        return cls._unchecked(
            observations,
            inputs,
            ArbitraryMask._unchecked(jnp.stack(visibility, axis=axis)),
        )

    def select(self, index: batch.SelT) -> typing.Self:
        """Select only trajectory batch dimensions."""
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        """Insert replicated batch dimensions at `axis`."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        """Remove singleton batch dimensions."""
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        """Reorder entries along one batch axis."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        """Move one batch axis to another batch position."""
        return batch.move_axis(self, source, destination)

    def valid(self) -> ContiguousMask:
        """Contiguous timesteps belonging to each trajectory."""
        return self.observations.valid()

    def transition_inputs(self) -> jax.Array:
        """Inputs for `x[t-1] -> x[t]`; input zero is unused."""
        return self.inputs.values[..., 1:, :]

    def valid_transitions(self) -> ContiguousMask:
        """Contiguous mask for valid transitions `x[t] -> x[t+1]`."""
        return self.valid().adjacent_pairs()

    def observation_weights(self) -> jax.Array:
        """Boolean weights combining structural validity and observation visibility."""
        return self.valid().apply(self.visible.values)

    def weighted_observations(self) -> WeightedObservations:
        """Observation values with validity-aware weights."""
        return WeightedObservations(
            values=self.observations.values,
            weights=self.observation_weights(),
        )
