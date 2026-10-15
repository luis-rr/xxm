"""Batched temporal values and known data for independent trajectories."""

import dataclasses
import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.lengths import FillT, Lengths, LengthsSpec
from xxm.core.mask import ArbitraryMask

BatchableT = typing.TypeVar('BatchableT', bound=batch.Batchable)


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
    lengths: Lengths

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch shape."""
        values_shape = self.values.shape[:-2]
        weights_shape = self.weights.shape[:-1]

        batch.require_same_shape(values_shape, weights_shape)

        return values_shape

    @property
    def max_steps(self) -> int:
        """Number of timesteps."""
        return self.values.shape[-2]

    @property
    def observation_dim(self) -> int:
        """Observation dimension."""
        return self.values.shape[-1]

    @property
    def dtype(self) -> jnp.dtype:
        """Observation dtype."""
        return self.values.dtype

    def validate(self) -> None:
        """Validate weighted-observation shapes and values on the host."""
        self.validate_shapes()

        support = self.lengths.to_mask(self.max_steps).values

        if bool(jnp.any(~support & (self.weights != 0))):
            raise ValueError('weights must be zero outside lengths')

    def validate_shapes(self) -> None:
        """Check static structure without reading traced length values."""
        if self.values.ndim < 2:
            raise ValueError('values must have shape (*B, T, D_y)')

        if self.weights.shape != self.values.shape[:-1]:
            raise ValueError('weights must have shape (*B, T) aligned with values')

        batch.require_same_shape(self.batch_shape, self.lengths.batch_shape)

    @classmethod
    def from_array(
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
            lengths=Lengths(jnp.asarray(values.shape[0], dtype=jnp.int32)),
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
    """Temporal values `(*B, T, *E)` with contiguous valid prefixes.

    Every batch item is one sequence. `lengths` stores valid-prefix lengths;
    padded tails are storage only. `get` retrieves a cropped sequence on the host.

    Construction normalizes arrays and validates static storage shape only.
    """

    values: jax.Array
    lengths: Lengths

    def __init__(
        self,
        values: jax.Array,
        lengths: Lengths,
    ) -> None:
        object.__setattr__(self, 'values', jnp.asarray(values))
        object.__setattr__(self, 'lengths', lengths)
        self.validate()

    @classmethod
    def _unchecked(
        cls,
        values: jax.Array,
        lengths: Lengths,
    ) -> typing.Self:
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
        return self.lengths.batch_shape

    @property
    def time_axis(self) -> int:
        return len(self.batch_shape)

    @property
    def max_steps(self) -> int:
        return self.values.shape[self.time_axis]

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.values.shape[self.time_axis + 1 :]

    def select_time(self, *, start: int, stop: int | None = None) -> typing.Self:
        """Slice stored time, retaining batch/event axes and valid-prefix lengths.

        Bounds follow ordinary slicing of the dense horizon.
        """
        start, stop, _ = slice(start, stop).indices(self.max_steps)

        extent = jnp.clip(self.lengths.values - start, 0, max(stop - start, 0))
        lengths = Lengths(extent)
        selection = (slice(None),) * self.time_axis + (slice(start, stop),)
        return type(self).from_padded(self.values[selection], lengths)

    def time_major(self) -> jax.Array:
        """Return values with layout `(T, *B, *E)` for temporal kernels."""
        return jnp.moveaxis(self.values, self.time_axis, 0)

    @property
    def lengths_spec(self) -> LengthsSpec:
        """Number of timesteps."""
        return LengthsSpec(self.lengths, max_steps=self.max_steps)

    def validate(self) -> None:
        """Validate temporal storage shape."""

        batch_shape = self.lengths.batch_shape
        batch_ndim = len(batch_shape)

        if self.values.ndim < batch_ndim + 1:
            raise ValueError(
                f'values must have shape (*B, T, *E). Got {self.values.shape}'
            )

        batch.require_same_shape(
            self.values.shape[:batch_ndim],
            batch_shape,
        )

    @classmethod
    def from_array(cls, values: jax.Array) -> typing.Self:
        """Construct one unbatched sequence `(T, *E)`."""
        values = jnp.asarray(values)
        if values.ndim < 1:
            raise ValueError(f'values must have shape (T, *E). Got {values.shape}')

        return cls.from_padded(
            values,
            Lengths(jnp.asarray(values.shape[0], dtype=jnp.int32)),
        )

    @classmethod
    def from_padded(
        cls,
        values: jax.Array,
        lengths: Lengths,
        fill: FillT = 0,
    ) -> typing.Self:
        """Construct explicitly padded temporal values `(*B, T, *E)`."""
        return cls(
            values=lengths.clear(values, fill=fill),
            lengths=lengths,
        )

    @classmethod
    def from_arrays(
        cls,
        values: typing.Sequence[jax.Array],
    ) -> typing.Self:
        """Pad a flat collection of `(T_i, *E)` arrays to a common time size."""
        sequences = tuple(cls.from_array(value) for value in values)
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
        event_shape = sequences[0].event_shape

        for sequence in sequences:
            batch.require_same_shape(
                batch_shape,
                sequence.batch_shape,
            )

            if sequence.event_shape != event_shape:
                raise ValueError('sequences must share an event shape')

        _, axis = batch.insertion(
            (len(sequences),),
            axis,
            len(batch_shape),
        )
        max_steps = max(sequence.max_steps for sequence in sequences)

        return cls._unchecked(
            values=jnp.stack(
                [
                    batch.pad_axis(
                        sequence.values,
                        max_steps,
                        axis=len(sequence.batch_shape),
                    )
                    for sequence in sequences
                ],
                axis=axis,
            ),
            lengths=Lengths(
                jnp.stack(
                    [sequence.lengths.values for sequence in sequences],
                    axis=axis,
                )
            ),
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

    def get(
        self,
        index: batch.SelT | None = None,
        *,
        squeeze: bool = True,
    ) -> jax.Array:
        """Retrieve one unpadded item."""
        selected = batch.get(self, index)

        result = selected.values[: int(selected.lengths.values)]

        return jnp.squeeze(result) if squeeze else result

    def items(self) -> typing.Iterator[tuple[tuple[int, ...], jax.Array]]:
        """Yield `(index, get(index))` pairs on the host in batch order.

        Indices are tuples with one integer per batch axis; the last axis
        varies fastest. An unbatched sequence yields index `()` once, and an
        empty batch yields no items. Arrays use the default squeezing of `get`.
        """
        for index in batch.iter_keys(self.batch_shape):
            yield index, self.get(index)

    def pad_time(self, max_steps: int) -> typing.Self:
        """Extend dense storage with zeros and false visibility, preserving lengths."""
        return self.__class__._unchecked(
            batch.pad_axis(self.values, max_steps, axis=len(self.batch_shape)),
            self.lengths,
        )

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
    def lengths(self) -> Lengths:
        """Observation/input valid-prefix lengths `(*B,)`."""
        return self.observations.lengths

    @property
    def lengths_spec(self) -> LengthsSpec:
        """Observation/input valid-prefix lengths `(*B,)`."""
        return LengthsSpec(self.observations.lengths, self.max_steps)

    @property
    def max_steps(self) -> int:
        return self.observations.max_steps

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
                lengths=self.observations.lengths,
            ),
            inputs=self.inputs,
            visible=self.visible,
        )

    def validate(self) -> None:
        """Validate temporal structure and public data policies on the host."""
        self.observations.validate()
        self.inputs.validate()
        if len(self.observations.event_shape) != 1:
            raise ValueError('observations must have shape (*B, T, D_y)')
        if len(self.inputs.event_shape) != 1:
            raise ValueError('inputs must have shape (*B, T, D_u)')
        if self.batch_shape != self.inputs.batch_shape:
            raise ValueError('observations and inputs must share their batch shape')
        if self.max_steps != self.inputs.max_steps:
            raise ValueError('observations and inputs must share their time dimension')
        if not jnp.array_equal(self.lengths.values, self.inputs.lengths.values):
            raise ValueError('observations and inputs must share their lengths')

        if self.visible.batch_shape != self.batch_shape:
            raise ValueError('visibility must share the dataset batch shape')
        if self.visible.max_steps != self.max_steps:
            raise ValueError('visibility must share the dataset time dimension')

        self.visible.validate(self.observations.values)

        valid = self.lengths.to_mask(self.max_steps).values
        if bool(jnp.any(self.visible.values & ~valid)):
            raise ValueError('visible must be false outside valid sequence prefixes')

        if not bool(jnp.all(jnp.isfinite(self.observations.values))):
            raise ValueError('observations must contain only finite values')
        if not bool(jnp.all(jnp.isfinite(self.inputs.values))):
            raise ValueError('inputs must contain only finite values')

    @classmethod
    def from_array(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Construct one unbatched trajectory from `(T, D_y)` observations."""
        sequence = Sequences.from_array(observations)
        return cls.from_padded(
            sequence.values,
            sequence.lengths,
            inputs=inputs,
            visible=visible,
        )

    @classmethod
    def from_arrays(
        cls,
        observations: typing.Sequence[jax.Array],
        *,
        inputs: typing.Sequence[jax.Array] | None = None,
        visible: typing.Sequence[jax.Array] | None = None,
    ) -> typing.Self:
        """Pad a flat collection of unequal-length observation trajectories."""
        observations = tuple(jnp.asarray(values) for values in observations)
        sequences = Sequences.from_arrays(observations)

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
            padded_inputs = Sequences.from_arrays(inputs).values

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
                [
                    batch.pad_axis(values, sequences.max_steps, axis=0)
                    for values in visible
                ],
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
        lengths: Lengths,
        *,
        inputs: jax.Array | None = None,
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Construct explicitly padded trajectories `(*B, T, D_y)`."""
        observation_sequences = Sequences.from_padded(observations, lengths)

        if inputs is None:
            inputs = jnp.empty(
                (*observation_sequences.values.shape[:-1], 0),
                dtype=observation_sequences.values.dtype,
            )

        input_sequences = Sequences.from_padded(inputs, observation_sequences.lengths)

        if visible is None:
            visibility = observation_sequences.lengths.to_mask(
                observation_sequences.max_steps
            )

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
                + ((0, observations.max_steps - data.max_steps),),
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

    def transition_inputs(self) -> jax.Array:
        """Inputs for `x[t-1] -> x[t]`; input zero is unused."""
        return self.inputs.values[..., 1:, :]

    def pad_time(self, max_steps: int) -> typing.Self:
        """Extend dense storage with zeros and false visibility, preserving lengths.

        The target must be a static Python integer. This transformation also
        supports working datasets with arbitrary model/sequence batch prefixes.
        """
        if max_steps < 0 or max_steps < self.max_steps:
            raise ValueError('max_steps must be nonnegative and cannot shrink time')

        if max_steps == self.max_steps:
            return self

        return self._unchecked(
            self.observations.pad_time(max_steps),
            self.inputs.pad_time(max_steps),
            self.visible.pad_time(max_steps),
        )

    def observation_weights(self) -> jax.Array:
        """Boolean weights combining structural validity and observation visibility."""
        return self.lengths.clear(self.visible.values)

    def weighted_observations(self) -> WeightedObservations:
        """Observation values with validity-aware weights."""
        return WeightedObservations(
            values=self.observations.values,
            weights=self.observation_weights(),
            lengths=self.lengths,
        )


class DatasetCollection(typing.NamedTuple):
    """
    Positional collection of independent recording sessions.

    Use `from_datasets` to normalize source sessions to flat batches `(N_s,)`
    and a common dense time horizon and input dimension. Observation dimensions
    and sequence counts may differ. Direct construction assumes normalized
    datasets; call `validate` explicitly to diagnose their layout.

    Session position is meaningful:

        datasets[i] <-> emissions[i]

    Sessions are not model batch dimensions. Collection structure is static
    PyTree structure; heterogeneous session arrays are never stacked directly.
    """

    datasets: tuple[Dataset, ...]

    @classmethod
    def from_array(
        cls,
        observations: jax.Array,
        *,
        inputs: jax.Array | None = None,
        visible: jax.Array | ArbitraryMask | None = None,
    ) -> typing.Self:
        """Construct a single-session unbatched trajectory from `(T, D_y)` observations."""
        data = Dataset.from_array(
            observations,
            inputs=inputs,
            visible=visible,
        )

        return cls.from_datasets((data,))

    @classmethod
    def from_arrays(
        cls,
        observations: typing.Sequence[jax.Array],
        *,
        inputs: typing.Sequence[jax.Array] | None = None,
        visible: typing.Sequence[jax.Array] | None = None,
    ) -> typing.Self:
        """Construct a single-session batched set of trajectories from many observations."""

        data = Dataset.from_arrays(
            observations,
            inputs=inputs,
            visible=visible,
        )

        return cls.from_datasets((data,))

    @classmethod
    def from_datasets(cls, datasets: typing.Sequence[Dataset]) -> typing.Self:
        """Promote single trajectories and pad sessions once, outside JIT."""
        datasets = tuple(datasets)
        if not datasets:
            raise ValueError('datasets must contain at least one session')
        normalized = []
        for session, data in enumerate(datasets):
            if len(data.batch_shape) > 1:
                raise ValueError(
                    f'session {session} must have at most one source batch axis; '
                    'flatten sequence axes explicitly before collection construction'
                )
            normalized.append(data if data.batch_shape else data.broadcast((1,)))
        input_dims = tuple(data.input_dim for data in normalized)
        if any(dim != input_dims[0] for dim in input_dims[1:]):
            raise ValueError(
                f'all sessions must share their input dimension; got {input_dims}'
            )
        max_steps = max(data.max_steps for data in normalized)
        collection = cls(tuple(data.pad_time(max_steps) for data in normalized))
        collection.validate()
        return collection

    @property
    def dtype(self) -> jnp.dtype:
        """Shared observation dtype across sessions."""
        return jnp.result_type(*(dataset.dtype for dataset in self.datasets))

    def weighted_observations(self) -> tuple[WeightedObservations, ...]:
        """Observation values with validity-aware weights."""
        return tuple(dataset.weighted_observations() for dataset in self.datasets)

    def get(
        self,
        session: int | None = None,
    ) -> Dataset:
        """
        Return one session.

        If `session` is omitted, require a singleton collection. This keeps
        ordinary single-session APIs ergonomic while making ambiguity explicit
        for multi-session models.
        """
        if session is None:
            return self.single()

        return self.datasets[session]

    def single(self) -> Dataset:
        """Require exactly one session and return it."""
        if len(self.datasets) != 1:
            raise ValueError(
                f'operation requires exactly one session; got {len(self.datasets)}'
            )

        return self.datasets[0]

    def validate(self) -> None:
        """Validate collection-level structural invariants."""
        if not self.datasets:
            raise ValueError('datasets must contain at least one session')

        for session, data in enumerate(self.datasets):
            if len(data.batch_shape) != 1:
                raise ValueError(
                    f'session {session} must have one source sequence axis; '
                    'use from_datasets and flatten sequence axes explicitly'
                )
            if data.max_steps != self.max_steps:
                raise ValueError('all sessions must share max_steps; use from_datasets')

        input_dims = tuple(data.input_dim for data in self.datasets)

        if any(input_dim != input_dims[0] for input_dim in input_dims[1:]):
            raise ValueError(
                f'all sessions must share their input dimension; got {input_dims}'
            )

    @property
    def input_dim(self) -> int:
        """Shared input dimension."""
        return self.datasets[0].input_dim

    @property
    def observation_dims(self) -> tuple[int, ...]:
        """Observation dimension of every session."""
        return tuple(data.observation_dim for data in self.datasets)

    @property
    def partition(self) -> batch.Partition:
        return batch.Partition(tuple(data.batch_shape[0] for data in self.datasets))

    @property
    def max_steps(self) -> int:
        """Common dense temporal horizon of normalized sessions."""
        return self.datasets[0].max_steps

    def pool_inputs(self) -> Sequences:
        return batch.concatenate([dataset.inputs for dataset in self.datasets])

    def pool_lengths_spec(self) -> LengthsSpec:
        return LengthsSpec(
            lengths=batch.concatenate(
                tuple(dataset.lengths for dataset in self.datasets),
                axis=0,
            ),
            max_steps=self.max_steps,
        )
