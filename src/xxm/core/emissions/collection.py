"""Positional collections of session-specific emission models."""

from __future__ import annotations

import typing

import jax

from xxm.core import batch
from xxm.core.affine import Affine

EmissionT = typing.TypeVar(
    'EmissionT',
    bound=batch.Batchable,
)
MappedEmissionT = typing.TypeVar('MappedEmissionT', bound=batch.Batchable)
ValueT = typing.TypeVar('ValueT')


class ComposableEmission(batch.Batchable, typing.Protocol):
    def compose_input(self, alignment: Affine) -> typing.Self: ...


class StateEmission(batch.Batchable, typing.Protocol):
    def permute_states(self, permutation: jax.Array) -> typing.Self: ...


ComposableEmissionT = typing.TypeVar('ComposableEmissionT', bound=ComposableEmission)
StateEmissionT = typing.TypeVar('StateEmissionT', bound=StateEmission)


class EmissionsCollection(
    typing.NamedTuple,
    typing.Generic[EmissionT],
):
    """
    Positional collection of session-specific emission models.

    Every emission describes one observation space for the same shared latent
    or state model. Observation dimensions and intrinsic emission parameter
    shapes may therefore differ across sessions.

    All emissions must share the same model batch shape. Direct construction
    assembles the tuple without validation; use `validate` at host boundaries.

    Session position is meaningful:

        emissions[i] <-> datasets[i]
    """

    emissions: tuple[EmissionT, ...]

    def get(
        self,
        session: int | None = None,
    ) -> EmissionT:
        """
        Return one session's emission model.

        If `session` is omitted, require a singleton collection.
        """
        if session is None:
            return self.single()

        return self.emissions[session]

    def single(self) -> EmissionT:
        """Require exactly one session and return its emission model."""
        if len(self.emissions) != 1:
            raise ValueError(
                f'operation requires exactly one session; got {len(self.emissions)}'
            )

        return self.emissions[0]

    def map(
        self,
        function: typing.Callable[[EmissionT], MappedEmissionT],
    ) -> EmissionsCollection[MappedEmissionT]:
        """Transform every session emission, preserving positional order."""
        return EmissionsCollection(
            tuple(function(emission) for emission in self.emissions)
        )

    def compose_input(
        self: EmissionsCollection[ComposableEmissionT],
        alignment: Affine,
    ) -> EmissionsCollection[ComposableEmissionT]:
        """Align the latent coordinates of every compatible emission."""
        return self.map(lambda emission: emission.compose_input(alignment))

    def permute_states(
        self: EmissionsCollection[StateEmissionT],
        permutation: jax.Array,
    ) -> EmissionsCollection[StateEmissionT]:
        """Relabel every state-dependent emission."""
        return self.map(lambda emission: emission.permute_states(permutation))

    def validate(self) -> None:
        """Validate collection-level structural invariants."""
        if not self.emissions:
            raise ValueError('emissions must contain at least one session')

        batch.require_same_shape(*(emission.batch_shape for emission in self.emissions))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shared model batch shape."""
        return self.emissions[0].batch_shape

    def select(
        self,
        index: batch.SelT,
    ) -> typing.Self:
        """Select model batch dimensions from every session emission."""
        return batch.select(self, index)

    def broadcast(
        self,
        shape,
        axis: int = 0,
    ) -> typing.Self:
        """Insert replicated model batch dimensions in every emission."""
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(
        self,
        axis=None,
    ) -> typing.Self:
        """Remove singleton model batch dimensions from every emission."""
        return batch.squeeze(self, axis=axis)

    def permute(
        self,
        permutation,
        axis: int = 0,
    ) -> typing.Self:
        """Reorder one model batch axis in every emission."""
        return batch.permute(self, permutation, axis=axis)

    def move_axis(
        self,
        source: int,
        destination: int,
    ) -> typing.Self:
        """Move one model batch axis in every emission."""
        return batch.move_axis(self, source, destination)
