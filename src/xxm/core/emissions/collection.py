"""Positional collections of session-specific emission models."""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp

from xxm.core import batch
from xxm.core.affine import Affine
from xxm.core.chains.discrete import DiscretePotential
from xxm.core.chains.gaussian import GaussianPotential
from xxm.core.data import WeightedObservations
from xxm.core.emissions.continuous import Emissions as ContinuousEmission
from xxm.core.emissions.continuous import LaplaceEmissionsT, QuadraticEmissionsT
from xxm.core.emissions.discrete import Emissions as DiscreteEmissions
from xxm.core.posteriors import ContinuousPosterior, DiscretePosterior

EmissionT = typing.TypeVar('EmissionT', bound=batch.Batchable)
DataT = typing.TypeVar('DataT', bound=batch.Batchable)
PosteriorT = typing.TypeVar('PosteriorT')
ResultT = typing.TypeVar('ResultT')
PotentialT = typing.TypeVar('PotentialT', bound=batch.Batchable)

ContinuousEmissionT = typing.TypeVar('ContinuousEmissionT', bound=ContinuousEmission)
DiscreteEmissionT = typing.TypeVar('DiscreteEmissionT', bound=DiscreteEmissions)

WOCollection = tuple[WeightedObservations, ...]


def _partition(data: tuple[batch.Batchable, ...]) -> batch.Partition:
    return batch.Partition(tuple(item.batch_shape[0] for item in data))


class EmissionsCollection(
    typing.NamedTuple,
    typing.Generic[EmissionT, DataT, PosteriorT],
):
    """
    Positional collection of session-specific emission models.

    Emissions share a model batch shape but may have different
    observation dimensions and intrinsic parameter shapes.

    Session position is meaningful:
        emissions[i] <-> datasets[i]
    """

    emissions: tuple[EmissionT, ...]

    def _dispatch(
        self,
        data: tuple[DataT, ...],
        function: typing.Callable[[EmissionT, DataT, int], ResultT],
    ) -> tuple[ResultT, ...]:
        """Evaluate each emission on Cartesian-broadcast session data."""

        results = []

        for i, (emission, item) in enumerate(zip(self.emissions, data, strict=True)):
            emission, item = batch.cartesian_broadcast(emission, item)
            results.append(function(emission, item, i))

        return tuple(results)

    def _compute_potential(
        self,
        data: tuple[DataT, ...],
        function: typing.Callable[[EmissionT, DataT], PotentialT],
    ) -> PotentialT:
        """Compute and concatenate session-specific potentials."""

        potentials = self._dispatch(
            data,
            lambda emission, item, _: function(emission, item),
        )

        return batch.concatenate(
            potentials,
            axis=len(self.batch_shape),
        )

    def _fit_params(
        self,
        posterior: PosteriorT,
        data: tuple[DataT, ...],
        function: typing.Callable[[EmissionT, PosteriorT, DataT], EmissionT],
    ) -> typing.Self:
        """Split pooled posterior and fit each emission across its sequences."""

        posteriors = _partition(data).split(posterior, axis=len(self.batch_shape))

        fitted = []

        for emission, q, item in zip(
            self.emissions,
            posteriors,
            data,
            strict=True,
        ):
            item = batch.broadcast(item, self.batch_shape, axis=0)
            fitted.append(function(emission, q, item))

        return type(self)(tuple(fitted))

    def get(self, session: int | None = None) -> EmissionT:
        """Return a session emission, requiring a singleton if omitted."""
        if session is None:
            return self.single()
        return self.emissions[session]

    def single(self) -> EmissionT:
        """Require exactly one session."""
        if len(self.emissions) != 1:
            raise ValueError(
                f'operation requires exactly one session; got {len(self.emissions)}'
            )

        return self.emissions[0]

    def map(
        self,
        function: typing.Callable[[EmissionT], EmissionT],
    ) -> typing.Self:
        """Apply a type-preserving transformation to every emission."""
        return type(self)(tuple(function(emission) for emission in self.emissions))

    def validate(self) -> None:
        """Validate collection-level structural invariants."""
        if not self.emissions:
            raise ValueError('emissions must contain at least one session')

        batch.require_same_shape(*(emission.batch_shape for emission in self.emissions))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Shared model batch shape."""
        return self.emissions[0].batch_shape

    def select(self, index: batch.SelT) -> typing.Self:
        return batch.select(self, index)

    def broadcast(self, shape, axis: int = 0) -> typing.Self:
        return batch.broadcast(self, shape, axis=axis)

    def squeeze(self, axis=None) -> typing.Self:
        return batch.squeeze(self, axis=axis)

    def permute(self, permutation, axis: int = 0) -> typing.Self:
        return batch.permute(self, permutation, axis=axis)

    def move_axis(self, source: int, destination: int) -> typing.Self:
        return batch.move_axis(self, source, destination)


class ContinuousEmissionsCollection(
    EmissionsCollection[
        ContinuousEmissionT,
        WeightedObservations,
        ContinuousPosterior,
    ],
    typing.Generic[ContinuousEmissionT],
):
    """Session-specific emissions for continuous latent variables."""

    def fit_params(
        self,
        posterior: ContinuousPosterior,
        observations: WOCollection,
    ) -> typing.Self:
        return self._fit_params(
            posterior,
            observations,
            lambda emission, q, obs: emission.fit_params(q, obs),
        )

    def compute_potential(
        self: ContinuousEmissionsCollection[QuadraticEmissionsT],
        observations: WOCollection,
    ) -> GaussianPotential:
        return self._compute_potential(
            observations,
            lambda emission, obs: emission.compute_potential(obs),
        )

    def compute_local_potential(
        self: ContinuousEmissionsCollection[LaplaceEmissionsT],
        observations: WOCollection,
        latents: jax.Array,
    ) -> GaussianPotential:
        latents_split = _partition(observations).split(
            latents,
            axis=len(self.batch_shape),
        )

        potentials = self._dispatch(
            observations,
            lambda emission, obs, i: emission.compute_local_potential(
                obs, latents_split[i]
            ),
        )

        return batch.concatenate(
            potentials,
            axis=len(self.batch_shape),
        )

    def log_likelihood(
        self,
        observations: WOCollection,
        latents: jax.Array,
    ) -> jax.Array:
        latents_split = _partition(observations).split(
            latents,
            axis=len(self.batch_shape),
        )

        values = self._dispatch(
            observations,
            lambda emission, obs, i: emission.log_likelihood(obs, latents_split[i]),
        )

        return jnp.concatenate(
            values,
            axis=len(self.batch_shape),
        )

    def expected_log_likelihood(
        self,
        observations: WOCollection,
        posterior: ContinuousPosterior,
    ) -> jax.Array:
        posterior_split = _partition(observations).split(
            posterior,
            axis=len(self.batch_shape),
        )

        values = self._dispatch(
            observations,
            lambda emission, obs, i: emission.expected_log_likelihood(
                obs, posterior_split[i]
            ),
        )

        return jnp.concatenate(
            values,
            axis=len(self.batch_shape),
        )

    def compose_input(self, alignment: Affine) -> typing.Self:
        """Transform the latent coordinates of every emission."""
        return self.map(lambda emission: emission.compose_input(alignment))


class DiscreteEmissionsCollection(
    EmissionsCollection[
        DiscreteEmissionT,
        WeightedObservations,
        DiscretePosterior,
    ],
    typing.Generic[DiscreteEmissionT],
):
    """Session-specific emissions for discrete latent states."""

    def fit_params(
        self,
        posterior: DiscretePosterior,
        observations: WOCollection,
    ) -> typing.Self:
        return self._fit_params(
            posterior,
            observations,
            lambda emission, q, obs: emission.fit_params(q, obs),
        )

    def compute_potential(self, observations: WOCollection) -> DiscretePotential:
        return self._compute_potential(
            observations,
            lambda emission, obs: emission.compute_potential(obs),
        )

    def log_likelihoods(self, observations: WOCollection) -> jax.Array:
        values = self._dispatch(
            observations,
            lambda emission, obs, _: emission.log_likelihoods(obs),
        )

        return jnp.concatenate(values, axis=len(self.batch_shape))

    def permute_states(self, permutation: jax.Array) -> typing.Self:
        """Relabel state-specific emission parameters."""
        return self.map(lambda emission: emission.permute_states(permutation))
